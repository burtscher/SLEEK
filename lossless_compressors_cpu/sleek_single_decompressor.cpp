/*
This file is part of SLEEK, a set of ultra-fast lossless and guaranteed-error-bounded lossy main-memory compression algorithms for floating-point data on GPUs.

BSD 3-Clause License

Copyright (c) 2026, Anju Mongandampulath Akathoott, Andrew Rodriguez, and Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/SLEEK/.

Publication: This work is described in detail in the following paper.
Anju Mongandampulath Akathoott, Andrew Rodriguez, and Martin Burtscher. "SLEEK: Compressing Memory Copies for Floating-Point Data on GPUs." Proceedings of the 40th IEEE International Parallel and Distributed Processing Symposium (IPDPS'26). May 2026.

Sponsor: This material is based upon work supported by the U.S. National Science Foundation under Grant Number 2403380 and by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under Award Number DE-SC0022223.
*/


#define NDEBUG

using byte = unsigned char;
using type_u = unsigned int;

#include <limits>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <sys/time.h>
#include <string>

static const int bytes_in_type = 4;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]


struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


static void h_decode(const byte* const __restrict__ input, byte* const __restrict__ output, long long& outsize)
{
  // input header
  long long* const head_in = (long long*)input;
  outsize = head_in[0];

  // initialize
  const long long chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];
  long long* const start = new long long [chunks];

  // convert chunk sizes into starting positions
  long long pfs = 0;
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    start[chunkID] = pfs;
    pfs += (long long)size_in[chunkID];
  }

  const int TB = sizeof(type_u) * 8;  // number of bits in type_u
  const int size = CS / sizeof(type_u);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(sizeof(type_u) == bytes_in_type);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<type_u>::value);
  
  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1)
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    long long chunk2 [CS / sizeof(long long)];
    byte* out = (byte*)chunk2;
    const long long base = chunkID * CS;
    const int osize = (int)std::min((long long)CS, outsize - base);
    int csize = size_in[chunkID];
    if (csize == osize) {
      // simply copy
      memcpy(&output[base], &data_in[start[chunkID]], osize);
    } else {
      byte* in = &data_in[start[chunkID]];
      // decode data values
      int startPos = 0;
      const type_u* const in_t = (type_u*)&in[SC];
      type_u* const out_t = (type_u*)out;
      for (int i = 0; i < SC; i++) {
        const int logn = in[i];
        const int beg = i * chunksize;
        const int end = beg + chunksize;
        if (logn == 0) {
          for (int j = beg; j < end; j++) {
            out_t[j] = 0;
          }
        } else if (logn == TB) {
          const int offs = startPos / TB - beg;
          for (int j = beg; j < end; j++) {
            type_u val = in_t[offs + j];
            if constexpr (TB == 32) {
              val = (val >> 1) ^ (((int)(val << 31)) >> 31);  // iTCMS
              if ((val & 0xff00'0000) != 0) {
                if (val >= 0x8000'0000) {
                  val += 0x0100'0000;
                }
                val += 0x8000'0000;
              }
              val = (val << 31) | (val >> 1);
            } else {
              val = (val >> 1) ^ (((long long)(val << 63)) >> 63);  // iTCMS
              if ((val & 0xffe0'0000'0000'0000) != 0) {
                if (val >= 0x8000'0000'0000'0000) {
                  val += 0x0020'0000'0000'0000;
                }
                val += 0x8000'0000'0000'0000;
              }
              val = (val << 63) | (val >> 1);
            }
            out_t[j] = val;
          }
        } else {
          int loc = startPos;
          const type_u mask = (logn == 64) ? (~0ULL) : ((1ULL << logn) - 1);
          for (int j = beg; j < end; j++) {
            const int pos = loc / TB;
            const int shift = loc % TB;
            type_u res = in_t[pos] >> shift;
            if (TB - shift < logn) {
              res |= in_t[pos + 1] << (TB - shift);
            }
            loc += logn;
            type_u val = res & mask;
            if constexpr (TB == 32) {
              val = (val >> 1) ^ (((int)(val << 31)) >> 31);  // iTCMS
              if ((val & 0xff00'0000) != 0) {
                if (val >= 0x8000'0000) {
                  val += 0x0100'0000;
                }
                val += 0x8000'0000;
              }
              val = (val << 31) | (val >> 1);
            } else {
              val = (val >> 1) ^ (((long long)(val << 63)) >> 63);  // iTCMS
              if ((val & 0xffe0'0000'0000'0000) != 0) {
                if (val >= 0x8000'0000'0000'0000) {
                  val += 0x0020'0000'0000'0000;
                }
                val += 0x8000'0000'0000'0000;
              }
              val = (val << 63) | (val >> 1);
            }
            out_t[j] = val;
          }
        }
        startPos += chunksize * logn;
      }

      // read header info
      csize = *(short*)&in[csize - 2];

      if (csize != osize) {fprintf(stderr, "ERROR: csize %d does not match osize %d in chunk %lld\n\n", csize, osize, chunkID);}
      memcpy(&output[base], out, csize);
    }
  }

  // finish
  delete [] start;
}


int main(int argc, char* argv [])
{
  printf("CPU SLEEK 1.0: single-precision lossless decompressor\n");
  printf("Copyright 2026 Texas State University\n\n");

  // read input from file
  if (argc < 3) {printf("USAGE: %s compressed_file_name decompressed_file_name\n\n", argv[0]); return -1;}

  // read input file
  FILE* const fin = fopen(argv[1], "rb");
  long long pre_size = 0;
  const long long pre_val = fread(&pre_size, sizeof(pre_size), 1, fin); assert(pre_val == sizeof(pre_size));
  fseek(fin, 0, SEEK_END);
  const long long hencsize = ftell(fin);  assert(hencsize > 0);
  byte* const hencoded = new byte [std::max(pre_size, hencsize)];
  fseek(fin, 0, SEEK_SET);
  const long long insize = fread(hencoded, 1, hencsize, fin);  assert(insize == hencsize);
  fclose(fin);
  printf("encoded size: %lld bytes\n", insize);

  #if defined(ARTIFACT)
    // Check if the third argument is "y" to enable performance analysis
    char* perf_str = argv[3];
    bool perf = false;
    if (perf_str != nullptr && strcmp(perf_str, "y") == 0) {
      perf = true;
    } else if (perf_str != nullptr && strcmp(perf_str, "y") != 0) {
      fprintf(stderr, "ERROR: Invalid argument. Use 'y' or nothing.\n");
      return -1;
    }
  #endif

  // allocate CPU memory
  byte* hdecoded = new byte [pre_size];
  long long hdecsize = 0;

  #if defined(ARTIFACT)
    if (perf) {
      // warm up
      byte* dummy = new byte [pre_size];
      long long dummy_size = 0;
      h_decode(hencoded, dummy, dummy_size);
      delete [] dummy;
    }
  #endif

  // time CPU decoding
  CPUTimer htimer;
  htimer.start();
  h_decode(hencoded, hdecoded, hdecsize);
  double hruntime = htimer.stop();

  printf("decoded size: %lld bytes\n", hdecsize);
  const float CR = (100.0 * insize) / hdecsize;
  printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

  #if defined(ARTIFACT)
    if (perf) {
      printf("decoding time: %.6f s\n", hruntime);
      double hthroughput = hdecsize * 0.000000001 / hruntime;
      printf("decoding throughput: %8.3f Gbytes/s\n", hthroughput);
    }
  #endif
  
  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(hdecoded, 1, hdecsize, fout);
  fclose(fout);

  delete [] hencoded;
  delete [] hdecoded;
  return 0;
}
