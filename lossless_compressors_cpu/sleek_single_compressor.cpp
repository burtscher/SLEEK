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

#define clz(val) \
    (sizeof(type_u) == sizeof(unsigned int) ? __builtin_clz(static_cast<unsigned int>(val)) : \
     __builtin_clzll(static_cast<unsigned long long>(val)))

#include <cassert>
#include <cstring>
#include <cstdio>
#include <algorithm>
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


static void h_encode(byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, long long& outsize)
{
  // initialize
  const long long chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];
  long long* const carry = new long long [chunks];
  memset(carry, 0, chunks * sizeof(long long));

  // encode chunk
  const int TB = sizeof(type_u) * 8;  // number of bits in type_u
  const int size = CS / sizeof(type_u);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;  // size of subchunk in words

  static_assert(sizeof(type_u) == bytes_in_type);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<type_u>::value);

  // process chunks in parallel
#pragma omp parallel for schedule(dynamic, 1)
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    type_u tmp_buffer_t [CS / sizeof(type_u)]; 
    byte out [CS / sizeof(byte)];
    const long long base = chunkID * CS;
    const int osize = (int)std::min((long long)CS, insize - base);
    byte* const in = &input[base];

    // clear unused part of input buffer
    if (osize < CS) {
      memset(in + osize, 0, CS - osize);
    }

    type_u* const in_t = (type_u*)in;

    // determine bits needed for each subchunk
    int bits = 0;
    for (int i = 0; i < SC; i++) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;

      type_u max_val = 0;
      for (int j = beg; j < end; j++) {
        type_u val = in_t[j];
        if constexpr (TB == 32) {
          val = ((val << 1) | (val >> 31));
          if ((val & 0xff00'0000) != 0) {
            val -= 0x8000'0000;
            if (((val & 0xff00'0000) == 0) || (val >= 0x8000'0000)) {
              val -= 0x0100'0000;
            }
          }
          val = (val << 1) ^ (((int)val) >> 31);  // TCMS
        } else {
          val = ((val << 1) | (val >> 63));
          if ((val & 0xffe0'0000'0000'0000) != 0) {
            val -= 0x8000'0000'0000'0000;
            if (((val & 0xffe0'0000'0000'0000) == 0) || (val >= 0x8000'0000'0000'0000)) {
              val -= 0x0020'0000'0000'0000;
            }
          }
          val = (val << 1) ^ (((long long)val) >> 63);  // TCMS
        }
        tmp_buffer_t[j] = val;
        max_val = std::max(max_val, val);
      }

      int cnt = TB;
      if (max_val != 0) {
        cnt = clz(max_val);
      }
      int ln = TB - cnt;  // logn value for subchunk
      bits += ln * chunksize;
      out[i] = ln;
    }

    const int newsize = (SC * 8 + bits + 16) / 8;

    // handle carry
    long long offs = 0LL;
    if (chunkID > 0) {
      do {
#pragma omp atomic read
        offs = carry[chunkID - 1];
      } while (offs == 0);
#pragma omp flush
    }


    // check if encoded data fits
    if ((newsize < CS) && (newsize < osize)) {
      // store carry of compressed data
#pragma omp atomic write
      carry[chunkID] = (offs + (long long)newsize);
      size_out[chunkID] = newsize;

      // clear out buffer
      type_u* const out_t = (type_u*)&out[SC];
      memset(out_t, 0, bits / 8);

      // encode data values
      int startPos = 0;
      for (int i = 0; i < SC; i++) {
        const int logn = out[i];
        if (logn > 0) {
          const int beg = i * chunksize;
          const int end = beg + chunksize;
          if (logn == TB) {
            const int offs = startPos / TB - beg;
            for (int j = beg; j < end; j++) {
              out_t[offs + j] = tmp_buffer_t[j];
            }
          } else {
            int loc = startPos;
            for (int j = beg; j < end; j++) {
              const type_u val = tmp_buffer_t[j];
              const int pos = loc / TB;
              const int shift = loc % TB;
              out_t[pos] |= val << shift;
              if (TB - shift < logn) {
                out_t[pos + 1] = val >> (TB - shift);
              }
              loc += logn;
            }
          }
        }
        startPos += chunksize * logn;
      }

      // output header info
      *(short*)&out[newsize - 2] = osize;

      memcpy(&data_out[offs], out, newsize);
    } else {
      // store original data
#pragma omp atomic write
      carry[chunkID] = (offs + (long long)osize);
      size_out[chunkID] = osize;
      memcpy(&data_out[offs], &input[base], osize);
    }
  }

  // output header
  head_out[0] = insize;

  // finish
  outsize = &data_out[carry[chunks - 1]] - output;
  delete [] carry;
}


int main(int argc, char* argv [])
{
  printf("CPU SLEEK 1.0: single-precision lossless compressor\n");
  printf("Copyright 2026 Texas State University\n\n");

  // read input from file
  if (argc < 3) {printf("USAGE: %s input_file_name compressed_file_name\n\n", argv[0]); return -1;}
  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {fprintf(stderr, "ERROR: input file too small\n\n"); return -1;}
  byte* const input = new byte [(fsize + CS - 1) / CS * CS];
  fseek(fin, 0, SEEK_SET);
  const long long insize = fread(input, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);
  printf("original size: %lld bytes\n", insize);

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
  const long long chunks = (insize + CS - 1) / CS;  // round up
  const long long maxsize = 3 * sizeof(int) + chunks * sizeof(short) + chunks * CS;
  byte* const hencoded = new byte [maxsize];
  long long hencsize = 0;

  #if defined(ARTIFACT)
    if (perf) {
      // warm up
      byte* dummy = new byte [(insize + CS - 1) / CS * CS];
      std::copy(input, input + insize, dummy);
      h_encode(dummy, insize, hencoded, hencsize);
      delete [] dummy;
    }
  #endif

  CPUTimer htimer;
  htimer.start();
  h_encode(input, insize, hencoded, hencsize);
  double hruntime = htimer.stop();

  printf("encoded size: %lld bytes\n", hencsize);
  const float CR = (100.0 * hencsize) / insize;
  printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

  #if defined(ARTIFACT)
    if (perf) {
      printf("encoding time: %.6f s\n", hruntime);
      double hthroughput = insize * 0.000000001 / hruntime;
      printf("encoding throughput: %8.3f Gbytes/s\n", hthroughput);
    }
  #endif

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(hencoded, 1, hencsize, fout);
  fclose(fout);

  delete [] input;
  delete [] hencoded;
  return 0;
}
