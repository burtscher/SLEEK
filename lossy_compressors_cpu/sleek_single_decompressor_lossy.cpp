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
using type_f = float;

#define to_float(val) \
    (sizeof(type_f) == sizeof(float) ? std::stof(val) : std::stod(val))

#include <climits>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <sys/time.h>
#include <string>
#include "MAXABS_f32.h"

const type_u mantissabits = 23;
static const int bytes_in_type = 4;
const int lShift = sizeof(type_u) * 8 - 1;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]


static inline type_u deQuantize(const type_u enc, const int thr_e, const int offs)
{
  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits
  int dec = 0;  // default value is 0
  if (enc != 0) {
    const int abs = (enc + 1) >> 1;  // absolute value
    if (abs >= (1 << m)) {  // above threshold
      dec = abs + offs;  // decode losslessly
    } else if (abs > 0) {  // non-zero lossy case
      const int shift = __builtin_clz(abs) - (31 - m);  // compute shift amount
      dec = abs << shift;  // shift to normalized position
      dec &= (1 << m) - 1;  // remove implied 1
      dec |= (thr_e - shift) << m;  // insert biased exponent
    }
    dec |= enc << (e + m);  // insert sign bit
  }
  return dec;
}


static inline void h_iSLEEK(int& csize, byte in [CS], byte out [CS], const int thr_e, const int offs)
{
  const int TB = sizeof(type_u) * 8;  // number of bits in type_u
  const int size = CS / sizeof(type_u);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(sizeof(type_u) == bytes_in_type);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<type_u>::value);

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
        const type_u val = in_t[offs + j];
        out_t[j] = deQuantize(val, thr_e, offs);
        
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
        const type_u val = res & mask;
        out_t[j] = deQuantize(val, thr_e, offs);
      }
    }
    startPos += chunksize * logn;
  }

  // read header info
  csize = *(short*)&in[csize - 2];
}


struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


static void h_decode(const byte* const __restrict__ input, byte* const __restrict__ output, long long& outsize, const int thr_e, const int offs)
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

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1) default(none) shared(chunks, outsize, size_in, data_in, start, output, thr_e, offs, stderr) 
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk1 [CS / sizeof(long long)];
    long long chunk2 [CS / sizeof(long long)];
    byte* in = (byte*)chunk1;
    byte* out = (byte*)chunk2;
    const long long base = chunkID * CS;
    const int osize = (int)std::min((long long)CS, outsize - base);
    int csize = size_in[chunkID];
    if (csize == osize) {
      // simply copy
      memcpy(&output[base], &data_in[start[chunkID]], osize);
    } else {
      // decompress
      memcpy(in, &data_in[start[chunkID]], csize);
      h_iSLEEK(csize, in, out, thr_e, offs);
      if (csize != osize) {fprintf(stderr, "ERROR: csize %d does not match osize %d in chunk %lld\n\n", csize, osize, chunkID);}
      memcpy(&output[base], out, csize);
    }
  }

  // finish
  delete [] start;
}


#if defined(ARTIFACT)
  static inline type_f computeNOAeb(const type_f* const input, const long long size, const type_f eb_param)
  {
    type_f min = input[0], max = input[0];
    #pragma omp parallel for default(none) shared(size, input) reduction(max: max) reduction(min: min)
    for (long long i = 1; i < size; i++) {
      const type_f val = input[i];
      if (val < min)
        min = val;
      else if (val > max)
        max = val;
    }
    return (max - min) * eb_param;
  }
#endif


int main(int argc, char* argv [])
{
  printf("CPU SLEEK 1.0: single-precision lossy decompressor\n");
  printf("Copyright 2026 Texas State University\n\n");

  #if defined(ARTIFACT)

    const int expected_argc = 5;
    const int idx_input = 1;
    const int idx_output = 2;
    const int idx_original = 3;
    const int idx_eb = 4;
    const int idx_perf = 5;

    // read input from file
    if (argc < expected_argc) {printf("USAGE: %s compressed_file_name decompressed_file_name orig_file_name errorbound\n\n", argv[0]); return -1;}

    // read original file
    FILE* const fin_orig = fopen(argv[idx_original], "rb");
    fseek(fin_orig, 0, SEEK_END);
    const long long fsize_orig = ftell(fin_orig);
    if (fsize_orig <= 0) {fprintf(stderr, "ERROR: original input file too small\n\n"); return -1;}
    byte* const input_orig = new byte [(fsize_orig + CS - 1) / CS * CS];
    fseek(fin_orig, 0, SEEK_SET);
    const long long insize_orig = fread(input_orig, 1, fsize_orig, fin_orig);  assert(insize_orig == fsize_orig);
    fclose(fin_orig);

    // compute input-specific eb
    const type_f eb_parameter = to_float(argv[idx_eb]);
    type_f* f_input = (type_f*)input_orig;
    const type_f eb = computeNOAeb(f_input, insize_orig / sizeof(type_f), eb_parameter);

  #else

    const int expected_argc = 4;
    const int idx_input = 1;
    const int idx_output = 2;
    const int idx_eb = 3;
    const int idx_perf = 4;

    // read input from file
    if (argc < expected_argc) {printf("USAGE: %s compressed_file_name decompressed_file_name errorbound\n\n", argv[0]); return -1;}

    // get eb
    const type_f eb = to_float(argv[idx_eb]);

  #endif

  // eb variables
  const int e = 8;  // exponent bits
  const int m = 23;  // mantissa bits
  const int eb_e = (*((int*)&eb) >> m) & ((1 << e) - 1);  // extract biased exponent
  const int thr_e = eb_e + (m + 1);  // biased exponent of threshold
  const int offs = (thr_e << m) - (1 << m);  // offset for lossless encoding
  if (thr_e >= (1 << e) - 1) {fprintf(stderr, "iQUANT_IABS_0_f32: ERROR: error_bound is too large\n"); return -1;}

  // read input file
  FILE* const fin = fopen(argv[idx_input], "rb");
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
    char* perf_str = argv[idx_perf];
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
      h_decode(hencoded, dummy, dummy_size, thr_e, offs);
      delete [] dummy;
    }
  #endif

  // time CPU decoding
  CPUTimer htimer;
  htimer.start();
  h_decode(hencoded, hdecoded, hdecsize, thr_e, offs);
  double hruntime = htimer.stop();

  #if defined(ARTIFACT)
    // verify
    MAXABS_f32(insize_orig, hdecoded, input_orig, eb);
  #endif
 
  printf("decoded size: %lld bytes\n", hdecsize);
  const type_f CR = (100.0 * insize) / hdecsize;
  printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

  #if defined(ARTIFACT)
    if (perf) {
      printf("decoding time: %.6f s\n", hruntime);
      double hthroughput = hdecsize * 0.000000001 / hruntime;
      printf("decoding throughput: %8.3f Gbytes/s\n", hthroughput);
    }
  #endif

  // write to file
  FILE* const fout = fopen(argv[idx_output], "wb");
  fwrite(hdecoded, 1, hdecsize, fout);
  fclose(fout);

  delete [] hencoded;
  delete [] hdecoded;
  return 0;
}
