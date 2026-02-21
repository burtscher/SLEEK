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
using type_u = unsigned long long;
using type_f = double;

#define clz(val) \
    (sizeof(type_u) == sizeof(unsigned int) ? __builtin_clz(static_cast<unsigned int>(val)) : \
     __builtin_clzll(static_cast<unsigned long long>(val)))

#define to_float(val) \
    (sizeof(type_f) == sizeof(float) ? std::stof(val) : std::stod(val))

#include <cassert>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <sys/time.h>
#include <climits>
#include <stdexcept>
#include <cmath>
#include <string>

const type_u mantissabits = 52;
const type_u remove_sign_bit_mask = 0x7fff'ffff'ffff'ffff;
static const int bytes_in_type = 8;
const int rShift = (sizeof(type_u) * 8 - 1);
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]


struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


static inline type_u quantize(const type_u val, const int eb_e, const int thr_e, const long long offs)
{
  const int e = 11;  // exponent bits
  const int m = 52;  // mantissa bits
  const long long abs = val & ((1ULL << (e + m)) - 1ULL);  // compute absolute value
  const int val_e = abs >> m;  // extract exponent
  long long enc = 0LL;  // default value is 0
  if (val_e >= thr_e) {  // at or above threshold
    enc = abs - offs;  // lossless encoding
  } else if (val_e >= eb_e) {  // lossy encoding
    long long mant = val & ((1LL << m) - 1LL);  // extract mantissa
    const int shift = thr_e - val_e;  // bias cancels out
    mant |= 1LL << m;  // insert implicit 1
    mant += 1LL << (shift - 1);  // round to nearest, ties round away from zero
    enc = mant >> shift;  // shift out unnecessary bits
  }
  enc = (enc << 1) | (~val >> (e + m));  // magnitude ~sign
  if (enc != 0LL) enc--;  // -0 -> +0 and fill gap
  return enc;
}


static void h_encode(byte* input, const long long insize, byte* const __restrict__ output, long long& outsize, const int eb_e, const int thr_e, const long long offs)
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
#pragma omp parallel for schedule(dynamic, 1) default(none) shared(chunks, insize, input, offs, thr_e, eb_e, carry, size_out, data_out)
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
        const type_u val = quantize(in_t[j], eb_e, thr_e, offs);
        tmp_buffer_t[j] = val;
        max_val = std::max(max_val, val);
      }

      int cnt = TB;
      if (max_val != 0) {
        cnt = clz(max_val);
      }
      const int ln = TB - cnt;  // logn value for subchunk
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
    if (newsize < osize) {
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

    printf("min_val: %.10f\n", min);
    printf("max_val: %.10f\n", max);
    printf("diff between max and min values: %.10f\n", (max - min));
    return (max - min) * eb_param;
  }
#endif


int main(int argc, char* argv [])
{
  printf("CPU SLEEK 1.0: double-precision lossy compressor\n");
  printf("Copyright 2026 Texas State University\n\n");

  // read input from file
  if (argc < 4) {printf("USAGE: %s input_file_name compressed_file_name error_bound\n\n", argv[0]); return -1;}
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
    const type_f parameter = to_float(argv[3]);
    type_f* t_input = (type_f*)input;
    const type_f eb = computeNOAeb(t_input, insize / sizeof(type_f), parameter);
  #else
    const type_f eb = to_float(argv[3]);
  #endif
  
  // eb variables
  const int e = 11;  // exponent bits
  const int m = 52;  // mantissa bits
  const int eb_e = (*((long long*)&eb) >> m) & ((1LL << e) - 1LL);  // extract biased exponent
  const int thr_e = eb_e + (m + 1);  // biased exponent of threshold
  const long long offs = ((long long)thr_e << m) - (1LL << m);  // offset for lossless encoding
  if (thr_e >= (1 << e) - 1) {fprintf(stderr, "QUANT_IABS_0_f64: ERROR: error_bound is too large\n"); return -1;}


  if (insize % sizeof(type_f) != 0) {fprintf(stderr, "ERROR: size of input must be a multiple of %ld bytes\n", sizeof(type_f)); return -1;}

  #if defined(ARTIFACT)
    // Check if the third argument is "y" to enable performance analysis
    char* perf_str = argv[4];
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
  // time CPU preprocessor encoding
  if (perf) {
    // warm up
    byte* dummy = new byte [(insize + CS - 1) / CS * CS];
    std::copy(input, input + insize, dummy);
    h_encode(dummy, insize, hencoded, hencsize, eb_e, thr_e, offs);
    delete [] dummy;
  }
  #endif

  CPUTimer htimer;
  htimer.start();
  h_encode(input, insize, hencoded, hencsize, eb_e, thr_e, offs);
  double hruntime = htimer.stop();

  printf("encoded size: %lld bytes\n", hencsize);
  const type_f CR = (100.0 * hencsize) / insize;
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
