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

#pragma once


#include "sleek_common_gpu.h"


template <typename T>
struct sleek_error_bound_info
{
  const int eb_e;
  const int thr_e;
  const T offs_e;
};


template <typename F>
static inline sleek_error_bound_info<typename std::conditional<sizeof(F) == 4, unsigned int, unsigned long long>::type> sleek_error_bound(const F eb)
{
  constexpr const int e = (sizeof(F) == 4) ? 8 : 11;
  constexpr const int m = (sizeof(F) == 4) ? 23 : 52;

  using U = typename std::conditional<sizeof(F) == 4, int, long long>::type;
  using T = typename std::conditional<sizeof(F) == 4, unsigned int, unsigned long long>::type;

  const int eb_e = (*((U*)&eb) >> m) & ((1LL << e) - 1LL);  // extract biased exponent
  const int thr_e = eb_e + (m + 1);  // biased exponent of threshold
  const T offs_e = ((U)thr_e << m) - (1LL << m);  // offset for lossless encoding
  if (thr_e >= (1 << e) - 1) {fprintf(stderr, "ERROR: error_bound is too large\n"); exit(-1);}

  return { eb_e, thr_e, offs_e };
}


static inline void sleek_compress_gpu(const int blocks, const sleek_error_bound_info<unsigned int> eb_info, const float* const __restrict__ d_input, const long long num_elements, byte* const __restrict__ d_encoded, long long * const __restrict__ d_encsize)
{
  const long long insize = num_elements * sizeof(float);
  const long long chunks = (insize + SLEEK_CS - 1) / SLEEK_CS + 1;  // round up, +1 for chunk counter

  long long* d_fullcarry;
  cudaMalloc((void **)&d_fullcarry, chunks * sizeof(long long));
  cudaMemsetAsync(d_fullcarry, 0, chunks * sizeof(long long));
  unsigned long long* const g_chunk_counter = (unsigned long long*) &d_fullcarry[chunks - 1];

  d_encode<unsigned int><<<blocks, SLEEK_TPB>>>((byte*) d_input, insize, d_encoded, d_encsize, d_fullcarry, g_chunk_counter, eb_info.eb_e, eb_info.thr_e, eb_info.offs_e);

  cudaFree(d_fullcarry);
}


static inline void sleek_decompress_gpu(const int blocks, const sleek_error_bound_info<unsigned int> eb_info, const byte* const __restrict__ d_encoded, float* const __restrict__ d_decoded, long long * const __restrict__ d_decsize)
{
  unsigned long long* g_chunk_counter;
  cudaMalloc((void **)&g_chunk_counter, sizeof(unsigned long long));

  cudaMemsetAsync(g_chunk_counter, 0, sizeof(unsigned long long));
  d_decode<unsigned int><<<blocks, SLEEK_TPB>>>(d_encoded, (byte*) d_decoded, d_decsize, g_chunk_counter, eb_info.thr_e, eb_info.offs_e);

  cudaFree(g_chunk_counter);
}


static inline void sleek_compress_gpu(const int blocks, const sleek_error_bound_info<unsigned long long> eb_info, const double* const __restrict__ d_input, const long long num_elements, byte* const __restrict__ d_encoded, long long * const __restrict__ d_encsize)
{
  const long long insize = num_elements * sizeof(double);
  const long long chunks = (insize + SLEEK_CS - 1) / SLEEK_CS + 1;  // round up, +1 for chunk counter

  long long* d_fullcarry;
  cudaMalloc((void **)&d_fullcarry, chunks * sizeof(long long));
  cudaMemsetAsync(d_fullcarry, 0, chunks * sizeof(long long));
  unsigned long long* const g_chunk_counter = (unsigned long long*) &d_fullcarry[chunks - 1];

  d_encode<unsigned long long><<<blocks, SLEEK_TPB>>>((byte*) d_input, insize, d_encoded, d_encsize, d_fullcarry, g_chunk_counter, eb_info.eb_e, eb_info.thr_e, eb_info.offs_e);

  cudaFree(d_fullcarry);
}


static inline void sleek_decompress_gpu(const int blocks, const sleek_error_bound_info<unsigned long long> eb_info, const byte* const __restrict__ d_encoded, double* const __restrict__ d_decoded, long long * const __restrict__ d_decsize)
{
  unsigned long long* g_chunk_counter;
  cudaMalloc((void **)&g_chunk_counter, sizeof(unsigned long long));

  cudaMemsetAsync(g_chunk_counter, 0, sizeof(unsigned long long));
  d_decode<unsigned long long><<<blocks, SLEEK_TPB>>>(d_encoded, (byte*) d_decoded, d_decsize, g_chunk_counter, eb_info.thr_e, eb_info.offs_e);

  cudaFree(g_chunk_counter);
}

