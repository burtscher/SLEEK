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


#define NDEBUG


#include <cassert>
#include <cuda.h>
#include <string>
#include <cmath>


using byte = unsigned char;
static const int SLEEK_CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int SLEEK_TPB = 512;  // threads per block [must be power of 2 and at least 128]
#define SLEEK_WS 32


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  #define __all(arg) __all_sync(~0, arg)
  #define __any(arg) __any_sync(~0, arg)
  #define __ballot(arg) __ballot_sync(~0, arg)
  #define __shfl_up(...) __shfl_up_sync(~0, __VA_ARGS__)
  #define __shfl_xor(...) __shfl_xor_sync(~0, __VA_ARGS__)
#endif


static inline void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}


static inline long long sleek_upper_bound_size(const long long insize)
{
  const long long chunks = (insize + SLEEK_CS - 1) / SLEEK_CS;  // round up
  return 3 * sizeof(int) + chunks * sizeof(short) + chunks * SLEEK_CS;
}


static inline int sleek_system_checks()
{
  // perform system checks

  const int endian = 1;

  if (*((char*)(&endian)) != 1) {fprintf(stderr, "ERROR: SLEEK only supports little-endian systems\n\n"); return -1;}

  if (sizeof(long long) != 8) {fprintf(stderr, "ERROR: long long must be 8 bytes\n\n"); return -1;}

  if (sizeof(int) != 4) {fprintf(stderr, "ERROR: int must be 4 bytes\n\n"); return -1;}

  if (sizeof(short) != 2) {fprintf(stderr, "ERROR: short must be 2 bytes\n\n"); return -1;}

  if (sizeof(char) != 1) {fprintf(stderr, "ERROR: char must be 1 byte\n\n"); return -1;}

  if (SLEEK_CS != 16384) {fprintf(stderr, "ERROR: SLEEK_CS must be 16384\n\n"); return -1;}

  return 0;
}


static inline int sleek_thread_blocks()
{
  // get GPU info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  CheckCuda(__LINE__);
  return SMs * (mTpSM / SLEEK_TPB);
}


// copy (len) bytes from shared memory (source) to global memory (destination)
// source must we word aligned
static inline __device__ void s2g(void* const __restrict__ destination, const void* const __restrict__ source, const int len)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  byte* const __restrict__ output = (byte*)destination;
  if (len < 128) {
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)output;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    const int* const __restrict__ in_w = (int*)input;
    if (bcnt == 0) {
      int* const __restrict__ out_w = (int*)output;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += SLEEK_TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    } else {
      const int shift = bcnt * 8;
      const int rlen = len - bcnt;
      int* const __restrict__ out_w = (int*)&output[bcnt];
      if (tid < bcnt) output[tid] = input[tid];
      if (tid < wcnt) out_w[tid] = __funnelshift_r(in_w[tid], in_w[tid + 1], shift);
      for (int i = tid + wcnt; i < rlen / 4; i += SLEEK_TPB) {
        out_w[i] = __funnelshift_r(in_w[i], in_w[i + 1], shift);
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    }
  }
}


// copy (len) bytes from global memory (source) to shared memory (destination) using separate shared memory buffer (temp)
// destination and temp must we word aligned, accesses up to SLEEK_CS + 3 bytes in temp
static inline __device__ void g2s(void* const __restrict__ destination, const void* const __restrict__ source, const int len, void* const __restrict__ temp)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  if (len < 128) {
    byte* const __restrict__ output = (byte*)destination;
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)input;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    int* const __restrict__ out_w = (int*)destination;
    if (bcnt == 0) {
      const int* const __restrict__ in_w = (int*)input;
      byte* const __restrict__ out = (byte*)destination;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += SLEEK_TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        out[i] = input[i];
      }
    } else {
      const int offs = 4 - bcnt;  //(4 - bcnt) & 3;
      const int shift = offs * 8;
      const int rlen = len - bcnt;
      const int* const __restrict__ in_w = (int*)&input[bcnt];
      byte* const __restrict__ buffer = (byte*)temp;
      byte* const __restrict__ buf = (byte*)&buffer[offs];
      int* __restrict__ buf_w = (int*)&buffer[4];  //(int*)&buffer[(bcnt + 3) & 4];
      if (tid < bcnt) buf[tid] = input[tid];
      if (tid < wcnt) buf_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < rlen / 4; i += SLEEK_TPB) {
        buf_w[i] = in_w[i];
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        buf[i] = input[i];
      }
      __syncthreads();
      buf_w = (int*)buffer;
      for (int i = tid; i < (len + 3) / 4; i += SLEEK_TPB) {
        out_w[i] = __funnelshift_r(buf_w[i], buf_w[i + 1], shift);
      }
    }
  }
}


static inline __device__ void propagate_carry(const int value, const long long chunkID, volatile long long* const __restrict__ fullcarry, long long* const __restrict__ s_fullc)
{
  if (threadIdx.x == SLEEK_TPB - 1) {  // last thread
    fullcarry[chunkID] = (chunkID == 0) ? (long long)value : (long long)-value;
  }

  if (chunkID != 0) {
    if (threadIdx.x + SLEEK_WS >= SLEEK_TPB) {  // last warp
      const int lane = threadIdx.x % SLEEK_WS;
      const long long cidm1ml = chunkID - 1 - lane;
      long long val = -1;
      __syncwarp();  // not optional
      do {
        if (cidm1ml >= 0) {
          val = fullcarry[cidm1ml];
        }
      } while ((__any(val == 0)) || (__all(val <= 0)));
      const int mask = __ballot(val > 0);
      const int pos = __ffs(mask) - 1;
      long long partc = (lane < pos) ? -val : 0;
      partc += __shfl_xor(partc, 1);
      partc += __shfl_xor(partc, 2);
      partc += __shfl_xor(partc, 4);
      partc += __shfl_xor(partc, 8);
      partc += __shfl_xor(partc, 16);

      if (lane == pos) {
        const long long fullc = partc + val;
        fullcarry[chunkID] = fullc + value;
        *s_fullc = fullc;
      }
    }
  }
}


template <typename T>
static __device__ inline T block_sum_reduction(T val, void* buffer)  // returns sum to all threads
{
  const int lane = threadIdx.x % SLEEK_WS;
  const int warp = threadIdx.x / SLEEK_WS;
  constexpr const int warps = SLEEK_TPB / SLEEK_WS;
  T* const s_carry = (T*)buffer;
  assert(SLEEK_WS >= warps);

  if constexpr (sizeof(T) == 4) {
    val += __reduce_xor_sync(~0, val);
  } else {
    val += __shfl_xor(val, 1);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 16);    
  }

  if (lane == 0) s_carry[warp] = val;
  __syncthreads();  // s_carry written

  if constexpr (warps > 1) {
    if (warp == 0) {
      val = (lane < warps) ? s_carry[lane] : 0;
      if constexpr (sizeof(T) == 4) {
        val += __reduce_xor_sync(~0, val);
      } else {
        val += __shfl_xor(val, 1);
        if constexpr (warps > 2) {
          val += __shfl_xor(val, 2);
          if constexpr (warps > 4) {
            val += __shfl_xor(val, 4);
            if constexpr (warps > 8) {
              val += __shfl_xor(val, 8);
              if constexpr (warps > 16) {
                val += __shfl_xor(val, 16);
              }
            }
          }
        }
      }
      s_carry[lane] = val;
    }
    __syncthreads();  // s_carry updated
  }

  return s_carry[0];
}


//
// lossless
//


template <typename T>
static __device__ inline bool d_SLEEK(int& csize, byte in [SLEEK_CS], byte out [SLEEK_CS], byte temp [SLEEK_CS])
{
  constexpr const int TB = sizeof(T) * 8;  // number of bits in T
  constexpr const int size = SLEEK_CS / sizeof(T);
  constexpr const int SC = 32;  // subchunks [do not change]
  constexpr const int chunksize = size / SC;

  static_assert(sizeof(T) >= 4);
  static_assert(chunksize % SLEEK_WS == 0);
  static_assert(SLEEK_WS == SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  const int tid = threadIdx.x;
  const int lane = tid % SLEEK_WS;
  const int warp = tid / SLEEK_WS;
  constexpr const int warps = SLEEK_TPB / SLEEK_WS;

  // clear unused part of input buffer
  T* const in_t = (T*)in;
  if (csize < SLEEK_CS) {
    if (csize % sizeof(T) == 0) {
      for (int i = csize / sizeof(T) + tid; i < size; i += SLEEK_TPB) {
        in_t[i] = 0;
      }
    } else {
      for (int i = csize + tid; i < SLEEK_CS; i += SLEEK_TPB) {
        in[i] = 0;
      }
    }
    __syncthreads();
  }

  // determine bits needed for each subchunk
  int ln = -1;
  for (int i = warp; i < SC; i += warps) {
    const int beg = i * chunksize;
    const int end = beg + chunksize;

    // max of values for each thread
    T max_val = 0;
    for (int j = beg + lane; j < end; j += SLEEK_WS) {
      T val = in_t[j];
      if constexpr (TB == 32) {
        val = __funnelshift_rc(val, val, 31);
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
      in_t[j] = val;
      max_val = max(max_val, val);
    }

    // warp level max
    if constexpr (TB == 32) {
      max_val = __reduce_max_sync(~0, max_val);
    } else {
      max_val = max(max_val, __shfl_xor(max_val, 1));
      max_val = max(max_val, __shfl_xor(max_val, 2));
      max_val = max(max_val, __shfl_xor(max_val, 4));
      max_val = max(max_val, __shfl_xor(max_val, 8));
      max_val = max(max_val, __shfl_xor(max_val, 16));
    }

    // figure out number of bits needed
    if (lane == i) {
      int cnt = TB;
      if (max_val != 0) {
        if constexpr (TB == 64) {
          cnt = __clzll(max_val);
        } else {
          cnt = __clz(max_val);
        }
      }
      ln = TB - cnt;  // logn value for each chunk
    }
  }
  if (ln >= 0) out[lane] = ln;
  __syncthreads();

  // warp prefix sum over bits
  int* const bits = (int*)temp;
  if (warp == 0) {
    const int org = out[lane] * chunksize;
    int val = org;
    int tmp = __shfl_up(val, 1);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up(val, 2);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up(val, 4);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up(val, 8);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up(val, 16);
    if (lane >= 16) val += tmp;
    bits[lane] = val - org;
    if (lane == SC - 1) bits[SC] = val;
  }
  __syncthreads();

  // check if encoded data fits
  const int tot = bits[SC];
  const int newsize = (SC * 8 + tot + 16) / 8;
  if (newsize >= SLEEK_CS) return false;

  // clear out buffer
  T* const out_t = (T*)&out[SC];
  for (int i = tid; i < tot / TB; i += SLEEK_TPB) out_t[i] = 0;
  __syncthreads();

  // encode data values
  for (int i = warp; i < SC; i += warps) {
    const int logn = out[i];
    if (logn > 0) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;
      if (logn == TB) {
        const int offs = bits[i] / TB - beg;
        for (int j = beg + lane; j < end; j += SLEEK_WS) {
          out_t[offs + j] = in_t[j];
        }
      } else {
        const int incr = SLEEK_WS * logn;
        int loc = bits[i] + lane * logn;
        for (int j = beg + lane; j < end; j += SLEEK_WS) {
          const T val = in_t[j];
          const int pos = loc / TB;
          const int shift = loc % TB;
          atomicOr_block(&out_t[pos], val << shift);
          if (TB - shift < logn) {
            atomicOr_block(&out_t[pos + 1], val >> (TB - shift));
          }
          loc += incr;
        }
      }
    }
  }

  // output header info
  if (tid == 0) {
    *(short*)&out[newsize - 2] = csize;
  }

  csize = newsize;
  return true;
}


template <typename T>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(SLEEK_TPB, 4)
#else
static __global__ __launch_bounds__(SLEEK_TPB, 3)
#endif
void d_encode(const byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, long long* const __restrict__ outsize, long long* const __restrict__ fullcarry, unsigned long long* const __restrict__ g_chunk_counter)
{
  // allocate shared memory buffer
  __shared__ long long chunk [2 * (SLEEK_CS / sizeof(long long)) + 17];

  // split into 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (SLEEK_CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (SLEEK_CS / sizeof(long long))];
  byte* const temp = (byte*)&chunk[2 * (SLEEK_CS / sizeof(long long))];

  // initialize
  const int tid = threadIdx.x;
  const long long last = 2 * (SLEEK_CS / sizeof(long long));
  const long long chunks = (insize + SLEEK_CS - 1) / SLEEK_CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];

  // loop over chunks
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(g_chunk_counter, 1LL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * SLEEK_CS;
    if (base >= insize) break;

    // load chunk
    const int osize = (int)min((long long)SLEEK_CS, insize - base);
    long long* const input_l = (long long*)&input[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += SLEEK_TPB) {
      out_l[i] = input_l[i];
    }
    if constexpr (sizeof(T) == 4) {
      const int extra = osize % 8;
      if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];
    }

    // encode chunk
    __syncthreads();  // chunk produced, chunk[last] consumed
    int csize = osize;
    byte* tmp = in; in = out; out = tmp;
    bool good = d_SLEEK<T>(csize, in, out, temp);
   __syncthreads();

    // handle carry
    if (!good || (csize >= osize)) csize = osize;
    propagate_carry(csize, chunkID, fullcarry, (long long*)temp);

    // reload chunk if incompressible
    if (tid == 0) size_out[chunkID] = csize;
    if (csize == osize) {
      // store original data
      long long* const out_l = (long long*)out;
      for (long long i = tid; i < osize / 8; i += SLEEK_TPB) {
        out_l[i] = input_l[i];
      }
      if constexpr (sizeof(T) == 4) {
        const int extra = osize % 8;
        if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];
      }
    }
    __syncthreads();  // "out" done, temp produced

    // store chunk
    const long long offs = (chunkID == 0) ? 0 : *((long long*)temp);
    s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + SLEEK_CS >= insize)) {
      // output header
      head_out[0] = insize;
      // compute compressed size
      *outsize = &data_out[fullcarry[chunkID]] - output;
    }
  } while (true);
}


template <typename T>
static __device__ inline void d_iSLEEK(int& csize, byte in [SLEEK_CS], byte out [SLEEK_CS], byte temp [SLEEK_CS])
{
  const int tid = threadIdx.x;
  const int lane = tid % SLEEK_WS;
  const int warp = tid / SLEEK_WS;
  constexpr const int warps = SLEEK_TPB / SLEEK_WS;

  constexpr const int TB = sizeof(T) * 8;  // number of bits in T
  constexpr const int size = SLEEK_CS / sizeof(T);
  constexpr const int SC = 32;  // subchunks [do not change]
  constexpr const int chunksize = size / SC;

  static_assert(sizeof(T) >= 4);
  static_assert(SLEEK_WS == SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  // warp prefix sum over bits
  int* const bits = (int*)temp;
  if (warp == 0) {
    const int org = in[lane] * chunksize;
    int val = org;
    int tmp = __shfl_up(val, 1);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up(val, 2);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up(val, 4);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up(val, 8);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up(val, 16);
    if (lane >= 16) val += tmp;
    bits[lane] = val - org;
  }
  __syncthreads();

  // decode data values
  const T* const in_t = (T*)&in[SC];
  T* const out_t = (T*)out;
  for (int i = warp; i < SC; i += warps) {
    const int logn = in[i];
    const int beg = i * chunksize;
    const int end = beg + chunksize;
    if (logn == 0) {
      for (int j = beg + lane; j < end; j += SLEEK_WS) {
        out_t[j] = 0;
      }
    } else if (logn == TB) {
      const int offs = bits[i] / TB - beg;
      for (int j = beg + lane; j < end; j += SLEEK_WS) {
        T val = in_t[offs + j];
        if constexpr (TB == 32) {
          val = (val >> 1) ^ (((int)(val << 31)) >> 31);  // iTCMS
          if ((val & 0xff00'0000) != 0) {
            if (val >= 0x8000'0000) {
              val += 0x0100'0000;
            }
            val += 0x8000'0000;
          }
          val = __funnelshift_lc(val, val, 31);
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
      const int incr = SLEEK_WS * logn;
      int loc = bits[i] + lane * logn;
      const T mask = (logn == 64) ? (~0ULL) : ((1ULL << logn) - 1);
      for (int j = beg + lane; j < end; j += SLEEK_WS) {
        const int pos = loc / TB;

        const T lo = in_t[pos];
        const T hi = in_t[pos + 1];

        const int shift = loc % TB;

        T res;
        if constexpr (TB == 32) {
          res = __funnelshift_rc(lo, hi, shift);
        } else {
          res = lo >> shift;
          if (TB - shift < logn) {
            res |= hi << (TB - shift);
          }
        }

        loc += incr;
        T val = res & mask;
        if constexpr (TB == 32) {
          val = (val >> 1) ^ (((int)(val << 31)) >> 31);  // iTCMS
          if ((val & 0xff00'0000) != 0) {
            if (val >= 0x8000'0000) {
              val += 0x0100'0000;
            }
            val += 0x8000'0000;
          }
          val = __funnelshift_lc(val, val, 31);
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
  }

  // read header info
  csize = *(short*)&in[csize - 2];
}


template <typename T>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(SLEEK_TPB, 4)
#else
static __global__ __launch_bounds__(SLEEK_TPB, 3)
#endif
void d_decode(const byte* const __restrict__ input, byte* const __restrict__ output, long long* const __restrict__ g_outsize, unsigned long long* const __restrict__ g_chunk_counter)
{
  // allocate shared memory buffer
  __shared__ long long chunk [2 * (SLEEK_CS / sizeof(long long)) + 16];
  const int last = 2 * (SLEEK_CS / sizeof(long long));

  // input header
  long long* const head_in = (long long*)input;
  const long long outsize = head_in[0];

  // initialize
  const long long chunks = (outsize + SLEEK_CS - 1) / SLEEK_CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  long long prevChunkID = 0;
  long long prevOffset = 0;
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(g_chunk_counter, 1LL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * SLEEK_CS;
    if (base >= outsize) break;

    // compute sum of all prior csizes (start where left off in previous iteration)
    long long sum = 0;
    for (long long i = prevChunkID + tid; i < chunkID; i += SLEEK_TPB) {
      sum += (long long)size_in[i];
    }
    int csize = (int)size_in[chunkID];

    // create the 3 shared memory buffers
    byte* in = (byte*)&chunk[0 * (SLEEK_CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (SLEEK_CS / sizeof(long long))];
    byte* temp = (byte*)&chunk[2 * (SLEEK_CS / sizeof(long long))];

    const long long offs = prevOffset + block_sum_reduction(sum, (long long*)out); //chunk[last + 1]);
    prevChunkID = chunkID;
    prevOffset = offs;
    __syncthreads();

    // load chunk
    g2s(in, &data_in[offs], csize, out);
    __syncthreads();  // chunk produced, chunk[last] consumed

    // decode
    const int osize = (int)min((long long)SLEEK_CS, outsize - base);
    if (csize < osize) {
      d_iSLEEK<T>(csize, in, out,temp);
    } else {
      byte* tmp = in; in = out; out = tmp; // swap if no decode
    }
    __syncthreads();

    if (csize != osize) {printf("ERROR: csize %d doesn't match osize %d in chunk %lld\n\n", csize, osize, chunkID); __trap();}
    long long* const output_l = (long long*)&output[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += SLEEK_TPB) {
      output_l[i] = out_l[i];
    }
    if constexpr (sizeof(T) == 4) {
      const int extra = osize % 8;
      if (tid < extra) output[base + osize - extra + tid] = out[osize - extra + tid];
    }
  } while (true);

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}


//
// lossy
//


template <typename T>
static __device__ inline T quantize(const T val, const int eb_e, const int thr_e, const T offs)
{
  using ST = typename std::conditional<sizeof(T) == 4, int, long long>::type;

  constexpr const int e = (sizeof(T) == 4) ? 8 : 11;
  constexpr const int m = (sizeof(T) == 4) ? 23 : 52;

  const ST abs = val & (((T) 1 << (e + m)) - 1);  // compute absolute value
  const int val_e = abs >> m;  // extract exponent
  ST enc = 0;  // default value is 0

  if (val_e >= thr_e) {  // at or above threshold
    enc = abs - offs;  // lossless encoding
  } else if (val_e >= eb_e) {  // lossy encoding
    ST mant = val & (((ST) 1 << m) - 1);  // extract mantissa
    const int shift = thr_e - val_e;  // bias cancels out
    mant |= (ST) 1 << m;  // insert implicit 1
    mant += (ST) 1 << (shift - 1);  // round to nearest, ties round away from zero
    enc = mant >> shift;  // shift out unnecessary bits
  }

  enc = (enc << 1) | (~val >> (e + m));  // magnitude ~sign
  if (enc != 0) enc--;  // -0 -> +0 and fill gap

  return enc;
}


template <typename T>
static __device__ inline bool d_SLEEK(int& csize, byte in [SLEEK_CS], byte out [SLEEK_CS], byte temp [SLEEK_CS], const int eb_e, const int thr_e, const T offs_e)
{
  constexpr const int TB = sizeof(T) * 8;  // number of bits in T
  constexpr const int size = SLEEK_CS / sizeof(T);
  constexpr const int SC = 32;  // subchunks [do not change]
  constexpr const int chunksize = size / SC;

  static_assert(sizeof(T) >= 4);
  static_assert(chunksize % SLEEK_WS == 0);
  static_assert(SLEEK_WS == SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  const int tid = threadIdx.x;
  const int lane = tid % SLEEK_WS;
  const int warp = tid / SLEEK_WS;
  const int warps = SLEEK_TPB / SLEEK_WS;

  // clear unused part of input buffer
  T* const in_t = (T*)in;
  if (csize < SLEEK_CS) {
    if (csize % sizeof(T) == 0) {
      for (int i = csize / sizeof(T) + tid; i < size; i += SLEEK_TPB) {
        in_t[i] = 0;
      }
    } else {
      for (int i = csize + tid; i < SLEEK_CS; i += SLEEK_TPB) {
        in[i] = 0;
      }
    }
    __syncthreads();
  }

  // determine bits needed for each subchunk
  int ln = -1;
  for (int i = warp; i < SC; i += warps) {
    const int beg = i * chunksize;
    const int end = beg + chunksize;

    // max of values for each thread
    T max_val = 0;
    for (int j = beg + lane; j < end; j += SLEEK_WS) {
      const T val = quantize(in_t[j], eb_e, thr_e, offs_e);
      in_t[j] = val;
      max_val = max(max_val, val);
    }

    // warp level max
    if constexpr (TB == 32) {
      max_val = __reduce_max_sync(~0, max_val);
    } else {
      max_val = max(max_val, __shfl_xor(max_val, 1));
      max_val = max(max_val, __shfl_xor(max_val, 2));
      max_val = max(max_val, __shfl_xor(max_val, 4));
      max_val = max(max_val, __shfl_xor(max_val, 8));
      max_val = max(max_val, __shfl_xor(max_val, 16));
    }

    // figure out number of bits needed
    if (lane == i) {
      int cnt = TB;
      if (max_val != 0) {
        if constexpr (TB == 64) {
          cnt = __clzll(max_val);
        } else {
          cnt = __clz(max_val);
        }
      }
      ln = TB - cnt;  // logn value for each chunk
    }
  }
  if (ln >= 0) out[lane] = ln;
  __syncthreads();

  // warp prefix sum over bits
  int* const bits = (int*)temp;
  if (warp == 0) {
    const int org = out[lane] * chunksize;
    int val = org;
    int tmp = __shfl_up(val, 1);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up(val, 2);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up(val, 4);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up(val, 8);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up(val, 16);
    if (lane >= 16) val += tmp;
    bits[lane] = val - org;
    if (lane == SC - 1) bits[SC] = val;
  }
  __syncthreads();

  // check if encoded data fits
  const int tot = bits[SC];
  const int newsize = (SC * 8 + tot + 16) / 8;
  if (newsize >= SLEEK_CS) return false;

  // clear out buffer
  T* const out_t = (T*)&out[SC];
  for (int i = tid; i < tot / TB; i += SLEEK_TPB) out_t[i] = 0;
  __syncthreads();

  // encode data values
  if constexpr (TB == 32) {
    for (int i = warp; i < SC; i += warps) {
      const int logn = out[i];
      if (logn > 0) {
        const int beg = i * chunksize;
        const int end = beg + chunksize;
        if (logn == TB) {
          const int offs = bits[i] / TB - beg;
          for (int j = beg + lane; j < end; j += SLEEK_WS) {
            out_t[offs + j] = in_t[j];
          }
        } else {
          const int incr = SLEEK_WS * 2 * logn;
          int loc = bits[i] + lane * 2 * logn;
          for (int j = beg + lane * 2; j < end; j += SLEEK_WS * 2) {
            const T val = in_t[j];
            const T valB = in_t[j + 1];
            const int loc2 = loc + logn;
            const int pos = loc / TB;
            const int shift = loc % TB;
            const int pos2 = loc2 / TB;
            const int shift2 = loc2 % TB;
            const int val1 = val << shift;
            const int val2 = valB << shift2;
            const int val3 = val >> (TB - shift);
            const int val4 = valB >> (TB - shift2);
            if (val1 != 0) atomicOr_block(&out_t[pos], val1);
            if (val2 != 0) atomicOr_block(&out_t[pos2], val2);
            if ((val3 != 0) && (shift != 0)) atomicOr_block(&out_t[pos + 1], val3);
            if ((val4 != 0) && (shift2 != 0)) atomicOr_block(&out_t[pos2 + 1], val4);
            loc += incr;
          }
        }
      }
    }
  } else {
    for (int i = warp; i < SC; i += warps) {
      const int logn = out[i];
      if (logn > 0) {
        const int beg = i * chunksize;
        const int end = beg + chunksize;
        if (logn == TB) {
          const int offs = bits[i] / TB - beg;
          for (int j = beg + lane; j < end; j += SLEEK_WS) {
            out_t[offs + j] = in_t[j];
          }
        } else {
          const int incr = SLEEK_WS * logn;
          int loc = bits[i] + lane * logn;
          for (int j = beg + lane; j < end; j += SLEEK_WS) {
            const T val = in_t[j];
            const int pos = loc / TB;
            const int shift = loc % TB;
            const int val1 = (int)(val << shift);
            const int val2 = (int)((val << shift) >> 32);
            const int val3 = (int)(val >> (TB - shift));
            const int val4 = (int)((val >> (TB - shift)) >> 32);
            if (val1 != 0) atomicOr_block((int*)&out_t[pos], val1);
            if (val2 != 0) atomicOr_block(((int*)&out_t[pos]) + 1, val2);
            if ((val3 != 0) && (shift != 0)) atomicOr_block((int*)&out_t[pos + 1], val3);
            if ((val4 != 0) && (shift != 0)) atomicOr_block(((int*)&out_t[pos + 1]) + 1, val4);
            loc += incr;
          }
        }
      }
    }
  }

  // output header info
  if (tid == 0) {
    *(short*)&out[newsize - 2] = csize;
  }

  csize = newsize;
  return true;
}


template <typename T>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(SLEEK_TPB, 4)
#else
static __global__ __launch_bounds__(SLEEK_TPB, 3)
#endif
void d_encode(const byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, long long* const __restrict__ outsize, long long* const __restrict__ fullcarry, unsigned long long* const __restrict__ g_chunk_counter, const int eb_e, const int thr_e, const T offs_e)
{
  // allocate shared memory buffer
  __shared__ long long chunk [2 * (SLEEK_CS / sizeof(long long)) + 17];

  // split into 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (SLEEK_CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (SLEEK_CS / sizeof(long long))];
  byte* const temp = (byte*)&chunk[2 * (SLEEK_CS / sizeof(long long))];

  // initialize
  const int tid = threadIdx.x;
  const long long last = 2 * (SLEEK_CS / sizeof(long long)) + 1;
  const long long chunks = (insize + SLEEK_CS - 1) / SLEEK_CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];

  // loop over chunks
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(g_chunk_counter, 1LL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * SLEEK_CS;
    if (base >= insize) break;

    // load chunk
    const int osize = (int)min((long long)SLEEK_CS, insize - base);
    long long* const input_l = (long long*)&input[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += SLEEK_TPB) {
      out_l[i] = input_l[i];
    }
    if constexpr (sizeof(T) == 4) {
      const int extra = osize % 8;
      if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];
    }

    // encode chunk
    __syncthreads();  // chunk produced, chunk[last] consumed
    int csize = osize;
    byte* tmp = in; in = out; out = tmp;
    bool good = d_SLEEK<T>(csize, in, out, temp, eb_e, thr_e, offs_e);
   __syncthreads();

    // handle carry
    if (!good || (csize >= osize)) csize = osize;
    propagate_carry(csize, chunkID, fullcarry, (long long*)temp);

    // reload chunk if incompressible
    if (tid == 0) size_out[chunkID] = csize;
    if (csize == osize) {
      // store original data
      long long* const out_l = (long long*)out;
      for (long long i = tid; i < osize / 8; i += SLEEK_TPB) {
        out_l[i] = input_l[i];
      }
      if constexpr (sizeof(T) == 4) {
        const int extra = osize % 8;
        if (tid < extra) out[(long long)osize - (long long)extra + (long long)tid] = input[base + (long long)osize - (long long)extra + (long long)tid];
      }
    }
    __syncthreads();  // "out" done, temp produced

    // store chunk
    const long long offs = (chunkID == 0) ? 0 : *((long long*)temp);
    s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + SLEEK_CS >= insize)) {
      // output header
      head_out[0] = insize;
      // compute compressed size
      *outsize = &data_out[fullcarry[chunkID]] - output;
    }
  } while (true);
}


template <typename T>
static __device__ inline T deQuantize(const T enc, const int thr_e, const T offs)
{
  using ST = typename std::conditional<sizeof(T) == 4, int, long long>::type;

  constexpr const int e = (sizeof(T) == 4) ? 8 : 11;
  constexpr const int m = (sizeof(T) == 4) ? 23 : 52;

  ST dec = 0;  // default value is 0
  if (enc != 0) {
    const ST abs = (enc + 1) >> 1;  // absolute value
    if (abs >= ((ST) 1 << m)) {  // above threshold
      dec = abs + offs;  // decode losslessly
    } else if (abs > 0) {  // non-zero lossy case

      int shift;
      if constexpr (sizeof(T) == 4) {
        shift = __clz(abs) - (31 - m);  // compute shift amount
      } else {
        shift = __clzll(abs) - (63 - m);  // compute shift amount
      }

      dec = abs << shift;  // shift to normalized position
      dec &= ((ST) 1 << m) - 1;  // remove implied 1
      dec |= (ST)(thr_e - shift) << m;  // insert biased exponent

    }
    dec |= enc << (e + m);  // insert sign bit
  }

  return dec;
}


template <typename T>
static __device__ inline void d_iSLEEK(int& csize, byte in [SLEEK_CS], byte out [SLEEK_CS], byte temp [SLEEK_CS], const int thr_e, const T offs_e)
{
  const int tid = threadIdx.x;
  const int lane = tid % SLEEK_WS;
  const int warp = tid / SLEEK_WS;
  const int warps = SLEEK_TPB / SLEEK_WS;

  constexpr const int TB = sizeof(T) * 8;  // number of bits in T
  constexpr const int size = SLEEK_CS / sizeof(T);
  constexpr const int SC = 32;  // subchunks [do not change]
  constexpr const int chunksize = size / SC;

  static_assert(sizeof(T) >= 4);
  static_assert(SLEEK_WS == SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  // warp prefix sum over bits
  int* const bits = (int*)temp;
  if (warp == 0) {
    const int org = in[lane] * chunksize;
    int val = org;
    int tmp = __shfl_up(val, 1);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up(val, 2);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up(val, 4);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up(val, 8);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up(val, 16);
    if (lane >= 16) val += tmp;
    bits[lane] = val - org;
  }
  __syncthreads();

  // decode data values
  const T* const in_t = (T*)&in[SC];
  T* const out_t = (T*)out;
  for (int i = warp; i < SC; i += warps) {
    const int logn = in[i];
    const int beg = i * chunksize;
    const int end = beg + chunksize;
    if (logn == 0) {
      for (int j = beg + lane; j < end; j += SLEEK_WS) {
        out_t[j] = 0;
      }
    } else if (logn == TB) {
      const int offs = bits[i] / TB - beg;
      for (int j = beg + lane; j < end; j += SLEEK_WS) {
        const T val = in_t[offs + j];
        out_t[j] = deQuantize(val, thr_e, offs_e);
      }
    } else {
      const int incr = SLEEK_WS * logn;
      int loc = bits[i] + lane * logn;
      const T mask = (logn == 64) ? (~0ULL) : ((1ULL << logn) - 1);
      for (int j = beg + lane; j < end; j += SLEEK_WS) {
        const int pos = loc / TB;

        const T lo = in_t[pos];
        const T hi = in_t[pos + 1];

        const int shift = loc % TB;

        T res;
        if constexpr (TB == 32) {
          res = __funnelshift_rc(lo, hi, shift);
        } else {
          res = lo >> shift;
          if (TB - shift < logn) {
            res |= hi << (TB - shift);
          }
        }

        loc += incr;
        const T val = res & mask;
        out_t[j] = deQuantize(val, thr_e, offs_e);
      }
    }
  }

  // read header info
  csize = *(short*)&in[csize - 2];
}


template <typename T>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(SLEEK_TPB, 4)
#else
static __global__ __launch_bounds__(SLEEK_TPB, 3)
#endif
void d_decode(const byte* const __restrict__ input, byte* const __restrict__ output, long long* const __restrict__ g_outsize, unsigned long long* const __restrict__ g_chunk_counter, const int thr_e, const T offs_e)
{
  // allocate shared memory buffer
  __shared__ long long chunk [2 * (SLEEK_CS / sizeof(long long)) + 16];
  const int last = 2 * (SLEEK_CS / sizeof(long long)) + 1;

  // input header
  long long* const head_in = (long long*)input;
  const long long outsize = head_in[0];

  // initialize
  const long long chunks = (outsize + SLEEK_CS - 1) / SLEEK_CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  long long prevChunkID = 0;
  long long prevOffset = 0;
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(g_chunk_counter, 1LL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * SLEEK_CS;
    if (base >= outsize) break;

    // compute sum of all prior csizes (start where left off in previous iteration)
    long long sum = 0;
    for (long long i = prevChunkID + tid; i < chunkID; i += SLEEK_TPB) {
      sum += (long long)size_in[i];
    }
    int csize = (int)size_in[chunkID];

    // create the 3 shared memory buffers
    byte* in = (byte*)&chunk[0 * (SLEEK_CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (SLEEK_CS / sizeof(long long))];
    byte* temp = (byte*)&chunk[2 * (SLEEK_CS / sizeof(long long))];

    const long long offs = prevOffset + block_sum_reduction(sum, (long long*)out); //chunk[last + 1]);
    prevChunkID = chunkID;
    prevOffset = offs;
    __syncthreads();

    // load chunk
    g2s(in, &data_in[offs], csize, out);
    __syncthreads();  // chunk produced, chunk[last] consumed

    // decode
    const int osize = (int)min((long long)SLEEK_CS, outsize - base);
    if (csize < osize) {
      d_iSLEEK<T>(csize, in, out, temp, thr_e, offs_e);
    } else {
      byte* tmp = in; in = out; out = tmp; // swap if no decode
    }
    __syncthreads();

    if (csize != osize) {printf("ERROR: csize %d doesn't match osize %d in chunk %lld\n\n", csize, osize, chunkID); __trap();}
    long long* const output_l = (long long*)&output[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += SLEEK_TPB) {
      output_l[i] = out_l[i];
    }
    if constexpr (sizeof(T) == 4) {
      const int extra = osize % 8;
      if (tid < extra) output[base + osize - extra + tid] = out[osize - extra + tid];
    }
  } while (true);

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}

