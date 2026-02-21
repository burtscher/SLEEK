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


#include <cmath>
#include <string>
#include <cassert>
#include <stdexcept>
#include <cuda.h>
#include <cuda/std/limits>


using byte = unsigned char;
using type_u = unsigned int;
using type_f = float;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]
#define WS 32


#include "MAXABS_f32.h"


#define to_float(val) \
    (sizeof(type_f) == sizeof(float) ? std::stof(val) : std::stod(val))


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  #define __shfl_up(...) __shfl_up_sync(~0, __VA_ARGS__)
  #define __shfl_xor(...) __shfl_xor_sync(~0, __VA_ARGS__)
#endif





static __device__ inline type_u block_sum_reduction(type_u val, void* buffer)  // returns sum to all threads
{
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const int warps = TPB / WS;
  type_u* const s_carry = (type_u*)buffer;
  assert(WS >= warps);

  val += __shfl_xor(val, 1);  // MB: use reduction on 8.6 CC
  val += __shfl_xor(val, 2);
  val += __shfl_xor(val, 4);
  val += __shfl_xor(val, 8);
  val += __shfl_xor(val, 16);

  if (lane == 0) s_carry[warp] = val;
  __syncthreads();  // s_carry written

  if constexpr (warps > 1) {
    if (warp == 0) {
      val = (lane < warps) ? s_carry[lane] : 0;
      val += __shfl_xor(val, 1);  // MB: use reduction on 8.6 CC
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
      s_carry[lane] = val;
    }
    __syncthreads();  // s_carry updated
  }

  return s_carry[0];
}


static __device__ inline type_u deQuantize(const type_u enc, const int thr_e, const int offs)
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


static __device__ inline void d_iSLEEK(int& csize, byte in [CS], byte out [CS], byte temp [CS], const int thr_e, const int offs)
{
  const int tid = threadIdx.x;
  const int lane = tid % WS;
  const int warp = tid / WS;
  const int warps = TPB / WS;

  const int TB = sizeof(type_u) * 8;  // number of bits in type_u
  const int size = CS / sizeof(type_u);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(sizeof(type_u) >= 4);
  static_assert(WS == SC);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<type_u>::value);

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
  const type_u* const in_t = (type_u*)&in[SC];
  type_u* const out_t = (type_u*)out;
  for (int i = warp; i < SC; i += warps) {
    const int logn = in[i];
    const int beg = i * chunksize;
    const int end = beg + chunksize;
    if (logn == 0) {
      for (int j = beg + lane; j < end; j += WS) {
        out_t[j] = 0;
      }
    } else if (logn == TB) {
      const int offs = bits[i] / TB - beg;
      for (int j = beg + lane; j < end; j += WS) {
        type_u val = in_t[offs + j];
        out_t[j] = deQuantize(val, thr_e, offs);
      }
    } else {
      const int incr = WS * logn;
      int loc = bits[i] + lane * logn;
      const type_u mask = (logn == 64) ? (~0ULL) : ((1ULL << logn) - 1);
      for (int j = beg + lane; j < end; j += WS) {
        const int pos = loc / TB;

        const type_u lo = in_t[pos];
        const type_u hi = in_t[pos + 1];

        const int shift = loc % TB;

        type_u res;
        if constexpr (TB == 32) {
          res = __funnelshift_rc(lo, hi, shift);
        } else {
          res = lo >> shift;
          if (TB - shift < logn) {
            res |= hi << (TB - shift);
          }
        }

        loc += incr;
        type_u val = res & mask;
        out_t[j] = deQuantize(val, thr_e, offs);
      }
    }
  }

  // read header info
  csize = *(short*)&in[csize - 2];
}


// copy (len) bytes from global memory (source) to shared memory (destination) using separate shared memory buffer (temp)
// destination and temp must we word aligned, accesses up to CS + 3 bytes in temp
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
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
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
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        buf_w[i] = in_w[i];
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        buf[i] = input[i];
      }
      __syncthreads();
      buf_w = (int*)buffer;
      for (int i = tid; i < (len + 3) / 4; i += TPB) {
        out_w[i] = __funnelshift_r(buf_w[i], buf_w[i + 1], shift);
      }
    }
  }
}


static __device__ unsigned long long g_chunk_counter;


static __global__ void d_reset()
{
  g_chunk_counter = 0LL;
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 4)
#else
static __global__ __launch_bounds__(TPB, 3)
#endif
void d_decode(const byte* const __restrict__ input, byte* const __restrict__ output, long long* const __restrict__ g_outsize, const int thr_e, const int offs)
{
  // allocate shared memory buffer
  __shared__ long long chunk [2 * (CS / sizeof(long long)) + 16];
  const int last = 2 * (CS / sizeof(long long)) + 1;

  // input header
  long long* const head_in = (long long*)input;
  const long long outsize = head_in[0];

  // initialize
  const long long chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  long long prevChunkID = 0;
  long long prevOffset = 0;
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const long long chunkID = chunk[last];
    const long long base = chunkID * CS;
    if (base >= outsize) break;

    // compute sum of all prior csizes (start where left off in previous iteration)
    long long sum = 0;
    for (long long i = prevChunkID + tid; i < chunkID; i += TPB) {
      sum += (long long)size_in[i];
    }
    int csize = (int)size_in[chunkID];

    // create the 3 shared memory buffers
    byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
    byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

    const long long offs = prevOffset + block_sum_reduction(sum, (long long*)out); //chunk[last + 1]);
    prevChunkID = chunkID;
    prevOffset = offs;
    __syncthreads();

    // load chunk
    g2s(in, &data_in[offs], csize, out);
    __syncthreads();  // chunk produced, chunk[last] consumed

    // decode
    const int osize = (int)min((long long)CS, outsize - base);
    if (csize < osize) {
      d_iSLEEK(csize, in, out,temp, thr_e, offs);
    } else {
      byte* tmp = in; in = out; out = tmp; // swap if no decode
    }
    __syncthreads();

    if (csize != osize) {printf("ERROR: csize %d doesn't match osize %d in chunk %lld\n\n", csize, osize, chunkID); __trap();}
    long long* const output_l = (long long*)&output[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      output_l[i] = out_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) output[base + osize - extra + tid] = out[osize - extra + tid];
  } while (true);

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}


struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg); cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg); cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};


static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
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
  printf("GPU SLEEK 1.0: single-precision lossy decompressor\n");
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

  // get GPU info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); return -1;}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / TPB);
  CheckCuda(__LINE__);

  // allocate GPU memory
  byte* ddecoded;
  cudaMallocHost((void **)&ddecoded, pre_size);
  byte* d_encoded;
  cudaMalloc((void **)&d_encoded, insize);
  cudaMemcpy(d_encoded, hencoded, insize, cudaMemcpyHostToDevice);
  byte* d_decoded;
  cudaMalloc((void **)&d_decoded, pre_size);
  long long* d_decsize;
  cudaMalloc((void **)&d_decsize, sizeof(long long));
  CheckCuda(__LINE__);

  #if defined(ARTIFACT)
    if (perf) {
      // warm up
      byte* d_decoded_dummy;
      cudaMalloc((void **)&d_decoded_dummy, pre_size);
      long long* d_decsize_dummy;
      cudaMalloc((void **)&d_decsize_dummy, sizeof(long long));
      d_decode<<<blocks, TPB>>>(d_encoded, d_decoded_dummy, d_decsize_dummy, thr_e, offs);
      cudaFree(d_decoded_dummy);
      cudaFree(d_decsize_dummy);
    }
  #endif

  // time GPU decoding
  GPUTimer dtimer;
  long long ddecsize = 0;
  dtimer.start();
  d_reset<<<1, 1>>>();
  d_decode<<<blocks, TPB>>>(d_encoded, d_decoded, d_decsize, thr_e, offs);
  cudaDeviceSynchronize();
  double runtime = dtimer.stop();

  cudaMemcpy(&ddecsize, d_decsize, sizeof(long long), cudaMemcpyDeviceToHost);
  // get decoded GPU result
  cudaMemcpy(ddecoded, d_decoded, ddecsize, cudaMemcpyDeviceToHost);
  printf("decoded size: %lld bytes\n", ddecsize);
  CheckCuda(__LINE__);

  #if defined(ARTIFACT)
    // verify
    MAXABS_f32(insize_orig, ddecoded, input_orig, eb);
  #endif

  const float CR = (100.0 * insize) / ddecsize;
  printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

  #if defined(ARTIFACT)
    if (perf) {
      printf("decoding time: %.6f s\n", runtime);
      double throughput = ddecsize * 0.000000001 / runtime;
      printf("decoding throughput: %8.3f Gbytes/s\n", throughput);
      CheckCuda(__LINE__);
    }
  #endif

  // write to file
  FILE* const fout = fopen(argv[idx_output], "wb");
  fwrite(ddecoded, 1, ddecsize, fout);
  fclose(fout);

  // clean up GPU memory
  cudaFree(d_encoded);
  cudaFree(d_decoded);
  cudaFree(d_decsize);
  CheckCuda(__LINE__);

  // clean up
  cudaFreeHost(ddecoded);
  return 0;
}
