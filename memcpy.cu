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

#include <cmath>
#include <string>
#include <cassert>
#include <cuda.h>


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


int main(int argc, char* argv [])
{
  printf("GPU_memcpy\n");
  printf("Copyright 2026 Texas State University\n\n");

  // read input from file
  if (argc < 2) {printf("USAGE: %s file_name\n\n", argv[0]); return -1;}

  // check GPU
  const int device = 0;
  cudaSetDevice(device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); exit(-1);}
  
  // read input file
  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {fprintf(stderr, "ERROR: input file too small\n\n"); return -1;}
  byte* const input = new byte [fsize];
  fseek(fin, 0, SEEK_SET);
  const long long insize = fread(input, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);
  printf("original size: %lld bytes\n", insize);

  // allocate GPU memory
  byte* d_encoded;
  cudaMalloc((void **)&d_encoded, insize);

  byte* d_decoded;
  cudaMalloc((void **)&d_decoded, insize);

  CheckCuda(__LINE__);

  GPUTimer dtimer;
  double runtime, throughput;

  // host to device

  dtimer.start();
  cudaMemcpy(d_encoded, input, insize, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  runtime = dtimer.stop();
  printf("HtD time: %.6f s\n", runtime);
  throughput = insize * 0.000000001 / runtime;
  printf("HtD throughput: %8.3f Gbytes/s\n", throughput);

  // device to device

  dtimer.start();
  cudaMemcpy(d_decoded, d_encoded, insize, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  runtime = dtimer.stop();
  printf("DtD time: %.6f s\n", runtime);
  throughput = insize * 0.000000001 / runtime;
  printf("DtD throughput: %8.3f Gbytes/s\n", throughput);

  // device to host

  dtimer.start();
  cudaMemcpy(input, d_encoded, insize, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  runtime = dtimer.stop();
  printf("DtH time: %.6f s\n", runtime);
  throughput = insize * 0.000000001 / runtime;
  printf("DtH throughput: %8.3f Gbytes/s\n", throughput);

  // clean up GPU memory
  cudaFree(d_encoded);
  cudaFree(d_decoded);
  CheckCuda(__LINE__);
  delete [] input;
  return 0;
}