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


#include "sleek_lossy_gpu.h"
#include <string>
#include <cmath>
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


#if defined(ARTIFACT)
  static inline double computeNOAeb(const double* const input, const long long size, const double eb_param)
  {
    double min = input[0], max = input[0];
    #pragma omp parallel for default(none) shared(size, input) reduction(max: max) reduction(min: min)
    for (long long i = 1; i < size; i++) {
      const double val = input[i];
      min = std::min(val, min);
      max = std::max(val, max);
    }

    printf("min_val: %.10f\n", min);
    printf("max_val: %.10f\n", max);
    printf("diff between max and min values: %.10f\n", (max - min));
    return (max - min) * eb_param;
  }
#endif


int main(int argc, char* argv [])
{
  printf("GPU SLEEK 1.0: double-precision lossy compressor\n");
  printf("Copyright 2026 Texas State University\n\n");

  if (argc != 4) {printf("USAGE: %s input_file_name compressed_file_name error_bound\n\n", argv[0]); return -1;}

  // system check
  if (sleek_system_checks() != 0) {return -1;}

  // read input file
  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {fprintf(stderr, "ERROR: input file too small\n\n"); return -1;}
  const long long num_elements = fsize / sizeof(double);
  double* const input = new double [num_elements];
  fseek(fin, 0, SEEK_SET);
  // all "*size" variables in bytes
  const long long insize = fread(input, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);
  printf("original size: %lld bytes\n", insize);

  if (insize % sizeof(double) != 0) {fprintf(stderr, "ERROR: size of input must be a multiple of %ld bytes\n", sizeof(double)); return -1;}

  // allocate GPU memory
  const long long maxsize = sleek_upper_bound_size(insize);
  byte* dencoded;
  cudaMallocHost((void **)&dencoded, maxsize);
  double* d_input;
  cudaMalloc((void **)&d_input, insize);
  cudaMemcpy(d_input, input, insize, cudaMemcpyHostToDevice);
  byte* d_encoded;
  cudaMalloc((void **)&d_encoded, maxsize);
  long long* d_encsize;
  cudaMalloc((void **)&d_encsize, sizeof(long long));
  CheckCuda(__LINE__);

  const int blocks = sleek_thread_blocks();

  #if defined(ARTIFACT)

    // eb variables
    const double parameter = std::stod(argv[3]);
    const double eb = computeNOAeb(input, num_elements, parameter);
    const auto eb_info = sleek_error_bound(eb);

    // warm up
    sleek_compress_gpu(blocks, eb_info, d_input, num_elements, d_encoded, d_encsize);
    cudaDeviceSynchronize();
    CheckCuda(__LINE__);

    GPUTimer dtimer;
    dtimer.start();
    sleek_compress_gpu(blocks, eb_info, d_input, num_elements, d_encoded, d_encsize);
    double runtime = dtimer.stop();

    long long dencsize = 0;
    cudaMemcpy(&dencsize, d_encsize, sizeof(long long), cudaMemcpyDeviceToHost);

    // get encoded GPU result
    cudaMemcpy(dencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);
    printf("encoded size: %lld bytes\n", dencsize);
    CheckCuda(__LINE__);

    const float CR = (100.0 * dencsize) / insize;
    printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

    printf("encoding time: %.6f s\n", runtime);
    double throughput = insize * 0.000000001 / runtime;
    printf("encoding throughput: %8.3f Gbytes/s\n", throughput);
    CheckCuda(__LINE__);

  #else

    // eb variables
    const double eb = std::stod(argv[3]);
    const auto eb_info = sleek_error_bound(eb);

    sleek_compress_gpu(blocks, eb_info, d_input, num_elements, d_encoded, d_encsize);
    long long dencsize = 0;
    cudaMemcpy(&dencsize, d_encsize, sizeof(long long), cudaMemcpyDeviceToHost);

    // get encoded GPU result
    cudaMemcpy(dencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);
    printf("encoded size: %lld bytes\n", dencsize);
    CheckCuda(__LINE__);

    const float CR = (100.0 * dencsize) / insize;
    printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

  #endif

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(dencoded, 1, dencsize, fout);
  fclose(fout);

  // clean up GPU memory
  cudaFree(d_input);
  cudaFree(d_encoded);
  cudaFree(d_encsize);
  CheckCuda(__LINE__);

  // clean up
  delete [] input;
  cudaFreeHost(dencoded);
  return 0;
}
