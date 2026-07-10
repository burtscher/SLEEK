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


#include "sleek_lossy_cpu.h"
#include <sys/time.h>
#include "MAXABS_f32.h"


struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


#if defined(ARTIFACT)
  static inline float computeNOAeb(const float* const input, const long long size, const float eb_param)
  {
    float min = input[0], max = input[0];
    #pragma omp parallel for default(none) shared(size, input) reduction(max: max) reduction(min: min)
    for (long long i = 1; i < size; i++) {
      const float val = input[i];
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
  printf("CPU SLEEK 1.0: single-precision lossy decompressor\n");
  printf("Copyright 2026 Texas State University\n\n");

  #if defined(ARTIFACT)
    if (argc != 5) {printf("USAGE: %s compressed_file_name decompressed_file_name error_bound original_file_name\n\n", argv[0]); return -1;}
  #else
    if (argc != 4) {printf("USAGE: %s compressed_file_name decompressed_file_name error_bound\n\n", argv[0]); return -1;}
  #endif

  // system check
  if (sleek_system_checks() != 0) {return -1;}

  // read input file
  FILE* const fin = fopen(argv[1], "rb");
  long long pre_size = 0;
  const long long pre_val = fread(&pre_size, sizeof(pre_size), 1, fin); assert(pre_val == 1);
  fseek(fin, 0, SEEK_END);
  const long long hencsize = ftell(fin);  assert(hencsize > 0);
  byte* const hencoded = new byte [std::max(pre_size, hencsize)];
  fseek(fin, 0, SEEK_SET);
  // all "*size" variables in bytes
  const long long insize = fread(hencoded, 1, hencsize, fin);  assert(insize == hencsize);
  fclose(fin);
  printf("encoded size: %lld bytes\n", insize);

  // allocate CPU memory
  float* hdecoded = new float [pre_size / sizeof(float)];
  long long hdecsize = 0;

  #if defined(ARTIFACT)

    // read original file
    FILE* const fin_orig = fopen(argv[4], "rb");
    fseek(fin_orig, 0, SEEK_END);
    const long long fsize_orig = ftell(fin_orig);
    if (fsize_orig <= 0) {fprintf(stderr, "ERROR: original input file too small\n\n"); return -1;}
    const long long num_elements = fsize_orig / sizeof(float);
    float* const input_orig = new float [num_elements];
    fseek(fin_orig, 0, SEEK_SET);
    const long long insize_orig = fread(input_orig, 1, fsize_orig, fin_orig);  assert(insize_orig == fsize_orig);
    fclose(fin_orig);

    // eb variables
    const float parameter = std::stof(argv[3]);
    const float eb = computeNOAeb((float*) input_orig, insize_orig / sizeof(float), parameter);
    const auto eb_info = sleek_error_bound(eb);

    // warm up
    float* dummy = new float [pre_size / sizeof(float)];
    long long dummy_size = 0;
    sleek_decompress_cpu(eb_info, hencoded, dummy, dummy_size);
    delete [] dummy;

    // time CPU decoding
    CPUTimer htimer;
    htimer.start();
    sleek_decompress_cpu(eb_info, hencoded, hdecoded, hdecsize);
    double hruntime = htimer.stop();

    // verify
    MAXABS_f32(insize_orig, (byte*) hdecoded, (byte*) input_orig, eb);

    printf("decoded size: %lld bytes\n", hdecsize);
    const float CR = (100.0 * insize) / hdecsize;
    printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

    printf("decoding time: %.6f s\n", hruntime);
    double hthroughput = hdecsize * 0.000000001 / hruntime;
    printf("decoding throughput: %8.3f Gbytes/s\n", hthroughput);

  #else

    // eb variables
    const float eb = std::stof(argv[3]);
    const auto eb_info = sleek_error_bound(eb);

    sleek_decompress_cpu(eb_info, hencoded, hdecoded, hdecsize);

    printf("decoded size: %lld bytes\n", hdecsize);
    const float CR = (100.0 * insize) / hdecsize;
    printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

  #endif

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(hdecoded, 1, hdecsize, fout);
  fclose(fout);

  delete [] hencoded;
  delete [] hdecoded;
  return 0;
}
