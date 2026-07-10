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


#include "sleek_lossless_cpu.h"
#include <sys/time.h>


struct CPUTimer
{
  timeval beg, end;
  CPUTimer() {}
  ~CPUTimer() {}
  void start() {gettimeofday(&beg, NULL);}
  double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};


int main(int argc, char* argv [])
{
  printf("CPU SLEEK 1.0: single-precision lossless decompressor\n");
  printf("Copyright 2026 Texas State University\n\n");

  if (argc != 3) {printf("USAGE: %s compressed_file_name decompressed_file_name\n\n", argv[0]); return -1;}

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

    // warm up
    float* dummy = new float [pre_size / sizeof(float)];
    long long dummy_size = 0;
    sleek_decompress_cpu(hencoded, dummy, dummy_size);
    delete [] dummy;

    // time CPU decoding
    CPUTimer htimer;
    htimer.start();
    sleek_decompress_cpu(hencoded, hdecoded, hdecsize);
    double hruntime = htimer.stop();

    printf("decoded size: %lld bytes\n", hdecsize);
    const float CR = (100.0 * insize) / hdecsize;
    printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

    printf("decoding time: %.6f s\n", hruntime);
    double hthroughput = hdecsize * 0.000000001 / hruntime;
    printf("decoding throughput: %8.3f Gbytes/s\n", hthroughput);

  #else

    sleek_decompress_cpu(hencoded, hdecoded, hdecsize);

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
