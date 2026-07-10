/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2026, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher
All rights reserved.

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

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


static void MAXABS_f32(const long long size, const byte* const __restrict__ recon, const byte* const __restrict__ orig, const float errorbound)
{
  using type_f = float;
  using type_i = int;
  assert(sizeof(type_f) == sizeof(type_i));

  if ((size % sizeof(type_f)) != 0) {fprintf(stderr, "ERROR: MAXABS_f32 requires data to be a multiple of %ld bytes long\n", sizeof(type_f)); exit(-1);}
  if (errorbound <= 0) {fprintf(stderr, "ERROR: MAXABS_f32 requires the maximum allowed absolute error to be greater than zero\n"); exit(-1);}

  const type_f* const orig_f = (type_f*)orig;
  const type_f* const recon_f = (type_f*)recon;
  const long long len = size / sizeof(type_f);

  type_f max_diff = 0.0;
  long long num_eb_violations = 0;
  type_f min = orig_f[0], max = orig_f[0];
  long long cnt1 = 0;

  for (long long i = 0; i < len; i++) {
    // collect min and max values in original input
    const type_f val = orig_f[i];
    if (val < min)
      min = val;
    else if (val > max)
      max = val;

    if (!std::isfinite(orig_f[i]) || !std::isfinite(recon_f[i])) {  // at least one value is INF or NaN
      if (recon_f[i] != orig_f[i]) { // Note: Comparison of two NaNs will always return non-equal
        if (!std::isnan(orig_f[i]) || !std::isnan(recon_f[i])) {  // at least one value isn't a NaN => it is infinity
          if (cnt1 == 0) {
            fprintf(stderr, "MAXABS_f32 ERROR: absolute error bound exceeded due to NaN or INF at position %lld: reconstructed value is '%.10f' vs original value: '%.10f'\n\n", i, recon_f[i], orig_f[i]);
          }
          cnt1++;
        }
      }
    } else {
      max_diff = std::max(max_diff, std::abs(orig_f[i] - recon_f[i]));
      const type_f lower = orig_f[i] - errorbound;
      const type_f upper = orig_f[i] + errorbound;
      if ((recon_f[i] < lower) || (recon_f[i] > upper) || (std::abs(orig_f[i] - recon_f[i]) > errorbound)) {
        num_eb_violations++;
      }
    }
  }

  if (num_eb_violations == 0 && cnt1 == 0) {
    printf("MAXABS_f32 verification passed\n");
  }
  printf("min_val: %.10f\n", min);
  printf("max_val: %.10f\n", max);
  printf("diff between max and min values in original input: %.10f\n", (max - min));
  printf("num_eb_violations: %lld (%.2f%%) \n", num_eb_violations, 100.0 * ((float)num_eb_violations / len));  // not counting overflows and underflows 
  printf("max_diff between original and reconstructed values: %.10f\n", max_diff);
  printf("inf_nan_violations: %lld (%.2f%%) \n", cnt1, 100.0 * (float)cnt1 / len);
  printf("Number of values in original file: %lld\n", len);
}
