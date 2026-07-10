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
#include <string>
#include <cmath>
#include <limits>
#include <cstring>
#include <cstdio>
#include <algorithm>


using byte = unsigned char;
static const int SLEEK_CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]


#define clz(val) \
    (sizeof(T) == sizeof(unsigned int) ? __builtin_clz(static_cast<unsigned int>(val)) : \
     __builtin_clzll(static_cast<unsigned long long>(val)))

#define to_float(val) \
    (sizeof(type_f) == sizeof(float) ? std::stof(val) : std::stod(val))


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


//
// lossless
//


template <typename T>
static void h_encode(byte* const __restrict__ input, const long long insize, byte* const __restrict__ output, long long& outsize)
{
  // initialize
  const long long chunks = (insize + SLEEK_CS - 1) / SLEEK_CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];
  long long* const carry = new long long [chunks];
  memset(carry, 0, chunks * sizeof(long long));

  // encode chunk
  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = SLEEK_CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;  // size of subchunk in words

  static_assert(sizeof(T) >= 4);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  T tmp_partial_chunk [size]; // used for the last chunk if not a full chunk

  // process chunks in parallel
#pragma omp parallel for schedule(dynamic, 1) default(none) shared(chunks, insize, input, carry, size_out, data_out, tmp_partial_chunk)
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    T tmp_buffer_t [size];
    byte out [SLEEK_CS];
    const long long base = chunkID * SLEEK_CS;
    const int osize = (int)std::min((long long)SLEEK_CS, insize - base);
    byte* in = &input[base];

    // clear unused part of input buffer
    if (osize < SLEEK_CS) {
      memset((byte*) tmp_partial_chunk, 0, SLEEK_CS);
      std::copy(in, in + osize, (byte*) tmp_partial_chunk);
      in = (byte*) tmp_partial_chunk;
    }

    T* const in_t = (T*)in;

    // determine bits needed for each subchunk
    int bits = 0;
    for (int i = 0; i < SC; i++) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;

      T max_val = 0;
      for (int j = beg; j < end; j++) {
        T val = in_t[j];
        if constexpr (TB == 32) {
          val = ((val << 1) | (val >> 31));
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
        tmp_buffer_t[j] = val;
        max_val = std::max(max_val, val);
      }

      int cnt = TB;
      if (max_val != 0) {
        cnt = clz(max_val);
      }
      int ln = TB - cnt;  // logn value for subchunk
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
    if ((newsize < SLEEK_CS) && (newsize < osize)) {
      // store carry of compressed data
#pragma omp atomic write
      carry[chunkID] = (offs + (long long)newsize);
      size_out[chunkID] = newsize;

      // clear out buffer
      T* const out_t = (T*)&out[SC];
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
              const T val = tmp_buffer_t[j];
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


template <typename T>
static void h_decode(const byte* const __restrict__ input, byte* const __restrict__ output, long long& outsize)
{
  // input header
  long long* const head_in = (long long*)input;
  outsize = head_in[0];

  // initialize
  const long long chunks = (outsize + SLEEK_CS - 1) / SLEEK_CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];
  long long* const start = new long long [chunks];

  // convert chunk sizes into starting positions
  long long pfs = 0;
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    start[chunkID] = pfs;
    pfs += (long long)size_in[chunkID];
  }

  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = SLEEK_CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(sizeof(T) >= 4);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);
  
  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1) default(none) shared(chunks, outsize, size_in, data_in, start, output, stderr) 
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    long long chunk2 [SLEEK_CS / sizeof(long long)];
    byte* out = (byte*)chunk2;
    const long long base = chunkID * SLEEK_CS;
    const int osize = (int)std::min((long long)SLEEK_CS, outsize - base);
    int csize = size_in[chunkID];
    if (csize == osize) {
      // simply copy
      memcpy(&output[base], &data_in[start[chunkID]], osize);
    } else {
      byte* in = &data_in[start[chunkID]];
      // decode data values
      int startPos = 0;
      const T* const in_t = (T*)&in[SC];
      T* const out_t = (T*)out;
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
            T val = in_t[offs + j];
            if constexpr (TB == 32) {
              val = (val >> 1) ^ (((int)(val << 31)) >> 31);  // iTCMS
              if ((val & 0xff00'0000) != 0) {
                if (val >= 0x8000'0000) {
                  val += 0x0100'0000;
                }
                val += 0x8000'0000;
              }
              val = (val << 31) | (val >> 1);
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
          int loc = startPos;
          const T mask = (logn == 64) ? (~0ULL) : ((1ULL << logn) - 1);
          for (int j = beg; j < end; j++) {
            const int pos = loc / TB;
            const int shift = loc % TB;
            T res = in_t[pos] >> shift;
            if (TB - shift < logn) {
              res |= in_t[pos + 1] << (TB - shift);
            }
            loc += logn;
            T val = res & mask;
            if constexpr (TB == 32) {
              val = (val >> 1) ^ (((int)(val << 31)) >> 31);  // iTCMS
              if ((val & 0xff00'0000) != 0) {
                if (val >= 0x8000'0000) {
                  val += 0x0100'0000;
                }
                val += 0x8000'0000;
              }
              val = (val << 31) | (val >> 1);
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
        startPos += chunksize * logn;
      }

      // read header info
      csize = *(short*)&in[csize - 2];

      if (csize != osize) {fprintf(stderr, "ERROR: csize %d does not match osize %d in chunk %lld\n\n", csize, osize, chunkID);}
      memcpy(&output[base], out, csize);
    }
  }

  // finish
  delete [] start;
}


//
// lossy
//


template <typename T>
static inline T quantize(const T val, const int eb_e, const int thr_e, const T offs)
{
  using U = typename std::conditional<sizeof(T) == 4, int, long long>::type;

  constexpr const int e = (sizeof(T) == 4) ? 8 : 11;
  constexpr const int m = (sizeof(T) == 4) ? 23 : 52;

  const U abs = val & ((1ULL << (e + m)) - 1ULL);  // compute absolute value
  const int val_e = abs >> m;  // extract exponent
  U enc = 0LL;  // default value is 0

  if (val_e >= thr_e) {  // at or above threshold
    enc = abs - offs;  // lossless encoding
  } else if (val_e >= eb_e) {  // lossy encoding
    U mant = val & ((1LL << m) - 1LL);  // extract mantissa
    const int shift = thr_e - val_e;  // bias cancels out
    mant |= 1LL << m;  // insert implicit 1
    mant += 1LL << (shift - 1);  // round to nearest, ties round away from zero
    enc = mant >> shift;  // shift out unnecessary bits
  }

  enc = (enc << 1) | (~val >> (e + m));  // magnitude ~sign
  if (enc != 0LL) enc--;  // -0 -> +0 and fill gap

  return enc;
}


template <typename T>
static void h_encode(byte* input, const long long insize, byte* const __restrict__ output, long long& outsize, const int eb_e, const int thr_e, const T offs_e)
{
  // initialize
  const long long chunks = (insize + SLEEK_CS - 1) / SLEEK_CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];
  long long* const carry = new long long [chunks];
  memset(carry, 0, chunks * sizeof(long long));

  // encode chunk
  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = SLEEK_CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;  // size of subchunk in words

  static_assert(sizeof(T) >= 4);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  T tmp_partial_chunk [size]; // used for the last chunk if not a full chunk

  // process chunks in parallel
#pragma omp parallel for schedule(dynamic, 1) default(none) shared(chunks, insize, input, offs_e, thr_e, eb_e, carry, size_out, data_out, tmp_partial_chunk)
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    T tmp_buffer_t [size];
    byte out [SLEEK_CS];
    const long long base = chunkID * SLEEK_CS;
    const int osize = (int)std::min((long long)SLEEK_CS, insize - base);
    byte* in = &input[base];

    // clear unused part of input buffer
    if (osize < SLEEK_CS) {
      memset((byte*) tmp_partial_chunk, 0, SLEEK_CS);
      std::copy(in, in + osize, (byte*) tmp_partial_chunk);
      in = (byte*) tmp_partial_chunk;
    }

    T* const in_t = (T*)in;

    // determine bits needed for each subchunk
    int bits = 0;
    for (int i = 0; i < SC; i++) {
      const int beg = i * chunksize;
      const int end = beg + chunksize;
      T max_val = 0;
      for (int j = beg; j < end; j++) {
        const T val = quantize(in_t[j], eb_e, thr_e, offs_e);
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
      T* const out_t = (T*)&out[SC];
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
              const T val = tmp_buffer_t[j];
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


template <typename T>
static inline T deQuantize(const T enc, const int thr_e, const T offs)
{
  using U = typename std::conditional<sizeof(T) == 4, int, long long>::type;

  constexpr const int e = (sizeof(T) == 4) ? 8 : 11;
  constexpr const int m = (sizeof(T) == 4) ? 23 : 52;

  U dec = 0LL;  // default value is 0
  if (enc != 0LL) {
    const U abs = (enc + 1) >> 1;  // absolute value
    if (abs >= (1LL << m)) {  // above threshold
      dec = abs + offs;  // decode losslessly
    } else if (abs > 0LL) {  // non-zero lossy case

      if constexpr (sizeof(T) == 4) {
        const int shift = __builtin_clz(abs) - (31 - m);  // compute shift amount
        dec = abs << shift;  // shift to normalized position
        dec &= (1 << m) - 1;  // remove implied 1
        dec |= (thr_e - shift) << m;  // insert biased exponent
      } else {
        const int shift = __builtin_clzll(abs) - (63 - m);  // compute shift amount
        dec = abs << shift;  // shift to normalized position
        dec &= (1LL << m) - 1LL;  // remove implied 1
        dec |= (long long)(thr_e - shift) << m;  // insert biased exponent
      }

    }
    dec |= enc << (e + m);  // insert sign bit
  }

  return dec;
}


template <typename T>
static void h_decode(const byte* const __restrict__ input, byte* const __restrict__ output, long long& outsize, const int thr_e, const T offs_e)
{
  // input header
  long long* const head_in = (long long*)input;
  outsize = head_in[0];

  // initialize
  const long long chunks = (outsize + SLEEK_CS - 1) / SLEEK_CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];
  long long* const start = new long long [chunks];

  // convert chunk sizes into starting positions
  long long pfs = 0;
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    start[chunkID] = pfs;
    pfs += (long long)size_in[chunkID];
  }

  const int TB = sizeof(T) * 8;  // number of bits in T
  const int size = SLEEK_CS / sizeof(T);
  const int SC = 32;  // subchunks [do not change]
  const int chunksize = size / SC;

  static_assert(sizeof(T) >= 4);
  static_assert(SC == sizeof(int) * 8);
  static_assert(std::is_unsigned<T>::value);

  // process chunks in parallel
  #pragma omp parallel for schedule(dynamic, 1) default(none) shared(chunks, outsize, size_in, data_in, start, output, thr_e, offs_e, stderr) 
  for (long long chunkID = 0; chunkID < chunks; chunkID++) {
    // load chunk
    long long chunk2 [SLEEK_CS / sizeof(long long)];
    byte* out = (byte*)chunk2;
    const long long base = chunkID * SLEEK_CS;
    const int osize = (int)std::min((long long)SLEEK_CS, outsize - base);
    int csize = size_in[chunkID];
    if (csize == osize) {
      // simply copy
      memcpy(&output[base], &data_in[start[chunkID]], osize);
    } else {
      const byte* const in = &data_in[start[chunkID]];

      // decode data values
      int startPos = 0;
      const T* const in_t = (T*)&in[SC];
      T* const out_t = (T*)out;
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
            const T val = in_t[offs + j];
            out_t[j] = deQuantize(val, thr_e, offs_e);
            
          }
        } else {
          int loc = startPos;
          const T mask = (logn == 64) ? (~0ULL) : ((1ULL << logn) - 1);
          for (int j = beg; j < end; j++) {
            const int pos = loc / TB;
            const int shift = loc % TB;
            T res = in_t[pos] >> shift;
            if (TB - shift < logn) {
              res |= in_t[pos + 1] << (TB - shift);
            }
            loc += logn;
            const T val = res & mask;
            out_t[j] = deQuantize(val, thr_e, offs_e);
          }
        }
        startPos += chunksize * logn;
      }

      // read header info
      csize = *(short*)&in[csize - 2];

      if (csize != osize) {fprintf(stderr, "ERROR: csize %d does not match osize %d in chunk %lld\n\n", csize, osize, chunkID);}
      memcpy(&output[base], out, csize);
    }
  }

  // finish
  delete [] start;
}
