#pragma once

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

#include <anns.hpp>
#include <string>
#include <cassert>
#include <iostream>
#include <cmath>

#if defined(__AVX512F__)
static const std::string g_simd_architecture = "avx512";
static const size_t bf_upbound_lim_float = 16;
static const size_t bf_upbound_lim_uint8 = 32;

#elif defined(__AVX__)
static const std::string g_simd_architecture = "avx";
static const size_t bf_upbound_lim_float = 16;
static const size_t bf_upbound_lim_uint8 = 32;

#elif defined(__SSE__)
static const std::string g_simd_architecture = "sse";
static const size_t bf_upbound_lim_float = 16;
static const size_t bf_upbound_lim_uint8 = 32;

#else
static const std::string g_simd_architecture = "none";
#endif

#if defined(__AVX512F__)
static inline __m512 _mm512_hadd_ps(__m512 a)
{
  __m512 sum1 = _mm512_add_ps(a, _mm512_maskz_shuffle_ps(0x55, a, a, _MM_SHUFFLE(2, 3, 0, 1)));
  __m512 sum2 = _mm512_add_ps(sum1, _mm512_maskz_shuffle_ps(0xAA, sum1, sum1, _MM_SHUFFLE(1, 0, 3, 2)));
  __m512 sum3 = _mm512_add_ps(sum2, _mm512_maskz_shuffle_ps(0xFF, sum2, sum2, _MM_SHUFFLE(0, 1, 2, 3)));
  return sum3;
}
#endif

// Reading function for SSE, AVX, and AVX512
// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read(size_t d, const float *x)
{
  assert(d < 4);

#if defined(_MSC_VER)
  // alignas (alignof(16)) float buf[4] = { 0, 0, 0, 0 };
  __declspec(align(16)) float buf[4] = {0, 0, 0, 0};
#else
  __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
#endif

  switch (d)
  {
  case 3:
    buf[2] = x[2];
  case 2:
    buf[1] = x[1];
  case 1:
    buf[0] = x[0];
  }
  return _mm_load_ps(buf);
  // cannot use AVX2 _mm_mask_set1_epi32
}

static inline __m128i masked_read(size_t d, const uint8_t *x)
{
  assert(d < 16);
#if defined(_MSC_VER)
  __declspec(align(16)) uint8_t buf[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#else
  __attribute__((__aligned__(16))) uint8_t buf[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif

  switch (d)
  {
  case 15:
    buf[14] = x[14];
  case 14:
    buf[13] = x[13];
  case 13:
    buf[12] = x[12];
  case 12:
    buf[11] = x[11];
  case 11:
    buf[10] = x[10];
  case 10:
    buf[9] = x[9];
  case 9:
    buf[8] = x[8];
  case 8:
    buf[7] = x[7];
  case 7:
    buf[6] = x[6];
  case 6:
    buf[5] = x[5];
  case 5:
    buf[4] = x[4];
  case 4:
    buf[3] = x[3];
  case 3:
    buf[2] = x[2];
  case 2:
    buf[1] = x[1];
  case 1:
    buf[0] = x[0];
  }
  return _mm_load_si128((const __m128i *)buf);
}

namespace anns
{

  namespace metrics
  {

#if defined(__AVX512F__)

    // AVX512 implementation by Yusuke
    static float euclidean(const float *x, const float *y, size_t d)
    {
      /// @brief when dimension < 20, the runtime of distance function show that bruteforce < simd.
      if (d < bf_upbound_lim_float)
      {
        float dist = 0;
        for (size_t i = 0; i < d; i++)
        {
          dist += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return dist;
      }

      __m512 msum1 = _mm512_setzero_ps();

      while (d >= 16)
      {
        __m512 mx = _mm512_loadu_ps(x);
        __m512 my = _mm512_loadu_ps(y);
        const __m512 a_m_b1 = _mm512_sub_ps(mx, my);
        // msum1 += a_m_b1 * a_m_b1;
        msum1 = _mm512_add_ps(msum1, _mm512_mul_ps(a_m_b1, a_m_b1));
        x += 16;
        y += 16;
        d -= 16;
      }
      __m256 msum2 = _mm512_extractf32x8_ps(msum1, 1);
      // msum2 += _mm512_extractf32x8_ps(msum1, 0);
      msum2 = _mm256_add_ps(msum2, _mm512_extractf32x8_ps(msum1, 0));

      while (d >= 8)
      {
        __m256 mx = _mm256_loadu_ps(x);
        __m256 my = _mm256_loadu_ps(y);
        // const __m256 a_m_b1 = mx - my;
        const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
        // msum2 += a_m_b1 * a_m_b1;
        msum2 = _mm256_add_ps(msum2, _mm256_mul_ps(a_m_b1, a_m_b1));
        x += 8;
        y += 8;
        d -= 8;
      }

      __m128 msum3 = _mm256_extractf128_ps(msum2, 1);
      // msum3 += _mm256_extractf128_ps(msum2, 0);
      msum3 = _mm_add_ps(msum3, _mm256_extractf128_ps(msum2, 0));

      if (d >= 4)
      {
        __m128 mx = _mm_loadu_ps(x);
        __m128 my = _mm_loadu_ps(y);
        // const __m128 a_m_b1 = mx - my;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum3 += a_m_b1 * a_m_b1;
        msum3 = _mm_add_ps(msum3, _mm_mul_ps(a_m_b1, a_m_b1));
        x += 4;
        y += 4;
        d -= 4;
      }

      if (d > 0)
      {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        // __m128 a_m_b1 = mx - my;
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum3 += a_m_b1 * a_m_b1;
        msum3 = _mm_add_ps(msum3, _mm_mul_ps(a_m_b1, a_m_b1));
      }

      msum3 = _mm_hadd_ps(msum3, msum3);
      msum3 = _mm_hadd_ps(msum3, msum3);
      return _mm_cvtss_f32(msum3);
    }

    static float euclidean(const uint8_t *x, const uint8_t *y, size_t d)
    {
      if (d < bf_upbound_lim_uint8)
      {
        float dist = 0;
        for (size_t i = 0; i < d; i++)
        {
          dist += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return dist;
      }

      __m512 msum1 = _mm512_setzero_ps();
      const __m512i m512i_zero = _mm512_setzero_si512();

      while (d >= 64)
      {
        __m512i mx = _mm512_loadu_si512((const __m512i *)x);
        __m512i my = _mm512_loadu_si512((const __m512i *)y);
        __m512i lo_m_32i16 = _mm512_subs_epi16(_mm512_unpacklo_epi8(mx, m512i_zero), _mm512_unpacklo_epi8(my, m512i_zero));
        __m512i hi_m_32i16 = _mm512_subs_epi16(_mm512_unpackhi_epi8(mx, m512i_zero), _mm512_unpackhi_epi8(my, m512i_zero));
        __m512i lo_m_32i16_2 = _mm512_mullo_epi16(lo_m_32i16, lo_m_32i16);
        __m512i hi_m_32i16_2 = _mm512_mullo_epi16(hi_m_32i16, hi_m_32i16);
        __m512i lolo_m_16i32 = _mm512_unpacklo_epi16(lo_m_32i16_2, m512i_zero);
        __m512i hilo_m_16i32 = _mm512_unpackhi_epi16(lo_m_32i16_2, m512i_zero);
        __m512i lohi_m_16i32 = _mm512_unpacklo_epi16(hi_m_32i16_2, m512i_zero);
        __m512i hohi_m_16i32 = _mm512_unpackhi_epi16(hi_m_32i16_2, m512i_zero);
        __m512i lo_m_32i32 = _mm512_add_epi32(lolo_m_16i32, hilo_m_16i32);
        __m512i hi_m_32i32 = _mm512_add_epi32(lohi_m_16i32, hohi_m_16i32);
        msum1 = _mm512_add_ps(msum1, _mm512_cvtepi32_ps(lo_m_32i32));
        msum1 = _mm512_add_ps(msum1, _mm512_cvtepi32_ps(hi_m_32i32));
        x += 64;
        y += 64;
        d -= 64;
      }

      __m256 msum2 = _mm512_extractf32x8_ps(msum1, 1);
      msum2 = _mm256_add_ps(msum2, _mm512_extractf32x8_ps(msum1, 0));
      const __m256i m256i_zero = _mm256_setzero_si256();

      while (d >= 32)
      {
        __m256i mx = _mm256_loadu_si256((const __m256i *)x);
        __m256i my = _mm256_loadu_si256((const __m256i *)y);
        __m256i lo_m_16i16 = _mm256_subs_epi16(_mm256_unpacklo_epi8(mx, m256i_zero), _mm256_unpacklo_epi8(my, m256i_zero));
        __m256i hi_m_16i16 = _mm256_subs_epi16(_mm256_unpackhi_epi8(mx, m256i_zero), _mm256_unpackhi_epi8(my, m256i_zero));
        __m256i lo_m_16i16_2 = _mm256_mullo_epi16(lo_m_16i16, lo_m_16i16);
        __m256i hi_m_16i16_2 = _mm256_mullo_epi16(hi_m_16i16, hi_m_16i16);
        __m256i lolo_m_8i32 = _mm256_unpacklo_epi16(lo_m_16i16_2, m256i_zero);
        __m256i lohi_m_8i32 = _mm256_unpackhi_epi16(lo_m_16i16_2, m256i_zero);
        __m256i hilo_m_8i32 = _mm256_unpacklo_epi16(hi_m_16i16_2, m256i_zero);
        __m256i hihi_m_8i32 = _mm256_unpackhi_epi16(hi_m_16i16_2, m256i_zero);
        __m256i lo_m_8i32 = _mm256_add_epi32(lolo_m_8i32, hilo_m_8i32);
        __m256i hi_m_8i32 = _mm256_add_epi32(lohi_m_8i32, hihi_m_8i32);
        msum2 = _mm256_add_ps(msum2, _mm256_cvtepi32_ps(lo_m_8i32));
        msum2 = _mm256_add_ps(msum2, _mm256_cvtepi32_ps(hi_m_8i32));
        x += 32;
        y += 32;
        d -= 32;
      }

      __m128 msum3 = _mm256_extractf32x4_ps(msum2, 1);
      msum3 = _mm_add_ps(msum3, _mm256_extractf32x4_ps(msum2, 0));
      const __m128i m128i_zero = _mm_setzero_si128();

      while (d >= 16)
      {
        __m128i mx = _mm_loadu_si128((const __m128i *)x);
        __m128i my = _mm_loadu_si128((const __m128i *)y);
        __m128i lo_m_8i16 = _mm_subs_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16 = _mm_subs_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(lo_m_8i16, lo_m_8i16);
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(hi_m_8i16, hi_m_8i16);
        __m128i lolo_m_4i32 = _mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero);
        __m128i hilo_m_4i32 = _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero);
        __m128i lohi_m_4i32 = _mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero);
        __m128i hohi_m_4i32 = _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero);
        __m128i lo_m_4i32 = _mm_add_epi32(lolo_m_4i32, hilo_m_4i32);
        __m128i hi_m_4i32 = _mm_add_epi32(lohi_m_4i32, hohi_m_4i32);
        msum3 = _mm_add_ps(msum3, _mm_cvtepi32_ps(lo_m_4i32));
        msum3 = _mm_add_ps(msum3, _mm_cvtepi32_ps(hi_m_4i32));
        x += 16;
        y += 16;
        d -= 16;
      }

      if (d > 0)
      {
        __m128i mx = masked_read(d, x);
        __m128i my = masked_read(d, y);
        __m128i lo_m_8i16 = _mm_subs_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16 = _mm_subs_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(lo_m_8i16, lo_m_8i16);
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(hi_m_8i16, hi_m_8i16);
        __m128i lolo_m_4i32 = _mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero);
        __m128i hilo_m_4i32 = _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero);
        __m128i lohi_m_4i32 = _mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero);
        __m128i hohi_m_4i32 = _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero);
        __m128i lo_m_4i32 = _mm_add_epi32(lolo_m_4i32, hilo_m_4i32);
        __m128i hi_m_4i32 = _mm_add_epi32(lohi_m_4i32, hohi_m_4i32);
        msum3 = _mm_add_ps(msum3, _mm_cvtepi32_ps(lo_m_4i32));
        msum3 = _mm_add_ps(msum3, _mm_cvtepi32_ps(hi_m_4i32));
      }

      msum3 = _mm_hadd_ps(msum3, msum3);
      msum3 = _mm_hadd_ps(msum3, msum3);

      return _mm_cvtss_f32(msum3);
    }

#elif defined(__AVX__)

    // AVX implementation
    static float euclidean(const float *x, const float *y, size_t d)
    {
      if (d < bf_upbound_lim_float)
      {
        float dist = 0;
        for (size_t i = 0; i < d; i++)
        {
          dist += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return dist;
      }

      __m256 msum1 = _mm256_setzero_ps();

      while (d >= 8)
      {
        __m256 mx = _mm256_loadu_ps(x);
        __m256 my = _mm256_loadu_ps(y);
        // const __m256 a_m_b1 = mx - my;
        const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
        // msum1 += a_m_b1 * a_m_b1;
        msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(a_m_b1, a_m_b1));
        x += 8;
        y += 8;
        d -= 8;
      }

      __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
      // msum2 += _mm256_extractf128_ps(msum1, 0);
      msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

      if (d >= 4)
      {
        __m128 mx = _mm_loadu_ps(x);
        __m128 my = _mm_loadu_ps(y);
        // const __m128 a_m_b1 = mx - my;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum2 += a_m_b1 * a_m_b1;
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
        x += 4;
        y += 4;
        d -= 4;
      }

      if (d > 0)
      {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        // __m128 a_m_b1 = mx - my;
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum2 += a_m_b1 * a_m_b1;
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(a_m_b1, a_m_b1));
      }

      msum2 = _mm_hadd_ps(msum2, msum2);
      msum2 = _mm_hadd_ps(msum2, msum2);
      return _mm_cvtss_f32(msum2);
    }

    static float euclidean(const uint8_t *x, const uint8_t *y, size_t d)
    {
      if (d < bf_upbound_lim_uint8)
      {
        float dist = 0;
        for (size_t i = 0; i < d; i++)
        {
          dist += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return dist;
      }

      __m256 msum1 = _mm256_setzero_ps();
      const __m256i m256i_zero = _mm256_setzero_si256();

      while (d >= 32)
      {
        __m256i mx = _mm256_loadu_si256((const __m256i *)x);
        __m256i my = _mm256_loadu_si256((const __m256i *)y);
        __m256i lo_m_16i16 = _mm256_subs_epi16(_mm256_unpacklo_epi8(mx, m256i_zero), _mm256_unpacklo_epi8(my, m256i_zero));
        __m256i hi_m_16i16 = _mm256_subs_epi16(_mm256_unpackhi_epi8(mx, m256i_zero), _mm256_unpackhi_epi8(my, m256i_zero));
        __m256i lo_m_16i16_2 = _mm256_mullo_epi16(lo_m_16i16, lo_m_16i16);
        __m256i hi_m_16i16_2 = _mm256_mullo_epi16(hi_m_16i16, hi_m_16i16);
        __m256i lolo_m_8i32 = _mm256_unpacklo_epi16(lo_m_16i16_2, m256i_zero);
        __m256i lohi_m_8i32 = _mm256_unpackhi_epi16(lo_m_16i16_2, m256i_zero);
        __m256i hilo_m_8i32 = _mm256_unpacklo_epi16(hi_m_16i16_2, m256i_zero);
        __m256i hihi_m_8i32 = _mm256_unpackhi_epi16(hi_m_16i16_2, m256i_zero);
        __m256i lo_m_8i32 = _mm256_add_epi32(lolo_m_8i32, hilo_m_8i32);
        __m256i hi_m_8i32 = _mm256_add_epi32(lohi_m_8i32, hihi_m_8i32);
        msum1 = _mm256_add_ps(msum1, _mm256_cvtepi32_ps(lo_m_8i32));
        msum1 = _mm256_add_ps(msum1, _mm256_cvtepi32_ps(hi_m_8i32));
        x += 32;
        y += 32;
        d -= 32;
      }

      __m128 msum2 = _mm256_extractf32x4_ps(msum1, 1);
      msum2 = _mm_add_ps(msum2, _mm256_extractf32x4_ps(msum1, 0));
      const __m128i m128i_zero = _mm_setzero_si128();

      while (d >= 16)
      {
        __m128i mx = _mm_loadu_si128((const __m128i *)x);
        __m128i my = _mm_loadu_si128((const __m128i *)y);
        __m128i lo_m_8i16 = _mm_subs_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16 = _mm_subs_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(lo_m_8i16, lo_m_8i16);
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(hi_m_8i16, hi_m_8i16);
        __m128i lolo_m_4i32 = _mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero);
        __m128i hilo_m_4i32 = _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero);
        __m128i lohi_m_4i32 = _mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero);
        __m128i hohi_m_4i32 = _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero);
        __m128i lo_m_4i32 = _mm_add_epi32(lolo_m_4i32, hilo_m_4i32);
        __m128i hi_m_4i32 = _mm_add_epi32(lohi_m_4i32, hohi_m_4i32);
        msum2 = _mm_add_ps(msum2, _mm_cvtepi32_ps(lo_m_4i32));
        msum2 = _mm_add_ps(msum2, _mm_cvtepi32_ps(hi_m_4i32));
        x += 16;
        y += 16;
        d -= 16;
      }

      if (d > 0)
      {
        __m128i mx = masked_read(d, x);
        __m128i my = masked_read(d, y);
        __m128i lo_m_8i16 = _mm_subs_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16 = _mm_subs_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(lo_m_8i16, lo_m_8i16);
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(hi_m_8i16, hi_m_8i16);
        __m128i lolo_m_4i32 = _mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero);
        __m128i hilo_m_4i32 = _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero);
        __m128i lohi_m_4i32 = _mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero);
        __m128i hohi_m_4i32 = _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero);
        __m128i lo_m_4i32 = _mm_add_epi32(lolo_m_4i32, hilo_m_4i32);
        __m128i hi_m_4i32 = _mm_add_epi32(lohi_m_4i32, hohi_m_4i32);
        msum2 = _mm_add_ps(msum2, _mm_cvtepi32_ps(lo_m_4i32));
        msum2 = _mm_add_ps(msum2, _mm_cvtepi32_ps(hi_m_4i32));
      }

      msum2 = _mm_hadd_ps(msum2, msum2);
      msum2 = _mm_hadd_ps(msum2, msum2);

      return _mm_cvtss_f32(msum2);
    }

#elif defined(__SSE__)

    // distance for vector<float>
    static float euclidean(const float *x, const float *y, size_t d)
    {
      if (d < bf_upbound_lim_float)
      {
        float dist = 0;
        for (size_t i = 0; i < d; i++)
        {
          dist += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return dist;
      }

      __m128 msum1 = _mm_setzero_ps();

      while (d >= 4)
      {
        __m128 mx = _mm_loadu_ps(x);
        __m128 my = _mm_loadu_ps(y);
        // const __m128 a_m_b1 = mx - my;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum1 += a_m_b1 * a_m_b1;
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
        x += 4;
        y += 4;
        d -= 4;
      }

      if (d > 0)
      {
        // add the last 1, 2 or 3 values
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        // __m128 a_m_b1 = mx - my;
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        // msum1 += a_m_b1 * a_m_b1;
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
      }

      msum1 = _mm_hadd_ps(msum1, msum1);
      msum1 = _mm_hadd_ps(msum1, msum1);
      return _mm_cvtss_f32(msum1);
    }

    // distance for vector<uint8_t>
    static float euclidean(const uint8_t *x, const uint8_t *y, size_t d)
    {
      if (d < bf_upbound_lim_uint8)
      {
        float dist = 0;
        for (size_t i = 0; i < d; i++)
        {
          dist += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return dist;
      }

      __m128 msum = _mm_setzero_ps();
      const __m128i m128i_zero = _mm_setzero_si128();

      while (d >= 16)
      {
        __m128i mx = _mm_loadu_si128((const __m128i *)x);
        __m128i my = _mm_loadu_si128((const __m128i *)y);
        __m128i lo_m_8i16 = _mm_subs_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16 = _mm_subs_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(lo_m_8i16, lo_m_8i16);
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(hi_m_8i16, hi_m_8i16);
        __m128i lolo_m_4i32 = _mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero);
        __m128i hilo_m_4i32 = _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero);
        __m128i lohi_m_4i32 = _mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero);
        __m128i hohi_m_4i32 = _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero);
        __m128i lo_m_4i32 = _mm_add_epi32(lolo_m_4i32, hilo_m_4i32);
        __m128i hi_m_4i32 = _mm_add_epi32(lohi_m_4i32, hohi_m_4i32);
        msum = _mm_add_ps(msum, _mm_cvtepi32_ps(lo_m_4i32));
        msum = _mm_add_ps(msum, _mm_cvtepi32_ps(hi_m_4i32));
        x += 16;
        y += 16;
        d -= 16;
      }

      if (d > 0)
      {
        __m128i mx = masked_read(d, x);
        __m128i my = masked_read(d, y);
        __m128i lo_m_8i16 = _mm_subs_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16 = _mm_subs_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(lo_m_8i16, lo_m_8i16);
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(hi_m_8i16, hi_m_8i16);
        __m128i lolo_m_4i32 = _mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero);
        __m128i hilo_m_4i32 = _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero);
        __m128i lohi_m_4i32 = _mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero);
        __m128i hohi_m_4i32 = _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero);
        __m128i lo_m_4i32 = _mm_add_epi32(lolo_m_4i32, hilo_m_4i32);
        __m128i hi_m_4i32 = _mm_add_epi32(lohi_m_4i32, hohi_m_4i32);
        msum = _mm_add_ps(msum, _mm_cvtepi32_ps(lo_m_4i32));
        msum = _mm_add_ps(msum, _mm_cvtepi32_ps(hi_m_4i32));
      }

      msum = _mm_hadd_ps(msum, msum);
      msum = _mm_hadd_ps(msum, msum);

      return _mm_cvtss_f32(msum);
    }

#else

    static float euclidean(const float *x, const float *y, size_t d)
    {
      float dist = 0, diff;
      for (size_t i = 0; i < d; i++)
      {
        diff = x[i] - y[i];
        dist += diff * diff;
      }
      return dist;
    }

    static float euclidean(const uint8_t *x, const uint8_t *y, size_t d)
    {
      float dist = 0, diff;
      for (size_t i = 0; i < d; i++)
      {
        diff = x[i] - y[i];
        dist += diff * diff;
      }
      return dist;
    }

#endif

#if defined(__AVX512F__)

    static float inner_product(const float *x, const float *y, size_t d)
    {
      if (d < bf_upbound_lim_float)
      {
        float dist = 0;
        for (size_t i = 0; i < d; ++i)
          dist += x[i] * y[i];
        return -dist;
      }

      __m512 msum1 = _mm512_setzero_ps();

      while (d >= 16)
      {
        __m512 mx = _mm512_loadu_ps(x);
        __m512 my = _mm512_loadu_ps(y);
        msum1 = _mm512_add_ps(msum1, _mm512_mul_ps(mx, my));
        x += 16;
        y += 16;
        d -= 16;
      }

      __m256 msum2 = _mm512_extractf32x8_ps(msum1, 1);
      msum2 = _mm256_add_ps(msum2, _mm512_extractf32x8_ps(msum1, 0));

      while (d >= 8)
      {
        __m256 mx = _mm256_loadu_ps(x);
        __m256 my = _mm256_loadu_ps(y);
        msum2 = _mm256_add_ps(msum2, _mm256_mul_ps(mx, my));
        x += 8;
        y += 8;
        d -= 8;
      }

      __m128 msum3 = _mm256_extractf128_ps(msum2, 1);
      msum3 = _mm_add_ps(msum3, _mm256_extractf128_ps(msum2, 0));

      while (d >= 4)
      {
        __m128 mx = _mm_loadu_ps(x);
        __m128 my = _mm_loadu_ps(y);
        msum3 = _mm_add_ps(msum3, _mm_mul_ps(mx, my));
        x += 4;
        y += 4;
        d -= 4;
      }

      if (d > 0)
      {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        msum3 = _mm_add_ps(msum3, _mm_mul_ps(mx, my));
      }

      msum3 = _mm_hadd_ps(msum3, msum3);
      msum3 = _mm_hadd_ps(msum3, msum3);
      return -_mm_cvtss_f32(msum3);
    }

    static float inner_product(const uint8_t *x, const uint8_t *y, size_t d)
    {
      if (d < bf_upbound_lim_uint8)
      {
        float dist = 0;
        for (size_t i = 0; i < d; i++)
        {
          dist += x[i] * y[i];
        }
        return -dist;
      }

      __m512 msum1 = _mm512_setzero_ps();
      const __m512i m512i_zero = _mm512_setzero_si512();

      while (d >= 64)
      {
        __m512i mx = _mm512_loadu_si512((const __m512i *)x);
        __m512i my = _mm512_loadu_si512((const __m512i *)y);
        __m512i lo_m_32i16_2 = _mm512_mullo_epi16(_mm512_unpacklo_epi8(mx, m512i_zero), _mm512_unpacklo_epi8(my, m512i_zero));
        __m512i hi_m_32i16_2 = _mm512_mullo_epi16(_mm512_unpackhi_epi8(mx, m512i_zero), _mm512_unpackhi_epi8(my, m512i_zero));
        __m512i lo_m_32i32 = _mm512_add_epi32(_mm512_unpacklo_epi16(lo_m_32i16_2, m512i_zero), _mm512_unpackhi_epi16(lo_m_32i16_2, m512i_zero));
        __m512i hi_m_32i32 = _mm512_add_epi32(_mm512_unpacklo_epi16(hi_m_32i16_2, m512i_zero), _mm512_unpackhi_epi16(hi_m_32i16_2, m512i_zero));
        msum1 = _mm512_add_ps(msum1, _mm512_cvtepi32_ps(lo_m_32i32));
        msum1 = _mm512_add_ps(msum1, _mm512_cvtepi32_ps(hi_m_32i32));
        x += 64;
        y += 64;
        d -= 64;
      }

      __m256 msum2 = _mm512_extractf32x8_ps(msum1, 1);
      msum2 = _mm256_add_ps(msum2, _mm512_extractf32x8_ps(msum1, 0));
      const __m256i m256i_zero = _mm256_setzero_si256();

      while (d >= 32)
      {
        __m256i mx = _mm256_loadu_si256((const __m256i *)x);
        __m256i my = _mm256_loadu_si256((const __m256i *)y);
        __m256i lo_m_16i16_2 = _mm256_mullo_epi16(_mm256_unpacklo_epi8(mx, m256i_zero), _mm256_unpacklo_epi8(my, m256i_zero));
        __m256i hi_m_16i16_2 = _mm256_mullo_epi16(_mm256_unpackhi_epi8(mx, m256i_zero), _mm256_unpackhi_epi8(my, m256i_zero));
        __m256i lo_m_8i32 = _mm256_add_epi32(_mm256_unpacklo_epi16(lo_m_16i16_2, m256i_zero), _mm256_unpacklo_epi16(hi_m_16i16_2, m256i_zero));
        __m256i hi_m_8i32 = _mm256_add_epi32(_mm256_unpackhi_epi16(lo_m_16i16_2, m256i_zero), _mm256_unpackhi_epi16(hi_m_16i16_2, m256i_zero));
        msum2 = _mm256_add_ps(msum2, _mm256_cvtepi32_ps(lo_m_8i32));
        msum2 = _mm256_add_ps(msum2, _mm256_cvtepi32_ps(hi_m_8i32));
        x += 32;
        y += 32;
        d -= 32;
      }

      __m128 msum3 = _mm256_extractf32x4_ps(msum2, 1);
      msum3 = _mm_add_ps(msum3, _mm256_extractf32x4_ps(msum2, 0));
      const __m128i m128i_zero = _mm_setzero_si128();

      while (d >= 16)
      {
        __m128i mx = _mm_loadu_si128((const __m128i *)x);
        __m128i my = _mm_loadu_si128((const __m128i *)y);
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero));
        __m128i hi_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero));
        msum3 = _mm_add_ps(msum3, _mm_cvtepi32_ps(lo_m_4i32));
        msum3 = _mm_add_ps(msum3, _mm_cvtepi32_ps(hi_m_4i32));
        x += 16;
        y += 16;
        d -= 16;
      }

      if (d > 0)
      {
        __m128i mx = masked_read(d, x);
        __m128i my = masked_read(d, y);
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero));
        __m128i hi_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero));
        msum3 = _mm_add_ps(msum3, _mm_cvtepi32_ps(lo_m_4i32));
        msum3 = _mm_add_ps(msum3, _mm_cvtepi32_ps(hi_m_4i32));
      }

      msum3 = _mm_hadd_ps(msum3, msum3);
      msum3 = _mm_hadd_ps(msum3, msum3);

      return -_mm_cvtss_f32(msum3);
    }

#elif defined(__AVX__)

    static float inner_product(const float *x, const float *y, size_t d)
    {
      if (d < bf_upbound_lim_float)
      {
        float dist = 0;
        for (size_t i = 0; i < d; i++)
        {
          dist += x[i] * y[i];
        }
        return -dist;
      }

      __m256 msum1 = _mm256_setzero_ps();

      while (d >= 8)
      {
        __m256 mx = _mm256_loadu_ps(x);
        __m256 my = _mm256_loadu_ps(y);
        msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(mx, my));
        x += 8;
        y += 8;
        d -= 8;
      }

      __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
      // msum2 += _mm256_extractf128_ps(msum1, 0);
      msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

      if (d >= 4)
      {
        __m128 mx = _mm_loadu_ps(x);
        __m128 my = _mm_loadu_ps(y);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
        x += 4;
        y += 4;
        d -= 4;
      }

      if (d > 0)
      {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
      }

      msum2 = _mm_hadd_ps(msum2, msum2);
      msum2 = _mm_hadd_ps(msum2, msum2);
      return -_mm_cvtss_f32(msum2);
    }

    static float inner_product(const uint8_t *x, const uint8_t *y, size_t d)
    {
      if (d < bf_upbound_lim_uint8)
      {
        float dist = 0;
        for (size_t i = 0; i < d; i++)
        {
          dist += x[i] * y[i];
        }
        return -dist;
      }

      __m256 msum1 = _mm256_setzero_ps();
      const __m256i m256i_zero = _mm256_setzero_si256();

      while (d >= 32)
      {
        __m256i mx = _mm256_loadu_si256((const __m256i *)x);
        __m256i my = _mm256_loadu_si256((const __m256i *)y);
        __m256i lo_m_16i16_2 = _mm256_mullo_epi16(_mm256_unpacklo_epi8(mx, m256i_zero), _mm256_unpacklo_epi8(my, m256i_zero));
        __m256i hi_m_16i16_2 = _mm256_mullo_epi16(_mm256_unpackhi_epi8(mx, m256i_zero), _mm256_unpackhi_epi8(my, m256i_zero));
        __m256i lo_m_8i32 = _mm256_add_epi32(_mm256_unpacklo_epi16(lo_m_16i16_2, m256i_zero), _mm256_unpacklo_epi16(hi_m_16i16_2, m256i_zero));
        __m256i hi_m_8i32 = _mm256_add_epi32(_mm256_unpackhi_epi16(lo_m_16i16_2, m256i_zero), _mm256_unpackhi_epi16(hi_m_16i16_2, m256i_zero));
        msum1 = _mm256_add_ps(msum1, _mm256_cvtepi32_ps(lo_m_8i32));
        msum1 = _mm256_add_ps(msum1, _mm256_cvtepi32_ps(hi_m_8i32));
        x += 32;
        y += 32;
        d -= 32;
      }

      __m128 msum2 = _mm256_extractf32x4_ps(msum1, 1);
      msum2 = _mm_add_ps(msum2, _mm256_extractf32x4_ps(msum1, 0));
      const __m128i m128i_zero = _mm_setzero_si128();

      while (d >= 16)
      {
        __m128i mx = _mm_loadu_si128((const __m128i *)x);
        __m128i my = _mm_loadu_si128((const __m128i *)y);
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero));
        __m128i hi_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero));
        msum2 = _mm_add_ps(msum2, _mm_cvtepi32_ps(lo_m_4i32));
        msum2 = _mm_add_ps(msum2, _mm_cvtepi32_ps(hi_m_4i32));
        x += 16;
        y += 16;
        d -= 16;
      }

      if (d > 0)
      {
        __m128i mx = masked_read(d, x);
        __m128i my = masked_read(d, y);
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero));
        __m128i hi_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero));
        msum2 = _mm_add_ps(msum2, _mm_cvtepi32_ps(lo_m_4i32));
        msum2 = _mm_add_ps(msum2, _mm_cvtepi32_ps(hi_m_4i32));
      }

      msum2 = _mm_hadd_ps(msum2, msum2);
      msum2 = _mm_hadd_ps(msum2, msum2);

      return -_mm_cvtss_f32(msum2);
    }

#elif defined(__SSE__)

    static float inner_product(const float *x, const float *y, size_t d)
    {
      if (d < bf_upbound_lim_float)
      {
        float dist = 0;
        for (size_t i = 0; i < d; i++)
        {
          dist += x[i] * y[i];
        }
        return -dist;
      }

      __m128 msum1 = _mm_setzero_ps();

      while (d >= 4)
      {
        __m128 mx = _mm_loadu_ps(x);
        __m128 my = _mm_loadu_ps(y);
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, my));
        x += 4;
        y += 4;
        d -= 4;
      }

      if (d > 0)
      {
        // add the last 1, 2 or 3 values
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, my));
      }

      msum1 = _mm_hadd_ps(msum1, msum1);
      msum1 = _mm_hadd_ps(msum1, msum1);
      return -_mm_cvtss_f32(msum1);
    }

    static float inner_product(const uint8_t *x, const uint8_t *y, size_t d)
    {
      if (d < bf_upbound_lim_uint8)
      {
        float dist = 0;
        for (size_t i = 0; i < d; i++)
        {
          dist += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return -dist;
      }

      __m128 msum = _mm_setzero_ps();
      const __m128i m128i_zero = _mm_setzero_si128();

      while (d >= 16)
      {
        __m128i mx = _mm_loadu_si128((const __m128i *)x);
        __m128i my = _mm_loadu_si128((const __m128i *)y);
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero));
        __m128i hi_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero));
        msum = _mm_add_ps(msum, _mm_cvtepi32_ps(lo_m_4i32));
        msum = _mm_add_ps(msum, _mm_cvtepi32_ps(hi_m_4i32));
        x += 16;
        y += 16;
        d -= 16;
      }

      if (d > 0)
      {
        __m128i mx = masked_read(d, x);
        __m128i my = masked_read(d, y);
        __m128i lo_m_8i16_2 = _mm_mullo_epi16(_mm_unpacklo_epi8(mx, m128i_zero), _mm_unpacklo_epi8(my, m128i_zero));
        __m128i hi_m_8i16_2 = _mm_mullo_epi16(_mm_unpackhi_epi8(mx, m128i_zero), _mm_unpackhi_epi8(my, m128i_zero));
        __m128i lo_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(lo_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(lo_m_8i16_2, m128i_zero));
        __m128i hi_m_4i32 = _mm_add_epi32(_mm_unpacklo_epi16(hi_m_8i16_2, m128i_zero), _mm_unpackhi_epi16(hi_m_8i16_2, m128i_zero));
        msum = _mm_add_ps(msum, _mm_cvtepi32_ps(lo_m_4i32));
        msum = _mm_add_ps(msum, _mm_cvtepi32_ps(hi_m_4i32));
      }

      msum = _mm_hadd_ps(msum, msum);
      msum = _mm_hadd_ps(msum, msum);

      return -_mm_cvtss_f32(msum);
    }

#else

    static float inner_product(const float *x, const float *y, size_t d)
    {
      float dist = 0;
      for (size_t i = 0; i < d; i++)
      {
        dist += x[i] * y[i];
      }
      return dist;
    }

    static float inner_product(const uint8_t *x, const uint8_t *y, size_t d)
    {
      float dist = 0;
      for (size_t i = 0; i < d; i++)
      {
        dist += x[i] * y[i];
      }
      return dist;
    }

#endif

    static float hamming(const float *x, const float *y, size_t d)
    {
      float dist = 0;
      for (size_t i = 0; i < d; i++)
      {
        dist += fabsf(x[i] - y[i]);
      }
      return dist;
    }

    static float hamming(const uint8_t *x, const uint8_t *y, size_t d)
    {
      float dist = 0;
      for (size_t i = 0; i < d; i++)
      {
        dist += fabsf(x[i] - y[i]);
      }
      return dist;
    }

    static float cosine(const float *x, const float *y, size_t d)
    {
      assert(d <= MAGIC_DIMENSION);
      static const float zero_float[MAGIC_DIMENSION] = {0};
      return inner_product(x, y, d) / sqrt(euclidean(x, zero_float, d) * euclidean(y, zero_float, d));
    }

    static float cosine(const uint8_t *x, const uint8_t *y, size_t d)
    {
      assert(d <= MAGIC_DIMENSION);
      static const uint8_t zero_uint8[MAGIC_DIMENSION] = {0};
      return inner_product(x, y, d) / sqrt(euclidean(x, zero_uint8, d) * euclidean(y, zero_uint8, d));
    }

  }

}