/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#ifndef _SIMD_H
#define _SIMD_H

#if defined(__x86_64__)

#if defined(__AVX512F__)
#include <x86intrin.h>
#define HAS_F32x16
typedef __m512 F32x16;
#define F32x16_ZERO() _mm512_setzero_ps()
#define F32x16_STORE(ptr, value) _mm512_store_ps((ptr), (value))
#define F32x16_LOAD(ptr) _mm512_load_ps((ptr))
#define F32x16_MADD(acc, a, b) _mm512_fmadd_ps((a), (b), (acc))
#define F32x16_BROAD(ptr) _mm512_broadcastss_ps(_mm_load_ss(ptr))
#define HAS_F64x8
typedef __m512 F64x8;
#define F64x8_ZERO() _mm512_setzero_pd()
#define F64x8_STORE(ptr, value) _mm512_store_pd((ptr), (value))
#define F64x8_LOAD(ptr) _mm512_load_pd((ptr))
#define F64x8_MADD(acc, a, b) _mm512_fmadd_pd((a), (b), (acc))
#define F64x8_BROAD(ptr) _mm512_broadcastss_pd(_mm_load_ss(ptr))
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
#include <x86intrin.h>
#define HAS_F32x8
typedef __m256 F32x8;
#define F32x8_ZERO() _mm256_setzero_ps()
#define F32x8_STORE(ptr, value) _mm256_store_ps((ptr), (value))
#define F32x8_LOAD(ptr) _mm256_load_ps((ptr))
#define F32x8_MADD(acc, a, b) _mm256_fmadd_ps((a), (b), (acc))
#define F32x8_BROAD(ptr) _mm256_broadcast_ss((ptr))
#endif

// Default for SSE only
#include <x86intrin.h>
#define HAS_F32x4
typedef __m128 F32x4;
#define F32x4_ZERO() _mm_setzero_ps()
#define F32x4_STORE(ptr, value) _mm_store_ps((ptr), (value))
#define F32x4_LOAD(ptr) _mm_load_ps((ptr))
#define F32x4_MADD(acc, a, b) _mm_add_ps((acc), _mm_mul_ps((a), (b)))
#define F32x4_BROAD(ptr) _mm_load_ps1((ptr))

#elif defined(__aarch64__)

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define HAS_F32x4
typedef float32x4_t F32x4;
#define F32x4_ZERO() vmovq_n_f32(0)
#define F32x4_STORE(ptr, value) vst1q_f32((ptr), (value))
#define F32x4_LOAD(ptr) vld1q_f32((ptr))
#define F32x4_MADD(acc, a, b) vfmaq_f32((acc), (a), (b))
#define F32x4_BROAD(ptr) vmovq_n_f32(*(ptr))
#endif

#endif

#if !defined(HAS_F32x4)
typedef struct { float f[4]; } F32x4;
#define F32x4_ZERO() ({ F32x4 _value; for (int i = 0; i < 4; i++) _value.f[i] = 0.0F; _value;})
#define F32x4_STORE(ptr, value) ((void)({ F32x4 _value = (value); float *_ptr = (ptr); for (int i = 0; i < 4; i++) _ptr[i] = _value.f[i]; 0;}))
#define F32x4_LOAD(ptr) ({ F32x4 _value; const float *_ptr = (ptr); for (int i = 0; i < 4; i++) _value.f[i] = _ptr[i]; _value;})
#define F32x4_MADD(acc, a, b) ({ F32x4 _acc = (acc), _a = (a), _b = (b); for (int i = 0; i < 4; i++) _acc.f[i] += _a.f[i] * _b.f[i]; _acc;})
#define F32x4_BROAD(ptr) ({ F32x4 _value; float _scalar = *(ptr); for (int i = 0; i < 4; i++) _value.f[i] = _scalar; _value;})
#endif

#if !defined(HAS_F32x8)
#if defined(HAS_F32x4)
typedef struct { F32x4 f[2]; } F32x8;
#define F32x8_ZERO() ({ F32x8 _value; _value.f[0] =  F32x4_ZERO(); _value.f[1] = F32x4_ZERO(); _value;})
#define F32x8_STORE(ptr, value) ((void)({ F32x8 _value = (value); float *_ptr = (ptr); F32x4_STORE(_ptr, _value.f[0]); F32x4_STORE(_ptr+4, _value.f[1]); 0;}))
#define F32x8_LOAD(ptr) ({ F32x8 _value; const float *_ptr = (ptr); _value.f[0] = F32x4_LOAD(_ptr); _value.f[1] = F32x4_LOAD(_ptr+4); _value;})
#define F32x8_MADD(acc, a, b) ({ F32x8 _acc = (acc), _a = (a), _b = (b); _acc.f[0] = F32x4_MADD(_acc.f[0], _a.f[0], _b.f[0]); _acc.f[1] = F32x4_MADD(_acc.f[1], _a.f[1], _b.f[1]); _acc;})
#define F32x8_BROAD(ptr) ({ F32x8 _value; const float *_ptr = (ptr); _value.f[0] = F32x4_BROAD(_ptr); _value.f[1] = F32x4_BROAD(_ptr); _value;})
#else
typedef struct { float f[8]; } F32x8;
#define F32x8_ZERO() ({ F32x8 _value; for (int i = 0; i < 8; i++) _value.f[i] = 0.0F; _value;})
#define F32x8_STORE(ptr, value) ((void)({ F32x8 _value = (value); float *_ptr = (ptr); for (int i = 0; i < 8; i++) _ptr[i] = _value.f[i]; 0;}))
#define F32x8_LOAD(ptr) ({ F32x8 _value; const float *_ptr = (ptr); for (int i = 0; i < 8; i++) _value.f[i] = _ptr[i]; _value;})
#define F32x8_MADD(acc, a, b) ({ F32x8 _acc = (acc), _a = (a), _b = (b); for (int i = 0; i < 8; i++) _acc.f[i] += _a.f[i] * _b.f[i]; _acc;})
#define F32x8_BROAD(ptr) ({ F32x8 _value; float _scalar = *(ptr); for (int i = 0; i < 8; i++) _value.f[i] = _scalar; _value;})
#endif
#endif

#if !defined(HAS_F32x16)
#if defined(HAS_F32x8)
typedef struct { F32x8 f[2]; } F32x16;
#define F32x16_ZERO() ({ F32x16 _value; _value.f[0] =  F32x8_ZERO(); _value.f[1] = F32x8_ZERO(); _value;})
#define F32x16_STORE(ptr, value) ((void)({ F32x16 _value = (value); float *_ptr = (ptr); F32x8_STORE(_ptr, _value.f[0]); F32x8_STORE(_ptr+8, _value.f[1]); 0;}))
#define F32x16_LOAD(ptr) ({ F32x16 _value; const float *_ptr = (ptr); _value.f[0] = F32x8_LOAD(_ptr); _value.f[1] = F32x8_LOAD(_ptr+8); _value;})
#define F32x16_MADD(acc, a, b) ({ F32x16 _acc = (acc), _a = (a), _b = (b); _acc.f[0] = F32x8_MADD(_acc.f[0], _a.f[0], _b.f[0]); _acc.f[1] = F32x8_MADD(_acc.f[1], _a.f[1], _b.f[1]); _acc;})
#define F32x16_BROAD(ptr) ({ F32x16 _value; const float *_ptr = (ptr); _value.f[0] = F32x8_BROAD(_ptr); _value.f[1] = F32x8_BROAD(_ptr); _value;})
#elif defined(HAS_F32x4)
typedef struct { F32x4 f[4]; } F32x16;
#define F32x16_ZERO() ({ F32x16 _value; _value.f[0] =  F32x4_ZERO(); _value.f[1] = F32x4_ZERO(); _value.f[2] =  F32x4_ZERO(); _value.f[3] = F32x4_ZERO(); _value;})
#define F32x16_STORE(ptr, value) ((void)({ F32x16 _value = (value); float *_ptr = (ptr); F32x4_STORE(_ptr, _value.f[0]); F32x4_STORE(_ptr+4, _value.f[1]); F32x4_STORE(_ptr+8, _value.f[2]); F32x4_STORE(_ptr+12, _value.f[3]); 0;}))
#define F32x16_LOAD(ptr) ({ F32x16 _value; const float *_ptr = (ptr); _value.f[0] = F32x4_LOAD(_ptr); _value.f[1] = F32x4_LOAD(_ptr+4); _value.f[2] = F32x4_LOAD(_ptr+8); _value.f[3] = F32x4_LOAD(_ptr+12); _value;})
#define F32x16_MADD(acc, a, b) ({ F32x16 _acc = (acc), _a = (a), _b = (b); _acc.f[0] = F32x4_MADD(_acc.f[0], _a.f[0], _b.f[0]); _acc.f[1] = F32x4_MADD(_acc.f[1], _a.f[1], _b.f[1]); _acc.f[2] = F32x4_MADD(_acc.f[2], _a.f[2], _b.f[2]); _acc.f[3] = F32x4_MADD(_acc.f[3], _a.f[3], _b.f[3]); _acc;})
#define F32x16_BROAD(ptr) ({ F32x16 _value; const float *_ptr = (ptr); _value.f[0] = F32x4_BROAD(_ptr); _value.f[1] = F32x4_BROAD(_ptr); _value.f[2] = F32x4_BROAD(_ptr); _value.f[3] = F32x4_BROAD(_ptr); _value;})
#else
typedef struct { float f[16]; } F32x16;
#define F32x16_ZERO() ({ F32x16 _value; for (int i = 0; i < 16; i++) _value.f[i] = 0.0F; _value;})
#define F32x16_STORE(ptr, value) ((void)({ F32x16 _value = (value); float *_ptr = (ptr); for (int i = 0; i < 16; i++) _ptr[i] = _value.f[i]; 0;}))
#define F32x16_LOAD(ptr) ({ F32x16 _value; const float *_ptr = (ptr); for (int i = 0; i < 16; i++) _value.f[i] = _ptr[i]; _value;})
#define F32x16_MADD(acc, a, b) ({ F32x16 _acc = (acc), _a = (a), _b = (b); for (int i = 0; i < 16; i++) _acc.f[i] += _a.f[i] * _b.f[i]; _acc;})
#define F32x16_BROAD(ptr) ({ F32x16 _value; float _scalar = *(ptr); for (int i = 0; i < 16; i++) _value.f[i] = _scalar; _value;})
#endif
#endif

#endif /* _SIMD_H */
