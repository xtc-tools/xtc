/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
/*
 * Estimation of flops (i.e number of fmas/secs).
 *
 * Call: gflops = evaluate_flops("float32")
 *
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "simd.h"

extern double fclock(void); /* from fclock.c */

/* DTYPES */
#define DTYPE_float 1
#define DTYPE_double 2

#define DTYPE_type DTYPE_float
#define DTYPE_name "float32"
#define DTYPE float

#define V16SIZE 16
#define V16TYPE F32x16
#define V16ZERO() F32x16_ZERO()
#define V16BROAD(ptr) F32x16_BROAD(ptr)
#define V16STORE(ptr, val) F32x16_STORE(ptr, val)
#define V16LOAD(ptr) F32x16_LOAD(ptr)
#define V16MADD(acc, a, b) F32x16_MADD(acc, a, b)

#define V8SIZE 8
#define V8TYPE F32x8
#define V8ZERO() F32x8_ZERO()
#define V8BROAD(ptr) F32x8_BROAD(ptr)
#define V8STORE(ptr, val) F32x8_STORE(ptr, val)
#define V8LOAD(ptr) F32x8_LOAD(ptr)
#define V8MADD(acc, a, b) F32x8_MADD(acc, a, b)

#define V4SIZE 4
#define V4TYPE F32x4
#define V4ZERO() F32x4_ZERO()
#define V4BROAD(ptr) F32x4_BROAD(ptr)
#define V4STORE(ptr, val) F32x4_STORE(ptr, val)
#define V4LOAD(ptr) F32x4_LOAD(ptr)
#define V4MADD(acc, a, b) F32x4_MADD(acc, a, b)


#define NOINLINE __attribute__((noinline))
#define mem_barrier() asm("":::"memory")
#define opaque_addr(ptr) ({typeof(ptr) _ptr = (ptr); __asm__ __volatile__("# opaque %0":"+p"(_ptr)::"memory"); _ptr; })

#define VSIZE V16SIZE
#define VTYPE V16TYPE
#define VZERO V16ZERO
#define VBROAD V16BROAD
#define VSTORE V16STORE
#define VLOAD V16LOAD
#define VMADD V16MADD

static NOINLINE int64_t eval_fmadd_parallel_16x16(void *args[], int64_t runs) {
    DTYPE *vals = (DTYPE *)args[1];
    DTYPE *inps = vals;
    VTYPE a = VBROAD(inps++);
    VTYPE acc1 = VBROAD(inps++);
    VTYPE acc2 = VBROAD(inps++);
    VTYPE acc3 = VBROAD(inps++);
    VTYPE acc4 = VBROAD(inps++);
    VTYPE acc5 = VBROAD(inps++);
    VTYPE acc6 = VBROAD(inps++);
    VTYPE acc7 = VBROAD(inps++);
    VTYPE acc8 = VBROAD(inps++);
    VTYPE acc9 = VBROAD(inps++);
    VTYPE acc10 = VBROAD(inps++);
    VTYPE acc11 = VBROAD(inps++);
    VTYPE acc12 = VBROAD(inps++);
    VTYPE acc13 = VBROAD(inps++);
    VTYPE acc14 = VBROAD(inps++);
    VTYPE acc15 = VBROAD(inps++);
    VTYPE acc16 = VBROAD(inps++);
    int64_t i = 0;
    do {
#define UNROLLED 32
        DTYPE *bp = opaque_addr(vals);
        acc1 = VMADD(acc1, a, VLOAD(bp)); bp += VSIZE;
        acc2 = VMADD(acc2, a, VLOAD(bp)); bp += VSIZE;
        acc3 = VMADD(acc3, a, VLOAD(bp)); bp += VSIZE;
        acc4 = VMADD(acc4, a, VLOAD(bp)); bp += VSIZE;
        acc5 = VMADD(acc5, a, VLOAD(bp)); bp += VSIZE;
        acc6 = VMADD(acc6, a, VLOAD(bp)); bp += VSIZE;
        acc7 = VMADD(acc7, a, VLOAD(bp)); bp += VSIZE;
        acc8 = VMADD(acc8, a, VLOAD(bp)); bp += VSIZE;
        acc9 = VMADD(acc9, a, VLOAD(bp)); bp += VSIZE;
        acc10 = VMADD(acc10, a, VLOAD(bp)); bp += VSIZE;
        acc11 = VMADD(acc11, a, VLOAD(bp)); bp += VSIZE;
        acc12 = VMADD(acc12, a, VLOAD(bp)); bp += VSIZE;
        acc13 = VMADD(acc13, a, VLOAD(bp)); bp += VSIZE;
        acc14 = VMADD(acc14, a, VLOAD(bp)); bp += VSIZE;
        acc15 = VMADD(acc15, a, VLOAD(bp)); bp += VSIZE;
        acc16 = VMADD(acc16, a, VLOAD(bp)); bp += VSIZE;
        acc1 = VMADD(acc1, a, VLOAD(bp)); bp += VSIZE;
        acc2 = VMADD(acc2, a, VLOAD(bp)); bp += VSIZE;
        acc3 = VMADD(acc3, a, VLOAD(bp)); bp += VSIZE;
        acc4 = VMADD(acc4, a, VLOAD(bp)); bp += VSIZE;
        acc5 = VMADD(acc5, a, VLOAD(bp)); bp += VSIZE;
        acc6 = VMADD(acc6, a, VLOAD(bp)); bp += VSIZE;
        acc7 = VMADD(acc7, a, VLOAD(bp)); bp += VSIZE;
        acc8 = VMADD(acc8, a, VLOAD(bp)); bp += VSIZE;
        acc9 = VMADD(acc9, a, VLOAD(bp)); bp += VSIZE;
        acc10 = VMADD(acc10, a, VLOAD(bp)); bp += VSIZE;
        acc11 = VMADD(acc11, a, VLOAD(bp)); bp += VSIZE;
        acc12 = VMADD(acc12, a, VLOAD(bp)); bp += VSIZE;
        acc13 = VMADD(acc13, a, VLOAD(bp)); bp += VSIZE;
        acc14 = VMADD(acc14, a, VLOAD(bp)); bp += VSIZE;
        acc15 = VMADD(acc15, a, VLOAD(bp)); bp += VSIZE;
        acc16 = VMADD(acc16, a, VLOAD(bp)); bp += VSIZE;
    } while (++i < runs);
    DTYPE *out = (DTYPE *)args[0];
    VSTORE(&out[0*VSIZE], acc1);
    VSTORE(&out[1*VSIZE], acc2);
    VSTORE(&out[2*VSIZE], acc3);
    VSTORE(&out[3*VSIZE], acc4);
    VSTORE(&out[4*VSIZE], acc5);
    VSTORE(&out[5*VSIZE], acc6);
    VSTORE(&out[6*VSIZE], acc7);
    VSTORE(&out[7*VSIZE], acc8);
    VSTORE(&out[8*VSIZE], acc9);
    VSTORE(&out[9*VSIZE], acc10);
    VSTORE(&out[10*VSIZE], acc11);
    VSTORE(&out[11*VSIZE], acc12);
    VSTORE(&out[12*VSIZE], acc13);
    VSTORE(&out[13*VSIZE], acc14);
    VSTORE(&out[14*VSIZE], acc15);
    VSTORE(&out[15*VSIZE], acc16);
    return runs * UNROLLED * VSIZE;
#undef UNROLLED
}

#undef VSIZE
#undef VTYPE
#undef VZERO
#undef VBROAD
#undef VSTORE
#undef VLOAD
#undef VMADD

#define VSIZE V8SIZE
#define VTYPE V8TYPE
#define VZERO V8ZERO
#define VBROAD V8BROAD
#define VSTORE V8STORE
#define VLOAD V8LOAD
#define VMADD V8MADD

static NOINLINE int64_t eval_fmadd_parallel_12x8(void *args[], int64_t runs) {
    DTYPE *vals = (DTYPE *)args[1];
    DTYPE *inps = vals;
    VTYPE a = VBROAD(inps++);
    VTYPE acc1 = VBROAD(inps++);
    VTYPE acc2 = VBROAD(inps++);
    VTYPE acc3 = VBROAD(inps++);
    VTYPE acc4 = VBROAD(inps++);
    VTYPE acc5 = VBROAD(inps++);
    VTYPE acc6 = VBROAD(inps++);
    VTYPE acc7 = VBROAD(inps++);
    VTYPE acc8 = VBROAD(inps++);
    VTYPE acc9 = VBROAD(inps++);
    VTYPE acc10 = VBROAD(inps++);
    VTYPE acc11 = VBROAD(inps++);
    VTYPE acc12 = VBROAD(inps++);
    int64_t i = 0;
    do {
#define UNROLLED 24
        DTYPE *bp = opaque_addr(vals);
        acc1 = VMADD(acc1, a, VLOAD(bp)); bp += VSIZE;
        acc2 = VMADD(acc2, a, VLOAD(bp)); bp += VSIZE;
        acc3 = VMADD(acc3, a, VLOAD(bp)); bp += VSIZE;
        acc4 = VMADD(acc4, a, VLOAD(bp)); bp += VSIZE;
        acc5 = VMADD(acc5, a, VLOAD(bp)); bp += VSIZE;
        acc6 = VMADD(acc6, a, VLOAD(bp)); bp += VSIZE;
        acc7 = VMADD(acc7, a, VLOAD(bp)); bp += VSIZE;
        acc8 = VMADD(acc8, a, VLOAD(bp)); bp += VSIZE;
        acc9 = VMADD(acc9, a, VLOAD(bp)); bp += VSIZE;
        acc10 = VMADD(acc10, a, VLOAD(bp)); bp += VSIZE;
        acc11 = VMADD(acc11, a, VLOAD(bp)); bp += VSIZE;
        acc12 = VMADD(acc12, a, VLOAD(bp)); bp += VSIZE;
        acc1 = VMADD(acc1, a, VLOAD(bp)); bp += VSIZE;
        acc2 = VMADD(acc2, a, VLOAD(bp)); bp += VSIZE;
        acc3 = VMADD(acc3, a, VLOAD(bp)); bp += VSIZE;
        acc4 = VMADD(acc4, a, VLOAD(bp)); bp += VSIZE;
        acc5 = VMADD(acc5, a, VLOAD(bp)); bp += VSIZE;
        acc6 = VMADD(acc6, a, VLOAD(bp)); bp += VSIZE;
        acc7 = VMADD(acc7, a, VLOAD(bp)); bp += VSIZE;
        acc8 = VMADD(acc8, a, VLOAD(bp)); bp += VSIZE;
        acc9 = VMADD(acc9, a, VLOAD(bp)); bp += VSIZE;
        acc10 = VMADD(acc10, a, VLOAD(bp)); bp += VSIZE;
        acc11 = VMADD(acc11, a, VLOAD(bp)); bp += VSIZE;
        acc12 = VMADD(acc12, a, VLOAD(bp)); bp += VSIZE;
    } while (++i < runs);
    DTYPE *out = (DTYPE *)args[0];
    VSTORE(&out[0*VSIZE], acc1);
    VSTORE(&out[1*VSIZE], acc2);
    VSTORE(&out[2*VSIZE], acc3);
    VSTORE(&out[3*VSIZE], acc4);
    VSTORE(&out[4*VSIZE], acc5);
    VSTORE(&out[5*VSIZE], acc6);
    VSTORE(&out[6*VSIZE], acc7);
    VSTORE(&out[7*VSIZE], acc8);
    VSTORE(&out[8*VSIZE], acc9);
    VSTORE(&out[9*VSIZE], acc10);
    VSTORE(&out[10*VSIZE], acc11);
    VSTORE(&out[11*VSIZE], acc12);
    return runs * UNROLLED * VSIZE;
#undef UNROLLED
}

#undef VSIZE
#undef VTYPE
#undef VZERO
#undef VBROAD
#undef VSTORE
#undef VLOAD
#undef VMADD

#define VSIZE V4SIZE
#define VTYPE V4TYPE
#define VZERO V4ZERO
#define VBROAD V4BROAD
#define VSTORE V4STORE
#define VLOAD V4LOAD
#define VMADD V4MADD

static NOINLINE int64_t eval_fmadd_parallel_6x4x4(void *args[], int64_t runs) {
    DTYPE *vals = (DTYPE *)args[1];
    DTYPE *inps = vals;
    VTYPE a1 = VBROAD(inps++);
    VTYPE a2 = VBROAD(inps++);
    VTYPE a3 = VBROAD(inps++);
    VTYPE a4 = VBROAD(inps++);
    VTYPE a5 = VBROAD(inps++);
    VTYPE a6 = VBROAD(inps++);
    VTYPE acc1x1 = VBROAD(inps++);
    VTYPE acc2x1 = VBROAD(inps++);
    VTYPE acc3x1 = VBROAD(inps++);
    VTYPE acc4x1 = VBROAD(inps++);
    VTYPE acc5x1 = VBROAD(inps++);
    VTYPE acc6x1 = VBROAD(inps++);
    VTYPE acc1x2 = VBROAD(inps++);
    VTYPE acc2x2 = VBROAD(inps++);
    VTYPE acc3x2 = VBROAD(inps++);
    VTYPE acc4x2 = VBROAD(inps++);
    VTYPE acc5x2 = VBROAD(inps++);
    VTYPE acc6x2 = VBROAD(inps++);
    VTYPE acc1x3 = VBROAD(inps++);
    VTYPE acc2x3 = VBROAD(inps++);
    VTYPE acc3x3 = VBROAD(inps++);
    VTYPE acc4x3 = VBROAD(inps++);
    VTYPE acc5x3 = VBROAD(inps++);
    VTYPE acc6x3 = VBROAD(inps++);
    VTYPE acc1x4 = VBROAD(inps++);
    VTYPE acc2x4 = VBROAD(inps++);
    VTYPE acc3x4 = VBROAD(inps++);
    VTYPE acc4x4 = VBROAD(inps++);
    VTYPE acc5x4 = VBROAD(inps++);
    VTYPE acc6x4 = VBROAD(inps++);
    int64_t i = 0;
    do {
#define UNROLLED 24
        DTYPE *bp = opaque_addr(vals);
        VTYPE b0 = VLOAD(bp); bp += VSIZE;
        VTYPE b1 = VLOAD(bp); bp += VSIZE;
        acc1x1 = VMADD(acc1x1, a1, b0);
        acc2x1 = VMADD(acc2x1, a2, b0);
        acc3x1 = VMADD(acc3x1, a3, b0);
        acc4x1 = VMADD(acc4x1, a4, b0);
        acc5x1 = VMADD(acc5x1, a5, b0);
        acc6x1 = VMADD(acc6x1, a6, b0);
        acc1x2 = VMADD(acc1x2, a1, b1);
        acc2x2 = VMADD(acc2x2, a2, b1);
        acc3x2 = VMADD(acc3x2, a3, b1);
        acc4x2 = VMADD(acc4x2, a4, b1);
        acc5x2 = VMADD(acc5x2, a5, b1);
        acc6x2 = VMADD(acc6x2, a6, b1);
        VTYPE b2 = VLOAD(bp); bp += VSIZE;
        VTYPE b3 = VLOAD(bp); bp += VSIZE;
        acc1x3 = VMADD(acc1x3, a1, b2);
        acc2x3 = VMADD(acc2x3, a2, b2);
        acc3x3 = VMADD(acc3x3, a3, b2);
        acc4x3 = VMADD(acc4x3, a4, b2);
        acc5x3 = VMADD(acc5x3, a5, b2);
        acc6x3 = VMADD(acc6x3, a6, b2);
        acc1x4 = VMADD(acc1x4, a1, b3);
        acc2x4 = VMADD(acc2x4, a2, b3);
        acc3x4 = VMADD(acc3x4, a3, b3);
        acc4x4 = VMADD(acc4x4, a4, b3);
        acc5x4 = VMADD(acc5x4, a5, b3);
        acc6x4 = VMADD(acc6x4, a6, b3);
    } while (++i < runs);
    DTYPE *out = (DTYPE *)args[0];
    VSTORE(out, acc1x1); out += VSIZE;
    VSTORE(out, acc2x1); out += VSIZE;
    VSTORE(out, acc3x1); out += VSIZE;
    VSTORE(out, acc4x1); out += VSIZE;
    VSTORE(out, acc5x1); out += VSIZE;
    VSTORE(out, acc6x1); out += VSIZE;
    VSTORE(out, acc1x2); out += VSIZE;
    VSTORE(out, acc2x2); out += VSIZE;
    VSTORE(out, acc3x2); out += VSIZE;
    VSTORE(out, acc4x2); out += VSIZE;
    VSTORE(out, acc5x2); out += VSIZE;
    VSTORE(out, acc6x2); out += VSIZE;
    VSTORE(out, acc1x3); out += VSIZE;
    VSTORE(out, acc2x3); out += VSIZE;
    VSTORE(out, acc3x3); out += VSIZE;
    VSTORE(out, acc4x3); out += VSIZE;
    VSTORE(out, acc5x3); out += VSIZE;
    VSTORE(out, acc6x3); out += VSIZE;
    VSTORE(out, acc1x4); out += VSIZE;
    VSTORE(out, acc2x4); out += VSIZE;
    VSTORE(out, acc3x4); out += VSIZE;
    VSTORE(out, acc4x4); out += VSIZE;
    VSTORE(out, acc5x4); out += VSIZE;
    VSTORE(out, acc6x4); out += VSIZE;
    return runs * UNROLLED * VSIZE;
#undef UNROLLED
}

static NOINLINE int64_t eval_fmadd_parallel_5x4x4(void *args[], int64_t runs) {
    DTYPE *vals = (DTYPE *)args[1];
    DTYPE *inps = vals;
    VTYPE a1 = VBROAD(inps++);
    VTYPE a2 = VBROAD(inps++);
    VTYPE a3 = VBROAD(inps++);
    VTYPE a4 = VBROAD(inps++);
    VTYPE a5 = VBROAD(inps++);
    VTYPE acc1x1 = VBROAD(inps++);
    VTYPE acc2x1 = VBROAD(inps++);
    VTYPE acc3x1 = VBROAD(inps++);
    VTYPE acc4x1 = VBROAD(inps++);
    VTYPE acc5x1 = VBROAD(inps++);
    VTYPE acc1x2 = VBROAD(inps++);
    VTYPE acc2x2 = VBROAD(inps++);
    VTYPE acc3x2 = VBROAD(inps++);
    VTYPE acc4x2 = VBROAD(inps++);
    VTYPE acc5x2 = VBROAD(inps++);
    VTYPE acc1x3 = VBROAD(inps++);
    VTYPE acc2x3 = VBROAD(inps++);
    VTYPE acc3x3 = VBROAD(inps++);
    VTYPE acc4x3 = VBROAD(inps++);
    VTYPE acc5x3 = VBROAD(inps++);
    VTYPE acc1x4 = VBROAD(inps++);
    VTYPE acc2x4 = VBROAD(inps++);
    VTYPE acc3x4 = VBROAD(inps++);
    VTYPE acc4x4 = VBROAD(inps++);
    VTYPE acc5x4 = VBROAD(inps++);
    int64_t i = 0;
    do {
#define UNROLLED 20
        DTYPE *bp = opaque_addr(vals);
        VTYPE b0 = VLOAD(bp); bp += VSIZE;
        VTYPE b1 = VLOAD(bp); bp += VSIZE;
        acc1x1 = VMADD(acc1x1, a1, b0);
        acc2x1 = VMADD(acc2x1, a2, b0);
        acc3x1 = VMADD(acc3x1, a3, b0);
        acc4x1 = VMADD(acc4x1, a4, b0);
        acc5x1 = VMADD(acc5x1, a5, b0);
        acc1x2 = VMADD(acc1x2, a1, b1);
        acc2x2 = VMADD(acc2x2, a2, b1);
        acc3x2 = VMADD(acc3x2, a3, b1);
        acc4x2 = VMADD(acc4x2, a4, b1);
        acc5x2 = VMADD(acc5x2, a5, b1);
        VTYPE b2 = VLOAD(bp); bp += VSIZE;
        VTYPE b3 = VLOAD(bp); bp += VSIZE;
        acc1x3 = VMADD(acc1x3, a1, b2);
        acc2x3 = VMADD(acc2x3, a2, b2);
        acc3x3 = VMADD(acc3x3, a3, b2);
        acc4x3 = VMADD(acc4x3, a4, b2);
        acc5x3 = VMADD(acc5x3, a5, b2);
        acc1x4 = VMADD(acc1x4, a1, b3);
        acc2x4 = VMADD(acc2x4, a2, b3);
        acc3x4 = VMADD(acc3x4, a3, b3);
        acc4x4 = VMADD(acc4x4, a4, b3);
        acc5x4 = VMADD(acc5x4, a5, b3);
    } while (++i < runs);
    DTYPE *out = (DTYPE *)args[0];
    VSTORE(out, acc1x1); out += VSIZE;
    VSTORE(out, acc2x1); out += VSIZE;
    VSTORE(out, acc3x1); out += VSIZE;
    VSTORE(out, acc4x1); out += VSIZE;
    VSTORE(out, acc5x1); out += VSIZE;
    VSTORE(out, acc1x2); out += VSIZE;
    VSTORE(out, acc2x2); out += VSIZE;
    VSTORE(out, acc3x2); out += VSIZE;
    VSTORE(out, acc4x2); out += VSIZE;
    VSTORE(out, acc5x2); out += VSIZE;
    VSTORE(out, acc1x3); out += VSIZE;
    VSTORE(out, acc2x3); out += VSIZE;
    VSTORE(out, acc3x3); out += VSIZE;
    VSTORE(out, acc4x3); out += VSIZE;
    VSTORE(out, acc5x3); out += VSIZE;
    VSTORE(out, acc1x4); out += VSIZE;
    VSTORE(out, acc2x4); out += VSIZE;
    VSTORE(out, acc3x4); out += VSIZE;
    VSTORE(out, acc4x4); out += VSIZE;
    VSTORE(out, acc5x4); out += VSIZE;
    return runs * UNROLLED * VSIZE;
#undef UNROLLED
}

#undef VSIZE
#undef VTYPE
#undef VZERO
#undef VBROAD
#undef VSTORE
#undef VLOAD
#undef VMADD

typedef int64_t (*eval_func_t)(void *args[], int64_t runs);
typedef struct {
    double elapsed;
    int64_t runs;
    int64_t fmas;
} eval_t;

#define NUMBER_FACTOR 2

static eval_t eval_func(eval_func_t func, void *args[], double min_ms, int64_t number)
{
    mem_barrier();
    (void)func(args, 1); // cold run
    mem_barrier();
    int64_t fmas;
    double elapsed;
    while(1) {
        elapsed = fclock();
        mem_barrier();
        fmas = func(args, number);
        mem_barrier();
        elapsed = fclock() - elapsed;
        if (elapsed * 1000 >= min_ms)
            break;
        number *= NUMBER_FACTOR;
    }
    eval_t eval;
    eval.elapsed = elapsed;
    eval.runs = number;
    eval.fmas = fmas;
    return eval;
}


static void *alloc_aligned(size_t size, size_t align) {
    size = (size + align  - 1) / align * align;
    return aligned_alloc(align, size);
}

static double evaluate_flops_float() {
    eval_t eval;
    void *args[2];

    DTYPE *inits = alloc_aligned(64 * V16SIZE * sizeof(DTYPE), 4096);
    for (int i = 0; i < 64 * V16SIZE; i++) {
        inits[i] = 1 + (i % 127); // range [1, 127]
    }
    DTYPE *outs = alloc_aligned(64 * V16SIZE * sizeof(DTYPE), 4096);
    args[0] = outs;
    args[1] = inits;
    eval = eval_func(eval_fmadd_parallel_16x16, args, 100/*ms*/, 1/*initial runs*/);
    double flops16x16 = eval.fmas / eval.elapsed;
    eval = eval_func(eval_fmadd_parallel_12x8, args, 100/*ms*/, 1/*initial runs*/);
    double flops12x8 = eval.fmas / eval.elapsed;
    eval = eval_func(eval_fmadd_parallel_6x4x4, args, 100/*ms*/, 1/*initial runs*/);
    double flops6x4x4 = eval.fmas / eval.elapsed;
    eval = eval_func(eval_fmadd_parallel_5x4x4, args, 100/*ms*/, 1/*initial runs*/);
    double flops5x4x4 = eval.fmas / eval.elapsed;

    double flops = flops16x16;
    if (flops12x8 > flops) flops = flops12x8;
    if (flops6x4x4 > flops) flops = flops6x4x4;
    if (flops5x4x4 > flops) flops = flops5x4x4;
    free(args[0]);
    free(args[1]);
    return flops;
}

double evaluate_flops(const char *dtype_name) {
    if (strcmp(dtype_name, "float32") == 0) {
        return evaluate_flops_float();
    }
    return 0;
}
