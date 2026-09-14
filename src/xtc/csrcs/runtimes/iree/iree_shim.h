/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
/*
 * Thin C shim over the IREE runtime C API (iree/runtime/api.h) exposing a
 * single-invocation entry point with the flat ``void func(void* ctx)`` ABI
 * expected by XTC's shared measurement loop (csrcs/runtimes/host/evaluate_perf.c).
 *
 * All the IREE state (instance, device, session and the prepared call with its
 * pre-built input/output buffer views) lives behind the opaque ``ctx`` handle,
 * so evaluate_perf.c needs no knowledge of IREE and stays unchanged.
 */
#ifndef XTC_IREE_SHIM_H
#define XTC_IREE_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Description of one host-resident tensor argument. */
typedef struct {
  void *data;           /* host pointer to the contiguous tensor buffer */
  int32_t rank;         /* number of dimensions */
  const int64_t *shape; /* shape[rank] */
  const char *dtype;    /* IREE element type spelling, e.g. "f32", "i32" */
} xtc_ndarray_desc_t;

/*
 * Build an invocation context for `entry_function` of the module in `vmfb_path`.
 *
 * `num_threads` selects the HAL device and sizes its worker pool: <= 1 ->
 * "local-sync" (everything on the caller thread); > 1 -> "local-task" with a
 * thread pool of `num_threads` workers pinned to that many high-performance
 * physical cores (falling back to a plain group count if P-cores can't be
 * detected). Input buffers are filled from the descriptors once here (outside
 * any timed region); output descriptors record where results are copied back
 * after each invocation.
 *
 * Returns an opaque handle, or NULL on error.
 */
void *xtc_iree_setup(const char *vmfb_path, const char *entry_function,
                     int num_threads, const xtc_ndarray_desc_t *inputs,
                     int n_inputs, const xtc_ndarray_desc_t *outputs,
                     int n_outputs);

/*
 * Run exactly one invocation on `ctx`. This variant does NOT copy results back
 * to the host, so it measures compute only; it is the one fed to the timed
 * measurement loop. Aborts the process on an IREE error (inputs were already
 * validated at setup, so a failure here is unexpected and unrecoverable).
 */
void xtc_iree_invoke(void *ctx);

/*
 * Like xtc_iree_invoke, but also copies the results back into the host output
 * pointers recorded at setup. Used once for correctness validation and
 * write-back, outside the timed region.
 */
void xtc_iree_invoke_readback(void *ctx);

/* Release all resources held by `ctx` (NULL-safe). */
void xtc_iree_teardown(void *ctx);

/*
 * Message describing the most recent failure, or NULL if none. The returned
 * pointer is owned by the shim and valid until the next shim call.
 */
const char *xtc_iree_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* XTC_IREE_SHIM_H */
