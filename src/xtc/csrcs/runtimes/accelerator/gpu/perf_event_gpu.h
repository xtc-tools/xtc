/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#ifndef _PERF_EVENT_GPU_H
#define _PERF_EVENT_GPU_H

#include <stdint.h>

#include "perf_event.h"

#ifdef __cplusplus
    extern "C" {
#endif
  extern void open_perf_events__gpu(int n_events, const perf_event_args_t *events, int *fds);
  extern void close_perf_events__gpu(int n_events, const int *fds);
  extern void reset_perf_events__gpu(int n_events, const int *fds, uint64_t *results);
  extern void start_perf_events__gpu(int n_events, const int *fds, uint64_t *results);
  extern void stop_perf_events__gpu(int n_events, const int *fds, uint64_t *results);
  extern int get_perf_event_config__gpu(const char *name, perf_event_args_t* event);
#ifdef __cplusplus
    }
#endif


#endif /* _PERF_EVENT_GPU_H */
