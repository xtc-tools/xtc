/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */

#ifndef __PERF_EVENT_MPPA_H__
#define __PERF_EVENT_MPPA_H__

#include "perf_event.h"

#define mppa_OPEN_PERF_EVENT         open_perf_event /* reuse host */
#define mppa_READ_PERF_EVENT         read_perf_event /* reuse host */
#define mppa_CLOSE_PERF_EVENT        close_perf_event /* reuse host */
#define mppa_OPEN_PERF_EVENTS        open_perf_events /* reuse host */
#define mppa_ENABLE_PERF_EVENTS      mppa_enable_perf_events
#define mppa_CLOSE_PERF_EVENTS       close_perf_events /* reuse host */
#define mppa_RESET_PERF_EVENTS       reset_perf_events /* reuse host */
#define mppa_START_PERF_EVENTS       start_perf_events /* reuse host */
#define mppa_STOP_PERF_EVENTS        stop_perf_events /* reuse host */
#define mppa_GET_PERF_EVENT_CONFIG   get_perf_event_config /* reuse host */
#define mppa_PERF_EVENT_ARGS_DESTROY perf_event_args_destroy /* reuse host */

extern void mppa_reset_perf_events();

void mppa_enable_perf_events(int n_events, const int *fds) {
  mppa_reset_perf_events();
  enable_perf_events(n_events, fds);
}

#endif

