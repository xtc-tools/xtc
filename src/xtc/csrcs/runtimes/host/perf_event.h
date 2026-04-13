/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#ifndef _PERF_EVENT_H
#define _PERF_EVENT_H

#include <stdint.h>

#define CONCAT2_IMPL(A,B) A ## B
#define CONCAT2(A,B) CONCAT2_IMPL(A,B)
#define CONCAT3_IMPL(A,B,C) A ## B ## C
#define CONCAT3(A,B,C) CONCAT3_IMPL(A,B,C)
#define STR_IMPL(A) #A
#define STR(A) STR_IMPL(A)

#define PERF_EVENT_MAX_EVENTS 256

#define PERF_EVENT_CYCLES 0
#define PERF_EVENT_CLOCKS 1
#define PERF_EVENT_INSTRS 2
#define PERF_EVENT_MIGRATIONS 3
#define PERF_EVENT_SWITCHES 4
#define PERF_EVENT_CACHE_ACCESS 5
#define PERF_EVENT_CACHE_MISSES 6
#define PERF_EVENT_BRANCH_INSTRS 7
#define PERF_EVENT_BRANCH_MISSES 8
#define PERF_EVENT_NUM 9

typedef struct {
    int type;
    int event;
} perf_event_type_event_t;

typedef enum {
    PERF_ARG_INVALID = -1,
    PERF_ARG_GENERIC,
    PERF_ARG_PTR,
    PERF_ARG_GPU
} perf_event_arg_mode_t;

typedef struct {
    perf_event_arg_mode_t mode;
    union {
        perf_event_type_event_t config_pair;
        const void* config_ptr;
    } args;
} perf_event_args_t;

typedef enum {
  PERF_EVENT_CLOSE = -1,
  PERF_EVENT_GPU = -2
} perf_event_fd_t;

extern int all_perf_events[PERF_EVENT_NUM];
extern int open_perf_event(perf_event_args_t event);
extern int open_cycles_event();
extern int open_clock_event();
extern uint64_t read_perf_event(int perf_fd);
extern void close_perf_event(int perf_fd);
extern void open_raw_perf_events(int n_events, const int *events_pairs, int *fds);
extern void open_perf_events(int n_events, const perf_event_args_t *events, int *fds);
extern void enable_perf_events(int n_events, const int *fds);
extern void close_perf_events(int n_events, const int *fds);
extern void reset_perf_events(int n_events, const int *fds, uint64_t *results);
extern void start_perf_events(int n_events, const int *fds, uint64_t *results);
extern void stop_perf_events(int n_events, const int *fds, uint64_t *results);
extern int get_perf_event_config(const char *name, perf_event_args_t* event);

extern void perf_event_args_destroy(perf_event_args_t args);


#endif
