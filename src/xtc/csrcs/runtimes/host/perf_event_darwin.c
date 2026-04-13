/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include <assert.h>

#include "perf_event.h"
#include "kperf.h"

int all_perf_events[PERF_EVENT_NUM] =
  {
   PERF_EVENT_CYCLES,
   PERF_EVENT_CLOCKS,
   PERF_EVENT_INSTRS,
   PERF_EVENT_MIGRATIONS,
   PERF_EVENT_SWITCHES,
   PERF_EVENT_CACHE_ACCESS,
   PERF_EVENT_CACHE_MISSES,
   PERF_EVENT_BRANCH_INSTRS,
   PERF_EVENT_BRANCH_MISSES,
  };

#define PERF_TYPE_HARDWARE 0
#define PERF_TYPE_SOFTWARE 0
#define PERF_COUNT_HW_CPU_CYCLES EventCycles
#define PERF_COUNT_HW_INSTRUCTIONS EventInsts
#define PERF_COUNT_SW_TASK_CLOCK -1
#define PERF_COUNT_SW_CPU_MIGRATIONS -1
#define PERF_COUNT_SW_CONTEXT_SWITCHES -1
#define PERF_COUNT_HW_CACHE_REFERENCES -1
#define PERF_COUNT_HW_CACHE_MISSES -1
#define PERF_COUNT_HW_BRANCH_INSTRUCTIONS -1
#define PERF_COUNT_HW_BRANCH_MISSES -1


typedef struct { int id; int type; int num; const char *name; } perf_event_decl_t;

static const perf_event_decl_t perf_events_decl[] =
  {
   { PERF_EVENT_CYCLES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles" },
   { PERF_EVENT_CLOCKS, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, "clocks" },
   { PERF_EVENT_INSTRS, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions" },
   { PERF_EVENT_MIGRATIONS, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, "migrations" },
   { PERF_EVENT_SWITCHES, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, "context_switches" },
   { PERF_EVENT_CACHE_ACCESS, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, "cache_access" },
   { PERF_EVENT_CACHE_MISSES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "cache_misses" },
   { PERF_EVENT_BRANCH_INSTRS, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, "branches" },
   { PERF_EVENT_BRANCH_MISSES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, "branches_misses" },
  };

static int inited;
static int perf_events[PERF_EVENT_MAX_EVENTS];
static struct perf_counters kperf_counters;

static int kperf_init(void) {
  if (!lib_init())
    return 1;
  if (!init_perf_counters(&kperf_counters))
    return 1;
  for (int fd = 0; fd < PERF_EVENT_MAX_EVENTS; fd++) {
    perf_events[fd] = -1;
  }
  return 0;
}

int open_perf_event(perf_event_args_t event) {
  assert(event.mode == PERF_ARG_GENERIC);

  int type = event.args.config_pair.type;
  int raw_event = event.args.config_pair.event;

  if (type < 0 || raw_event < 0)
    return -1;

  if (inited == 0) {
    if (kperf_init() != 0)
      return -1;
    inited = 1;
  }

  int fd;
  for (fd = 0; fd < PERF_EVENT_MAX_EVENTS; fd++) {
    if (perf_events[fd] == -1)
      break;
  }
  if (fd >= PERF_EVENT_MAX_EVENTS) {
    return -1;
  }

  perf_events[fd] = raw_event;
  return fd;
}

uint64_t read_perf_event(int perf_fd) {
  assert(perf_fd >= 0 && perf_fd < PERF_EVENT_MAX_EVENTS);
  int event = perf_events[perf_fd];
  assert(event >= 0);
  read_perf_counters_before(&kperf_counters);
  uint64_t count = kperf_counters.counters0[kperf_counters.counter_map[event]];
  return count;
}

void close_perf_event(int perf_fd)
{
  assert(perf_fd >= 0 && perf_fd < PERF_EVENT_MAX_EVENTS);
  perf_events[perf_fd] = -1;
}

void open_perf_events(int n_events, const perf_event_args_t *events, int *fds) {
  assert(n_events <= PERF_EVENT_MAX_EVENTS);
  for (int i = 0; i < n_events; i++) {
    fds[i] = open_perf_event(events[i]);
    assert(fds[i] != -1);
  }
}

void enable_perf_events(int n_events, const int *fds) {
  /* Nothing to do */
}

void close_perf_events(int n_events, const int *fds) {
  for(int i = 0; i < n_events; i++) {
    if (fds[i] >= 0) {
      close_perf_event(fds[i]);
    }
  }
}

void reset_perf_events(int n_events, const int *fds, uint64_t *results)
{
  for(int i = 0; i < n_events; i++) {
    results[i] = 0;
  }
}

void start_perf_events(int n_events, const int *fds, uint64_t *results) {
  if (!inited) return;
  read_perf_counters_before(&kperf_counters);
}

void stop_perf_events(int n_events, const int *fds, uint64_t *results) {
  if (!inited) return;
  read_perf_counters_after(&kperf_counters);
  for(int i = 0; i < n_events; i++) {
    if (fds[i] >= 0) {
      int event = perf_events[fds[i]];
      uint64_t start = kperf_counters.counters0[kperf_counters.counter_map[event]];
      uint64_t stop = kperf_counters.counters1[kperf_counters.counter_map[event]];
      results[i] += stop - start;
    }
  }
}

int get_perf_event_config(const char *name, perf_event_args_t *event) {
  for (int e = 0; e < sizeof(perf_events_decl) / sizeof(*perf_events_decl);
       e++) {
    if (strcmp(name, perf_events_decl[e].name) == 0) {
      event->mode = PERF_ARG_GENERIC;
      event->args.config_pair.event = perf_events_decl[e].num;
      event->args.config_pair.type = perf_events_decl[e].type;
      return 0;
    }
  }

  return 1;
}

void perf_event_args_destroy(perf_event_args_t args) {
    // not supported on Darwin
}
