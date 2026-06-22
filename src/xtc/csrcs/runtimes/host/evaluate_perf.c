/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "perf_event.h"
#include "perf_metrics.h"

extern double fclock(void); /* from fclock.c */

typedef void (*func0_t)(void);
typedef void (*func1_t)(void *);
typedef void (*func2_t)(void *,void *);
typedef void (*func3_t)(void *,void *,void *);
typedef void (*func4_t)(void *,void *,void *,void *);
typedef void (*func5_t)(void *,void *,void *,void *,void *);
typedef void (*func6_t)(void *,void *,void *,void *,void *,void *);

typedef union {
  int64_t v_int64;
  double v_float64;
  void *v_handle;
  char *v_str;
} PackedArg;

typedef int (*packed_func_t)(PackedArg *, int *, int, PackedArg *, int *);

#define mem_barrier() asm("":::"memory")

#define NUMBER_FACTOR 2

#ifdef ACCELERATOR_NAME
#define PERF_EVENT_ACCELERATOR_H CONCAT2(perf_event_, ACCELERATOR_NAME)
#include STR(PERF_EVENT_ACCELERATOR_H.h)

#define OPEN_PERF_EVENT         CONCAT3(ACCELERATOR_NAME,_,OPEN_PERF_EVENT)
#define READ_PERF_EVENT         CONCAT3(ACCELERATOR_NAME,_,READ_PERF_EVENT)
#define CLOSE_PERF_EVENT        CONCAT3(ACCELERATOR_NAME,_,CLOSE_PERF_EVENT)
#define OPEN_PERF_EVENTS        CONCAT3(ACCELERATOR_NAME,_,OPEN_PERF_EVENTS)
#define ENABLE_PERF_EVENTS      CONCAT3(ACCELERATOR_NAME,_,ENABLE_PERF_EVENTS)
#define CLOSE_PERF_EVENTS       CONCAT3(ACCELERATOR_NAME,_,CLOSE_PERF_EVENTS)
#define RESET_PERF_EVENTS       CONCAT3(ACCELERATOR_NAME,_,RESET_PERF_EVENTS)
#define START_PERF_EVENTS       CONCAT3(ACCELERATOR_NAME,_,START_PERF_EVENTS)
#define STOP_PERF_EVENTS        CONCAT3(ACCELERATOR_NAME,_,STOP_PERF_EVENTS)
#define GET_PERF_EVENT_CONFIG   CONCAT3(ACCELERATOR_NAME,_,GET_PERF_EVENT_CONFIG)
#define PERF_EVENT_ARGS_DESTROY CONCAT3(ACCELERATOR_NAME,_,PERF_EVENT_ARGS_DESTROY)
#else
#define OPEN_PERF_EVENT         open_perf_event
#define READ_PERF_EVENT         read_perf_event
#define CLOSE_PERF_EVENT        close_perf_event
#define OPEN_PERF_EVENTS        open_perf_events
#define ENABLE_PERF_EVENTS      enable_perf_events
#define CLOSE_PERF_EVENTS       close_perf_events
#define RESET_PERF_EVENTS       reset_perf_events
#define START_PERF_EVENTS       start_perf_events
#define STOP_PERF_EVENTS        stop_perf_events
#define GET_PERF_EVENT_CONFIG   get_perf_event_config
#define PERF_EVENT_ARGS_DESTROY perf_event_args_destroy
#endif

#define define_evaluateN(FUNC, ...)                                     \
{                                                                       \
  assert(repeat > 0);                                                   \
  assert(number >= 0);                                                  \
  assert(min_repeat_ms >= 0);                                           \
                                                                        \
  int fd = -1;                                                          \
  perf_event_args_t *events = NULL;                                     \
  int *perf_fds = NULL;                                                 \
  uint64_t* values = NULL;                                              \
  if (events_num > 0) {                                                 \
    perf_fds = alloca(events_num*sizeof(*perf_fds));                    \
    events = alloca(events_num*sizeof(*events));                        \
    values = alloca(events_num*sizeof(*values));                        \
    for(int e = 0; e < events_num; e++) {                               \
      GET_PERF_EVENT_CONFIG(events_names[e], &events[e]);               \
    }                                                                   \
  }                                                                     \
  OPEN_PERF_EVENTS(events_num, events, perf_fds);                       \
                                                                        \
  if (number > 0) {                                                     \
    mem_barrier();                                                      \
    (void)func(__VA_ARGS__);                                            \
    mem_barrier();                                                      \
  } else {                                                              \
    number = 1;                                                         \
  }                                                                     \
                                                                        \
  ENABLE_PERF_EVENTS(events_num, perf_fds);                             \
  for (int r = 0; r < repeat; r++) {                                    \
    double elapsed;                                                     \
    int attempts = number;                                              \
    while (1) {                                                         \
      RESET_PERF_EVENTS(events_num, perf_fds, values);                  \
      elapsed = fclock();                                               \
      START_PERF_EVENTS(events_num, perf_fds, values);                  \
      for (int a = 0; a < attempts; a++) {                              \
        mem_barrier();                                                  \
        (void)func(__VA_ARGS__);                                        \
        mem_barrier();                                                  \
      }                                                                 \
      STOP_PERF_EVENTS(events_num, perf_fds, values);                   \
      elapsed = fclock() - elapsed;                                     \
      if (elapsed * 1000 >= (double)min_repeat_ms)                      \
        break;                                                          \
      attempts *= NUMBER_FACTOR;                                        \
    }                                                                   \
    if (events_num > 0) {                                               \
      for (int e = 0; e < events_num; e++) {                            \
        if (perf_fds[e] == -1)                                          \
          results[r*events_num+e] = -1;                                 \
        else                                                            \
          results[r*events_num+e] = ((double)values[e]) / attempts;     \
      }                                                                 \
    } else {                                                            \
      results[r] = elapsed / attempts;                                  \
    }                                                                   \
  }                                                                     \
  for (int e = 0; e < events_num; e++) {                                \
    PERF_EVENT_ARGS_DESTROY(events[e]);                                 \
  }                                                                     \
  CLOSE_PERF_EVENTS(events_num, perf_fds);                              \
}

void evaluate_packed_perf(double *results, int events_num, const char *events_names[],
                          int repeat, int number, int min_repeat_ms,
                          packed_func_t func, PackedArg *args, int *codes, int nargs)
{
  PackedArg res;
  int res_code = 0;
  res.v_int64 = 0;
  define_evaluateN(func, args, codes, nargs, &res, &res_code);
}

void evaluate0_perf(double *results, int events_num, const char *events_names[],
                    int repeat, int number, int min_repeat_ms,
                    func0_t func)
{
  define_evaluateN(func);
}

void evaluate1_perf(double *results, int events_num, const char *events_names[],
                    int repeat, int number, int min_repeat_ms,
                    func1_t func, void *arg0)
{
  define_evaluateN(func, arg0);
}

void evaluate2_perf(double *results, int events_num, const char *events_names[],
                    int repeat, int number, int min_repeat_ms,
                    func2_t func, void *arg0, void *arg1)
{
  define_evaluateN(func, arg0, arg1);
}

void evaluate3_perf(double *results, int events_num, const char *events_names[],
                    int repeat, int number, int min_repeat_ms,
                    func3_t func, void *arg0, void *arg1, void *arg2)
{
  define_evaluateN(func, arg0, arg1, arg2);
}

void evaluate4_perf(double *results, int events_num, const char *events_names[],
                    int repeat, int number, int min_repeat_ms,
                    func4_t func, void *arg0, void *arg1, void *arg2, void *arg3)
{
  define_evaluateN(func, arg0, arg1, arg2, arg3);
}

void evaluate5_perf(double *results, int events_num, const char *events_names[],
                    int repeat, int number, int min_repeat_ms,
                    func5_t func, void *arg0, void *arg1, void *arg2, void *arg3, void *arg4)
{
  define_evaluateN(func, arg0, arg1, arg2, arg3, arg4);
}

void evaluate6_perf(double *results, int events_num, const char *events_names[],
                    int repeat, int number, int min_repeat_ms,
                    func6_t func, void *arg0, void *arg1, void *arg2, void *arg3, void *arg4, void *arg5)
{
  define_evaluateN(func, arg0, arg1, arg2, arg3, arg4, arg5);
}

typedef struct {
  int is_derived;
  metric_resolver_t resolver;
  int hw_offset;
  int out_offset;
  int start_pass;
} metric_map_t;

void evaluate_perf(double *results, int events_num, const char *events_names[],
                   int repeat, int number, int min_repeat_ms,
                   void (*func)(), void **args, int nargs)
{
  // No event only time
  if (events_num == 0) {
    switch (nargs) {
      case 0: evaluate0_perf(results, 0, NULL, repeat, number, min_repeat_ms, (func0_t)func); break;
      case 1: evaluate1_perf(results, 0, NULL, repeat, number, min_repeat_ms, (func1_t)func, args[0]); break;
      case 2: evaluate2_perf(results, 0, NULL, repeat, number, min_repeat_ms, (func2_t)func, args[0], args[1]); break;
      case 3: evaluate3_perf(results, 0, NULL, repeat, number, min_repeat_ms, (func3_t)func, args[0], args[1], args[2]); break;
      case 4: evaluate4_perf(results, 0, NULL, repeat, number, min_repeat_ms, (func4_t)func, args[0], args[1], args[2], args[3]); break;
      case 5: evaluate5_perf(results, 0, NULL, repeat, number, min_repeat_ms, (func5_t)func, args[0], args[1], args[2], args[3], args[4]); break;
      case 6: evaluate6_perf(results, 0, NULL, repeat, number, min_repeat_ms, (func6_t)func, args[0], args[1], args[2], args[3], args[4], args[5]); break;
      default: assert(0); break;
    }
    return;
  }

  int max_passes = 1;
  int total_hw_events = 0;
  int total_out_results = 0;
  int has_standard_pmus = 0;
  metric_map_t *map = (metric_map_t *)malloc(events_num * sizeof(metric_map_t));

  for (int i = 0; i < events_num; i++) {
      map[i].hw_offset = total_hw_events;
      map[i].out_offset = total_out_results;

      if (resolve_metric(events_names[i], &map[i].resolver) && map[i].resolver.is_supported) {
        map[i].is_derived = 1;

        // isolate the TMA within it own passes
        map[i].start_pass = max_passes;
        max_passes += map[i].resolver.num_passes;

        total_hw_events += map[i].resolver.num_hw_events;
        total_out_results += map[i].resolver.num_results;
      } else {
        // All PMUs in pass 0
        map[i].is_derived = 0;
        map[i].start_pass = 0;
        has_standard_pmus = 1;

        total_hw_events += 1;
        total_out_results += 1;
      }
    }

  double *hw_results = (double *)malloc(repeat * total_hw_events * sizeof(double));

    for (int pass = 0; pass < max_passes; pass++) {
      // No PMU, pass 0 is reserved to PMU
      if (pass == 0 && !has_standard_pmus) continue;

      int pass_events_num = 0;
      const char **pass_events_names = (const char **)malloc(total_hw_events * sizeof(char *));
      int *pass_hw_offsets = (int *)malloc(total_hw_events * sizeof(int));

      for (int i = 0; i < events_num; i++) {
        if (map[i].is_derived) {
          // If current pass is owned by this TMA
          if (pass >= map[i].start_pass && pass < map[i].start_pass + map[i].resolver.num_passes) {
            int local_pass = pass - map[i].start_pass;

            int ev_start = 0;
            for (int p = 0; p < local_pass; p++) ev_start += map[i].resolver.events_per_pass[p];
            int ev_count = map[i].resolver.events_per_pass[local_pass];

            for (int j = 0; j < ev_count; j++) {
              pass_events_names[pass_events_num] = map[i].resolver.hw_events[ev_start + j];
              pass_hw_offsets[pass_events_num] = map[i].hw_offset + ev_start + j;
              pass_events_num++;
            }
          }
        } else {
          if (pass == 0) {
            pass_events_names[pass_events_num] = events_names[i];
            pass_hw_offsets[pass_events_num] = map[i].hw_offset;
            pass_events_num++;
          }
        }
      }

      if (pass_events_num > 0) {
        double *pass_results = (double *)malloc(repeat * pass_events_num * sizeof(double));

        int all_failed = 1;
        perf_event_args_t *test_args = (perf_event_args_t *)malloc(pass_events_num * sizeof(perf_event_args_t));
        int *test_fds = (int *)malloc(pass_events_num * sizeof(int));

        for (int j = 0; j < pass_events_num; j++) GET_PERF_EVENT_CONFIG(pass_events_names[j], &test_args[j]);
        OPEN_PERF_EVENTS(pass_events_num, test_args, test_fds);

        for (int j = 0; j < pass_events_num; j++) if (test_fds[j] >= 0) all_failed = 0;

        for (int j = 0; j < pass_events_num; j++) PERF_EVENT_ARGS_DESTROY(test_args[j]);
        CLOSE_PERF_EVENTS(pass_events_num, test_fds);
        free(test_args);
        free(test_fds);

        if (all_failed) {
          fprintf(stderr,"[DEBUG] execution bypassed all events failed for the pass %d\n",pass_events_num);
          // Bypass heavy kernel execution if counters failed
          for (int r = 0; r < repeat; r++) {
            for (int j = 0; j < pass_events_num; j++) pass_results[r * pass_events_num + j] = -1.0;
          }
        } else {
            switch (nargs) {
              case 0: evaluate0_perf(pass_results, pass_events_num, pass_events_names, repeat, number, min_repeat_ms, (func0_t)func); break;
              case 1: evaluate1_perf(pass_results, pass_events_num, pass_events_names, repeat, number, min_repeat_ms, (func1_t)func, args[0]); break;
              case 2: evaluate2_perf(pass_results, pass_events_num, pass_events_names, repeat, number, min_repeat_ms, (func2_t)func, args[0], args[1]); break;
              case 3: evaluate3_perf(pass_results, pass_events_num, pass_events_names, repeat, number, min_repeat_ms, (func3_t)func, args[0], args[1], args[2]); break;
              case 4: evaluate4_perf(pass_results, pass_events_num, pass_events_names, repeat, number, min_repeat_ms, (func4_t)func, args[0], args[1], args[2], args[3]); break;
              case 5: evaluate5_perf(pass_results, pass_events_num, pass_events_names, repeat, number, min_repeat_ms, (func5_t)func, args[0], args[1], args[2], args[3], args[4]); break;
              case 6: evaluate6_perf(pass_results, pass_events_num, pass_events_names, repeat, number, min_repeat_ms, (func6_t)func, args[0], args[1], args[2], args[3], args[4], args[5]); break;
            }
        }
        // Gather results from passes
        for (int r = 0; r < repeat; r++) {
          for (int e = 0; e < pass_events_num; e++) {
            hw_results[r * total_hw_events + pass_hw_offsets[e]] =
                pass_results[r * pass_events_num + e];
          }
        }
        free(pass_results);
      }
      free(pass_events_names);
      free(pass_hw_offsets);
    }

  // Postprocessing
  for (int r = 0; r < repeat; r++) {
    for (int i = 0; i < events_num; i++) {
      const double *run_raw = &hw_results[r * total_hw_events + map[i].hw_offset];
      double *run_final = &results[r * total_out_results + map[i].out_offset];

      if (map[i].is_derived) {
        int failed = 0;
        for (int ev = 0; ev < map[i].resolver.num_hw_events; ev++) {
          if (run_raw[ev] == -1.0) failed = 1;
        }
        if (failed) {
          for (int res_idx = 0; res_idx < map[i].resolver.num_results; res_idx++) {
            run_final[res_idx] = -1.0;
          }
        } else {
          map[i].resolver.compute_formula(run_raw, run_final);
        }
      } else {
        run_final[0] = run_raw[0];
      }
    }
  }

  free(hw_results);
  free(map);
}

void evaluate(double *results,
                   int repeat, int number, int min_repeat_ms,
                   void (*func)(), void **args, int nargs)
{
    evaluate_perf(results, 0, NULL,
                  repeat, number, min_repeat_ms,
                  func, args, nargs);
}
void evaluate_packed(double *results,
                     int repeat, int number, int min_repeat_ms,
                     packed_func_t func, PackedArg *args, int *codes, int nargs)
{
    evaluate_packed_perf(results, 0, NULL,
                         repeat, number, min_repeat_ms,
                         func, args, codes, nargs);
}

int get_total_results_size(int events_num, const char *events_names[])
{
    int total = 0;
    for (int i = 0; i < events_num; i++) {
        total += get_perf_metric_results_count(events_names[i]);
    }
    return total;
}
