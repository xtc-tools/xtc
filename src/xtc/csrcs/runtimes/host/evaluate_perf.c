/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "perf_event.h"

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
  assert(number > 0);                                                   \
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
  mem_barrier();                                                        \
  (void)func(__VA_ARGS__);                                              \
  mem_barrier();                                                        \
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


void evaluate_perf(double *results, int events_num, const char *events_names[],
                   int repeat, int number, int min_repeat_ms,
                   void (*func)(), void **args, int nargs)
{
  switch (nargs) {
  case 0:
    evaluate0_perf(results, events_num, events_names,
                   repeat, number, min_repeat_ms,
                   (func0_t)func);
    break;
  case 1:
    evaluate1_perf(results, events_num, events_names,
                   repeat, number, min_repeat_ms,
                   (func1_t)func, args[0]);
    break;
  case 2:
    evaluate2_perf(results, events_num, events_names,
                   repeat, number, min_repeat_ms,
                   (func2_t)func, args[0], args[1]);
    break;
  case 3:
    evaluate3_perf(results, events_num, events_names,
                   repeat, number, min_repeat_ms,
                   (func3_t)func, args[0], args[1], args[2]);
    break;
  case 4:
    evaluate4_perf(results, events_num, events_names,
                   repeat, number, min_repeat_ms,
                   (func4_t)func, args[0], args[1], args[2], args[3]);
    break;
  case 5:
    evaluate5_perf(results, events_num, events_names,
                   repeat, number, min_repeat_ms,
                   (func5_t)func, args[0], args[1], args[2], args[3], args[4]);
    break;
  case 6:
    evaluate6_perf(results, events_num, events_names,
                   repeat, number, min_repeat_ms,
                   (func6_t)func, args[0], args[1], args[2], args[3], args[4], args[5]);
    break;
  default:
    assert(0);
    break;
  }
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
