/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include <assert.h>
#include <stdint.h>

extern double fclock(void); /* from fclock.c */

typedef void (*func0_t)();
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

typedef int (*packed_func_t)(PackedArg *, int *, int,PackedArg *,int *);

#define mem_barrier() asm("":::"memory")

#define NUMBER_FACTOR 2

#define define_evaluateN(FUNC, ...)					\
  {									\
  assert(repeat > 0);							\
  assert(number > 0);							\
  assert(min_repeat_ms >= 0);						\
									\
  mem_barrier();							\
  (void)func(__VA_ARGS__);						\
  mem_barrier();							\
									\
  for (int r = 0; r < repeat; r++) {					\
    double elapsed;							\
    int attempts = number;						\
    while (1) {								\
      elapsed = fclock();						\
      for (int a = 0; a < attempts; a++) {				\
	mem_barrier();							\
	(void)func(__VA_ARGS__);					\
	mem_barrier();							\
      }									\
      elapsed = fclock() - elapsed;					\
      if (elapsed * 1000 >= (double)min_repeat_ms)			\
	break;								\
      attempts *= NUMBER_FACTOR;					\
    }									\
    results[r] = elapsed / attempts;					\
  }									\
}

void evaluate_packed(double *results, int repeat,
		     int number, int min_repeat_ms,
		     packed_func_t func, PackedArg *args, int *codes, int nargs)
{
  PackedArg res;
  int res_code = 0;
  res.v_int64 = 0;
  define_evaluateN(func, args, codes, nargs, &res, &res_code);
}

void evaluate0(double *results, int repeat,
	       int number, int min_repeat_ms,
	       func0_t func)
{
  define_evaluateN(func);
}

void evaluate1(double *results, int repeat,
	       int number, int min_repeat_ms,
	       func1_t func, void *arg0)
{
  define_evaluateN(func, arg0);
}

void evaluate2(double *results, int repeat,
	       int number, int min_repeat_ms,
	       func2_t func, void *arg0, void *arg1)
{
  define_evaluateN(func, arg0, arg1);
}

void evaluate3(double *results, int repeat,
	       int number, int min_repeat_ms,
	       func3_t func, void *arg0, void *arg1, void *arg2)
{
  define_evaluateN(func, arg0, arg1, arg2);
}

void evaluate4(double *results, int repeat,
	       int number, int min_repeat_ms,
	       func4_t func, void *arg0, void *arg1, void *arg2, void *arg3)
{
  define_evaluateN(func, arg0, arg1, arg2, arg3);
}

void evaluate5(double *results, int repeat,
	       int number, int min_repeat_ms,
	       func5_t func, void *arg0, void *arg1, void *arg2, void *arg3, void *arg4)
{
  define_evaluateN(func, arg0, arg1, arg2, arg3, arg4);
}

void evaluate6(double *results, int repeat,
	       int number, int min_repeat_ms,
	       func6_t func, void *arg0, void *arg1, void *arg2, void *arg3, void *arg4, void *arg5)
{
  define_evaluateN(func, arg0, arg1, arg2, arg3, arg4, arg5);
}


void evaluate(double *results, int repeat,
	      int number, int min_repeat_ms,
	      void (*func)(), void **args, int nargs)
{
  switch (nargs) {
  case 0:
    evaluate0(results, repeat, number, min_repeat_ms,
	      func);
    break;
  case 1:
    evaluate1(results, repeat, number, min_repeat_ms,
	      func, args[0]);
    break;
  case 2:
    evaluate2(results, repeat, number, min_repeat_ms,
	      func, args[0], args[1]);
    break;
  case 3:
    evaluate3(results, repeat, number, min_repeat_ms,
	      func, args[0], args[1], args[2]);
    break;
  case 4:
    evaluate4(results, repeat, number, min_repeat_ms,
	      func, args[0], args[1], args[2], args[3]);
    break;
  case 5:
    evaluate5(results, repeat, number, min_repeat_ms,
	      func, args[0], args[1], args[2], args[3], args[4]);
    break;
  case 6:
    evaluate6(results, repeat, number, min_repeat_ms,
	      func, args[0], args[1], args[2], args[3], args[4], args[5]);
    break;
  default:
    assert(0);
    break;
  }
}
