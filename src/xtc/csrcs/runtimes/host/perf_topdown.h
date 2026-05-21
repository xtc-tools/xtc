#ifndef _PERF_TOPDOWN_H
#define _PERF_TOPDOWN_H

#include <stdio.h>

extern void perf_topdown_init(void);
extern void perf_topdown_fini(void);
extern void perf_topdown_start(void);
extern void perf_topdown_stop(void);
extern void perf_topdown_dump(FILE *file);

#endif
