#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <x86intrin.h>
#include <assert.h>

#include "perf_topdown.h"

/* Ref to: https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/topdown.txt */
#define RDPMC_FIXED	(1 << 30)	/* return fixed counters */
#define RDPMC_METRIC	(1 << 29)	/* return metric counters */
#define FIXED_COUNTER_SLOTS		3
#define METRIC_COUNTER_TOPDOWN_L1_L2	0
#define SLOTS_PER_CYCLES 4 // todo : arch dependent ?

#define GET_METRIC(m, i) (((m) >> (i*8)) & 0xff)

/* L1 Topdown metric events */
#define TOPDOWN_RETIRING(val)	((double)GET_METRIC(val, 0) / 0xff)
#define TOPDOWN_BAD_SPEC(val)	((double)GET_METRIC(val, 1) / 0xff)
#define TOPDOWN_FE_BOUND(val)	((double)GET_METRIC(val, 2) / 0xff)
#define TOPDOWN_BE_BOUND(val)	((double)GET_METRIC(val, 3) / 0xff)

/*
 * L2 Topdown metric events.
 * Available on Sapphire Rapids and later platforms.
 */
#define TOPDOWN_HEAVY_OPS(val)		((double)GET_METRIC(val, 4) / 0xff)
#define TOPDOWN_BR_MISPREDICT(val)	((double)GET_METRIC(val, 5) / 0xff)
#define TOPDOWN_FETCH_LAT(val)		((double)GET_METRIC(val, 6) / 0xff)
#define TOPDOWN_MEM_BOUND(val)		((double)GET_METRIC(val, 7) / 0xff)

/* static inline uint64_t read_slots(void) */
/* { */
/* 	return _rdpmc(RDPMC_FIXED | FIXED_COUNTER_SLOTS); */
/* } */

/* static inline uint64_t read_metrics(void) */
/* { */
/* 	return _rdpmc(RDPMC_METRIC | METRIC_COUNTER_TOPDOWN_L1_L2); */
/* } */

static int sys_perf_event_open(struct perf_event_attr *hw_event,
			       pid_t pid, int cpu, int group_fd,
			       unsigned long flags)
{
  long fd;
  fd = syscall(SYS_perf_event_open, hw_event, pid, cpu,
	       group_fd, flags);
  if (fd < 0) {
    perror("sys_perf_event_open failure");
    exit(EXIT_FAILURE);
  }
  return (int)fd;
}

/* static void *map_perf_fd(int perf_fd) */
/* { */
/*   void *ptr; */
/*   ptr = mmap(0, getpagesize(), PROT_READ, MAP_SHARED, perf_fd, 0); */
/*   if (ptr == NULL) { */
/*     perror("map_perf_fd failure"); */
/*     exit(EXIT_FAILURE); */
/*   } */
/*   return ptr; */
/* } */

/* static void unmap_perf_fd(void *ptr) */
/* { */
/*   munmap(ptr, getpagesize()); */
/* } */

struct perf_topdown_handle_s {
  uint64_t slots_start;
  uint64_t slots_stop;
  //uint64_t metrics_start;
  //uint64_t metrics_stop;
  uint64_t retiring_start;
  uint64_t retiring_stop;
  uint64_t fe_bound_start;
  uint64_t fe_bound_stop;
  uint64_t issued_start;
  uint64_t issued_stop;
  uint64_t bad_spec_start;
  uint64_t bad_spec_stop;
  int slots_fd;
  int retiring_fd;
  int fe_bound_fd;
  int issued_fd;
  int bad_spec_fd;
  //int metrics_fd;
  //void *slots_p;
  //void *metrics_p;
};
typedef struct perf_topdown_handle_s perf_topdown_handle_t;

static perf_topdown_handle_t global_handle;

static inline uint64_t read_perf_fd(int fd)
{
  uint64_t value;
  ssize_t n;
  n = read(fd, &value, sizeof(value));
  assert(n == sizeof(value));
  return value;
}

static inline void read_perf_group_fd(int fd, uint64_t *counters, int num)
{
  uint64_t fields[1 + num];
  ssize_t n;
  n = read(fd, &fields, sizeof(fields));
  assert(n == sizeof(fields));
  assert(fields[0] == num);
  for (int c = 0; c < num; c++) {
    counters[c] = fields[1+c];
  }
}

void perf_topdown_init(void)
{
  perf_topdown_handle_t *perf = &global_handle;
  struct perf_event_attr slots_attrs = {
    .type = PERF_TYPE_RAW,
    .size = sizeof(struct perf_event_attr),
    .config = 0x003c, //0x400, topdown-total-slots X86_CONFIG(.event=0x3c)
    .exclude_kernel = 1,
    .read_format = PERF_FORMAT_GROUP,
  };
  /* struct perf_event_attr metrics_attrs = { */
  /*   .type = PERF_TYPE_RAW, */
  /*   .size = sizeof(struct perf_event_attr), */
  /*   .config = 0x02c2, //0x8000, // retiring */
  /*   .exclude_kernel = 1, */
  /* }; */
  struct perf_event_attr retiring_attrs = {
    .type = PERF_TYPE_RAW,
    .size = sizeof(struct perf_event_attr),
    .config = 0x02c2, // topdown-slots-retired X86_CONFIG(.umask=2, .event=0xc2)
    .exclude_kernel = 1,
    .read_format = PERF_FORMAT_GROUP,
  };
  struct perf_event_attr issued_attrs = {
    .type = PERF_TYPE_RAW,
    .size = sizeof(struct perf_event_attr),
    .config = 0x010e, // topdown-slots-issued X86_CONFIG(.umask=1, .event=0x0e)
    .exclude_kernel = 1,
    .read_format = PERF_FORMAT_GROUP,
  };
  struct perf_event_attr fe_bound_attrs = {
    .type = PERF_TYPE_RAW,
    .size = sizeof(struct perf_event_attr),
    .config = 0x019c, // topdown-fetch-bubbles X86_CONFIG(.umask=1, .event=0x9c)
    .exclude_kernel = 1,
    .read_format = PERF_FORMAT_GROUP,
  };
  struct perf_event_attr bad_spec_attrs = {
    .type = PERF_TYPE_RAW,
    .size = sizeof(struct perf_event_attr),
    .config = 0x0100019d, // topdown-recovery-bubbles X86_CONFIG(.cmask=1, .umask=1, .event=0x9d)
    .exclude_kernel = 1,
    .read_format = PERF_FORMAT_GROUP,
  };
  memset(perf, 0, sizeof(*perf));
  perf->slots_fd = sys_perf_event_open(&slots_attrs, 0, -1, -1, 0);
  perf->retiring_fd = sys_perf_event_open(&retiring_attrs, 0, -1, perf->slots_fd, 0);
  perf->fe_bound_fd = sys_perf_event_open(&fe_bound_attrs, 0, -1, perf->slots_fd, 0);
  perf->issued_fd = sys_perf_event_open(&issued_attrs, 0, -1, perf->slots_fd, 0);
  perf->bad_spec_fd = sys_perf_event_open(&bad_spec_attrs, 0, -1, perf->slots_fd, 0);
}

void perf_topdown_fini(void)
{
  perf_topdown_handle_t *perf = &global_handle;
  close(perf->bad_spec_fd);
  close(perf->issued_fd);
  close(perf->fe_bound_fd);
  close(perf->retiring_fd);
  close(perf->slots_fd);
}

void perf_topdown_start(void)
{
  perf_topdown_handle_t *perf = &global_handle;
  uint64_t counters[5];
  read_perf_group_fd(perf->slots_fd, counters, sizeof(counters)/sizeof(*counters));
  perf->slots_start = counters[0];
  perf->retiring_start = counters[1];
  perf->fe_bound_start = counters[2];
  perf->issued_start = counters[3];
  perf->bad_spec_start = counters[4];
}

void perf_topdown_stop(void)
{
  perf_topdown_handle_t *perf = &global_handle;
  uint64_t counters[5];
  read_perf_group_fd(perf->slots_fd, counters, sizeof(counters)/sizeof(*counters));
  perf->slots_stop = counters[0];
  perf->retiring_stop = counters[1];
  perf->fe_bound_stop = counters[2];
  perf->issued_stop = counters[3];
  perf->bad_spec_stop = counters[4];
}

void perf_topdown_dump(FILE *file)
{
  perf_topdown_handle_t *perf = &global_handle;
  uint64_t cycles = perf->slots_stop - perf->slots_start;
  uint64_t slots = cycles * SLOTS_PER_CYCLES;   // assume Skylake, Broadwell, Cascade Lake arch
  uint64_t retiring = perf->retiring_stop - perf->retiring_start;
  uint64_t bad_spec = perf->bad_spec_stop - perf->bad_spec_start;
  uint64_t fe_bound = perf->fe_bound_stop - perf->fe_bound_start;
  uint64_t issued = perf->issued_stop - perf->issued_start;
  uint64_t be_bound = slots - retiring - fe_bound - bad_spec;
  double r_retiring = (double)retiring / slots;
  double r_bad_spec = (double)bad_spec / slots;
  double r_fe_bound = (double)fe_bound / slots;
  double r_issued = (double)issued / slots;
  double r_be_bound = (double)be_bound / slots;
  fprintf(file, "Cycles: %lu\n", cycles);
  fprintf(file, "Slots: %lu (%.2f%%)\n", slots, 100.0);
  fprintf(file, "Issued: %lu (%.2f%%)\n", issued, r_issued*100);
  fprintf(file, "Retiring: %lu (%.2f%%)\n", retiring, r_retiring*100);
  fprintf(file, "FE_bound: %lu (%.2f%%)\n", fe_bound, r_fe_bound*100);
  fprintf(file, "Bad_spec: %lu (%.2f%%)\n", bad_spec, r_bad_spec*100);
  fprintf(file, "BE_bound: %lu (%.2f%%)\n", be_bound, r_be_bound*100);
}
