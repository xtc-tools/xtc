/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include "perf_event.h"
#include <assert.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <syscall.h>
#include <unistd.h>

#if HAS_PFM
#include <perfmon/pfmlib.h>
#endif

#if HAS_GPU
#include "perf_event_gpu.h"
#endif /* HAS_GPU */

int all_perf_events[PERF_EVENT_NUM] = {
    PERF_EVENT_CYCLES,       PERF_EVENT_CLOCKS,        PERF_EVENT_INSTRS,
    PERF_EVENT_MIGRATIONS,   PERF_EVENT_SWITCHES,      PERF_EVENT_CACHE_ACCESS,
    PERF_EVENT_CACHE_MISSES, PERF_EVENT_BRANCH_INSTRS, PERF_EVENT_BRANCH_MISSES,
};

typedef struct {
  int id;
  int type;
  int num;
  const char *name;
} perf_event_decl_t;

static const perf_event_decl_t perf_events_decl[] = {
    {PERF_EVENT_CYCLES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"},
    {PERF_EVENT_CLOCKS, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK, "clocks"},
    {PERF_EVENT_INSTRS, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS,
     "instructions"},
    {PERF_EVENT_MIGRATIONS, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS,
     "migrations"},
    {PERF_EVENT_SWITCHES, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES,
     "context_switches"},
    {PERF_EVENT_CACHE_ACCESS, PERF_TYPE_HARDWARE,
     PERF_COUNT_HW_CACHE_REFERENCES, "cache_access"},
    {PERF_EVENT_CACHE_MISSES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES,
     "cache_misses"},
    {PERF_EVENT_BRANCH_INSTRS, PERF_TYPE_HARDWARE,
     PERF_COUNT_HW_BRANCH_INSTRUCTIONS, "branches"},
    {PERF_EVENT_BRANCH_MISSES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES,
     "branches_misses"},
};

typedef struct {
  struct perf_event_attr *attr;
  char **fstr;
  size_t size;
  int idx;
  int cpu;
  int flags;
} local_pfm_perf_encode_arg_t;

static int sys_perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                               int cpu, int group_fd, unsigned long flags) {
  long fd;
  fd = syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  return (int)fd;
}

static void init_perf_event_attr(struct perf_event_attr *attr_ptr)
{
  memset(attr_ptr, 0, sizeof(*attr_ptr));
  attr_ptr->size = sizeof(*attr_ptr);
  attr_ptr->exclude_kernel = 1;
  attr_ptr->exclude_hv = 1;
  attr_ptr->exclude_idle = 1;
  attr_ptr->inherit = 1;
  attr_ptr->disabled = 0;
}


static int open_perf_event_group(perf_event_args_t event, int group_fd) {
  if (event.mode == PERF_ARG_INVALID) {
      return -1;
  } else if (event.mode == PERF_ARG_GENERIC) {
    struct perf_event_attr attr;
    init_perf_event_attr(&attr);
    attr.type = event.args.config_pair.type;
    attr.config = event.args.config_pair.event;
    return sys_perf_event_open(&attr, 0 /*pid*/, -1 /*cpu*/, group_fd /*group fd*/, 0 /*flags*/);
  } else {
    local_pfm_perf_encode_arg_t *perf_gen = (local_pfm_perf_encode_arg_t *)event.args.config_ptr;
    return sys_perf_event_open(perf_gen->attr,
                               0 /*pid*/, perf_gen->cpu /*cpu*/, group_fd /*group fd*/,
                               perf_gen->flags /*flags*/);
  }
}

int open_perf_event(perf_event_args_t event) {
    return open_perf_event_group(event, -1);
}

static __attribute__((constructor)) void perf_event_init(void) {
#if HAS_PFM
  int res = pfm_initialize();
  if (res != PFM_SUCCESS) {
    fprintf(stderr, "ERROR: cannot initialize libpfm: pfm_initialize(): %s\n",
            pfm_strerror(res));
    exit(EXIT_FAILURE);
  }
#endif /* HAS_PFM */
}

static __attribute__((destructor)) void perf_event_fini(void) {
#if HAS_PFM
  pfm_terminate();
#endif /* HAS_PFM */
}

uint64_t read_perf_event(int perf_fd) {
  uint64_t value;
  ssize_t n;
  n = read(perf_fd, &value, sizeof(value));
  assert(n == sizeof(value));
  return value;
}

void close_perf_event(int perf_fd) { close(perf_fd); }

void open_perf_events(int n_events, const perf_event_args_t *events, int *fds) {
  assert(n_events <= PERF_EVENT_MAX_EVENTS);
  int group_fd = -1; // group leader
  for (int i = 0; i < n_events; i++) {
    #if HAS_GPU
    if (events[i].mode == PERF_ARG_GPU) // FIXME do it more efficiently
      continue;
    #endif /* HAS_GPU */
    fds[i] = open_perf_event_group(events[i], group_fd);
    if (group_fd == -1 && fds[i] >= 0) {
        group_fd = fds[i];
    }
  }
  #if HAS_GPU
  open_perf_events__gpu(n_events, events, fds);
  #endif /* HAS_GPU */
}

void enable_perf_events(int n_events, const int *fds) {
  /* Nothing to do */
}

void close_perf_events(int n_events, const int *fds) {
  for (int i = 0; i < n_events; i++) {
    #if HAS_GPU
    if (fds[i] == PERF_EVENT_GPU)
      continue;
    #endif /* HAS_GPU */
    if (fds[i] >= 0) {
      close_perf_event(fds[i]);
    }
  }
  #if HAS_GPU
  close_perf_events__gpu(n_events, fds);
  #endif /* HAS_GPU */
}

uint64_t _tmp_results[PERF_EVENT_MAX_EVENTS];

void reset_perf_events(int n_events, const int *fds, uint64_t *results) {
  for (int i = 0; i < n_events; i++) {
    results[i] = 0;
  }
  #if HAS_GPU
  reset_perf_events__gpu(n_events, fds, results);
  #endif /* HAS_GPU */
}

void start_perf_events(int n_events, const int *fds, uint64_t *results) {
  // Start GPU perf event before because to reduce overhead on CPU perf measurement
  #if HAS_GPU
  start_perf_events__gpu(n_events, fds, results);
  #endif /* HAS_GPU */

  for (int i = 0; i < n_events; i++) {
    #if HAS_GPU
    if (fds[i] == PERF_EVENT_GPU)
      continue;
    #endif /* HAS_GPU */
    if (fds[i] >= 0) {
      _tmp_results[i] = read_perf_event(fds[i]);
    }
  }
}

void stop_perf_events(int n_events, const int *fds, uint64_t *results) {
  for (int i = 0; i < n_events; i++) {
    #if HAS_GPU
    if (fds[i] == PERF_EVENT_GPU)
      continue;
    #endif /* HAS_GPU */
    if (fds[i] >= 0) {
      _tmp_results[i] = read_perf_event(fds[i]) - _tmp_results[i];
      results[i] += _tmp_results[i];
    }
  }

  // Stop GPU perf event after because to reduce overhead on CPU perf measurement
  #if HAS_GPU
  stop_perf_events__gpu(n_events, fds, results);
  #endif /* HAS_GPU */
}

#define X86_RAW(event, umask, cmask) \
    ((((uint64_t)(event) >> 8) << 32) | \
     ((uint64_t)(cmask) << 24) | \
     ((uint64_t)(umask) << 8) | \
     ((uint64_t)(event) & 0xFF))

typedef struct {
    const char *name;
    uint64_t raw_config;
} pmu_event_def_t;

// INTEL SKYLAKE / CASCADE LAKE (Pre-Ice Lake, no hw PERF_METRICS)
static const pmu_event_def_t skl_raw_events[] = {
    { "@skl_slots",           X86_RAW(0x3C, 0x00, 0) },
    { "@skl_fe_bound",        X86_RAW(0x9C, 0x01, 0) },
    { "@skl_issued",          X86_RAW(0x0E, 0x01, 0) },
    { "@skl_retiring",        X86_RAW(0xC2, 0x02, 0) },
    // with flags (any=1, edge=1)
    { "@skl_recovery",        0x0120010D             },
    { "@skl_machine_clears",  0x010401C3             },
    // L2 & L3 Memory Bound
    { "@skl_mem_stalls",      X86_RAW(0xA3, 0x14, 0x14) },
    { "@skl_core_stalls",     X86_RAW(0xA2, 0x01, 0) },
    { "@skl_fetch_lat_data",  X86_RAW(0x80, 0x04, 0) },
    { "@skl_fetch_lat_tag",   X86_RAW(0x83, 0x04, 0) },
    { "@skl_heavy_ops",       X86_RAW(0x79, 0x30, 0) },
    { "@skl_br_misp",         X86_RAW(0xC5, 0x00, 0) },
    { "@skl_stalls_mem_any",  X86_RAW(0xA3, 0x14, 0x14) },
    { "@skl_stalls_l1d_miss", X86_RAW(0xA3, 0x0C, 0x0C) },
    { "@skl_stalls_l2_miss",  X86_RAW(0xA3, 0x05, 0x05) },
    { "@skl_stalls_l3_miss",  X86_RAW(0xA3, 0x06, 0x06) },
    { "@skl_bound_on_stores", X86_RAW(0xA6, 0x40, 0) }
};

// INTEL MODERN (Ice Lake, Sapphire Rapids, Alder/Raptor Lake)
static const pmu_event_def_t icl_raw_events[] = {
    { "@icl_slots",           X86_RAW(0x00, 0x04, 0) },
    { "@icl_retiring",        X86_RAW(0x00, 0x80, 0) },
    { "@icl_bad_spec",        X86_RAW(0x00, 0x81, 0) },
    { "@icl_fe_bound",        X86_RAW(0x00, 0x82, 0) },
    { "@icl_be_bound",        X86_RAW(0x00, 0x83, 0) },
    { "@icl_heavy_ops",       X86_RAW(0x00, 0x84, 0) },
    { "@icl_br_mispredict",   X86_RAW(0x00, 0x85, 0) },
    { "@icl_fetch_lat",       X86_RAW(0x00, 0x86, 0) },
    { "@icl_mem_bound",       X86_RAW(0x00, 0x87, 0) },
    // Fallback L3 Memory events (same as Skylake)
    { "@icl_cyc",             X86_RAW(0x3C, 0x00, 0) },
    { "@icl_stalls_mem_any",  X86_RAW(0xA3, 0x14, 0x14) },
    { "@icl_stalls_l1d_miss", X86_RAW(0xA3, 0x0C, 0x0C) },
    { "@icl_stalls_l2_miss",  X86_RAW(0xA3, 0x05, 0x05) },
    { "@icl_stalls_l3_miss",  X86_RAW(0xA3, 0x06, 0x06) },
    { "@icl_bound_on_stores", X86_RAW(0xA6, 0x40, 0) }
};

// AMD ZEN 4 (Family 19h)
static const pmu_event_def_t zen4_raw_events[] = {
    { "@zen4_cyc",            X86_RAW(0x076, 0x00, 0) }, // LS_NOT_HALTED_CYC: Core active cycles
    { "@zen4_fe",             X86_RAW(0x1A0, 0x01, 0) }, // DE_NO_DISPATCH_PER_SLOT.NO_OPS_FROM_FRONTEND: Dispatch slots empty due to FE
    { "@zen4_disp",           X86_RAW(0x0AA, 0x07, 0) }, // DE_SRC_OP_DISP.ALL: OPs dispatched from any source (Decoder, OpCache, uCode)
    { "@zen4_ret",            X86_RAW(0x0C1, 0x00, 0) }, // EX_RET_OPS: Total OPs retired
    { "@zen4_be_mem",         X86_RAW(0x1A0, 0x02, 0) },
    { "@zen4_be_cpu",         X86_RAW(0x1A0, 0x04, 0) },
    { "@zen4_fe_lat",         X86_RAW(0x1A0, 0x01, 0x06) }, // DE_NO_DISPATCH_PER_SLOT.NO_OPS_FROM_FRONTEND (Cmask=6): Full pipeline stall from FE
    { "@zen4_fe_tot",         X86_RAW(0x1A0, 0x01, 0) },
    { "@zen4_bs_misp",        X86_RAW(0x0C3, 0x00, 0) },
    { "@zen4_bs_resync",      X86_RAW(0x091, 0x00, 0) },
    { "@zen4_ret_micro",      X86_RAW(0x1C2, 0x00, 0) }  // EX_RET_UCODE_OPS: Retired macro-ops originating from microcode sequencer (Heavy Ops)
};

#define ARM_RAW(event) (event)

static const pmu_event_def_t arm_raw_events[] = {
    // TopdownL1
    { "@arm_cyc",       ARM_RAW(0x11) }, // CPU_CYCLES
    { "@arm_fe",        ARM_RAW(0x23) }, // STALL_FRONTEND
    { "@arm_be",        ARM_RAW(0x24) }, // STALL_BACKEND
    { "@arm_inst",      ARM_RAW(0x08) }, // INST_RETIRED
    { "@arm_brmisp",    ARM_RAW(0x10) }, // BR_MIS_PRED

    // TopdownL2
    { "@arm_be_mem",    ARM_RAW(0x45) }, // STALL_BACKEND_MEM
    { "@arm_be_cpu",    ARM_RAW(0x44) }, // STALL_BACKEND_CPU
    { "@arm_l1i_miss",  ARM_RAW(0x01) }  // L1I_CACHE_REFILL
};

/*
 * Source
 * Intel : https://github.com/torvalds/linux/blob/m aster/arch/x86/events/intel/core.c
 * AMD : https://github.com/torvalds/linux/blob/m aster/arch/x86/events/amd/core.c
 *       tools/perf/pmu-events/arch/x86/amdzen4/pipeline.json
 *
 * https://github.com/intel/perfmon
 *
 * todo ARM : https://developer.arm.com/documentation/ddi0434/a/performance-monitoring-unit/performance-monitoring-register-descriptions/event-type-select-register?lang=en
 */
static inline int find_raw_event(const char *name, const pmu_event_def_t *table, int table_size, perf_event_args_t *event) {
    for (int i = 0; i < table_size; i++) {
        if (strcmp(name, table[i].name) == 0) {
            event->mode = PERF_ARG_GENERIC;
            event->args.config_pair.type = PERF_TYPE_RAW;
            event->args.config_pair.event = table[i].raw_config;
            return 0;
        }
    }
    return 1;
}

static inline int set_config_by_arch(const char *name, perf_event_args_t *event) {
    // Old Intel (Skylake / Cascade Lake)
    if (strncmp(name, "@skl_", 5) == 0) {
        return find_raw_event(name, skl_raw_events, sizeof(skl_raw_events) / sizeof(skl_raw_events[0]), event);
    }
    // Modern Intel (Ice Lake / Raptor Lake)
    else if (strncmp(name, "@icl_", 5) == 0) {
        return find_raw_event(name, icl_raw_events, sizeof(icl_raw_events) / sizeof(icl_raw_events[0]), event);
    }
    // AMD Zen 4
    else if (strncmp(name, "@zen4_", 6) == 0) {
        return find_raw_event(name, zen4_raw_events, sizeof(zen4_raw_events) / sizeof(zen4_raw_events[0]), event);
    }

    return -1; // unknow prefix
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

 int arch_specific_config = set_config_by_arch(name,event);
 if(arch_specific_config != -1) return arch_specific_config;

#if HAS_GPU
  if (strncmp(name, "gpu.", 4) == 0) {
    return get_perf_event_config__gpu(name, event);
  }
#endif /* HAS_GPU */

  #if HAS_PFM
  struct perf_event_attr *attr = malloc(sizeof(struct perf_event_attr));
  init_perf_event_attr(attr);

  local_pfm_perf_encode_arg_t *arg = malloc(sizeof(local_pfm_perf_encode_arg_t));

  memset(arg, 0, sizeof(local_pfm_perf_encode_arg_t));

  arg->size = sizeof(local_pfm_perf_encode_arg_t);
  arg->attr = attr;

  int ret = pfm_get_os_event_encoding(name, PFM_PLM3, PFM_OS_PERF_EVENT, arg);

  if (ret == PFM_SUCCESS) {
    event->mode = PERF_ARG_PTR;
    event->args.config_ptr = (const void *)arg;
    return 0;
  }
  free(arg);
  free(attr);
  #endif /* HAS_PFM */

  event->mode = PERF_ARG_INVALID;
  return 1;
}

void perf_event_args_destroy(perf_event_args_t args) {
    if (args.mode == PERF_ARG_PTR && args.args.config_ptr != NULL) {
        local_pfm_perf_encode_arg_t* pfm = (local_pfm_perf_encode_arg_t*) args.args.config_ptr;
        free((void*)pfm->attr);
        free((void*)args.args.config_ptr);
        args.args.config_ptr = NULL;
    }
}
