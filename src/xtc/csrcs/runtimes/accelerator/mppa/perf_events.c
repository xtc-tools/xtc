/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include "host_structures.h"
#include "mppa_management_host.h"

#include <mppa_offload_host.h>

#include <stdlib.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#define MAX_NB_PM_COUNTERS 8 /* value of _COS_PM_NB for Coolidge V2 */

typedef struct {
    const char *name;
    int id;
    const char *description;
} kvx_pm_counter_t;

#define NB_COUNTERS_AVAILABLE 55
static const kvx_pm_counter_t kvx_pm_counters[NB_COUNTERS_AVAILABLE] = {
    { "PCC"      , 0  /* _COS_PM_PCC      */, "Processor Clock CycleSAT" },
    { "ICC"      , 1  /* _COS_PM_ICC      */, "Idle Clock Cycle" },
    { "EBE"      , 2  /* _COS_PM_EBE      */, "Executed Bundle Event" },
    { "ENIE"     , 3  /* _COS_PM_ENIE     */, "Executed N Instructions Event" },
    { "ENSE"     , 4  /* _COS_PM_ENSE     */, "Executed N Syllables Event" },
    { "ICHE"     , 5  /* _COS_PM_ICHE     */, "I$ Hit Event" },
    { "ICME"     , 6  /* _COS_PM_ICME     */, "I$ Miss Event" },
    { "ICMABE"   , 7  /* _COS_PM_ICMABE   */, "I$ Memory Accesses Burst Event" },
    { "MNGIC"    , 8  /* _COS_PM_MNGIC    */, "Memory Not Granting Instruction cache access Cycle" },
    { "MIMHE"    , 9  /* _COS_PM_MIMHE    */, "MMU Instruction Micro-tlb Hit Event" },
    { "MIMME"    , 10 /* _COS_PM_MIMME    */, "MMU Instruction Micro-tlb Miss Event" },
    { "IATSC"    , 11 /* _COS_PM_IATSC    */, "Instruction Address Translation Stall Cycle" },
    { "FE"       , 12 /* _COS_PM_FE       */, "Fetch Event" },
    { "PBSC"     , 13 /* _COS_PM_PBSC     */, "Prefetch Buffer Starvation Cycle" },
    { "PNVC"     , 14 /* _COS_PM_PNVC     */, "Pipeline No Valid Cycle" },
    { "PSC"      , 15 /* _COS_PM_PSC      */, "Pipeline Starvation Cycle" },
    { "TADBE"    , 16 /* _COS_PM_TADBE    */, "Taken Applicative Direct Branch Event" },
    { "TABE"     , 17 /* _COS_PM_TABE     */, "Taken Applicative Branch Event" },
    { "TBE"      , 18 /* _COS_PM_TBE      */, "Taken Branch Event" },
    { "MDMHE"    , 19 /* _COS_PM_MDMHE    */, "MMU Data Micro-tlb Hit Event" },
    { "MDMME"    , 20 /* _COS_PM_MDMME    */, "MMU Data Micro-tlb Miss Event" },
    { "DATSC"    , 21 /* _COS_PM_DATSC    */, "Data Address Translation Stall Cycle" },
    { "DCLHE"    , 22 /* _COS_PM_DCLHE    */, "D$ Load Hit Event" },
    { "DCHE"     , 23 /* _COS_PM_DCHE     */, "D$ Hit Event" },
    { "DCLME"    , 24 /* _COS_PM_DCLME    */, "D$ Load Miss Event" },
    { "DCME"     , 25 /* _COS_PM_DCME     */, "D$ Miss Event" },
    { "DARSC"    , 26 /* _COS_PM_DARSC    */, "Data Access Related Stall Cycle" },
    { "LDSC"     , 27 /* _COS_PM_LDSC     */, "Load Dependency Stall Cycle" },
    { "DCNGC"    , 28 /* _COS_PM_DCNGC    */, "Data Cache Not Granting Cycle" },
    { "DMAE"     , 29 /* _COS_PM_DMAE     */, "Data Misaligned Access Event" },
    { "LCFSC"    , 30 /* _COS_PM_LCFSC    */, "Load Cam Full Stall Cycle" },
    { "MNGDC"    , 31 /* _COS_PM_MNGDC    */, "Memory Not Granting Data cache access Cycle" },
    { "MACC"     , 32 /* _COS_PM_MACC     */, "Memory Accesses Conflict Cycle" },
    { "TACC"     , 33 /* _COS_PM_TACC     */, "TLB Accesses Conflict Cycle" },
    { "IWC"      , 34 /* _COS_PM_IWC      */, "Idle Wait Cycle" },
    { "WISC"     , 35 /* _COS_PM_WISC     */, "Wait Instruction Stall Cycle" },
    { "SISC"     , 36 /* _COS_PM_SISC     */, "Synchronization Instruction Stall Cycle" },
    { "DDSC"     , 37 /* _COS_PM_DDSC     */, "Data Dependency Stall Cycle" },
    { "SC"       , 38 /* _COS_PM_SC       */, "Stall Cycle" },
    { "ELE"      , 39 /* _COS_PM_ELE      */, "Executed Load Event" },
    { "ELNBE"    , 40 /* _COS_PM_ELNBE    */, "Executed Load N Bytes Event" },
    { "ELUE"     , 41 /* _COS_PM_ELUE     */, "Executed Load Uncached Event" },
    { "ELUNBE"   , 42 /* _COS_PM_ELUNBE   */, "Executed Load Uncached N Bytes Event" },
    { "ESE"      , 43 /* _COS_PM_ESE      */, "Executed Store Event" },
    { "ESNBE"    , 44 /* _COS_PM_ESNBE    */, "Executed Store N Bytes Event" },
    { "EAE"      , 45 /* _COS_PM_EAE      */, "Executed Atomics Event" },
    { "CIRE"     , 46 /* _COS_PM_CIRE     */, "Coherency Invalidation Request Event" },
    { "CIE"      , 47 /* _COS_PM_CIE      */, "Coherency Invalidation Event" },
    { "SE"       , 48 /* _COS_PM_SE       */, "Stop Event" },
    { "RE"       , 49 /* _COS_PM_RE       */, "Reset Event" },
    { "FSC"      , 50 /* _COS_PM_FSC      */, "Fetch Stall Cycle" },
    /* PMC available only on Coolidge V2 */
    { "CPIRE"    , 51 /* _COS_PM_CPIRE    */, "Coherency Precise Invalidation Request Event" },
    { "CPIE"     , 52 /* _COS_PM_CPIE     */, "Coherency Precise Invalidation Event" },
    { "HUPEVICT" , 53 /* _COS_PM_HUPEVICT */, "Hit-Under-Prefetch Line Eviction Event" },
    { "HUPHIT"   , 54 /* _COS_PM_HUPHIT   */, "Hit-Under-Prefetch Hit Event" },
};

void mppa_setup_perf_events(char* event_names[], int n_events) {
    assert(n_events <= MAX_NB_PM_COUNTERS);
    int counter_ids[MAX_NB_PM_COUNTERS] = {0};
    for (int i = 0; i < n_events; i++) {
        const char *event_name = event_names[i];
        int j = 0;
        for (j = 0; j < NB_COUNTERS_AVAILABLE; j++) {
            if (strcmp(event_name, kvx_pm_counters[j].name) == 0) {
                counter_ids[i] = kvx_pm_counters[j].id;
                break;
            }
        }
        if (j == NB_COUNTERS_AVAILABLE) {
            printf("Event %s not found in the list of available events\n", event_name);
            assert(false);
        }
    }
    // Call runtime function
    mppa_setup_pm_counters(counter_ids, n_events);
}

void mppa_read_perf_events_results(void* dst_handle) {
    mppa_copy_out_pm_counters_buffer(dst_handle);
}

void mppa_reset_perf_events() {
  mppa_reset_pm_counters_buffer();
}

// uint64_t mppa_get_frequency() already exists in the runtime, just need to call it
