/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
//#include <stdio.h>

#include "perf_metrics.h"


#if defined(__x86_64__) || defined(__i386__)
    #define ARCH_IS_X86 1
    #include <cpuid.h>
    #include <x86intrin.h>
#else
    #define ARCH_IS_X86 0
#endif

#if defined(__aarch64__) || defined(__arm__)
    #define ARCH_IS_ARM 1
#else
    #define ARCH_IS_ARM 0
#endif

#if ARCH_IS_X86
#include <cpuid.h>



#define GET_METRIC(m, i) (((m) >> (i*8)) & 0xff)

/* L1 Topdown metric events */
#define TOPDOWN_RETIRING(val)	((float)GET_METRIC(val, 0) / 0xff)
#define TOPDOWN_BAD_SPEC(val)	((float)GET_METRIC(val, 1) / 0xff)
#define TOPDOWN_FE_BOUND(val)	((float)GET_METRIC(val, 2) / 0xff)
#define TOPDOWN_BE_BOUND(val)	((float)GET_METRIC(val, 3) / 0xff)

/*
 * L2 Topdown metric events.
 * Available on Sapphire Rapids and later platforms.
 */
#define TOPDOWN_HEAVY_OPS(val)		((float)GET_METRIC(val, 4) / 0xff)
#define TOPDOWN_BR_MISPREDICT(val)	((float)GET_METRIC(val, 5) / 0xff)
#define TOPDOWN_FETCH_LAT(val)		((float)GET_METRIC(val, 6) / 0xff)
#define TOPDOWN_MEM_BOUND(val)		((float)GET_METRIC(val, 7) / 0xff)

#define RDPMC_FIXED	(1 << 30)	/* return fixed counters */
#define RDPMC_METRIC	(1 << 29)	/* return metric counters */

#define FIXED_COUNTER_SLOTS		4
#define METRIC_COUNTER_TOPDOWN_L1_L2	0

static inline uint64_t read_slots(void)
{
	return _rdpmc(RDPMC_FIXED | FIXED_COUNTER_SLOTS);
}

static inline uint64_t read_metrics(void)
{
	return _rdpmc(RDPMC_METRIC | METRIC_COUNTER_TOPDOWN_L1_L2);
}
/*
_rdpmc calls should not be mixed with reading the metrics and slots counters
through system calls, as the kernel will reset these counters after each system
call. */



#ifdef __linux__
#include <linux/perf_event.h>
static void get_cpu_family_model(int *family, int *model) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        int base_family = (eax >> 8) & 0xF;
        int base_model  = (eax >> 4) & 0xF;

        int ext_family = (eax >> 20) & 0xFF;
        int ext_model  = (eax >> 16) & 0xF;

        *family = base_family;
        *model = base_model;

        if (base_family == 0xF) {
            *family += ext_family;
        }
        if (base_family == 0x6 || base_family == 0xF) {
            *model += (ext_model << 4);
        }
    } else {
        *family = 0;
        *model = 0;
    }
}
#endif

typedef enum {
    UARCH_UNKNOWN = 0,
    UARCH_INTEL_OLD,        // No native tma counters
    UARCH_INTEL_MODERN,     // Native tma counters
    UARCH_AMD_ZEN_1_2,
    UARCH_AMD_ZEN_3,
    UARCH_AMD_ZEN_4,
    UARCH_ARM
} cpu_uarch_t;

static cpu_uarch_t current_uarch = UARCH_UNKNOWN;
static int is_uarch_initialized = 0;

static cpu_uarch_t detect_hardware_architecture(void) {
#if ARCH_IS_X86
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx)) return UARCH_UNKNOWN;

    int family = 0, model = 0;
    get_cpu_family_model(&family, &model);

    // Intel: ebx="Genu", edx="ineI", ecx="ntel"
    if (ebx == 0x756e6547 && edx == 0x49656e69 && ecx == 0x6c65746e) {
        if (family == 6) {
            switch (model) {
                case 0x4E: case 0x5E: case 0x55: // Skylake, Cascade Lake, Cooper Lake
                case 0x8E: case 0x9E:            // Kaby, Coffee, Whiskey, Amber Lake
                case 0xA5: case 0xA6:            // Comet Lake
                case 0x3D: case 0x47: case 0x4F: // Broadwell
                case 0x56:                       // Broadwell-DE
                case 0x66:                       // Cannon Lake
                    return UARCH_INTEL_OLD;
                    
                case 0x7E: case 0x7D: case 0x9D: // Ice Lake Client
                case 0x6A: case 0x6C:            // Ice Lake Server
                case 0xA7:                       // Rocket Lake
                case 0x8C: case 0x8D:            // Tiger Lake
                case 0x8A:                       // Lakefield
                case 0x8F:                       // Sapphire Rapids
                case 0xCF:                       // Emerald Rapids
                case 0xAD: case 0xAE:            // Granite Rapids
                case 0x97: case 0x9A:            // Alder Lake
                case 0xB7: case 0xBA: case 0xBF: // Raptor Lake
                case 0xD7:                       // Bartlett Lake
                case 0xAA: case 0xAC:            // Meteor Lake
                case 0xB5: case 0xC5: case 0xC6: // Arrow Lake
                case 0xBD:                       // Lunar Lake
                case 0xCC: case 0xE5:            // Panther Lake
                case 0xD5:                       // Wildcat Lake
                    return UARCH_INTEL_MODERN;
            }
        }
        else if (family == 18) {
            switch (model) {
                case 0x01: case 0x03:
                    return UARCH_INTEL_MODERN; // Nova Lake
            }
        } 
        else if (family == 19) {
            switch (model) {
                case 0x01:
                    return UARCH_INTEL_MODERN; // Diamond Rapids
            }
        }
    }
    // AMD: ebx="Auth", edx="enti", ecx="cAMD"
    else if (ebx == 0x68747541 && edx == 0x69746e65 && ecx == 0x444d4163) {
        if (family == 0x17) {
            return UARCH_AMD_ZEN_1_2; // Zen 1 (Naples, Summit Ridge), Zen+ (Pinnacle), Zen 2 (Rome, Matisse)
        } 
        else if (family == 0x19) {
            if (model >= 0x10 && model <= 0x1F) return UARCH_AMD_ZEN_4; // Genoa
            if (model >= 0x60 && model <= 0x7F) return UARCH_AMD_ZEN_4; // Phoenix, Dragon Range
            if (model >= 0xA0 && model <= 0xAF) return UARCH_AMD_ZEN_4; // Bergamo
            
            return UARCH_AMD_ZEN_3; // Milan, Vermeer... (Default for family 19h not Zen 4)
        }
        // TODO: add family 0x1A pour Zen 5 (Turin, Granite Ridge)
    }
#elif ARCH_IS_ARM
    return UARCH_ARM; // need sysfs for more info
#endif
    return UARCH_UNKNOWN;
}

static inline cpu_uarch_t get_current_uarch(void) {
    if (!is_uarch_initialized) {
        current_uarch = detect_hardware_architecture();
        is_uarch_initialized = 1;
    }
    return current_uarch;
}

// Skylake arch

static const char *skl_tma_l1_events[] = {
    "@skl_slots",      // 0x003c (Cycles)
    "@skl_fe_bound",   // 0x019c
    "@skl_issued",     // 0x010e
    "@skl_retiring",   // 0x02c2
    "@skl_recovery"    // 0x0100019d
};

static void compute_skl_tma_l1(const double *raw_values, double *final_results) {
    double cycles = raw_values[0];
    double fe_bound_raw = raw_values[1];
    double issued_raw = raw_values[2];
    double retiring_raw = raw_values[3];
    double bad_spec_raw = raw_values[4];

    // Pipeline width = 4 on Intel Skylake
    double slots = 4.0 * cycles;

    if (slots > 0) {
        double r_fe_bound = fe_bound_raw / slots;
        double r_retiring = retiring_raw / slots;
        double r_bad_spec = bad_spec_raw / slots;
        double r_be_bound = (slots - retiring_raw - fe_bound_raw - bad_spec_raw) / slots;

        final_results[0] = r_retiring * 100.0;
        final_results[1] = r_bad_spec * 100.0;
        final_results[2] = r_fe_bound * 100.0;
        final_results[3] = r_be_bound * 100.0;
    } else {
        final_results[0] = final_results[1] = final_results[2] = final_results[3] = 0.0;
    }
}

static const int skl_l2_passes[] = {4, 4, 4}; // 3 passes of 4 counters
static const char *skl_tma_l2_events[] = {
    // Pass 1 : L1 Base + Backend
    "@skl_slots",
    "cycle_activity:stalls_mem_any",  // Memory Bound
    "resource_stalls:any",            // Core Bound base
    "@skl_fe_bound",

    // Pass 2 : Frontend
    "@skl_slots",
    "icache_16b:ifdata_stall",        // Fetch Latency
    "icache_64b:iftag_stall",         // Fetch Latency
    "@skl_retiring",

    // Pass 3 : Retiring + Bad Speculation
    "@skl_slots",
    "idq:ms_uops",                    // Heavy Ops
    "br_misp_retired:all_branches",   // Branch Mispredicts
    "machine_clears:count"            // Machine Clears
};


static void compute_skl_tma_l2(const double *raw, double *final) {
    // Pass 1 : 0 (slots), 1 (mem), 2 (core), 3 (fe)
    // Pass 2 : 4 (slots), 5 (lat_d), 6 (lat_t), 7 (ret)
    // Pass 3 : 8 (slots), 9 (heavy), 10 (misp), 11 (clears)

    // Use the slot of each group to not mess up ratio
    double slots_p1 = raw[0] * 4.0;
    double slots_p2 = raw[4] * 4.0;
    double slots_p3 = raw[8] * 4.0;

    if (slots_p1 > 0 && slots_p2 > 0 && slots_p3 > 0) {
        // Frontend
        double fe_bound = raw[3] / slots_p1;
        double fetch_lat = (raw[5] + raw[6]) / slots_p2;
        double fetch_bw = fe_bound - fetch_lat;

        // Retiring
        double retiring = raw[7] / slots_p2;
        double heavy_ops = raw[9] / slots_p3;
        double light_ops = retiring - heavy_ops;

        // Bad Speculation
        double br_misp = raw[10] / slots_p3;
        double m_clears = raw[11] / slots_p3;
        double bad_spec = br_misp + m_clears;

        // Backend
        double mem_bound = raw[1] / slots_p1;
        double core_bound = raw[2] / slots_p1;

        // Global backend
        double be_bound = 1.0 - (fe_bound + bad_spec + retiring);
        double total_be_raw = mem_bound + core_bound;
        if (total_be_raw > 0) {
            mem_bound = be_bound * (mem_bound / total_be_raw);
            core_bound = be_bound * (core_bound / total_be_raw);
        } else {
            mem_bound = be_bound; core_bound = 0;
        }

        final[0] = light_ops * 100.0;
        final[1] = heavy_ops * 100.0;
        final[2] = m_clears * 100.0;
        final[3] = br_misp * 100.0;
        final[4] = fetch_bw * 100.0;
        final[5] = fetch_lat * 100.0;
        final[6] = core_bound * 100.0;
        final[7] = mem_bound * 100.0;

        for(int i=0; i<8; i++) if (final[i] < 0.0) final[i] = 0.0;
    } else {
        for(int i=0; i<8; i++) final[i] = 0.0;
    }
}

static const int skl_l3_mem_passes[] = {4, 3};

static const char *skl_tma_l3_mem_events[] = {
    // Pass 1
    "@skl_slots",             //
    "@skl_stalls_mem_any",    // 0x140014a3
    "@skl_stalls_l1d_miss",   // 0x0c000c14 (cmask=0x0c, umask=0x0c, event=0x14)
    "@skl_stalls_l2_miss",    // 0x05000514 (cmask=0x05, umask=0x05, event=0x14)

    // Pass 2
    "@skl_slots",             //
    "@skl_stalls_l3_miss",    // 0x06000614 (cmask=0x06, umask=0x06, event=0x14)
    "@skl_bound_on_stores"    // 0x08a2 (RESOURCE_STALLS.SB)
};

// Compute only memory bound from L3
static void compute_skl_tma_l3_mem(const double *raw, double *final) {
    double cycles_p1 = raw[0];
    double cycles_p2 = raw[4];

    if (cycles_p1 > 0 && cycles_p2 > 0) {
        double mem_any  = raw[1] / cycles_p1;
        double l1d_miss = raw[2] / cycles_p1;
        double l2_miss  = raw[3] / cycles_p1;

        double l3_miss  = raw[5] / cycles_p2;
        double stores   = raw[6] / cycles_p2;

        final[0] = (mem_any - l1d_miss) * 100.0; // L1 Bound
        final[1] = (l1d_miss - l2_miss) * 100.0; // L2 Bound
        final[2] = (l2_miss - l3_miss) * 100.0;  // L3 Bound
        final[3] = l3_miss * 100.0;              // External RAM (DRAM) Bound
        final[4] = stores * 100.0;               // Store Bound

        for (int i=0; i<5; i++) if (final[i] < 0.0) final[i] = 0.0;
    } else {
        for (int i=0; i<5; i++) final[i] = 0.0;
    }
}

static const int skl_l3_passes[] = {5, 5, 5, 5, 5, 5, 5, 5};

static const char *skl_tma_l3_events[] = {
    // Pass 1: Memory Bounds
    "@skl_slots",            // raw[0]
    "@skl_stalls_mem_any",
    "@skl_stalls_l1d_miss",
    "@skl_stalls_l2_miss",
    "@skl_stalls_l3_miss",

    // Pass 2: Execution Stalls
    "@skl_slots",           // raw[5]
    "@skl_divider_active",
    "@skl_scoreboard",
    "@skl_bound_on_stores",
    "@skl_core_stalls",

    // Pass 3: Frontend Latency
    "@skl_slots",            // raw[10]
    "@skl_icache_miss",
    "@skl_itlb_miss",
    "@skl_clear_resteer",
    "@skl_lcp",

    // Pass 4: Frontend Bandwidth
    "@skl_slots",            // raw[15]
    "@skl_dsb2mite",
    "@skl_ms_switches",
    "@skl_idq_mite",
    "@skl_idq_dsb",

    // Pass 5: Instruction Mix
    "@skl_slots",            // raw[20]
    "@skl_idq_ms",
    "@skl_macro_fused",
    "@skl_mem_inst",
    "@skl_br_inst",

    // Pass 6: Floating Point 1 (Scalar & 128b)
    "@skl_slots",            // raw[25]
    "@skl_fp_scalar_s",
    "@skl_fp_scalar_d",
    "@skl_fp_128_s",
    "@skl_fp_128_d",

    // Pass 7: Floating Point 2 (256b & 512b)
    "@skl_slots",           // raw[30]
    "@skl_fp_256_s",
    "@skl_fp_256_d",
    "@skl_fp_512_s",
    "@skl_fp_512_d",

    // Pass 8: Parent Nodes (Other)
    "@skl_slots",          // raw[35]
    "@skl_retiring",       // raw[36] (uops_retired.retire_slots)
    "@skl_issued",         // raw[37] (uops_issued.any)
    "@skl_br_misp",        // raw[38] (br_misp_retired.all_branches)
    "@skl_nukes_mem"       // raw[39] (machine_clears.memory_ordering)
};

static void compute_skl_tma_l3(const double *raw, double *final) {
    double cyc_p1 = raw[0];  double cyc_p2 = raw[5];  double cyc_p3 = raw[10];
    double cyc_p4 = raw[15]; double cyc_p5 = raw[20]; double cyc_p6 = raw[25];
    double cyc_p7 = raw[30]; double cyc_p8 = raw[35];

    assert(cyc_p1 != cyc_p2 != cyc_p3 != cyc_p4 != cyc_p5 != cyc_p6 != cyc_p7 != cyc_p8 != 0);
    if (cyc_p1 > 0 && cyc_p2 > 0 && cyc_p3 > 0 && cyc_p4 > 0 && cyc_p5 > 0 && cyc_p6 > 0 && cyc_p7 > 0 && cyc_p8 > 0) {

        // Memory Subsystem
        double l1_bound   = (raw[1] - raw[2]) / cyc_p1;
        double l2_bound   = (raw[2] - raw[3]) / cyc_p1;
        double l3_bound   = (raw[3] - raw[4]) / cyc_p1;
        double dram_bound = raw[4] / cyc_p1;
        double store_bnd  = raw[8] / cyc_p2;

        // Execution / ALU
        double divider    = raw[6] / cyc_p2;
        double serial     = raw[7] / cyc_p2;
        double ports_util = (raw[9] - raw[6] - raw[7]) / cyc_p2; // core_stalls - divider - serial

        // Frontend Fetch
        double icache     = raw[11] / cyc_p3;
        double itlb       = raw[12] / cyc_p3;
        double lcp        = raw[14] / cyc_p3;
        double dsb2mite   = raw[16] / cyc_p4;
        double ms_swit    = raw[17] / cyc_p4;
        double dsb        = raw[19] / (cyc_p4 * 4.0); // IDQ.DSB is in uops, divise by slots
        double mite       = raw[18] / (cyc_p4 * 4.0); // IDQ.MITE is in uops

        // Retiring & Instruction Mix
        double ms_uops    = raw[21] / (cyc_p5 * 4.0);
        double fused      = raw[22] / (cyc_p5 * 4.0);
        double mem_ops    = raw[23] / (cyc_p5 * 4.0);
        double non_fused  = (raw[24] - raw[22]) / (cyc_p5 * 4.0); // br_inst - fused

        // Floating Point Arithmetic
        double fp_arith   = (raw[26] + raw[27] + raw[28] + raw[29]) / (cyc_p6 * 4.0) +
                            (raw[31] + raw[32] + raw[33] + raw[34]) / (cyc_p7 * 4.0);

        // Bad Speculation(L3)
        double resteers   = raw[13] / cyc_p3;

        // Others
        double retiring   = raw[36];
        double issued     = raw[37];
        double br_misp    = raw[38];
        double nukes_mem  = raw[39];

        // Few Uops Instructions
        // (Retiring - Fused) / Slots
        double few_uops = (retiring - raw[22]) / (cyc_p8 * 4.0);
        if (few_uops < 0.0) few_uops = 0.0;

        // Other Light Ops
        // Retiring Light - (FP_Arith + Mem_Ops + Fused + Non_Fused)
        double ret_light = (retiring - raw[21]) / (cyc_p8 * 4.0); // Retiring - ms_uops
        double other_light = ret_light - (fp_arith + mem_ops + fused + non_fused);
        if (other_light < 0.0) other_light = 0.0;

        // Other Mispredicts
        // Total Bad Speculation - Branch Mispredicts - Machine Clears
        double bad_spec_tot = (issued - retiring + (4.0 * raw[13])) / (cyc_p8 * 4.0); // raw[13] = recovery_cycles
        double r_br_misp = br_misp / (cyc_p8 * 4.0);
        double m_clears = raw[11] / (cyc_p3 * 4.0); // raw[11] (clear_resteer)
        double other_misp = bad_spec_tot - r_br_misp - m_clears;
        if (other_misp < 0.0) other_misp = 0.0;

        // Other Nukes
        // Total Machine Clears - Memory Ordering Clears
        double other_nukes = m_clears - (nukes_mem / (cyc_p8 * 4.0));
        if (other_nukes < 0.0) other_nukes = 0.0;

        // Order :
        // 0:resteers, 1:divider, 2:dram, 3:dsb, 4:dsb_switches, 5:few_uops, 6:fp_arith, 7:fused
        // 8:icache, 9:itlb, 10:l1, 11:l2, 12:l3, 13:lcp, 14:mem_ops, 15:ms_uops, 16:mite
        // 17:ms_swit, 18:non_fused_br, 19:other_light, 20:oth_misp, 21:nukes, 22:pmm, 23:ports, 24:serial, 25:store

        final[0] = resteers * 100.0;
        final[1] = divider * 100.0;
        final[2] = dram_bound * 100.0;
        final[3] = dsb * 100.0;
        final[4] = dsb2mite * 100.0;
        final[5] = few_uops * 100.0;
        final[6] = fp_arith * 100.0;
        final[7] = fused * 100.0;
        final[8] = icache * 100.0;
        final[9] = itlb * 100.0;
        final[10] = l1_bound * 100.0;
        final[11] = l2_bound * 100.0;
        final[12] = l3_bound * 100.0;
        final[13] = lcp * 100.0;
        final[14] = mem_ops * 100.0;
        final[15] = ms_uops * 100.0;
        final[16] = mite * 100.0;
        final[17] = ms_swit * 100.0;
        final[18] = non_fused * 100.0;
        final[19] = other_light * 100.0;
        final[20] = other_misp * 100.0;
        final[21] = other_nukes * 100.0;
        final[22] = 0.0; // pmm (Intel Optane DC, discontuated tech)
        final[23] = ports_util * 100.0;
        final[24] = serial * 100.0;
        final[25] = store_bnd * 100.0;

        for (int i=0; i<26; i++) if (final[i] < 0.0) final[i] = 0.0;
    } else {
        for (int i=0; i<26; i++) final[i] = 0.0;
    }
}

//  Modern Inter arch

static const char *intel_modern_tma_l1_events[] = {
    "@icl_slots",      // 0x0400 (Group Leader)
    "@icl_retiring",   // 0x8000
    "@icl_bad_spec",   // 0x8100
    "@icl_fe_bound",   // 0x8200
    "@icl_be_bound"    // 0x8300
};

static void compute_intel_modern_tma_l1(const double *raw_values, double *final_results) {
    double slots    = raw_values[0];
    double retiring = raw_values[1];
    double bad_spec = raw_values[2];
    double fe_bound = raw_values[3];
    double be_bound = raw_values[4];

    if (slots > 0) {
        final_results[0] = (retiring / slots) * 100.0;
        final_results[1] = (bad_spec / slots) * 100.0;
        final_results[2] = (fe_bound / slots) * 100.0;
        final_results[3] = (be_bound / slots) * 100.0;
    } else {
        final_results[0] = final_results[1] = final_results[2] = final_results[3] = 0.0;
    }
}

static const char *intel_modern_tma_l2_events[] = {
    "@icl_slots",
    "@icl_retiring",
    "@icl_heavy_ops",
    "@icl_bad_spec",
    "@icl_br_mispredict",
    "@icl_fe_bound",
    "@icl_fetch_lat",
    "@icl_be_bound",
    "@icl_mem_bound"
};

static void compute_intel_modern_tma_l2(const double *raw, double *final) {
    double slots = raw[0];

    if (slots > 0) {
      double retiring = raw[1];
      double heavy_ops = raw[2];
      double bad_spec = raw[3];
      double br_mispredict = raw[4];
      double fe_bound = raw[5];
      double fetch_lat = raw[6];
      double be_bound = raw[7];
      double mem_bound = raw[8];

      // Retiring
      final[0] = ((retiring - heavy_ops) / slots) * 100.0; // Light Ops
      final[1] = (heavy_ops / slots) * 100.0;              // Heavy Ops

      // Bad Speculation
      final[2] = ((bad_spec - br_mispredict) / slots) * 100.0; // Machine Clears
      final[3] = (br_mispredict / slots) * 100.0; // Branch Mispredicts

      // Frontend Bound
      final[4] = ((fe_bound - fetch_lat) / slots) * 100.0; // Fetch Bandwidth
      final[5] = (fetch_lat / slots) * 100.0;              // Fetch Latency

      // Backend Bound
      final[6] = ((be_bound - mem_bound) / slots) * 100.0; // Core Bound
      final[7] = (mem_bound / slots) * 100.0;              // Memory Bound

      for (int i = 0; i < 8; i++) {
        if (final[i] < 0.0)
          final[i] = 0.0;
      }
    } else {
        for (int i = 0; i < 8; i++) final[i] = 0.0;
    }
}

static const int icl_l3_mem_passes[] = {4, 3};

static const char *intel_modern_tma_l3_mem_events[] = {
    // Pass 1
    "@icl_cyc",
    "@icl_stalls_mem_any",
    "@icl_stalls_l1d_miss",
    "@icl_stalls_l2_miss",

    // Pass 2
    "@icl_cyc",
    "@icl_stalls_l3_miss",
    "@icl_bound_on_stores"
};

// intel_modern_tma_l3_mem use the same compute function as skl

static const int icl_l3_passes[] = {5, 5, 5, 5, 5, 5, 5, 5};

static const char *intel_modern_tma_l3_events[] = {
    // Pass 1: Memory Bounds
    "@icl_cyc", "@icl_stalls_mem_any", "@icl_stalls_l1d_miss", "@icl_stalls_l2_miss", "@icl_stalls_l3_miss",
    // Pass 2: Execution Stalls
    "@icl_cyc", "@icl_divider_active", "@icl_core_stalls", "@icl_bound_on_stores", "@icl_scoreboard",
    // Pass 3: Frontend Latency
    "@icl_cyc", "@icl_icache_miss", "@icl_itlb_miss", "@icl_clear_resteer", "@icl_lcp",
    // Pass 4: Frontend Bandwidth
    "@icl_cyc", "@icl_dsb2mite", "@icl_ms_switches", "@icl_idq_mite", "@icl_idq_dsb",
    // Pass 5: Instruction Mix
    "@icl_cyc", "@icl_idq_ms", "@icl_macro_fused", "@icl_mem_inst", "@icl_br_inst",
    // Pass 6: Floating Point 1
    "@icl_cyc", "@icl_fp_scalar_s", "@icl_fp_scalar_d", "@icl_fp_128_s", "@icl_fp_128_d",
    // Pass 7: Floating Point 2
    "@icl_cyc", "@icl_fp_256_s", "@icl_fp_256_d", "@icl_fp_512_s", "@icl_fp_512_d",
    // Pass 8: Parent Nodes (Other)
    "@icl_cyc", "@icl_retiring_uops", "@icl_issued_any", "@icl_br_misp", "@icl_nukes_mem"
};

// intel_modern_tma_l3 use the same compute function as skl

// Zen arch

static const char *amd_zen4_tma_l1_events[] = {
    "@zen4_cyc",   // 0x0076 (Cycles)
    "@zen4_fe",    // 0x1000001A0 (Dispatch slots empty because frontend is stalled)
    "@zen4_disp",  // 0x07AA (Ops dispatched from any source)
    "@zen4_ret"    // 0x00C1 (Ops retired)
};

static inline void compute_amd_tma_l1_generic(const double *raw_values, double *final_results, double pipeline_width) {
    double cycles   = raw_values[0];
    double fe_raw   = raw_values[1];
    double disp_raw = raw_values[2];
    double ret_raw  = raw_values[3];

    double slots = pipeline_width * cycles;

    if (slots > 0) {
        double r_fe_bound = fe_raw / slots;

        double r_bad_spec = (disp_raw - ret_raw) / slots;
        //assert(r_bad_spec >= 0.0);
        if (r_bad_spec < 0.0) r_bad_spec = 0.0;
        double r_retiring = ret_raw / slots;

        double r_be_bound = 1.0 - (r_fe_bound + r_bad_spec + r_retiring);
        //assert(r_be_bound >= 0.0);
        if (r_be_bound < 0.0) r_be_bound = 0.0;

        final_results[0] = r_retiring * 100.0;
        final_results[1] = r_bad_spec * 100.0;
        final_results[2] = r_fe_bound * 100.0;
        final_results[3] = r_be_bound * 100.0;
    } else {
        final_results[0] = final_results[1] = final_results[2] = final_results[3] = 0.0;
    }
}

static void compute_amd_zen34_tma_l1(const double *raw, double *final) {
    compute_amd_tma_l1_generic(raw, final, 6.0); // Zen 3 / Zen 4
}

static void compute_amd_zen1_tma_l1(const double *raw, double *final) {
    compute_amd_tma_l1_generic(raw, final, 4.0); // Zen 1 / Zen 2
}

static const int amd_zen4_l2_passes[] = {5, 4};

static const char *amd_zen4_tma_l2_events[] = {
    "@zen4_cyc",
    "@zen4_be_mem",     // mem stall
    "@zen4_be_cpu",     // cpu stall
    "@zen4_fe_lat",     // fetch latency
    "@zen4_fe_tot",     // total frontend (no cmask)

    "@zen4_cyc",
    "@zen4_bs_misp",    // branch mispredict
    "@zen4_bs_resync",  // pipeline restart (machine clear)
    "@zen4_ret_micro"   // Heavy ops
};

static void compute_amd_zen4_tma_l2(const double *raw, double *final) {
    double slots_p1 = raw[0] * 6.0; // pipeline width
    double slots_p2 = raw[4] * 6.0;

    if (slots_p1 > 0 && slots_p2 > 0) {
        // Backend Bound
        double be_mem = raw[1] / slots_p1;
        double be_cpu = raw[2] / slots_p1;

        // Frontend Bound (Bandwitdth = total - latency)
        double fe_lat = raw[3] / slots_p1;
        double fe_tot = raw[4] / slots_p1;
        double fe_bw  = fe_tot - fe_lat;
        if (fe_bw < 0.0) fe_bw = 0.0;

        // Bad Speculation
        double bs_misp = raw[6] / slots_p2;
        double bs_clear = raw[7] / slots_p2;

        // Retiring
        double ret_heavy = raw[8] / slots_p2;

        double total_retiring = 1.0 - (be_mem + be_cpu + fe_tot + bs_misp + bs_clear);
        double ret_light = total_retiring - ret_heavy;

        // L2 : Light, Heavy, Clears, Mispredicts, FE Bandwidth, FE Latency, Core Bound, Memory Bound
        final[0] = ret_light * 100.0;
        final[1] = ret_heavy * 100.0;
        final[2] = bs_clear * 100.0;
        final[3] = bs_misp * 100.0;
        final[4] = fe_bw * 100.0;
        final[5] = fe_lat * 100.0;
        final[6] = be_cpu * 100.0;
        final[7] = be_mem * 100.0;

        for(int i=0; i<8; i++) if (final[i] < 0.0) final[i] = 0.0;
    } else {
        for(int i=0; i<8; i++) final[i] = 0.0;
    }
}


// ARM arch

static const int arm_l1_passes[] = {5};
static const char *arm_tma_l1_events[] = {
    "@arm_cyc", "@arm_fe", "@arm_be", "@arm_inst", "@arm_brmisp"
};

static void compute_arm_tma_l1(const double *raw, double *final) {
    double cyc = raw[0];
    if (cyc > 0) {
        double fe_bound = raw[1] / cyc;
        double be_bound = raw[2] / cyc;

        // Cycle where at leat 1 instruction has been issued
        double remaining = 1.0 - fe_bound - be_bound;
        if (remaining < 0.0) remaining = 0.0;

        // Estimation : 10 cycles lost per branch miss
        // TODO : check doc
        // Cortex-A76 = 14 cycles (https://www.7-cpu.com/cpu/Cortex-A76.html)
        // Cortex-A53 = 7 cycles (https://www.7-cpu.com/cpu/Cortex-A53.html)
        // HPE RL300 ?
        double bad_spec = (raw[4] * 10.0) / cyc;
        if (bad_spec > remaining) bad_spec = remaining;

        double retiring = remaining - bad_spec;

        final[0] = retiring * 100.0;
        final[1] = bad_spec * 100.0;
        final[2] = fe_bound * 100.0;
        final[3] = be_bound * 100.0;
    } else {
        final[0] = final[1] = final[2] = final[3] = 0.0;
    }
}

static const int arm_l2_passes[] = {5, 3};
static const char *arm_tma_l2_events[] = {
    // Pass 1 : Backend (Memory vs CPU)
    "@arm_cyc",
    "@arm_be",
    "@arm_be_mem",
    "@arm_be_cpu",
    "@arm_fe",

    // Pass 2 : Bad Speculation + Frontend latency
    "@arm_cyc",
    "@arm_l1i_miss",
    "@arm_brmisp"
};

static void compute_arm_tma_l2(const double *raw, double *final) {
    double cyc1 = raw[0];
    double cyc2 = raw[5];

    if (cyc1 > 0 && cyc2 > 0) {
        // Backend Bound
        double be_tot = raw[1];
        double be_mem = raw[2];
        double be_cpu = raw[3];

        double r_be_mem = 0.0, r_be_cpu = 0.0;
        double r_be_tot = be_tot / cyc1;

        // Backend normalisation
        if (be_mem + be_cpu > 0) {
            r_be_mem = r_be_tot * (be_mem / (be_mem + be_cpu));
            r_be_cpu = r_be_tot * (be_cpu / (be_mem + be_cpu));
        } else {
            r_be_mem = r_be_tot;
        }

        // Frontend Bound
        double fe_tot = raw[4] / cyc1;
        // Estimation : 15 cycles to get instruction from L2/RAM
        // TODO : check doc
        double fe_lat = (raw[6] * 15.0) / cyc2;
        if (fe_lat > fe_tot) fe_lat = fe_tot;
        double fe_bw = fe_tot - fe_lat;

        // Bad Speculation + Retiring
        double remaining = 1.0 - fe_tot - r_be_tot;
        if (remaining < 0.0) remaining = 0.0;

        double bad_spec = (raw[7] * 10.0) / cyc2;
        if (bad_spec > remaining) bad_spec = remaining;

        double retiring = remaining - bad_spec;

        //  Light, Heavy, Clears, Mispredicts, FE BW, FE Lat, Core Bound, Memory Bound
        final[0] = retiring * 100.0; // Every instruction is Light
        final[1] = 0.0;              // No heavy ops
        final[2] = 0.0;              // Clear
        final[3] = bad_spec * 100.0; // Mispredicts
        final[4] = fe_bw * 100.0;    // Frontend Bandwidth
        final[5] = fe_lat * 100.0;   // Frontend Latency
        final[6] = r_be_cpu * 100.0; // Core bound
        final[7] = r_be_mem * 100.0; // Memory bound

        for(int i=0; i<8; i++) if (final[i] < 0.0) final[i] = 0.0;
    } else {
        for(int i=0; i<8; i++) final[i] = 0.0;
    }
}

#endif //__linux__


// Resolver logic

typedef struct {
    const char *metric_name;
    cpu_uarch_t uarch;
    
    int num_results;
    int num_passes;
    int num_hw_events;
    const int *events_per_pass;
    const char **hw_events;
    void (*compute_formula)(const double *raw_values, double *final_results);
} metric_registry_entry_t;

static const int pass_1[] = {1};
static const int pass_4[] = {4};
static const int pass_5[] = {5};
static const int pass_9[] = {9};

static const metric_registry_entry_t METRICS_REGISTRY[] = {
    // Topdown L1
    {"TopdownL1", UARCH_INTEL_OLD,              4, 1, 5, pass_5, skl_tma_l1_events,          compute_skl_tma_l1},
    {"TopdownL1", UARCH_INTEL_MODERN,           4, 1, 5, pass_5, intel_modern_tma_l1_events, compute_intel_modern_tma_l1},
    {"TopdownL1", UARCH_AMD_ZEN_4,              4, 1, 4, pass_4, amd_zen4_tma_l1_events,     compute_amd_zen34_tma_l1},
    {"TopdownL1", UARCH_ARM,                    4, 1, 5, arm_l1_passes, arm_tma_l1_events,   compute_arm_tma_l1},

    // Topdown L2
    {"TopdownL2", UARCH_INTEL_OLD,              8, 3, 12, skl_l2_passes, skl_tma_l2_events,          compute_skl_tma_l2},
    {"TopdownL2", UARCH_INTEL_MODERN,           8, 1,  9, pass_9,        intel_modern_tma_l2_events, compute_intel_modern_tma_l2},
    {"TopdownL2", UARCH_AMD_ZEN_4,              8, 2,  9, amd_zen4_l2_passes, amd_zen4_tma_l2_events, compute_amd_zen4_tma_l2},

    // Topdown L3
    {"TopdownL3", UARCH_INTEL_OLD,              26, 8, 40, skl_l3_passes, skl_tma_l3_events,          compute_skl_tma_l3},
    {"TopdownL3", UARCH_INTEL_MODERN,           26, 8, 40, icl_l3_passes, intel_modern_tma_l3_events, compute_skl_tma_l3},

    // Topdown L3 Mem
    {"TopdownL3_Mem", UARCH_INTEL_OLD,          5, 2, 7, skl_l3_mem_passes, skl_tma_l3_mem_events,          compute_skl_tma_l3_mem},
    {"TopdownL3_Mem", UARCH_INTEL_MODERN,       5, 2, 7, icl_l3_mem_passes, intel_modern_tma_l3_mem_events, compute_skl_tma_l3_mem},
};
static const int METRICS_REGISTRY_SIZE = sizeof(METRICS_REGISTRY) / sizeof(METRICS_REGISTRY[0]);



/*
 * Tested TopdownL1 :
 *  - INTEL_SKYLAKE_CASCADE (skylake, kaby-lake)L1,L2
 *  - INTEL_ICELAKE_SAPPHIRE (raptor lake)      L1,L2
 *  - AMD_ZEN_4    (Zen 4)                      L1,L2
 *
 *  Untested :
 *  - AMD_ZEN_1_2
 *
 *  Not implemented :
 *  - Aarch64
 *  - Zen 3
 *  - L3 support
 *
 *
 */
int resolve_metric(const char *metric_name, metric_resolver_t *out_resolver) {
#ifndef __linux__
    return 0;
#else
    cpu_uarch_t current_cpu = get_current_uarch();

    for (int i = 0; i < METRICS_REGISTRY_SIZE; i++) {
        const metric_registry_entry_t *entry = &METRICS_REGISTRY[i];

        if (entry->uarch == current_cpu && strcmp(entry->metric_name, metric_name) == 0) {
            
            out_resolver->is_supported = 1;
            out_resolver->num_results = entry->num_results;
            out_resolver->num_passes = entry->num_passes;
            out_resolver->num_hw_events = entry->num_hw_events;
            out_resolver->events_per_pass = entry->events_per_pass;
            out_resolver->hw_events = entry->hw_events;
            out_resolver->compute_formula = entry->compute_formula;
            
            return 1;
        }
    }

    out_resolver->is_supported = 0;
    return 0;
#endif
}

int get_perf_metric_results_count(const char *metric_name) {
    metric_resolver_t resolver;
    if (resolve_metric(metric_name, &resolver) && resolver.is_supported) {
        return resolver.num_results;
    }
    return 1;
}
