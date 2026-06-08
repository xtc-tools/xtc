#include <assert.h>
#include <string.h>
#include <linux/perf_event.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <x86intrin.h>

#include "perf_metrics.h"


#if defined(__x86_64__) || defined(__i386__)
    #define ARCH_IS_X86 1
    #include <cpuid.h>
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
    INTEL_UNKNOWN = 0,
    INTEL_SKYLAKE_CASCADE,
    INTEL_ICELAKE_SAPPHIRE
} intel_arch_t;

intel_arch_t detect_intel_microarchitecture(void) {
#if ARCH_IS_X86
    int family, model;
    get_cpu_family_model(&family, &model);

    if (family == 6) {
        switch (model) {
            // Arch without TMA counter
            case 0x4E: case 0x5E: case 0x55: // Skylake-X / Cascade Lake-X / Cooper Lake
            case 0x8E: case 0x9E:            // Whiskey Lake / Kaby Lake / Coffee Lake
            case 0x3D: case 0x47: case 0x4F: // Broadwell
            case 0x56:                       // Broadwell-DE
                return INTEL_SKYLAKE_CASCADE;

            // Arch with TMA counter
            case 0x7E: case 0x7D: case 0x9D: // Ice Lake
            case 0x6A: case 0x6C: case 0x8C: // Ice Lake SP/D
            case 0x8F:                       // Sapphire Rapids
            case 0x97: case 0x9A:            // Alder Lake
            case 0xB7: case 0xBA: case 0xBF: // Raptor Lake
            case 0xAA: case 0xAC:            // Meteor Lake
                return INTEL_ICELAKE_SAPPHIRE;
        }
    }
#endif
    return INTEL_UNKNOWN;
}

typedef enum {
    AMD_UNKNOWN = 0,
    AMD_ZEN_1_2,
    AMD_ZEN_3,
    AMD_ZEN_4
} amd_arch_t;

amd_arch_t detect_amd_microarchitecture(void) {
#if ARCH_IS_X86
    int family, model;
    get_cpu_family_model(&family, &model);

    if (family == 0x17) {
        // Zen 1 : Naples, Summit Ridge
        // Zen+ : Pinnacle Ridge
        // Zen 2 : Rome, Matisse, Naples
        return AMD_ZEN_1_2;
    }

    if (family == 0x19) {
        if (model >= 0x10 && model <= 0x1F) return AMD_ZEN_4; // Genoa
        if (model >= 0x60 && model <= 0x7F) return AMD_ZEN_4; // Phoenix/Dragon Range
        if (model >= 0xA0 && model <= 0xAF) return AMD_ZEN_4; // Bergamo

        // Todo : double check if missing models
        return AMD_ZEN_3; // (Milan, Vermeer...)
    }
    // todo family 0x1A for Zen 5 (Turin, Granite Ridge)
#endif
    return AMD_UNKNOWN;
}

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




int detect_if_intel(void) {
#if ARCH_IS_X86
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        // ebx = "Genu" (0x756e6547)
        // edx = "ineI" (0x49656e69)
        // ecx = "ntel" (0x6c65746e)
        if (ebx == 0x756e6547 && edx == 0x49656e69 && ecx == 0x6c65746e) {
            return 1;
        }
    }
#endif
    return 0;
}

int detect_if_amd(void) {
#if ARCH_IS_X86
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        // ebx = "Auth" (0x68747541)
        // edx = "enti" (0x69746e65)
        // ecx = "cAMD" (0x444d4163)
        if (ebx == 0x68747541 && edx == 0x69746e65 && ecx == 0x444d4163) {
            return 1;
        }
    }
#endif
    return 0;
}

int detect_if_arm(void) {
    return ARCH_IS_ARM;
}


static const int pass_1[] = {1};
static const int pass_4[] = {4};
static const int pass_5[] = {5};
static const int pass_9[] = {9};


// === Skylake arch ===


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
        // 1. Frontend
        double fe_bound = raw[3] / slots_p1;
        double fetch_lat = (raw[5] + raw[6]) / slots_p2;
        double fetch_bw = fe_bound - fetch_lat;

        // 2. Retiring
        double retiring = raw[7] / slots_p2;
        double heavy_ops = raw[9] / slots_p3;
        double light_ops = retiring - heavy_ops;

        // 3. Bad Speculation
        double br_misp = raw[10] / slots_p3;
        double m_clears = raw[11] / slots_p3;
        double bad_spec = br_misp + m_clears;

        // 4. Backend
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

static const int skl_l3_mem_passes[] = {4, 1};

static const char *skl_tma_l3_mem_events[] = {
    // Pass 1
    "@skl_slots",             //
    "@skl_stalls_mem_any",    // 0x14001414 (cmask=0x14, umask=0x14, event=0x14)
    "@skl_stalls_l1d_miss",   // 0x0c000c14 (cmask=0x0c, umask=0x0c, event=0x14)
    "@skl_stalls_l2_miss",    // 0x05000514 (cmask=0x05, umask=0x05, event=0x14)

    // Pass 2
    "@skl_slots",             //
    "@skl_stalls_l3_miss",    // 0x06000614 (cmask=0x06, umask=0x06, event=0x14)
    "@skl_bound_on_stores"    // 0x08a2 (RESOURCE_STALLS.SB)
};

// Compute only memory bound from L3
static void compute_skl_tma_l3_mem(const double *raw, double *final) {
    double slots_p1 = raw[0] * 4.0;
    double slots_p2 = raw[4] * 4.0;

    if (slots_p1 > 0 && slots_p2 > 0) {
        double mem_any  = raw[1] / slots_p1;
        double l1d_miss = raw[2] / slots_p1;
        double l2_miss  = raw[3] / slots_p1;

        double l3_miss  = raw[5] / slots_p2;
        double stores   = raw[6] / slots_p2;

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


//  === Modern Inter arch ===

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

// === Zen arch ===

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
        //  Backend Bound
        double be_mem = raw[1] / slots_p1;
        double be_cpu = raw[2] / slots_p1;

        // Frontend Bound (Bandwitdth = total - latency)
        double fe_lat = raw[3] / slots_p1;
        double fe_tot = raw[4] / slots_p1;
        double fe_bw  = fe_tot - fe_lat;
        if (fe_bw < 0.0) fe_bw = 0.0;

        // 3. Bad Speculation
        double bs_misp = raw[6] / slots_p2;
        double bs_clear = raw[7] / slots_p2;

        // 4. Retiring
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



// === Core logic ===


/* AMD metrics
 * backend_bound_memory = Memory bound
 * backend_bound_cpu = Core Bound
 * frontend_bound_latency = Fetch Latency
 * frontend_bound_bw = Bandwidth
 * bad_speculation_mispredicts = Branch Mispredicts
 */




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
    if (strcmp(metric_name, "TopdownL1") == 0) {

        if (detect_if_intel()) {
            fprintf(stderr,"[DEBUG] INTEL detected\n");
            intel_arch_t uarch = detect_intel_microarchitecture();

            if (uarch == INTEL_SKYLAKE_CASCADE) {
                fprintf(stderr,"[DEBUG] Old Intel detected\n");
                out_resolver->is_supported = 1;
                out_resolver->num_hw_events = 5;
                out_resolver->hw_events = skl_tma_l1_events;
                out_resolver->num_results = 4;
                out_resolver->num_passes = 1;
                out_resolver->events_per_pass = pass_5;
                out_resolver->compute_formula = compute_skl_tma_l1;
                return 1;
            } else if (uarch == INTEL_ICELAKE_SAPPHIRE) {
                fprintf(stderr,"[DEBUG] Modern Intel detected\n");
                out_resolver->is_supported = 1;
                out_resolver->num_hw_events = 5;
                out_resolver->hw_events = intel_modern_tma_l1_events;
                out_resolver->num_results = 4;
                out_resolver->num_passes = 1;
                out_resolver->events_per_pass = pass_5;
                out_resolver->compute_formula = compute_intel_modern_tma_l1;
                return 1;
            }
        }
        else if (detect_if_amd()) {
            amd_arch_t uarch = detect_amd_microarchitecture();

            if (uarch == AMD_ZEN_4) {
                fprintf(stderr,"[DEBUG] AMD_ZEN_4 detected\n");
                out_resolver->is_supported = 1;
                out_resolver->num_hw_events = 4;
                out_resolver->hw_events = amd_zen4_tma_l1_events;
                out_resolver->num_results = 4;
                out_resolver->num_passes = 1;
                out_resolver->events_per_pass = pass_4;
                out_resolver->compute_formula = compute_amd_zen34_tma_l1;
                return 1;
            }
            else if (uarch == AMD_ZEN_1_2) {
                fprintf(stderr,"[DEBUG] AMD_ZEN_1_2 detected (unsuported)\n");
                //out_resolver->is_supported = 1;
                //out_resolver->num_hw_events = 4;
                //out_resolver->hw_events = amd_zen4_tma_l1_events; // Todo change to zen1
                //out_resolver->num_results = 4;
                //out_resolver->num_passes = 1;
                //out_resolver->events_per_pass = pass_4;
                //out_resolver->compute_formula = compute_amd_zen34_tma_l1; // Todo change to zen1
                return 0;
            }
            // else if (uarch == AMD_ZEN_3) ...
        }
        else if (detect_if_arm()) {
            fprintf(stderr,"[DEBUG] ARM detected\n");
            // todo
            // fopen /sys/devices/system/cpu/cpu0/regs/identification/midr_el1
            // 0x410fd0c0 == Cortex-X1
            // 0x610f2200 == Apple M1
            return 0;
        }
    }
    else if (strcmp(metric_name, "TopdownL2") == 0) {
        if (detect_if_intel()) { // L2 intel
            intel_arch_t uarch = detect_intel_microarchitecture();

            if (uarch == INTEL_SKYLAKE_CASCADE) {
                fprintf(stderr,"[DEBUG] INTEL_SKYLAKE_CASCADE L2 (Multi-Pass)\n");
                out_resolver->is_supported = 1;
                out_resolver->num_hw_events = 12;
                out_resolver->hw_events = skl_tma_l2_events;
                out_resolver->num_results = 8;
                out_resolver->num_passes = 3;
                out_resolver->events_per_pass = skl_l2_passes; // Tableau {4, 4, 4}
                out_resolver->compute_formula = compute_skl_tma_l2;
                return 1;
            }
            else if (uarch == INTEL_ICELAKE_SAPPHIRE) {
                fprintf(stderr,"[DEBUG] INTEL_ICELAKE_SAPPHIRE L2\n");
                out_resolver->is_supported = 1;
                out_resolver->num_hw_events = 9;
                out_resolver->hw_events = intel_modern_tma_l2_events;
                out_resolver->num_results = 8;
                out_resolver->num_passes = 1;
                out_resolver->events_per_pass = pass_9;
                out_resolver->compute_formula = compute_intel_modern_tma_l2;
                return 1;
            }
        }

        else if (detect_if_amd()) { // L2 amd
            amd_arch_t uarch = detect_amd_microarchitecture();
            if (uarch == AMD_ZEN_4) {
                fprintf(stderr,"[DEBUG] AMD_ZEN_4 L2 (Multi-Pass)\n");
                out_resolver->is_supported = 1;
                out_resolver->num_hw_events = 9;
                out_resolver->hw_events = amd_zen4_tma_l2_events;
                out_resolver->num_results = 8;
                out_resolver->num_passes = 2;
                out_resolver->events_per_pass = amd_zen4_l2_passes;
                out_resolver->compute_formula = compute_amd_zen4_tma_l2;
                return 1;
            }
        }
        return 0;
    }
    else if (strcmp(metric_name, "TopdownL3_Mem") == 0) {
        if (detect_if_intel()) {
            intel_arch_t uarch = detect_intel_microarchitecture();
            if (uarch == INTEL_SKYLAKE_CASCADE) {
                fprintf(stderr,"[DEBUG] INTEL_SKYLAKE_CASCADE L3_Mem (Multi-Pass)\n");
                out_resolver->is_supported = 1;
                out_resolver->num_hw_events = 7;
                out_resolver->hw_events = skl_tma_l3_mem_events;
                out_resolver->num_results = 5;
                out_resolver->num_passes = 2;
                out_resolver->events_per_pass = skl_l3_mem_passes;
                out_resolver->compute_formula = compute_skl_tma_l3_mem;
                return 1;
            }
        }
        return 0;
        }

    // Unsuported hardware / metric or the event is a pmu
    return 0;
}

int get_perf_metric_results_count(const char *metric_name) {
    metric_resolver_t resolver;
    if (resolve_metric(metric_name, &resolver) && resolver.is_supported) {
        return resolver.num_results;
    }
    return 1;
}
