#include <assert.h>
#include <string.h>
#include <linux/perf_event.h>
#include <unistd.h>
#include <stdint.h>
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

/*
 * Tested TopdownL1 :
 *  - INTEL_SKYLAKE_CASCADE (skylake)
 *  - INTEL_ICELAKE_SAPPHIRE (raptor lake)
 *
 *  Untested :
 *  - AMD_ZEN_1_2
 *  - AMD_ZEN_4
 *
 *  Not implemented :
 *  - Aarch64
 *  - Zen 3
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
                out_resolver->compute_formula = compute_skl_tma_l1;
                return 1;
            } else if (uarch == INTEL_ICELAKE_SAPPHIRE) {
                fprintf(stderr,"[DEBUG] Modern Intel detected\n");
                out_resolver->is_supported = 1;
                out_resolver->num_hw_events = 5;
                out_resolver->hw_events = intel_modern_tma_l1_events;
                out_resolver->num_results = 4;
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
                out_resolver->compute_formula = compute_amd_zen34_tma_l1;
                return 1;
            }
            else if (uarch == AMD_ZEN_1_2) {
                fprintf(stderr,"[DEBUG] AMD_ZEN_1_2 detected\n");
                out_resolver->is_supported = 1;
                out_resolver->num_hw_events = 4;
                out_resolver->hw_events = amd_zen4_tma_l1_events;
                out_resolver->num_results = 4;
                out_resolver->compute_formula = compute_amd_zen34_tma_l1; //compute_amd_zen1_tma_l1;
                return 1;
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
        if (detect_if_intel()) {
            intel_arch_t uarch = detect_intel_microarchitecture();

            if (uarch == INTEL_SKYLAKE_CASCADE) {
                fprintf(stderr,"[DEBUG] Unsuported INTEL_SKYLAKE_CASCADE for L2\n");
                // todo : need multiplexing
                return 0;
            }
            else if (uarch == INTEL_ICELAKE_SAPPHIRE) {
                out_resolver->is_supported = 1;
                out_resolver->num_hw_events = 9;
                out_resolver->hw_events = intel_modern_tma_l2_events;
                out_resolver->num_results = 8;
                out_resolver->compute_formula = compute_intel_modern_tma_l2;
                return 1;
            }
        }
        return 0;
    }

    fprintf(stderr,"[DEBUG] Unsuported hardware\n");
    return 0; // Unsuported hardware
}


int get_perf_metric_results_count(const char *metric_name) {
    metric_resolver_t resolver;
    if (resolve_metric(metric_name, &resolver) && resolver.is_supported) {
        return resolver.num_results;
    }
    return 1;
}
