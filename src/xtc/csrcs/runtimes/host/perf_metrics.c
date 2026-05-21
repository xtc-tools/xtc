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

#include <time.h>
#include <stdio.h>
static void compute_test_tma_doc(){
    uint64_t slots_a = read_slots();
	uint64_t metric_a = read_metrics();

	{
    	srand(time(NULL));
    	for(int i = 0; i<100000;i++){
            asm("":::"memory");
            volatile int a = rand();
            asm("":::"memory");

    	}
	}

	uint64_t slots_b = read_slots();
	uint64_t metric_b = read_metrics();

	// compute scaled metrics for measurement a
	uint64_t retiring_slots_a = GET_METRIC(metric_a, 0) * slots_a;
	uint64_t bad_spec_slots_a = GET_METRIC(metric_a, 1) * slots_a;
	uint64_t fe_bound_slots_a = GET_METRIC(metric_a, 2) * slots_a;
	uint64_t be_bound_slots_a = GET_METRIC(metric_a, 3) * slots_a;

	// compute delta scaled metrics between b and a
	uint64_t retiring_slots = GET_METRIC(metric_b, 0) * slots_b - retiring_slots_a;
	uint64_t bad_spec_slots = GET_METRIC(metric_b, 1) * slots_b - bad_spec_slots_a;
	uint64_t fe_bound_slots = GET_METRIC(metric_b, 2) * slots_b - fe_bound_slots_a;
	uint64_t be_bound_slots = GET_METRIC(metric_b, 3) * slots_b - be_bound_slots_a;

	uint64_t slots_delta = slots_b - slots_a;
	float retiring_ratio = (float)retiring_slots / slots_delta;
	float bad_spec_ratio = (float)bad_spec_slots / slots_delta;
	float fe_bound_ratio = (float)fe_bound_slots / slots_delta;
	float be_bound_ratio = (float)be_bound_slots / slots_delta;

	printf("Retiring %.2f%% Bad Speculation %.2f%% FE Bound %.2f%% BE Bound %.2f%%\n",
		retiring_ratio * 100.,
		bad_spec_ratio * 100.,
		fe_bound_ratio * 100.,
		be_bound_ratio * 100.);

	/*
    The individual ratios of L2 metric events for the measurement period can be
    recreated from L1 and L2 metric counters. (Available on Sapphire Rapids and
    later platforms)
    */

	// compute scaled metrics for measurement a
	uint64_t heavy_ops_slots_a = GET_METRIC(metric_a, 4) * slots_a;
	uint64_t br_mispredict_slots_a = GET_METRIC(metric_a, 5) * slots_a;
	uint64_t fetch_lat_slots_a = GET_METRIC(metric_a, 6) * slots_a;
	uint64_t mem_bound_slots_a = GET_METRIC(metric_a, 7) * slots_a;

	// compute delta scaled metrics between b and a
	uint64_t heavy_ops_slots = GET_METRIC(metric_b, 4) * slots_b - heavy_ops_slots_a;
	uint64_t br_mispredict_slots = GET_METRIC(metric_b, 5) * slots_b - br_mispredict_slots_a;
	uint64_t fetch_lat_slots = GET_METRIC(metric_b, 6) * slots_b - fetch_lat_slots_a;
	uint64_t mem_bound_slots = GET_METRIC(metric_b, 7) * slots_b - mem_bound_slots_a;

	slots_delta = slots_b - slots_a;
	float heavy_ops_ratio = (float)heavy_ops_slots / slots_delta;
	float light_ops_ratio = retiring_ratio - heavy_ops_ratio;

	float br_mispredict_ratio = (float)br_mispredict_slots / slots_delta;
	float machine_clears_ratio = bad_spec_ratio - br_mispredict_ratio;

	float fetch_lat_ratio = (float)fetch_lat_slots / slots_delta;
	float fetch_bw_ratio = fe_bound_ratio - fetch_lat_ratio;

	float mem_bound_ratio = (float)mem_bound_slots / slots_delta;
	float core_bound_ratio = be_bound_ratio - mem_bound_ratio;

	printf("Heavy Operations %.2f%% Light Operations %.2f%% "
	       "Branch Mispredict %.2f%% Machine Clears %.2f%% "
	       "Fetch Latency %.2f%% Fetch Bandwidth %.2f%% "
	       "Mem Bound %.2f%% Core Bound %.2f%%\n",
		heavy_ops_ratio * 100.,
		light_ops_ratio * 100.,
		br_mispredict_ratio * 100.,
		machine_clears_ratio * 100.,
		fetch_lat_ratio * 100.,
		fetch_bw_ratio * 100.,
		mem_bound_ratio * 100.,
		core_bound_ratio * 100.);

}

#include "perf_topdown.h"
static void compute_test_tma_guil(){
    perf_topdown_init();
    perf_topdown_start();

	{
    	srand(time(NULL));
    	for(int i = 0; i<100000;i++){
            asm("":::"memory");
            volatile int a = rand();
            asm("":::"memory");

    	}
	}

    perf_topdown_stop();
    perf_topdown_dump(fdopen(1, "w"));
    perf_topdown_fini();
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

static const char *amd_zen4_tma_l1_events[] = {
    "@zen4_cyc",   // 0x0076 (Cycles)
    "@zen4_fe",    // 0x1000001A0 (Dispatch slots empty because frontend is stalled)
    "@zen4_disp",  // 0x07AA (Ops dispatched from any source)
    "@zen4_ret"    // 0x00C1 (Ops retired)
};

static void compute_amd_zen4_tma_l1(const double *raw_values, double *final_results) {
    double cycles   = raw_values[0];
    double fe_raw   = raw_values[1];
    double disp_raw = raw_values[2];
    double ret_raw  = raw_values[3];

    // Pipeline width = 6 on AMD Zen 4
    double slots = 6.0 * cycles;

    if (slots > 0) {
        double r_fe_bound = fe_raw / slots;
        double r_bad_spec = (disp_raw - ret_raw) / slots;
        assert(r_bad_spec >= 0.0);
        //if (r_bad_spec < 0.0) r_bad_spec = 0.0;
        double r_retiring = ret_raw / slots;
        double r_be_bound = 1.0 - (r_fe_bound + r_bad_spec + r_retiring);
        //if (r_be_bound < 0.0) r_be_bound = 0.0;
        assert(r_be_bound >= 0.0);

        final_results[0] = r_retiring * 100.0;
        final_results[1] = r_bad_spec * 100.0;
        final_results[2] = r_fe_bound * 100.0;
        final_results[3] = r_be_bound * 100.0;
    } else {
        final_results[0] = final_results[1] = final_results[2] = final_results[3] = 0.0;
    }
}


int resolve_metric(const char *metric_name, metric_resolver_t *out_resolver) {
    //compute_test_tma_guil();
    if (strcmp(metric_name, "TopdownL1") == 0) {
        int is_intel = detect_if_intel();
        int is_amd = detect_if_amd();
        int is_arm = detect_if_arm();
        assert(is_intel + is_amd + is_arm == 1);

        if (is_intel) {
            out_resolver->is_supported = 1;
            out_resolver->num_hw_events = 5;
            out_resolver->hw_events = skl_tma_l1_events;
            out_resolver->num_results = 4;
            out_resolver->compute_formula = compute_skl_tma_l1;
            return 1;
        }
        else if (is_amd) {
            out_resolver->is_supported = 1;
            out_resolver->num_hw_events = 4;
            out_resolver->hw_events = amd_zen4_tma_l1_events;
            out_resolver->num_results = 4;
            out_resolver->compute_formula = compute_amd_zen4_tma_l1;
            return 1;
        }
        else if (is_arm) {
            // todo
            return 0;
        }

        // Fallback : call perf stat in subprocess ?
        return 0; // Unsuported hardware
    }

    return 0; // Unknow metric
}

int get_perf_metric_results_count(const char *metric_name) {
    metric_resolver_t resolver;
    if (resolve_metric(metric_name, &resolver) && resolver.is_supported) {
        return resolver.num_results;
    }
    return 1;
}
