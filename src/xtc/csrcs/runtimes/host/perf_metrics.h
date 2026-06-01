/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#ifndef _PERF_METRICS_H
#define _PERF_METRICS_H

typedef struct {
    int is_supported;
    int num_results;

    int num_passes;
    int num_hw_events;          // Total events with all passes
    const int *events_per_pass; // events per passes (ex: {4, 4, 4})
    const char **hw_events;     // flat array with all events name

    void (*compute_formula)(const double *raw_values, double *final_results);
} metric_resolver_t;

int resolve_metric(const char *metric_name, metric_resolver_t *out_resolver);
int get_perf_metric_results_count(const char *metric_name);



#endif
