/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <syscall.h>
#include <unistd.h>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cupti_profiler_host.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>

#include "perf_event.h"
#include "runtime.h"
#include "perf_event_gpu.h"

#define LOG RUNTIME_LOG

#define MAX_NUM_OF_RANGES 20

// Global variables for Range Profiler
CUpti_RangeProfiler_Object* g_pRangeProfilerObject = NULL;

std::vector<uint8_t> g_counterDataImage;
std::vector<uint8_t> g_configImage;

std::string g_chipName;

std::vector<const char*> g_metrics =
{
  /*  Some example metrics: */

   // "sm__warps_launched.sum",
   // "sm__ctas_launched.sum",
   // "sm__cycles_elapsed.sum",
   // "gr__cycles_elapsed.max",
   // "dram__bytes_read.sum"
};

int _current_range_index = 0;

// CUPTI Helper function declarations
void initialize_and_enable_range_profiler();
void create_counter_data_image(size_t max_num_of_ranges_in_counter_data_image);
void create_config_image();

extern "C" void open_perf_events__gpu(int n_events, const perf_event_args_t *events, int *fds) {
  if (g_metrics.size() == 0)
    return;

  size_t cnt = 0;
  for (int i = 0; i < n_events; i++) {
    if (events[i].mode == PERF_ARG_GPU)
    {
      fds[i] = PERF_EVENT_GPU;
      cnt++;
    }
  }

  if (cnt != g_metrics.size())
    throw std::runtime_error("Cannot open a subset of metrics (NYI)");

  /* Configure CUPTI Range Profiler */

  // Initialize and Enable Range Profiler
  initialize_and_enable_range_profiler();

  // Create config image
  create_config_image();

  // Create counter data image
  create_counter_data_image(MAX_NUM_OF_RANGES);

  CUpti_RangeProfiler_SetConfig_Params setConfig = { CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE };
  setConfig.pRangeProfilerObject = g_pRangeProfilerObject;
  setConfig.configSize = g_configImage.size();
  setConfig.pConfig = g_configImage.data();
  setConfig.counterDataImageSize = g_counterDataImage.size();
  setConfig.pCounterDataImage = g_counterDataImage.data();
  setConfig.range = CUPTI_AutoRange;
  setConfig.replayMode = CUPTI_KernelReplay;
  setConfig.maxRangesPerPass = MAX_NUM_OF_RANGES;
  setConfig.numNestingLevels = 1;
  setConfig.minNestingLevel = 1;
  setConfig.passIndex = 0;
  setConfig.targetNestingLevel = 0;
  cuptiRangeProfilerSetConfig(&setConfig);

  _current_range_index = 0;
}

extern "C" void close_perf_events__gpu(int n_events, const int *fds) {
  if (g_metrics.size() == 0)
    return;

  // Disable Range profiler
  CUpti_RangeProfiler_Disable_Params disableRangeProfiler = { CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE };
  disableRangeProfiler.pRangeProfilerObject = g_pRangeProfilerObject;
  cuptiRangeProfilerDisable(&disableRangeProfiler);

  // Deinitialize profiler
  CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = { CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE };
  cuptiProfilerDeInitialize(&profilerDeInitializeParams);
}

extern "C" void reset_perf_events__gpu(int n_events, const int *fds, uint64_t *results) {
  if (g_metrics.size() == 0)
    return;

  // Get information about profiled ranges
  CUpti_RangeProfiler_GetCounterDataInfo_Params cdiParams = { CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE };
  cdiParams.pCounterDataImage = g_counterDataImage.data();
  cdiParams.counterDataImageSize = g_counterDataImage.size();
  cuptiRangeProfilerGetCounterDataInfo(&cdiParams);
  
  _current_range_index = cdiParams.numTotalRanges;
}

extern "C" void start_perf_events__gpu(int n_events, const int *fds, uint64_t *results) {
  if (g_metrics.size() == 0)
    return;

  for (int i = 0; i < n_events; i++)
    if (fds[i] == PERF_EVENT_GPU)
      results[i] = 0;
  
  CUpti_RangeProfiler_Start_Params startRangeProfiler = { CUpti_RangeProfiler_Start_Params_STRUCT_SIZE };
  startRangeProfiler.pRangeProfilerObject = g_pRangeProfilerObject;
  cuptiRangeProfilerStart(&startRangeProfiler);
}

static void read_perf_events__gpu(int n_events, const int *fds, uint64_t *results) {
  if (g_metrics.size() == 0)
    return;

  // Decode profiling data
  CUpti_RangeProfiler_DecodeData_Params decodeData = { CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE };
  decodeData.pRangeProfilerObject = g_pRangeProfilerObject;
  cuptiRangeProfilerDecodeData(&decodeData);

  // Get information about profiled ranges
  CUpti_RangeProfiler_GetCounterDataInfo_Params cdiParams = { CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE };
  cdiParams.pCounterDataImage = g_counterDataImage.data();
  cdiParams.counterDataImageSize = g_counterDataImage.size();
  cuptiRangeProfilerGetCounterDataInfo(&cdiParams);
  LOG("Number of profiled ranges: %zu\n", cdiParams.numTotalRanges);

  CUpti_Profiler_Host_Initialize_Params hostInitializeParams = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
  hostInitializeParams.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
  hostInitializeParams.pChipName = g_chipName.c_str();
  hostInitializeParams.pCounterAvailabilityImage = nullptr;
  cuptiProfilerHostInitialize(&hostInitializeParams);
  CUpti_Profiler_Host_Object* pHostObject = hostInitializeParams.pHostObject;

  std::vector<double> accumulatedMetricValues(g_metrics.size());
  memset(accumulatedMetricValues.data(), 0, sizeof(double) * g_metrics.size());
  for (size_t rangeIndex = _current_range_index; rangeIndex < cdiParams.numTotalRanges; ++rangeIndex) {
    CUpti_RangeProfiler_CounterData_GetRangeInfo_Params getRangeInfoParams = {CUpti_RangeProfiler_CounterData_GetRangeInfo_Params_STRUCT_SIZE};
    getRangeInfoParams.counterDataImageSize = g_counterDataImage.size();
    getRangeInfoParams.pCounterDataImage = g_counterDataImage.data();
    getRangeInfoParams.rangeIndex = rangeIndex;
    getRangeInfoParams.rangeDelimiter = "/";
    cuptiRangeProfilerCounterDataGetRangeInfo(&getRangeInfoParams);

    std::vector<double> metricValues(g_metrics.size());
    CUpti_Profiler_Host_EvaluateToGpuValues_Params evalauateToGpuValuesParams {CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    evalauateToGpuValuesParams.pHostObject = pHostObject;
    evalauateToGpuValuesParams.pCounterDataImage = g_counterDataImage.data();
    evalauateToGpuValuesParams.counterDataImageSize = g_counterDataImage.size();
    evalauateToGpuValuesParams.ppMetricNames = g_metrics.data();
    evalauateToGpuValuesParams.numMetrics = g_metrics.size();
    evalauateToGpuValuesParams.rangeIndex = rangeIndex;
    evalauateToGpuValuesParams.pMetricValues = metricValues.data();
    cuptiProfilerHostEvaluateToGpuValues(&evalauateToGpuValuesParams);
    LOG("Range: %s\n", getRangeInfoParams.rangeName);
    LOG("Metric Values:\n");
    for (size_t i = 0; i < g_metrics.size(); ++i) {
      accumulatedMetricValues[i] += metricValues[i];
      LOG("\t%s: %f\n", g_metrics[i], metricValues[i]);
    }
    LOG("\n");
  }

  int result_index = 0;
  LOG("Accumulated Metric Values:\n");
  for (size_t i = 0; i < g_metrics.size(); ++i) {
    while (fds[result_index] != PERF_EVENT_GPU)
      result_index++;
    results[result_index] = (uint64_t)accumulatedMetricValues[i]; // FIXME uint64_t is not adapted for some metrics
    result_index++;
    LOG("\t%s: %f\n", g_metrics[i], accumulatedMetricValues[i]);
  }

  CUpti_Profiler_Host_Deinitialize_Params deinitializeParams = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
  deinitializeParams.pHostObject = pHostObject;
  cuptiProfilerHostDeinitialize(&deinitializeParams);
  pHostObject = nullptr;

}
extern "C" void stop_perf_events__gpu(int n_events, const int *fds, uint64_t *results) {
  if (g_metrics.size() == 0)
    return;

  CUpti_RangeProfiler_Stop_Params stopRangeProfiler = { CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE };
  stopRangeProfiler.pRangeProfilerObject = g_pRangeProfilerObject;
  cuptiRangeProfilerStop(&stopRangeProfiler);

  read_perf_events__gpu(n_events, fds, results);
}

extern "C" int get_perf_event_config__gpu(const char *name, perf_event_args_t *event) {
  assert(strncmp(name, "gpu.", 4) == 0);
  const char* gpu_name = name + 4;

  event->mode = PERF_ARG_GPU;
  if (strcmp(gpu_name, "cycles") == 0)
    g_metrics.push_back("gpu__cycles_elapsed.sum");
  else if (strcmp(gpu_name, "clocks") == 0)
    g_metrics.push_back("gpu__time_duration.sum");
  else
    g_metrics.push_back(gpu_name);

  return 0;
}

// ============================================================================
//                          CUpti Helper Functions
// ============================================================================

void create_config_image()
{
  CUpti_Profiler_Host_Initialize_Params hostInitializeParams = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
  hostInitializeParams.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
  hostInitializeParams.pChipName = g_chipName.c_str();
  hostInitializeParams.pCounterAvailabilityImage = nullptr;
  cuptiProfilerHostInitialize(&hostInitializeParams);
  CUpti_Profiler_Host_Object* pHostObject = hostInitializeParams.pHostObject;

  CUpti_Profiler_Host_ConfigAddMetrics_Params configAddMetricsParams {CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
  configAddMetricsParams.pHostObject = pHostObject;
  configAddMetricsParams.ppMetricNames = g_metrics.data();
  configAddMetricsParams.numMetrics = g_metrics.size();
  cuptiProfilerHostConfigAddMetrics(&configAddMetricsParams);

  CUpti_Profiler_Host_GetConfigImageSize_Params getConfigImageSizeParams {CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
  getConfigImageSizeParams.pHostObject = pHostObject;
  cuptiProfilerHostGetConfigImageSize(&getConfigImageSizeParams);
  g_configImage.resize(getConfigImageSizeParams.configImageSize);

  CUpti_Profiler_Host_GetConfigImage_Params getConfigImageParams = {CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
  getConfigImageParams.pHostObject = pHostObject;
  getConfigImageParams.pConfigImage = g_configImage.data();
  getConfigImageParams.configImageSize = g_configImage.size();
  cuptiProfilerHostGetConfigImage(&getConfigImageParams);

  CUpti_Profiler_Host_GetNumOfPasses_Params getNumOfPassesParam {CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
  getNumOfPassesParam.pConfigImage = g_configImage.data();
  getNumOfPassesParam.configImageSize = g_configImage.size();
  cuptiProfilerHostGetNumOfPasses(&getNumOfPassesParam);
  LOG("Num of Passes: %lu\n", getNumOfPassesParam.numOfPasses);

  CUpti_Profiler_Host_Deinitialize_Params deinitializeParams = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
  deinitializeParams.pHostObject = pHostObject;
  cuptiProfilerHostDeinitialize(&deinitializeParams);
  pHostObject = nullptr;
}


void initialize_and_enable_range_profiler()
{
  cuInit(/*flags=*/0);
  CUdevice device;
  cuDeviceGet(&device, /*ordinal=*/0);
  CUcontext ctx;
  cuDevicePrimaryCtxRetain(&ctx, device);
  cuCtxPushCurrent(ctx);

  // Initialize CUPTI Profiler
  CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
  cuptiProfilerInitialize(&profilerInitializeParams);

  CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
  getChipNameParams.deviceIndex = (size_t)device;
  cuptiDeviceGetChipName(&getChipNameParams);
  g_chipName = std::string(getChipNameParams.pChipName);
  LOG("Chip Name: %s\n", g_chipName.c_str());

  // Enable Range profiler
  CUpti_RangeProfiler_Enable_Params enableRange = { CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE };
  enableRange.ctx = ctx;
  cuptiRangeProfilerEnable(&enableRange);
  g_pRangeProfilerObject = enableRange.pRangeProfilerObject;

  cuCtxPopCurrent(nullptr);
}

void create_counter_data_image(size_t max_num_of_ranges_in_counter_data_image)
{
  // Get counter data size
  CUpti_RangeProfiler_GetCounterDataSize_Params ctDataSize = { CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE };
  ctDataSize.pRangeProfilerObject = g_pRangeProfilerObject;
  ctDataSize.pMetricNames = g_metrics.data();
  ctDataSize.numMetrics = g_metrics.size();
  ctDataSize.maxNumOfRanges = max_num_of_ranges_in_counter_data_image;
  ctDataSize.maxNumRangeTreeNodes = max_num_of_ranges_in_counter_data_image;
  cuptiRangeProfilerGetCounterDataSize(&ctDataSize);

  // Initialize counter data image
  g_counterDataImage.resize(ctDataSize.counterDataSize);
  CUpti_RangeProfiler_CounterDataImage_Initialize_Params initCtImg = { CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
  initCtImg.pRangeProfilerObject = g_pRangeProfilerObject;
  initCtImg.pCounterData = g_counterDataImage.data();
  initCtImg.counterDataSize = g_counterDataImage.size();
  cuptiRangeProfilerCounterDataImageInitialize(&initCtImg);
}

