/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "runtime.h"
#include "dlpack.h"
#include "alloc.h"

#define LOG RUNTIME_LOG
#define DEBUG RUNTIME_DEBUG

// (2*1024*1024)
#define DEFAULT_ALIGNMENT (256)

typedef struct {
  DLTensor dl_tensor;
  int32_t reference_count;
} CNDArray;

static int64_t CNDArray_alloc_alignment = DEFAULT_ALIGNMENT;

void CNDArray_set_alloc_alignment(int64_t alignment) {
  assert(alignment >= 1);
  CNDArray_alloc_alignment = alignment;
}

int64_t CNDArray_get_alloc_alignment(void) {
  return CNDArray_alloc_alignment;
}

void CNDArray_reset_alloc_alignment(void) {
  CNDArray_alloc_alignment = DEFAULT_ALIGNMENT;
}

int64_t CNDArray_data_size(CNDArray* array) {
  int64_t num_elems = 1;
  for (int32_t idx = 0; idx < array->dl_tensor.ndim; idx++) {
    num_elems *= array->dl_tensor.shape[idx];
  }
  return (num_elems * array->dl_tensor.dtype.bits + 7) / 8;
}

int64_t CNDArray_data_alignment(CNDArray* array) {
  int64_t alignment = (array->dl_tensor.dtype.bits + 7) / 8;
  alignment = CNDArray_alloc_alignment > alignment ? CNDArray_alloc_alignment: alignment;
  return alignment;
}

int CNDArray_init_null(CNDArray *array, int32_t ndim, const int64_t* shape, DLDataType dtype, DLDevice dev)
{
  memset(array, 0, sizeof(*array));
  array->dl_tensor.ndim = ndim;
  array->dl_tensor.shape = CHostMemoryAllocate(sizeof(*shape) * ndim);
  memcpy(array->dl_tensor.shape, shape, sizeof(*shape) * ndim);
  array->dl_tensor.dtype = dtype;
  array->dl_tensor.device = dev;
  array->dl_tensor.data = 0;
  return 0;
}

int CNDArray_init(CNDArray* array, int32_t ndim, const int64_t* shape, DLDataType dtype, DLDevice dev)
{
  int status = CNDArray_init_null(array, ndim, shape, dtype, dev);
  if (status != 0) {
    return status;
  }
  int64_t total_elem_bytes = CNDArray_data_size(array);
  int64_t alignment = CNDArray_data_alignment(array);
  array->dl_tensor.data =
    CHostMemoryAllocateAligned(total_elem_bytes, alignment);
  return 0;
}

int CNDArray_fini(CNDArray* array)
{
  CHostMemoryFree(array->dl_tensor.data);
  array->dl_tensor.data = NULL;
  CHostMemoryFree(array->dl_tensor.shape);
  array->dl_tensor.shape = NULL;
}

void CNDArray_IncrementReference(CNDArray* array)
{
  array->reference_count++;
}

uint32_t CNDArray_DecrementReference(CNDArray* array) {
  if (array->reference_count > 0) {
    array->reference_count--;
  }

  return array->reference_count;
}

int CNDArray_release(CNDArray* array) {
  if (CNDArray_DecrementReference(array) > 0) {
    return 0;
  }
  int status;
  status = CNDArray_fini(array);
  return status;
}

void CNDArray_copy_from_data(CNDArray* array, void *data)
{
  LOG("CNDArray_copy_from_data: array ptr = %p, data ptr = %p\n", array, data);
  int64_t total_elem_bytes = CNDArray_data_size(array);
  memcpy(array->dl_tensor.data, data, total_elem_bytes);
}

void CNDArray_copy_to_data(CNDArray* array, void *data)
{
  LOG("CNDArray_copy_to_data: array ptr = %p, data ptr = %p\n", array, data);
  int64_t total_elem_bytes = CNDArray_data_size(array);
  memcpy(data, array->dl_tensor.data, total_elem_bytes);
}

void CNDArray_fill_zero(CNDArray* array)
{
  int64_t total_elem_bytes = CNDArray_data_size(array);
  memset(array->dl_tensor.data, 0, total_elem_bytes);
}

CNDArray *CNDArray_new(int32_t ndim, const int64_t* shape, DLDataType dtype, DLDevice dev)
{
  LOG("CNDArray_new: dim = %d, shape ptr = %p\n", ndim, shape);
  if (DEBUG) {
    for (int i = 0; i < ndim; i++) {
      LOG("  Shape[%d] = %ld\n", i, shape[i]);
    }
  }
  CNDArray *array = CHostMemoryAllocate(sizeof(*array));
  int status;
  status = CNDArray_init(array, ndim, shape, dtype, dev);
  if (status != 0) {
    CHostMemoryFree(array);
    return NULL;
  }
  CNDArray_IncrementReference(array);
  LOG("  Returns: ptr %p\n", array);
  return array;
}

void CNDArray_del(CNDArray* array)
{
  LOG("CNDArray_del: array ptr = %p\n", array);
  CNDArray_release(array);
  assert(array->reference_count == 0);
  CHostMemoryFree(array);
}

