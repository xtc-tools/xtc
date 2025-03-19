/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include <stdlib.h>
#include <assert.h>
#include "runtime.h"
#include "alloc.h"

#define LOG RUNTIME_LOG
#define DEBUG RUNTIME_DEBUG

void *CHostMemoryAllocate(size_t size)
{
  LOG("CHostMemoryAllocate: size = %ld\n", size);
  void *ptr;
  ptr = malloc(size);
  assert(ptr != NULL);
  LOG("  Returns: ptr %p\n", ptr);
  return ptr;
}

void *CHostMemoryAllocateAligned(size_t size, size_t alignment)
{
  LOG("CHostMemoryAllocateAligned: size = %ld, alignment = %ld\n", size, alignment);
  size_t aligned_size = (size + alignment - 1) / alignment * alignment;
  void *ptr = aligned_alloc(alignment, aligned_size);
  assert(ptr != NULL);
  LOG("  Returns: ptr %p\n", ptr);
  return ptr;
}

void CHostMemoryFree(void *ptr)
{
  LOG("CHostMemoryFree: ptr = %p\n", ptr);
  assert(ptr != NULL);
  free(ptr);
}
