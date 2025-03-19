/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#ifndef _RUNTIME_ALLOC_H
#define _RUNTIME_ALLOC_H

#include <stdint.h>

extern void *CHostMemoryAllocate(size_t size);
extern void *CHostMemoryAllocateAligned(size_t size, size_t alignment);
extern void CHostMemoryFree(void *ptr);
#endif
