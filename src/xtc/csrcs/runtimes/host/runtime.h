/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#ifndef _RUNTIME_H
#define _RUNTIME_H

#ifndef RUNTIME_DEBUG
#define RUNTIME_DEBUG 0
#endif

#if RUNTIME_DEBUG
#include <stdio.h>
#define RUNTIME_LOG(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__)
#else
#define RUNTIME_LOG(fmt, ...) ((void)0)
#endif

#endif

