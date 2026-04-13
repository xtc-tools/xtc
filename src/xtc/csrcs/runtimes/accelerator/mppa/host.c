/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include "host_structures.h"
#include "mlir_host_header.h"
#include "mppa_management_host.h"

#include <mppa_offload_host.h>

#include <stdlib.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

void *mppa_common_structures = NULL;
mppa_offload_accelerator_t *mppa_accelerator = NULL;
mppa_offload_sysqueue_t *master_sysqueue = NULL;
size_t mppa_alloc_alignment = 64;

bool mppa_init_device()
{
    if (mppa_common_structures == NULL) {
        mppa_common_structures = mppa_pre_init();
        // Get ctx
        mppa_offload_ctx_ptr = (mppa_offload_host_context_t*) ((void**)mppa_common_structures)[0];
        assert(mppa_offload_ctx_ptr != NULL);
        // Get accelerator
        mppa_accelerator = mppa_offload_get_accelerator(mppa_offload_ctx_ptr, 0);
        assert(mppa_accelerator != NULL);
        // Get main sysqueue
        master_sysqueue = mppa_offload_get_sysqueue(mppa_accelerator, 0);
        assert(master_sysqueue != NULL);
    }
    return true;
}

bool mppa_deinit_device()
{
    mppa_de_init();
    mppa_common_structures = NULL;
    return true;
}

void* mppa_get_common_structures()
{
    return mppa_common_structures;
}

void* mppa_create_memory_handle()
{
    void *handle = malloc(sizeof(mppa_buffer_t));
    assert(handle != NULL);
    return handle;
}

bool mppa_destroy_memory_handle(void *handle)
{
    free(handle);
    return true;
}

void mppa_set_alloc_alignment(size_t alignment)
{
    mppa_alloc_alignment = alignment;
}

bool mppa_memory_allocate(void *handle, size_t size)
{
    assert(handle != NULL);
    assert(master_sysqueue != NULL);
    mppa_buffer_t *buffer = (mppa_buffer_t *)handle;
    if (mppa_offload_alloc(master_sysqueue, size, mppa_alloc_alignment, MPPA_OFFLOAD_ALLOC_DDR, &(buffer->voffset), &(buffer->offset)) != 0) {
        assert(0 && "Fail to alloc buffer\n");
        return false;
    }
    return true;
}

bool mppa_memory_free(void *handle)
{
    assert(handle != NULL);
    assert(master_sysqueue != NULL);
    mppa_buffer_t *buffer = (mppa_buffer_t *)handle;
    if (mppa_offload_free(master_sysqueue, MPPA_OFFLOAD_ALLOC_DDR, buffer->voffset) != 0) {
        assert(0 && "Fail to dealloc buffer\n");
        return false;
    }
    return true;
}

bool mppa_memory_copy_to(void *handle, void *src, size_t size)
{
    assert(handle != NULL);
    assert(mppa_accelerator != NULL);
    mppa_buffer_t *buffer = (mppa_buffer_t *)handle;
    if (mppa_offload_write(mppa_accelerator, src, buffer->offset, size, NULL) != 0) {
        assert(0 && "Failed write buffer\n");
        return false;
    }
    return true;
}

bool mppa_memory_copy_from(void *handle, void *dst, size_t size)
{
    assert(handle != NULL);
    assert(mppa_accelerator != NULL);
    mppa_buffer_t *buffer = (mppa_buffer_t *)handle;
    if (mppa_offload_read(mppa_accelerator, dst, buffer->offset, size, NULL) != 0) {
        assert(0 && "Failed read buffer\n");
        return false;
    }
    return true;
}

bool mppa_memory_fill_zero(void *handle, size_t size)
{
    void* tmp = calloc(size, 1);
    assert(tmp != NULL);
    bool res = mppa_memory_copy_to(handle, tmp, size);
    free(tmp);
    return res;
}

void* mppa_memory_data_pointer(void *handle)
{
    assert(handle != NULL);
    mppa_buffer_t *buffer = (mppa_buffer_t *)handle;
    return (void*)buffer->voffset;
}
