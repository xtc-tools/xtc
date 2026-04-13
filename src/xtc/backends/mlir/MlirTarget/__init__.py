#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os

from .MlirTarget import MlirTarget


def get_target_from_name(name: str) -> type[MlirTarget]:
    if name == "llvmir":
        from .MlirLLVMTarget import MlirLLVMTarget

        return MlirLLVMTarget
    elif name == "c":
        from .MlirCTarget import MlirCTarget

        return MlirCTarget
    elif name == "nvgpu":
        from .MlirNVGPUTarget import MlirNVGPUTarget

        return MlirNVGPUTarget
    elif name == "mppa":
        from .MlirMppaTarget import MlirMppaTarget

        return MlirMppaTarget
    else:
        raise NameError(f"'{name}' is not a known target")


def get_default_target():
    return get_target_from_name(os.getenv("XTC_MLIR_TARGET", "llvmir"))
