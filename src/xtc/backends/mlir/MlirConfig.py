#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from dataclasses import dataclass, field
from xtc.utils.tools import (
    get_mlir_prefix,
)


@dataclass(frozen=True)
class MlirConfig:
    shared_lib: bool = False
    executable: bool = False
    mlir_install_dir: str | None = None
    to_disassemble: str = ""
    save_temps: bool = False
    save_temps_dir: str = "./save_temps_dir"
    bare_ptr: bool = True
    print_source_ir: bool = False
    print_transformed_ir: bool = False
    print_assembly: bool = False
    visualize_jumps: bool = True
    print_lowered_ir: bool = False
    debug: bool = False
    color: bool = False
    concluding_passes: list[str] = field(default_factory=list)
    always_vectorize: bool = False
    vectors_size: int | None = None
    arch: str = "native"
    cpu: str = "native"
    selected_device: int | None = None
    required_extensions: list[str] = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(
            self, "mlir_install_dir", get_mlir_prefix(self.mlir_install_dir)
        )
