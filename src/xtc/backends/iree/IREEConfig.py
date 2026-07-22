#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from dataclasses import dataclass, field

__all__ = [
    "IREEConfig",
]


@dataclass(frozen=True)
class IREEConfig:
    """Configuration for the IREE compiler.

    Only the CPU (``llvm-cpu``) target is supported for now.
    """

    target_backend: str = "llvm-cpu"
    target_cpu: str = "host"
    target_cpu_features: str | None = None
    target_triple: str | None = None
    dump_file: str | None = None
    print_source_ir: bool = False
    extra_args: list[str] = field(default_factory=list)

    def iree_compile_args(self) -> list[str]:
        args = [f"--iree-llvmcpu-target-cpu={self.target_cpu}"]
        # A triple different from the host enables cross-compilation; when it is
        # left unset IREE targets the host triple (native compilation).
        if self.target_triple is not None:
            args.append(f"--iree-llvmcpu-target-triple={self.target_triple}")
        if self.target_cpu_features is not None:
            args.append(
                f"--iree-llvmcpu-target-cpu-features={self.target_cpu_features}"
            )
        args.extend(self.extra_args)
        return args
