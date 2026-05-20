#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from functools import wraps
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any, cast

from xtc.integration.pytorch.compile import (
    MatmulKernelArtifacts,
    export_matmul_kernel,
)

MATMUL_C_SHIM_DECL = textwrap.dedent(
    """\
    AOTITorchError
    aoti_torch_cpu_matmul_cpp(
        AtenTensorHandle out,
        AtenTensorHandle x,
        AtenTensorHandle w,
        AtenTensorHandle* b)"""
)

_active_artifacts: MatmulKernelArtifacts | None = None
_cpp_wrapper_patch_installed = False


def _update_aot_custom_op_libs(artifacts: MatmulKernelArtifacts) -> None:
    import torch._inductor.config as inductor_config

    libs = list(inductor_config.aot_inductor.custom_op_libs or [])
    if artifacts.shim_lib_name not in libs:
        libs.append(artifacts.shim_lib_name)
    inductor_config.aot_inductor.custom_op_libs = libs

    cache = str(artifacts.cache_dir.resolve())
    existing = os.environ.get("LIBRARY_PATH", "")
    if cache not in existing.split(":"):
        os.environ["LIBRARY_PATH"] = f"{cache}:{existing}" if existing else cache


def set_active_inductor_artifacts(artifacts: MatmulKernelArtifacts | None) -> None:
    global _active_artifacts
    _active_artifacts = artifacts
    if artifacts is not None:
        _update_aot_custom_op_libs(artifacts)


def get_active_inductor_link_flags() -> list[str]:
    if _active_artifacts is None:
        return []
    art = _active_artifacts
    shim = str(art.shim_lib_path.resolve())
    lib_dir = str(art.cache_dir.resolve())
    # Link the shim archive directly (same pattern as HalideCodeCache extra_flags).
    return [
        shim,
        f"-Wl,-rpath,{lib_dir}",
        f"-Wl,-rpath,{art.xtc_lib_path.parent.resolve()}",
    ]


def _torch_build_paths() -> tuple[list[str], list[str], list[str]]:
    from torch.utils.cpp_extension import include_paths, library_paths

    includes = include_paths()
    lib_dirs = library_paths()
    libs = ["torch", "torch_cpu", "c10"]
    return includes, lib_dirs, libs


def _c_type_for_dtype(xtc_dtype: str) -> str:
    if xtc_dtype == "float32":
        return "float"
    if xtc_dtype == "float64":
        return "double"
    raise ValueError(f"unsupported xtc dtype for inductor shim: {xtc_dtype}")


def _write_generated_shim(artifacts: MatmulKernelArtifacts, src_path: Path) -> None:
    c_type = _c_type_for_dtype(artifacts.xtc_dtype)
    header = artifacts.export_name
    payload = artifacts.payload_name

    content = textwrap.dedent(
        f"""\
        #include "{header}.h"

        #include <torch/csrc/inductor/aoti_torch/c/shim.h>

        namespace {{

        using Float = {c_type};

        }}  // namespace

        extern "C" {{

        AOTITorchError aoti_torch_cpu_matmul_cpp(
            AtenTensorHandle out,
            AtenTensorHandle x,
            AtenTensorHandle w,
            AtenTensorHandle* b) {{

          void* x_ptr_v = nullptr;
          void* w_ptr_v = nullptr;
          void* out_ptr_v = nullptr;
          if (aoti_torch_get_data_ptr(x, &x_ptr_v)) return 1;
          if (aoti_torch_get_data_ptr(w, &w_ptr_v)) return 1;
          if (aoti_torch_get_data_ptr(out, &out_ptr_v)) return 1;

          auto* x_ptr = static_cast<Float*>(x_ptr_v);
          auto* w_ptr = static_cast<Float*>(w_ptr_v);
          auto* out_ptr = static_cast<Float*>(out_ptr_v);

          {payload}(x_ptr, w_ptr, out_ptr);

          return 0;
        }}

        }}  // extern "C"
        """
    )
    src_path.write_text(content, encoding="utf-8")


def build_matmul_aoti_shim(
    artifacts: MatmulKernelArtifacts,
    *,
    force: bool = False,
) -> Path:
    if not force and artifacts.shim_lib_path.is_file():
        return artifacts.shim_lib_path

    gen_dir = artifacts.cache_dir / f"shim_build_{artifacts.export_name}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    src_path = gen_dir / "matmul_aoti_shim.cpp"
    _write_generated_shim(artifacts, src_path)

    includes, lib_dirs, libs = _torch_build_paths()
    ext = artifacts.shim_lib_path.suffix.lstrip(".")
    if sys.platform == "darwin":
        shared_flag = "-dynamiclib"
        rpath_flag = f"-Wl,-rpath,{artifacts.xtc_lib_path.parent}"
    else:
        shared_flag = "-shared"
        rpath_flag = f"-Wl,-rpath,{artifacts.xtc_lib_path.parent}"

    cmd: list[str] = [
        os.environ.get("CXX", "g++"),
        "-O2",
        "-std=c++17",
        "-fPIC",
        shared_flag,
        "-o",
        str(artifacts.shim_lib_path),
        str(src_path),
        f"-I{artifacts.export_dir / 'include'}",
        f"-L{artifacts.xtc_lib_path.parent}",
        f"-l{artifacts.export_name}",
        rpath_flag,
    ]
    for inc in includes:
        cmd.append(f"-I{inc}")
    for lib_dir in lib_dirs:
        cmd.extend(["-L", lib_dir])
    for lib in libs:
        cmd.append(f"-l{lib}")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "failed to build XTC matmul AOTI shim\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return artifacts.shim_lib_path


def ensure_inductor_matmul_artifacts(
    x_shape: tuple[int, ...],
    w_shape: tuple[int, int],
    dtype: object,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> MatmulKernelArtifacts:
    import torch

    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"expected torch.dtype, got {type(dtype)!r}")
    artifacts = export_matmul_kernel(
        x_shape, w_shape, dtype, cache_dir=cache_dir, force=force
    )
    set_active_inductor_artifacts(artifacts)
    return artifacts


ensure_inductor_linear_artifacts = ensure_inductor_matmul_artifacts


def _partition_extra_flags(
    flags: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split compile flags from link-only flags (-L/-l/-Wl)."""
    compile_flags: list[str] = []
    link_flags: list[str] = []
    i = 0
    while i < len(flags):
        flag = flags[i]
        if flag == "-L" and i + 1 < len(flags):
            link_flags.extend([flag, flags[i + 1]])
            i += 2
            continue
        if (
            flag.startswith("-L")
            or flag.startswith("-l")
            or flag.startswith("-Wl,")
            or flag.endswith(".so")
            or flag.endswith(".a")
        ):
            link_flags.append(flag)
        else:
            compile_flags.append(flag)
        i += 1
    return tuple(compile_flags), tuple(link_flags)


def _patch_cpp_wrapper_code_cache() -> None:
    global _cpp_wrapper_patch_installed
    if _cpp_wrapper_patch_installed:
        return

    from torch._inductor.codecache import CppCodeCache

    _orig_load_async = cast(
        Callable[[type[Any], str, str, Any, Sequence[str], str | None], Any],
        getattr(CppCodeCache.load_async, "__func__", CppCodeCache.load_async),
    )

    @wraps(_orig_load_async)
    def _load_async_impl(
        cls: type[Any],
        main_code: str,
        device_type: str = "cpu",
        submit_fn: Any = None,
        extra_flags: Sequence[str] = (),
        optimized_code: str | None = None,
    ) -> Any:
        merged = tuple(extra_flags) + tuple(get_active_inductor_link_flags())
        compile_flags, link_flags = _partition_extra_flags(merged)
        flags = compile_flags + link_flags
        if not link_flags:
            return _orig_load_async(
                cls,
                main_code,
                device_type,
                submit_fn,
                flags,
                optimized_code,
            )

        from torch._inductor import codecache as codecache_mod

        orig_precompile = codecache_mod._precompile_header

        def precompile_without_link(
            header: str,
            hashable_cmd_line: str,
            min_optimize: bool = False,
            **compile_command: Any,
        ) -> str:
            cmd = dict(compile_command)
            pch_flags, _ = _partition_extra_flags(tuple(cmd.get("extra_flags", ())))
            cmd["extra_flags"] = pch_flags
            return orig_precompile(
                header, hashable_cmd_line, min_optimize=min_optimize, **cmd
            )

        codecache_mod._precompile_header = precompile_without_link  # type: ignore[assignment]
        try:
            return _orig_load_async(
                cls,
                main_code,
                device_type,
                submit_fn,
                flags,
                optimized_code,
            )
        finally:
            codecache_mod._precompile_header = orig_precompile

    CppCodeCache.load_async = classmethod(_load_async_impl)  # type: ignore[method-assign,assignment]
    _cpp_wrapper_patch_installed = True


def _declare_matmul_shim_on_wrapper(wrapper: object) -> None:
    if getattr(wrapper, "_xtc_matmul_shim_declared", False):
        return
    wrapper._xtc_matmul_shim_declared = True  # type: ignore[attr-defined]
    decl = f"""
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
extern "C" {{
{MATMUL_C_SHIM_DECL};
}}
"""
    wrapper.prefix._lines.insert(0, decl)  # type: ignore[attr-defined]


def _patch_cpp_wrapper_extern_call() -> None:
    from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu

    if getattr(CppWrapperCpu, "_xtc_extern_patch_installed", False):
        return
    CppWrapperCpu._xtc_extern_patch_installed = True  # type: ignore[attr-defined]

    orig = CppWrapperCpu.generate_c_shim_extern_kernel_call

    def patched(
        self: object,
        kernel: str,
        args: list[str],
        device: str,
        **kwargs: object,
    ) -> None:
        shim = self.get_c_shim_func_name(kernel, device)  # type: ignore[attr-defined]
        if shim == "aoti_torch_cpu_matmul_cpp":
            _declare_matmul_shim_on_wrapper(self)
        return orig(self, kernel, args, device, **kwargs)  # type: ignore[arg-type]

    CppWrapperCpu.generate_c_shim_extern_kernel_call = patched  # type: ignore[method-assign]


def register_aot_inductor_cpp_shims() -> None:
    """Register C shim declarations for AOT Inductor (use inside config.patch)."""
    import torch
    import torch._inductor.config as inductor_config

    op = torch.ops.xtc.matmul.default
    shims = dict(inductor_config.aot_inductor.custom_ops_to_c_shims)
    shims[op] = [MATMUL_C_SHIM_DECL]
    inductor_config.aot_inductor.custom_ops_to_c_shims = shims
    if _active_artifacts is not None:
        _update_aot_custom_op_libs(_active_artifacts)


def register_inductor_cpp_hooks() -> None:
    _patch_cpp_wrapper_code_cache()
    _patch_cpp_wrapper_extern_call()
