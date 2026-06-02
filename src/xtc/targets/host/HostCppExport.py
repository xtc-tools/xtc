#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from functools import reduce
import operator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from xtc.utils.ext_tools import get_shlib_extension
from xtc.utils.numpy import np_init

if TYPE_CHECKING:
    from .HostModule import HostModule

__all__ = ["HostCppExporter"]

_DTYPE_MAP: dict[str, tuple[str, str]] = {
    "float32": ("float", "f"),
    "float64": ("double", "d"),
}


@dataclass(frozen=True)
class _TensorArg:
    name: str
    c_name: str
    shape: tuple[int, ...]
    dtype: str
    c_type: str
    numel: int
    is_input: bool


def _c_ident(name: str) -> str:
    ident = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if ident and ident[0].isdigit():
        ident = f"_{ident}"
    return ident or "tensor"


def _macro_prefix(c_name: str) -> str:
    return f"XTC_{c_name.upper()}"


class HostCppExporter:
    """Export a CPU shared-library module for linking from external C/C++ code."""

    def __init__(
        self,
        module: HostModule,
        out_dir: Path,
        *,
        name: str | None = None,
        seed: int = 0,
    ) -> None:
        self._module = module
        self._out_dir = out_dir
        self._export_name = name or module.name
        self._seed = seed

    def export(self) -> None:
        inputs_spec_fn = self._module._np_inputs_spec
        outputs_spec_fn = self._module._np_outputs_spec
        reference_impl = self._module._reference_impl
        if inputs_spec_fn is None or outputs_spec_fn is None:
            raise ValueError("module is missing np_inputs_spec / np_outputs_spec")
        if reference_impl is None:
            raise ValueError("module is missing reference_impl (graph required)")

        inputs_spec = inputs_spec_fn()
        outputs_spec = outputs_spec_fn()
        input_names, output_names = self._resolve_arg_names(
            len(inputs_spec), len(outputs_spec)
        )

        input_args = [
            self._tensor_arg(name, spec, is_input=True)
            for name, spec in zip(input_names, inputs_spec)
        ]
        output_args = [
            self._tensor_arg(name, spec, is_input=False)
            for name, spec in zip(output_names, outputs_spec)
        ]

        self._out_dir.mkdir(parents=True, exist_ok=True)
        (self._out_dir / "include").mkdir(exist_ok=True)
        (self._out_dir / "lib").mkdir(exist_ok=True)
        (self._out_dir / "data" / "inputs").mkdir(parents=True, exist_ok=True)
        (self._out_dir / "data" / "outputs").mkdir(parents=True, exist_ok=True)

        self._copy_shared_lib()
        self._write_golden_data(input_args, output_args, reference_impl)
        self._write_header(input_args, output_args)
        self._write_test_cpp(input_args, output_args)
        self._write_makefile()
        self._write_readme()

    def _resolve_arg_names(
        self, num_inputs: int, num_outputs: int
    ) -> tuple[list[str], list[str]]:
        graph = self._module._graph
        if graph is not None:
            return (
                [node.name for node in graph.inputs_nodes],
                [node.name for node in graph.outputs_nodes],
            )
        return (
            [f"in{i}" for i in range(num_inputs)],
            [f"out{i}" for i in range(num_outputs)],
        )

    def _tensor_arg(
        self, name: str, spec: dict[str, Any], *, is_input: bool
    ) -> _TensorArg:
        dtype = spec["dtype"]
        if dtype not in _DTYPE_MAP:
            raise NotImplementedError(
                f"export does not support dtype {dtype!r} (supported: {list(_DTYPE_MAP)})"
            )
        shape = tuple(spec["shape"])
        c_type = _DTYPE_MAP[dtype][0]
        c_name = _c_ident(name)
        numel = reduce(operator.mul, shape, 1)
        return _TensorArg(
            name=name,
            c_name=c_name,
            shape=shape,
            dtype=dtype,
            c_type=c_type,
            numel=numel,
            is_input=is_input,
        )

    def _copy_shared_lib(self) -> None:
        ext = get_shlib_extension()
        dest = self._out_dir / "lib" / f"lib{self._export_name}.{ext}"
        shutil.copy2(self._module.file_name, dest)

    def _write_golden_data(
        self,
        input_args: list[_TensorArg],
        output_args: list[_TensorArg],
        reference_impl: Callable[..., None],
    ) -> None:
        np.random.seed(self._seed)
        inputs: list[np.ndarray[Any, Any]] = []
        for arg in input_args:
            arr = np_init(shape=arg.shape, dtype=arg.dtype)
            path = self._out_dir / "data" / "inputs" / f"{arg.c_name}.bin"
            arr.tofile(path)
            inputs.append(arr)

        outputs: list[np.ndarray[Any, Any]] = []
        for arg in output_args:
            arr = np.zeros(arg.shape, dtype=arg.dtype)
            outputs.append(arr)

        reference_impl(*inputs, *outputs)

        for arg, arr in zip(output_args, outputs):
            path = self._out_dir / "data" / "outputs" / f"{arg.c_name}.bin"
            arr.tofile(path)

    def _write_header(
        self, input_args: list[_TensorArg], output_args: list[_TensorArg]
    ) -> None:
        payload = self._module.payload_name
        all_args = input_args + output_args
        params = ", ".join(f"{a.c_type}* {a.c_name}" for a in all_args)
        meta_lines: list[str] = []
        for arg in all_args:
            prefix = _macro_prefix(arg.c_name)
            meta_lines.append(f"#define {prefix}_NDIM {len(arg.shape)}")
            for i, dim in enumerate(arg.shape):
                meta_lines.append(f"#define {prefix}_SHAPE_{i} {dim}")
            meta_lines.append(f'#define {prefix}_DTYPE "{arg.dtype}"')
            meta_lines.append(f"#define {prefix}_NUMEL {arg.numel}")

        meta = "\n".join(meta_lines)
        content = f"""\
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {{
#endif

void {payload}({params});

/* Tensor metadata (shapes, dtypes, element counts). */
{meta}

#ifdef __cplusplus
}}
#endif
"""
        path = self._out_dir / "include" / f"{self._export_name}.h"
        path.write_text(content, encoding="utf-8")

    def _write_test_cpp(
        self, input_args: list[_TensorArg], output_args: list[_TensorArg]
    ) -> None:
        payload = self._module.payload_name
        header = self._export_name
        all_args = input_args + output_args
        c_types = {a.c_type for a in all_args}
        if len(c_types) != 1:
            raise NotImplementedError(
                "export test.cpp requires a single floating-point dtype across all tensors"
            )
        c_type = c_types.pop()

        load_blocks: list[str] = []
        ptr_names: list[str] = []
        check_blocks: list[str] = []

        for arg in input_args:
            load_blocks.append(self._load_vector_block(arg, c_type, "inputs"))
            ptr_names.append(f"{arg.c_name}.data()")

        for arg in output_args:
            load_blocks.append(
                self._load_vector_block(arg, c_type, "outputs", zero_init=True)
            )
            ptr_names.append(f"{arg.c_name}.data()")

        for arg in output_args:
            check_blocks.append(
                f"""\
  {{
    std::vector<{c_type}> ref = load_binary<{c_type}>(
        "data/outputs/{arg.c_name}.bin", {arg.numel});
    if (!allclose({arg.c_name}, ref)) {{
      std::cerr << "FAIL {arg.c_name}\\n";
      return 1;
    }}
    std::cout << "OK {arg.c_name}\\n";
  }}"""
            )

        load_src = "\n\n".join(load_blocks)
        checks = "\n".join(check_blocks)
        call_args = ", ".join(ptr_names)

        content = f"""\
#include "{header}.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template <typename T>
std::vector<T> load_binary(const std::string& path, size_t expected_numel) {{
  std::ifstream in(path, std::ios::binary);
  if (!in) {{
    throw std::runtime_error("cannot open " + path);
  }}
  in.seekg(0, std::ios::end);
  const auto nbytes = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  const size_t numel = nbytes / sizeof(T);
  if (numel != expected_numel) {{
    throw std::runtime_error("size mismatch for " + path);
  }}
  std::vector<T> data(numel);
  in.read(reinterpret_cast<char*>(data.data()),
          static_cast<std::streamsize>(nbytes));
  return data;
}}

template <typename T>
bool allclose(const std::vector<T>& got, const std::vector<T>& ref,
              T rtol = static_cast<T>(1e-5), T atol = static_cast<T>(1e-6)) {{
  if (got.size() != ref.size()) {{
    return false;
  }}
  for (size_t i = 0; i < got.size(); ++i) {{
    const T diff = std::abs(got[i] - ref[i]);
    const T tol = atol + rtol * std::abs(ref[i]);
    if (diff > tol) {{
      std::cerr << "  mismatch at index " << i << ": got=" << got[i]
                << " ref=" << ref[i] << "\\n";
      return false;
    }}
  }}
  return true;
}}

int main() {{
  try {{
{load_src}

    {payload}({call_args});

{checks}
    std::cout << "All checks passed.\\n";
    return 0;
  }} catch (const std::exception& ex) {{
    std::cerr << "Error: " << ex.what() << "\\n";
    return 1;
  }}
}}
"""
        (self._out_dir / "test.cpp").write_text(content, encoding="utf-8")

    def _load_vector_block(
        self, arg: _TensorArg, c_type: str, subdir: str, *, zero_init: bool = False
    ) -> str:
        if zero_init:
            init = f"std::vector<{c_type}>({arg.numel})"
        else:
            init = (
                f'load_binary<{c_type}>("data/{subdir}/{arg.c_name}.bin", {arg.numel})'
            )
        return f"    std::vector<{c_type}> {arg.c_name} = {init};"

    def _write_makefile(self) -> None:
        ext = get_shlib_extension()
        if ext == "dylib":
            rpath = "-Wl,-rpath,@loader_path/lib"
        else:
            rpath = "-Wl,-rpath,'$$ORIGIN/lib'"
        content = f"""\
NAME      := {self._export_name}
CXX       ?= c++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra
INCLUDES  := -Iinclude
LDFLAGS   := -Llib -l$(NAME) {rpath}

test: test.cpp
\t$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

.PHONY: run clean
run: test
\t./test

clean:
\trm -f test
"""
        (self._out_dir / "Makefile").write_text(content, encoding="utf-8")

    def _write_readme(self) -> None:
        content = f"""\
# XTC exported kernel: {self._export_name}

Artifacts produced by `module.export()` for CPU linking from C/C++.

## Build and test

```bash
make
make run
```

- `include/{self._export_name}.h` — kernel declaration and tensor metadata macros
- `lib/lib{self._export_name}.{get_shlib_extension()}` — compiled shared library
- `data/inputs/*.bin` / `data/outputs/*.bin` — golden inputs and reference outputs
- `test.cpp` — loads golden data, calls the kernel, compares against reference
"""
        (self._out_dir / "README.md").write_text(content, encoding="utf-8")
