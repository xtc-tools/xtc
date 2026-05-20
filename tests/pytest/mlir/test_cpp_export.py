#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mlir_utils import matmul_impl, requires_mlir
from xtc.utils.ext_tools import get_shlib_extension

I, J, K, DTYPE = 4, 16, 8, "float32"


@requires_mlir
@pytest.mark.skipif(sys.platform != "linux", reason="cpp export integration test (linux)")
def test_cpp_export_matmul(tmp_path: Path) -> None:
    impl = matmul_impl(I, J, K, DTYPE, "matmul_export")
    sch = impl.get_scheduler()
    sch.set_dims(["i", "j", "k"])
    sched = sch.schedule()

    comp = impl.get_compiler(shared_lib=True, dump_file=str(tmp_path / "matmul_export"))
    module = comp.compile(sched)

    export_dir = tmp_path / "export"
    module.export(export_dir)

    ext = get_shlib_extension()
    export_name = "matmul_export"
    assert (export_dir / "include" / f"{export_name}.h").is_file()
    assert (export_dir / "lib" / f"lib{export_name}.{ext}").is_file()
    assert (export_dir / "test.cpp").is_file()
    assert (export_dir / "Makefile").is_file()
    assert (export_dir / "README.md").is_file()
    assert list((export_dir / "data" / "inputs").glob("*.bin"))
    assert list((export_dir / "data" / "outputs").glob("*.bin"))

    build = subprocess.run(
        ["make", "test"],
        cwd=export_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, f"make failed:\n{build.stdout}\n{build.stderr}"

    run = subprocess.run(
        ["./test"],
        cwd=export_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, f"test binary failed:\n{run.stdout}\n{run.stderr}"
    assert "All checks passed." in run.stdout
