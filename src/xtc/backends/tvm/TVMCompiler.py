#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, cast
from typing_extensions import override
import tempfile
from pathlib import Path
import shutil
import subprocess
import shlex
import sys
from functools import partial
from packaging.version import Version

from xtc.targets.host import HostModule

import xtc.backends.tvm as backend
import xtc.itf as itf
from xtc.utils.text import jinja_generate_file
from xtc.utils.tarfile import TarFile

from xtc.utils.host_tools import disassemble, target_triple

from .TVMOpsCompiler import (
    TVMExprCompiler,
    TVMScheduledExpr,
    TVMScheduledExprTE,
    TVMScheduledExprTIR,
)
from .TVMOps import (
    TVMBaseExpr,
)

import tvm


__all__ = [
    "TVMCompiler",
]

TVM_VERSION = Version(tvm.__version__.split("+", 1)[0])


class TVMCompiler(itf.comp.Compiler):
    def __init__(
        self,
        backend: "backend.TVMBackend",
        **kwargs: Any,
    ) -> None:
        self._backend = backend
        self.payload_name = self._backend.payload_name
        self.save_temps = kwargs.get("save_temps", False)
        self.save_temps_dir = kwargs.get("save_temps_dir", "./save_temps_dir")
        self.bare_ptr = kwargs.get("bare_ptr", False)
        self.dump_file = kwargs.get("dump_file")
        assert self.dump_file is not None, f"must pass the dump_file name"
        self.print_source_ir = kwargs.get("print_source_ir", False)
        self.print_transformed_ir = kwargs.get("print_transformed_ir", False)
        self.print_assembly = kwargs.get("print_assembly", False)
        self.print_file = kwargs.get("print_file", sys.stdout)
        self.color = kwargs.get("color", False)
        self.shared_lib = kwargs.get("shared_lib", False)
        self.ar_lib = kwargs.get("ar_lib", False)
        self.executable = kwargs.get("executable", False)
        self.emit_c = kwargs.get("emit_c", False)
        self.target = kwargs.get("target", "native")
        self.arch = kwargs.get("arch", "native")
        self.tvm_target_options = self._get_tvm_target_options(self.target, self.arch)
        self.tvm_target = "llvm"
        self.tvm_tgt = f"{self.tvm_target} {self.tvm_target_options}"
        assert not self.executable, f"executable generation not supported yet for TVM"
        assert self.shared_lib or self.emit_c or self.ar_lib, (
            f"shared_lib/ar_lib or C generation is mandatory for TVM"
        )
        assert not (self.shared_lib and self.ar_lib), (
            f"cannot have both shlib and arlib"
        )

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    def _save_temp(self, fname: str, content: str) -> None:
        if not self.save_temps:
            return
        Path(self.save_temps_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{self.save_temps_dir}/{fname}", "w") as outf:
            outf.write(content)

    @override
    def compile(self, schedule: itf.schd.Schedule) -> itf.comp.Module:
        assert isinstance(schedule, backend.TVMSchedule)
        assert self.dump_file is not None
        save_temp = self._save_temp
        op = self._backend._tvm_base
        func_name = self.payload_name
        packed_func_name = f"packed_{func_name}" if self.bare_ptr else func_name

        if self.shared_lib:
            type = "shlib"
        elif self.ar_lib:
            type = "arlib"
        else:
            # May emit c in addition to lib
            assert self.emit_c
            type = "csrc"

        Path(self.dump_file).parent.mkdir(parents=True, exist_ok=True)
        dump_base = Path(self.dump_file).stem
        lib_path = self.dump_file
        if type in ["arlib", "shlib"]:
            emit_c_base = f"{lib_path}.export_c"
        else:
            emit_c_base = lib_path
        if self.bare_ptr:
            packed_lib_path = f"{lib_path}_packed"
            emit_c_packed_base = f"{emit_c_base}_packed"
        else:
            packed_lib_path = lib_path
            emit_c_packed_base = emit_c_base
        expr_compiler = TVMExprCompiler(op, tir_schedule=self._backend._tir_schedule)
        schedulable = expr_compiler.generate()
        if self.print_source_ir or self.save_temps:
            lowered = schedulable.schedule().dumps()
            if self.print_source_ir:
                self._print(lowered)
            save_temp(f"{dump_base}.initial.txt", lowered)
        schedule = cast(backend.TVMSchedule, schedule)
        save_temp(f"{dump_base}.sched.txt", str(schedule))
        if self.print_transformed_ir:
            self._print(schedule)
        sch = schedulable.schedule(schedule)
        if self.print_transformed_ir or self.save_temps:
            lowered = sch.dumps()
            if self.print_transformed_ir:
                self._print(lowered)
            save_temp(f"{dump_base}.scheduled.txt", lowered)
        if self.emit_c:
            self._build_c(
                sch,
                func_name=packed_func_name,
                fname=emit_c_packed_base,
            )
        if type in ["shlib", "arlib"]:
            built = self._build(sch, func_name=packed_func_name)
            if self.save_temps:
                for idx, mod in enumerate(built._collect_dso_modules()):
                    llvm_ir = str(mod.get_source("ll"))
                    save_temp(f"{dump_base}.lib{idx}.ll", llvm_ir)
                    # This will generate a .tar with the .o files
                    # built.export_library(f"{save_temps_dir}/{packed_lib_path}.tar")
            if self.print_assembly:
                with tempfile.TemporaryDirectory() as tdir:
                    tmpname = f"{tdir}/built"
                    fname = f"{packed_func_name}_compute_"
                    self._export_library(built, tmpname, type="shlib")
                    ext = ".dylib" if sys.platform == "darwin" else ".so"
                    disassembly = disassemble(
                        f"{tmpname}{ext}",
                        function=fname,
                        section=".text",
                        color=self.color,
                    )
                    print(disassembly, flush=True)
            self._export_library(built, packed_lib_path, type=type)

        csrcs, shlibs, arlibs, headers = [], [], [], []
        tvm_prefix = Path(tvm.__path__[0])
        if self.bare_ptr:
            wrapper = PackedOperatorWrapper(
                op,
                func_name,
                packed_func_name,
                cc_prefix=self._cc_prefix(),
            )
            if type == "shlib":
                ext = ".dylib" if sys.platform == "darwin" else ".so"
                wrapper.build(lib_path, packed_lib_path, type=type)
                shlibs = [f"{packed_lib_path}{ext}"]
            elif type == "arlib":
                wrapper.build(lib_path, packed_lib_path, type=type)
                arlibs = [f"{packed_lib_path}.a"]
            if self.emit_c:
                wrapper.build(emit_c_base, emit_c_packed_base, type="csrc")
                csrcs = [f"{emit_c_packed_base}.c"]
        if Path(f"{lib_path}.h").exists():
            headers = [f"{lib_path}.h"]
        if type in ["shlib", "arlib"]:
            ext = ".dylib" if sys.platform == "darwin" else ".so"
            shlibs += [f"{tvm_prefix}/libtvm_runtime{ext}"]
            # As of now shared_lib/ar_lib takes priority over emit_c
            return HostModule(
                dump_base,
                func_name,
                f"{lib_path}{ext}" if type == "shlib" else f"{lib_path}.a",
                type,
                shlibs=shlibs,
                arlibs=arlibs,
                headers=headers,
                bare_ptr=self.bare_ptr,
                graph=self._backend._graph,
            )
        assert type == "csrc"
        headers_path = [
            str(path)
            for path in [
                tvm_prefix / "include",
                tvm_prefix / "3rdparty" / "dlpack" / "include",
            ]
        ]
        return HostModule(
            dump_base,
            func_name,
            f"{lib_path}.c",
            "csrc",
            bare_ptr=self.bare_ptr,
            csrcs=csrcs,
            headers=headers,
            headers_path=headers_path,
            graph=self._backend._graph,
        )

    def _print(self, *content: Any) -> None:
        print(*content, flush=True, file=self.print_file)

    def _build(
        self,
        sch: TVMScheduledExpr,
        func_name: str | None = None,
    ) -> Any:
        op = sch.schedulable.expr
        if func_name is None:
            func_name = op.name
        return self._tvm_build_crt(
            sch,
            cname=func_name,
            target=self.tvm_tgt,
        )

    def _build_c(
        self,
        sch: TVMScheduledExpr,
        func_name: str | None = None,
        fname: str | None = None,
    ) -> None:
        if func_name is None:
            func_name = sch.schedulable.expr.name
        if fname is None:
            fname = func_name
        self._tvm_emit_c(sch, self.tvm_tgt, func_name, fname)

    @classmethod
    def _get_tvm_target_options(cls, target: str, arch: str) -> str:
        """
        Returm the tvm target options given the target and arch
        """
        if target == "native":
            assert arch in ["native", ""]
            return cls._get_tvm_native_target_options()
        else:
            assert arch != "native", f"can't pass native arch for non native target"
        tvm_cpu = ""
        tvm_attrs = ""
        tvm_triple = target_triple(target)
        if target in ["x86_64"]:
            if arch == "avx512":
                tvm_cpu = "skylake-avx512"
            elif arch == "avx2":
                tvm_cpu = "core-avx2"
        elif target in ["aarch64"]:
            if arch == "neon":
                tvm_cpu = "cortex-a72"
                tvm_attrs = "+neon"
        target_options = []
        if tvm_triple:
            target_options.append(f"-mtriple={tvm_triple}")
        if tvm_cpu:
            target_options.append(f"-mcpu={tvm_cpu}")
        if tvm_attrs:
            target_options.append(f"-mattr={tvm_attrs}")
        return " ".join(target_options)

    @classmethod
    def _get_tvm_native_target_options(cls) -> str:
        """
        Returm the tvm target options to pass to llvm.
        """
        from cpuinfo import get_cpu_info

        info = get_cpu_info()
        arch = info["arch_string_raw"]
        flags = info.get("flags", [])
        triple = target_triple(arch)
        cpu, attrs = "", ""
        if arch == "x86_64":
            if "avx512f" in flags:
                cpu = "skylake-avx512"
            elif "avx2" in flags:
                cpu = "core-avx2"
        elif arch == "aarch64":
            if "asimd" in flags:
                cpu = "cortex-a72"
                attrs = "+neon"
        target_options = []
        if triple:
            target_options.append(f"-mtriple={triple}")
        if cpu:
            target_options.append(f"-mcpu={cpu}")
        if attrs:
            target_options.append(f"-mattr={attrs}")
        return " ".join(target_options)

    @classmethod
    def _tvm_build_crt_args(cls, target: str) -> dict[str, Any]:
        # We use system-lib with crt runtime such that DSO loading works
        # The generated .so can then be used:
        # - for static compilation as soon as the tvm runtime is provided
        # - for dynamic loading from python
        # Recent version of tvm (i.e. 0.19) have a Runtime object
        # Older version (i.e. 0.16) support passing runtime options in target
        try:
            from tvm.relay.backend import Runtime

            runtime_kwargs = {
                "runtime": Runtime("crt", {"system-lib": True}),
            }
        except:
            runtime_kwargs = {}
            if TVM_VERSION < Version("0.21"):
                target = f"{target} --system-lib --runtime=c"

        return {
            "target": target,
            **runtime_kwargs,
        }

    @classmethod
    def _tvm_build_crt(cls, sch: TVMScheduledExpr, target: str, cname: str) -> Any:
        build_kwargs = cls._tvm_build_crt_args(target)
        config = {}
        if target.startswith("c "):
            config.update(
                {
                    "tir.disable_vectorize": True,
                }
            )
        with tvm.transform.PassContext(opt_level=3, config=config):
            if isinstance(sch, TVMScheduledExprTE):
                tensors = sch.schedulable._params
                built = tvm.build(sch._schedule, tensors, name=cname, **build_kwargs)  # type: ignore
            else:
                assert isinstance(sch, TVMScheduledExprTIR)
                func = sch._schedule.mod[sch.schedulable.expr.name]
                func = func.with_attr("global_symbol", cname)
                mod = tvm.IRModule({cname: func})
                built = tvm.build(mod, **build_kwargs)
        return built

    @classmethod
    def _tvm_emit_c(
        cls,
        sch: TVMScheduledExpr,
        target: str,
        cname: str,
        fname: str,
    ) -> Any:
        # Ignore initial target as of now and generate target agnostic C
        target = "c -keys=arch -march=generic -mcpu=generic"
        built = cls._tvm_build_crt(sch, target, cname)
        out_dir = Path(fname).parent
        out_base = Path(fname).stem
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            tar_file = tmp_dir_path / f"{cname}.tar"
            built.export_library(tar_file)
            with TarFile.open(tar_file) as tf:
                members = [info for info in tf.getmembers() if info.name.endswith(".c")]
                tf.extractall(tmp_dir, members=members, filter="data")
            out_dir.mkdir(parents=True, exist_ok=True)
            cfile = tmp_dir_path / "lib1.c"
            if not cfile.exists():
                cfile = tmp_dir_path / "lib0.c"
            shutil.copy(cfile, out_dir / f"{out_base}.c")

    def _cc_prefix(self) -> str:
        map = {
            "x86_64": "x86_64-linux-gnu-",
            "aarch64": "aarch64-linux-gnu-",
            "native": "",
        }
        assert self.target in map, (
            f"unsupported target for cross compilation: {self.target}"
        )
        return map[self.target]

    def _export_library(self, mod: Any, basename: str, type: str):
        from tvm.contrib import cc

        prefix = self._cc_prefix()
        if type == "arlib":
            fcompile = partial(cc.create_staticlib, ar=f"{prefix}ar")
            mod.export_library(f"{basename}.a", fcompile=fcompile)
            assert Path(f"{basename}.a").exists()
        else:
            fcompile = None
            if prefix != "":
                fcompile = cc.cross_compiler(f"{prefix}g++")
            assert type == "shlib"
            ext = ".dylib" if sys.platform == "darwin" else ".so"
            mod.export_library(f"{basename}{ext}", fcompile=fcompile)


class PackedOperatorWrapper:
    TEMPLATES_DIR = Path(__file__).parents[2] / "templates" / "tvm"

    def __init__(
        self,
        operation: TVMBaseExpr,
        func_name: str,
        packed_func_name: str,
        cc_prefix: str = "",
    ) -> None:
        self.operation = operation
        self.func_name = func_name
        self.packed_func_name = packed_func_name
        self._cc_prefix = cc_prefix

    def generate_c(self, output_base: str) -> None:
        config = {
            "inputs": self.operation.np_inputs_spec(),
            "outputs": self.operation.np_outputs_spec(),
            "func_name": self.func_name,
            "packed_func_name": self.packed_func_name,
        }
        jinja_generate_file(
            f"{output_base}.c",
            str(self.TEMPLATES_DIR / "packed_op_wrapper.c.jinja"),
            **config,
        )
        jinja_generate_file(
            f"{output_base}.h",
            str(self.TEMPLATES_DIR / "unpacked_op.h.jinja"),
            **config,
        )

    def build(self, lib_fname: str, packed_lib_fname: str, type: str) -> None:
        unpacked_lib_dir = Path(lib_fname).parent
        unpacked_lib_base = Path(lib_fname).stem
        packed_lib_dir = Path(packed_lib_fname).parent
        packed_lib_name = Path(packed_lib_fname).stem
        assert packed_lib_dir == unpacked_lib_dir, (
            f"must generate wrapper at the same location as packed lib"
        )
        with tempfile.TemporaryDirectory() as tdir:
            output_base = str(Path(tdir) / Path(lib_fname).stem)
            self.generate_c(output_base)
            shutil.copy(f"{output_base}.h", unpacked_lib_dir / f"{unpacked_lib_base}.h")
            if type == "csrc":
                shutil.copy(
                    f"{output_base}.c", unpacked_lib_dir / f"{unpacked_lib_base}.c"
                )
            elif type == "shlib":
                ext = ".dylib" if sys.platform == "darwin" else ".so"
                cmd = (
                    f"{self._cc_prefix}gcc --shared -fPIC -O2 {output_base}.c "
                    f"-o {unpacked_lib_base}{ext} "
                    f"{packed_lib_name}{ext} -Wl,--rpath,$ORIGIN"
                )
                p = subprocess.run(
                    shlex.split(cmd),
                    text=True,
                    capture_output=True,
                    cwd=unpacked_lib_dir,
                )
                if p.returncode != 0:
                    raise RuntimeError(
                        f"Failed command {cmd}:\n{p.stdout}\n{p.stderr}\n"
                    )
            elif type == "arlib":
                cmd = (
                    f"{self._cc_prefix}gcc -c -O2 {output_base}.c "
                    f"-o {unpacked_lib_base}.o"
                )
                p = subprocess.run(
                    shlex.split(cmd),
                    text=True,
                    capture_output=True,
                    cwd=unpacked_lib_dir,
                )
                if p.returncode != 0:
                    raise RuntimeError(
                        f"Failed command {cmd}:\n{p.stdout}\n{p.stderr}\n"
                    )
                cmd = (
                    f"{self._cc_prefix}ar -crs {unpacked_lib_base}.a "
                    f"{unpacked_lib_base}.o"
                )
                p = subprocess.run(
                    shlex.split(cmd),
                    text=True,
                    capture_output=True,
                    cwd=unpacked_lib_dir,
                )
                if p.returncode != 0:
                    raise RuntimeError(
                        f"Failed command {cmd}:\n{p.stdout}\n{p.stderr}\n"
                    )
