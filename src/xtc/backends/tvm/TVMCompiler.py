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
from xtc.utils.files import relative_to

from xtc.utils.host_tools import (
    disassemble,
    target_triple,
    cc_command,
    binutils_command,
)

from .TVMOpsCompiler import (
    TVMExprCompiler,
    TVMScheduledExpr,
    TVMScheduledExprTIR,
)
from .TVMOps import (
    TVMBaseExpr,
)

import tvm
import tvm_ffi


__all__ = [
    "TVMCompiler",
]


TVM_VERSION = Version(tvm.__version__.split("+", 1)[0])
assert TVM_VERSION >= Version("0.26")


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
        self.tvm_target = "llvm"
        self.tvm_target_options = self._get_tvm_target_options(self.target, self.arch)
        self.tvm_tgt = self._get_tvm_target(self.tvm_target, self.tvm_target_options)
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

    @override
    def get_source_ir(self, schedule: itf.schd.Schedule) -> str:
        # The initial lowered Tensor IR, before the schedule is applied.
        op = self._backend._tvm_base
        expr_compiler = TVMExprCompiler(op)
        return expr_compiler.generate().schedule().dumps()

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
        tvm_ffi_func_name = f"__tvm_ffi_{func_name}"
        compute_func_name = f"{func_name}_compute_"

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
            emit_c_base = f"{lib_path}_export_c"
        else:
            emit_c_base = lib_path
        packed_lib_path = f"{lib_path}_tvm_ffi"
        emit_c_packed_base = f"{emit_c_base}_tvm_ffi"
        expr_compiler = TVMExprCompiler(op)
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
                func_name=func_name,
                fname=emit_c_packed_base,
            )
        if type in ["shlib", "arlib"]:
            built = self._build(sch, func_name=func_name)
            if self.save_temps:
                for idx, mod in enumerate(built._collect_dso_modules()):
                    llvm_ir = str(mod.inspect_source("ll"))
                    save_temp(f"{dump_base}.lib{idx}.ll", llvm_ir)
                    # This will generate a .tar with the .o files
                    # built.export_library(f"{save_temps_dir}/{packed_lib_path}.tar")
            self._export_archive(built, f"{packed_lib_path}.a")

        wrapper = PackedOperatorWrapper(
            op,
            func_name,
            tvm_ffi_func_name,
            arch=self.target,
            bare_ptr=self.bare_ptr,
        )
        if type in ["shlib", "arlib"] and self.emit_c:
            wrapper.build(emit_c_base, emit_c_packed_base, type="csrc")
        module_file, module_args = wrapper.build(lib_path, packed_lib_path, type=type)
        assert Path(module_file).with_suffix("") == Path(lib_path)
        if type == "shlib" and self.print_assembly:
            disassembly = disassemble(
                module_file,
                function=compute_func_name,
                section="text",
                color=self.color,
                arch=self.target,
            )
            print(disassembly, flush=True)
        return HostModule(
            dump_base,
            func_name,
            module_file,
            type,
            **module_args,
            bare_ptr=self.bare_ptr,
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
    def _get_tvm_target(cls, kind: str, options: dict) -> dict:
        return {
            **dict(kind=kind),
            **options,
        }

    @classmethod
    def _get_tvm_target_options(cls, target: str, arch: str) -> dict:
        """
        Returm the tvm target options given the target and arch
        """
        if target == "native":
            assert arch in ["native", ""]
            return cls._get_tvm_native_target_options()
        else:
            assert arch != "native", f"can't pass native arch for non native target"
        tvm_cpu = ""
        tvm_attrs = []
        tvm_triple = target_triple(target)
        if target in ["x86_64"]:
            if arch == "avx512":
                tvm_cpu = "skylake-avx512"
            elif arch == "avx2":
                tvm_cpu = "core-avx2"
        elif target in ["aarch64"]:
            if arch == "neon":
                tvm_cpu = "cortex-a72"
                tvm_attrs += ["+neon"]
        return {
            **(dict(mtriple=tvm_triple) if tvm_triple else {}),
            **(dict(mcpu=tvm_cpu) if tvm_cpu else {}),
            **(dict(mattr=tvm_attrs) if tvm_attrs else {}),
        }

    @classmethod
    def _get_tvm_native_target_options(cls) -> dict:
        """
        Returm the tvm target options to pass to llvm.
        """
        from cpuinfo import get_cpu_info

        info = get_cpu_info()
        arch = info["arch_string_raw"]
        flags = info.get("flags", [])
        tvm_triple = target_triple(arch)
        tvm_cpu, tvm_attrs = "", []
        if arch == "x86_64":
            if "avx512f" in flags:
                tvm_cpu = "skylake-avx512"
            elif "avx2" in flags:
                tvm_cpu = "core-avx2"
        elif arch == "aarch64":
            if "asimd" in flags:
                tvm_cpu = "cortex-a72"
                tvm_attrs += ["+neon"]
        return {
            **(dict(mtriple=tvm_triple) if tvm_triple else {}),
            **(dict(mcpu=tvm_cpu) if tvm_cpu else {}),
            **(dict(mattr=tvm_attrs) if tvm_attrs else {}),
        }

    @classmethod
    def _tvm_build_crt_args(cls, target: dict) -> dict[str, Any]:
        # We use system-lib with crt runtime such that DSO loading works
        # As of TVM >= 0.26 this is not needed anymore
        # The generated .so can then be used:
        # - for static compilation as soon as the tvm runtime is provided
        # - for dynamic loading from python
        runtime_kwargs: dict[str, Any] = {}
        return {
            "target": target,
            **runtime_kwargs,
        }

    @classmethod
    def _tvm_build_crt(cls, sch: TVMScheduledExpr, target: dict, cname: str) -> Any:
        build_kwargs = cls._tvm_build_crt_args(target)
        config = {}
        if target["kind"] == "c":
            config.update(
                {
                    "tirx.disable_vectorize": True,
                }
            )
        with tvm.transform.PassContext(opt_level=3, config=config):
            assert isinstance(sch, TVMScheduledExprTIR)
            func = sch._schedule.mod[sch.schedulable.expr.name]
            func = func.with_attr("global_symbol", cname)
            mod = tvm.IRModule({cname: func})
            built = tvm.tirx.build(mod, **build_kwargs)
        return built

    @classmethod
    def _tvm_emit_c(
        cls,
        sch: TVMScheduledExpr,
        target: dict,
        cname: str,
        fname: str,
    ) -> Any:
        # Ignore initial target as of now and generate target agnostic C
        target = dict(kind="c", keys=["arch"], march="generic", mcpu="generic")
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

    def _export_archive(self, mod: Any, archive_path: str):
        from tvm.support import cc

        fcompile = partial(
            cc.create_staticlib,
            ar=binutils_command("ar", self.target),
        )
        mod.export_library(archive_path, fcompile=fcompile)
        assert Path(archive_path).exists()


class PackedOperatorWrapper:
    TEMPLATES_DIR = Path(__file__).parents[2] / "templates" / "tvm"

    def __init__(
        self,
        operation: TVMBaseExpr,
        func_name: str,
        packed_func_name: str,
        arch: str = "",
        bare_ptr: bool = False,
    ) -> None:
        self.operation = operation
        self.func_name = func_name
        self.packed_func_name = packed_func_name
        self._arch = arch
        self._bare_ptr = bare_ptr

    def generate_c(self, output_base: str, header: bool = False) -> None:
        config = {
            "inputs": self.operation.np_inputs_spec(),
            "outputs": self.operation.np_outputs_spec(),
            "func_name": self.func_name,
            "tvm_ffi_func_name": self.packed_func_name,
        }
        if self._bare_ptr:
            jinja_generate_file(
                f"{output_base}.c",
                str(self.TEMPLATES_DIR / "packed_op_wrapper.c.jinja"),
                **config,
            )
            if header:
                jinja_generate_file(
                    f"{output_base}.h",
                    str(self.TEMPLATES_DIR / "unpacked_op.h.jinja"),
                    **config,
                )
        else:
            jinja_generate_file(
                f"{output_base}.c",
                str(self.TEMPLATES_DIR / "tvm_ffi_op_wrapper.c.jinja"),
                **config,
            )

    def build(
        self, lib_fname: str, packed_lib_fname: str, type: str
    ) -> tuple[str, dict[str, Any]]:
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        unpacked_lib_dir = Path(lib_fname).parent
        unpacked_lib_base = Path(lib_fname).stem
        packed_lib_dir = Path(packed_lib_fname).parent
        packed_lib_name = Path(packed_lib_fname).stem
        packed_ar_name = f"{packed_lib_fname}.a"
        assert packed_lib_dir == unpacked_lib_dir, (
            f"must generate wrapper at the same location as packed lib"
        )
        if type in ["shlib", "arlib"]:
            assert Path(packed_ar_name).exists()
        tdir = tempfile.mkdtemp(dir=unpacked_lib_dir)
        try:
            tvm_ffi_prefix = Path(tvm_ffi.__path__[0])
            tvm_ffi_libdir = tvm_ffi_prefix / "lib"
            tvm_prefix = Path(tvm.__path__[0])
            tvm_libdir = tvm_prefix / "lib"
            output_base = str(Path(tdir) / Path(lib_fname).stem)
            host_runtime_dir = Path(__file__).parents[2] / "csrcs" / "runtimes" / "host"
            tvm_runtime_init_c = str(host_runtime_dir / "tvm_runtime_init.c")
            self.generate_c(output_base, header=(type == "csrc"))
            headers, headers_path, csrcs, shlibs, arlibs = [], [], [], [], []
            if self._bare_ptr and type == "csrc":
                shutil.copy(
                    f"{output_base}.h", unpacked_lib_dir / f"{unpacked_lib_base}.h"
                )
                headers += [f"{lib_fname}.h"]
            if type == "csrc":
                shutil.copy(
                    f"{output_base}.c", unpacked_lib_dir / f"{unpacked_lib_base}.c"
                )
                module_file = f"{lib_fname}.c"
                csrcs += [
                    tvm_runtime_init_c,
                    f"{packed_lib_fname}.c",
                ]
                headers_path += [
                    str(path)
                    for path in [
                        tvm_prefix / "include",
                        tvm_ffi_prefix / "include",
                    ]
                ]
                shlibs += [
                    f"{tvm_libdir}/libtvm_runtime{ext}",
                    f"{tvm_ffi_libdir}/libtvm_ffi{ext}",
                ]
            elif type == "shlib":
                output_dir = unpacked_lib_dir
                object_fnames = [
                    str(relative_to(fname, output_dir))
                    for fname in self._build_objects(
                        [f"{output_base}.c"] + self._runtime_sources(),
                        tdir,
                    )
                ]
                opts = "-O2"
                sh_opts = "--shared -fPIC"
                ext = ".so"
                if sys.platform == "darwin":
                    sh_opts += " -undefined dynamic_lookup"
                    ext = ".dylib"
                shlib_fname = f"{unpacked_lib_base}{ext}"
                shlib_dest = str(relative_to(shlib_fname, output_dir))
                cmd = (
                    f"{cc_command(self._arch)} {sh_opts} {opts} "
                    f"{' '.join(object_fnames)}  "
                    f"{relative_to(packed_lib_fname, output_dir)}.a "
                    f"-o {unpacked_lib_base}{ext}"
                )
                p = subprocess.run(
                    shlex.split(cmd),
                    text=True,
                    capture_output=True,
                    cwd=output_dir,
                )
                if p.returncode != 0:
                    raise RuntimeError(
                        f"Failed command {cmd} (cwd: {output_dir}:\n"
                        f"{p.stdout}\n"
                        f"{p.stderr}\n"
                    )
                module_file = f"{lib_fname}{ext}"
                shlibs += [
                    f"{tvm_libdir}/libtvm_runtime{ext}",
                    f"{tvm_ffi_libdir}/libtvm_ffi{ext}",
                ]
            else:
                assert type == "arlib"
                archive_fname = self._build_archive(
                    [f"{output_base}.c"] + self._runtime_sources(),
                    f"{lib_fname}.a",
                )
                module_file = archive_fname
                arlibs += [f"{packed_lib_fname}.a"]
                shlibs += [
                    f"{tvm_libdir}/libtvm_runtime{ext}",
                    f"{tvm_ffi_libdir}/libtvm_ffi{ext}",
                ]
        except Exception:
            raise
        else:
            shutil.rmtree(tdir)
        module_args = {
            **(dict(headers=headers) if headers else {}),
            **(dict(headers_path=headers_path) if headers_path else {}),
            **(dict(csrcs=csrcs) if csrcs else {}),
            **(dict(shlibs=shlibs) if shlibs else {}),
            **(dict(arlibs=arlibs) if arlibs else {}),
        }
        return module_file, module_args

    def _runtime_sources(self) -> list[str]:
        host_runtime_dir = Path(__file__).parents[2] / "csrcs" / "runtimes" / "host"
        tvm_runtime_init_c = str(host_runtime_dir / "tvm_runtime_init.c")
        return [tvm_runtime_init_c]

    def _build_object(self, source_fname: str, object_fname: str) -> str:
        assert object_fname.endswith(".o")
        opts = "-O2"
        pic_opts = "-fPIC"
        output_dir = Path(object_fname).parent
        object_dest = str(relative_to(object_fname, output_dir))
        source_inp = str(relative_to(source_fname, output_dir))
        cmd = (
            f"{cc_command(self._arch)} -c {opts} {pic_opts} "
            f"{source_inp} "
            f"-o {object_dest}"
        )
        p = subprocess.run(
            shlex.split(cmd),
            text=True,
            capture_output=True,
            cwd=output_dir,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"Failed command {cmd} (cwd: {output_dir} :\n{p.stdout}\n{p.stderr}\n"
            )
        return object_fname

    def _build_objects(self, source_fnames: list[str], output_dir: str) -> list[str]:
        return [
            self._build_object(fname, str(Path(output_dir) / f"{Path(fname).stem}.o"))
            for fname in source_fnames
        ]

    def _build_archive(self, source_fnames: list[str], archive_fname: str) -> str:
        assert archive_fname.endswith(".a")
        output_dir = Path(archive_fname).parent
        archive_dest = str(relative_to(archive_fname, output_dir))
        tdir = tempfile.mkdtemp(dir=output_dir)
        try:
            object_fnames = [
                str(relative_to(fname, output_dir))
                for fname in self._build_objects(source_fnames, tdir)
            ]
            cmd = (
                f"{binutils_command('ar', self._arch)} -crs {archive_dest} "
                f"{' '.join(object_fnames)} "
            )
            p = subprocess.run(
                shlex.split(cmd),
                text=True,
                capture_output=True,
                cwd=output_dir,
            )
            if p.returncode != 0:
                raise RuntimeError(
                    f"Failed command {cmd} (cwd: {output_dir}:\n"
                    f"{p.stdout}\n"
                    f"{p.stderr}\n"
                )
        except Exception:
            raise
        else:
            shutil.rmtree(tdir)
        return archive_fname
