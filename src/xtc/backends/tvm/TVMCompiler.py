#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, cast
from typing_extensions import override
import tempfile
from pathlib import Path
import subprocess
import shlex

from xtc.targets.host import HostModule

import xtc.backends.tvm as backend
import xtc.itf as itf

from xtc.utils.host_tools import disassemble

from .TVMOps import TVMBaseExpr

__all__ = [
    "TVMCompiler",
]


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
        self.color = kwargs.get("color", False)
        self.shared_lib = kwargs.get("shared_lib", False)
        self.executable = kwargs.get("executable", False)
        self.emit_c = kwargs.get("emit_c", False)
        assert not self.executable, f"executable generation not supported yet for TVM"
        assert self.shared_lib, f"shared_lib generation is mandatory for TVM"

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

        dump_base = Path(self.dump_file).name
        lib_path = self.dump_file
        packed_lib_path = f"{lib_path}_packed" if self.bare_ptr else lib_path
        operation = op.generate()
        if self.print_source_ir or self.save_temps:
            sch = op.schedule(operation)
            lowered = str(op.lower(operation, sch))
            if self.print_source_ir:
                print(lowered, flush=True)
            save_temp(f"{dump_base}.initial.txt", lowered)
        schedule = cast(backend.TVMSchedule, schedule)
        save_temp(f"{dump_base}.sched.txt", str(schedule))
        if self.print_transformed_ir:
            print(schedule, flush=True)
        sch = op.schedule(operation, schedule)
        if self.print_transformed_ir or self.save_temps:
            lowered = str(op.lower(operation, sch))
            if self.print_transformed_ir:
                print(lowered, flush=True)
            save_temp(f"{dump_base}.scheduled.txt", lowered)
        if self.emit_c:
            op.emit_c(
                operation, sch, func_name=packed_func_name, dir=f"{lib_path}.export_c"
            )
        built = op.build(operation, sch, func_name=packed_func_name)
        if self.save_temps:
            for idx, mod in enumerate(built._collect_dso_modules()):
                llvm_ir = str(mod.get_source("ll"))
                save_temp(f"{dump_base}.lib{idx}.ll", llvm_ir)
            # This will generate a .tar with the .o files
            # built.export_library(f"{save_temps_dir}/{packed_lib_path}.tar")
        if self.print_assembly:
            with tempfile.TemporaryDirectory() as tdir:
                soname = f"{tdir}/built.so"
                fname = f"{packed_func_name}_compute_"
                built.export_library(soname)
                disassembly = disassemble(
                    soname,
                    function=fname,
                    section=".text",
                    color=self.color,
                )
                print(disassembly, flush=True)
        built.export_library(f"{packed_lib_path}.so")
        if self.bare_ptr:
            wrapper = PackedOperatorWrapper(
                op, func_name, packed_func_name, f"{packed_lib_path}.so"
            )
            wrapper.build(lib_path)

        return HostModule(
            dump_base,
            func_name,
            f"{lib_path}.so",
            "shlib",
            bare_ptr=self.bare_ptr,
            graph=self._backend._graph,
        )


def jinja_generate_file(fname: str, template_fname: str, **kwargs: Any) -> None:
    from jinja2 import Environment, FileSystemLoader

    file_path = Path(fname)
    template_path = Path(template_fname)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    template = Environment(loader=FileSystemLoader(template_path.parent)).get_template(
        template_path.name
    )
    with open(file_path, mode="w", encoding="utf-8") as file:
        file.write(template.render(kwargs))


class PackedOperatorWrapper:
    TEMPLATES_DIR = Path(__file__).parents[2] / "templates" / "tvm"

    def __init__(
        self,
        operation: TVMBaseExpr,
        func_name: str,
        packed_func_name: str,
        packed_lib_fname: str,
    ) -> None:
        self.operation = operation
        self.func_name = func_name
        self.packed_func_name = packed_func_name
        self.packed_lib_path = Path(packed_lib_fname)

    def generate_c(self, output_c_fname: str) -> None:
        config = {
            "inputs": self.operation.np_inputs_spec(),
            "outputs": self.operation.np_outputs_spec(),
            "func_name": self.func_name,
            "packed_func_name": self.packed_func_name,
        }
        jinja_generate_file(
            output_c_fname,
            str(self.TEMPLATES_DIR / "packed_op_wrapper.c.jinja"),
            **config,
        )

    def build(self, lib_fname: str) -> None:
        output_dir = Path(lib_fname).parent
        output_so = f"{Path(lib_fname).stem}.so"
        packed_lib_dir = self.packed_lib_path.parent
        packed_lib_name = self.packed_lib_path.name
        assert packed_lib_dir == output_dir, (
            f"must generate wrapper at the same location as packed lib"
        )
        with tempfile.TemporaryDirectory() as tdir:
            output_c = f"{Path(tdir) / Path(lib_fname).stem}.c"
            self.generate_c(output_c)
            cmd = f"gcc --shared -fPIC -O2 {output_c} -o {output_so} {packed_lib_name} -Wl,-rpath,$ORIGIN"
            p = subprocess.run(
                shlex.split(cmd), text=True, capture_output=True, cwd=output_dir
            )
        if p.returncode != 0:
            raise RuntimeError(f"Failed command {cmd}:\n{p.stdout}\n{p.stderr}\n")
