#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import Any, cast
import sys
import os
import tempfile
import shutil
from pathlib import Path

import xtc.backends.mlir as backend
import xtc.itf as itf

from xtc.backends.mlir.MlirProgram import MlirProgram, RawMlirProgram
from xtc.backends.mlir.MlirScheduler import MlirSchedule
from xtc.backends.mlir.MlirConfig import MlirConfig

from xtc.backends.mlir.MlirCompilerPasses import (
    MlirProgramInsertTransformPass,
    MlirProgramApplyTransformPass,
)

from xtc.backends.mlir.MlirTarget import (
    MlirTarget,
    get_target_from_name,
    get_default_target,
)
from xtc.utils.ext_tools import get_shlib_extension


class MlirCompiler(itf.comp.Compiler):
    def __init__(
        self,
        backend: "backend.MlirBackend",
        target: str | None = None,
        **kwargs: Any,
    ):
        self._backend = backend
        self.dump_file = kwargs.pop("dump_file", None)
        kwargs["bare_ptr"] = True  # Not supported for now
        kwargs["to_disassemble"] = kwargs.get("to_disassemble", backend.payload_name)
        self._config = MlirConfig(**kwargs)
        if target is None:
            self._target = get_default_target()(self._config)
        else:
            self._target = get_target_from_name(target)(self._config)
        assert self._target is not None
        self._compiler_kwargs = kwargs

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    @property
    def target(self) -> MlirTarget:
        return self._target

    @override
    def compile(
        self,
        schedule: itf.schd.Schedule,
    ) -> itf.comp.Module:
        shared_lib = self._config.shared_lib
        executable = self._config.executable
        temp_dir = None
        if self.dump_file is None:
            temp_dir = tempfile.mkdtemp()
            self.dump_file = f"{temp_dir}/{self._backend.payload_name}"
        program = self.generate_program()
        compiler = MlirProgramCompiler(
            mlir_program=program,
            mlir_schedule=cast(MlirSchedule, schedule),
            concluding_passes=self._backend.concluding_passes,
            always_vectorize=self._backend.always_vectorize,
            config=self._config,
            target=self._target,
            dump_file=self.dump_file,
        )
        assert compiler.dump_file is not None
        compiler.compile()
        io_specs_args = {}
        if self._backend._graph is None:
            # Pass backend defined inputs/outputs specs when not a Graph
            io_specs_args.update(
                {
                    "np_inputs_spec": self._backend.np_inputs_spec,
                    "np_outputs_spec": self._backend.np_outputs_spec,
                }
            )
        module = self._target.create_module(
            Path(compiler.dump_file).name,
            self._backend.payload_name,
            f"{compiler.dump_file}.{get_shlib_extension()}",
            "shlib",
            bare_ptr=self._config.bare_ptr,
            graph=self._backend._graph,
            **io_specs_args,
        )
        if temp_dir is not None:
            shutil.rmtree(temp_dir)
        return module

    def generate_program(self) -> RawMlirProgram:
        # xdsl_func input must be read only
        return MlirProgram(self._backend.xdsl_func, self._backend.no_alias)


class MlirProgramCompiler:
    def __init__(
        self,
        config: MlirConfig,
        target: MlirTarget,
        mlir_program: RawMlirProgram,
        mlir_schedule: MlirSchedule | None = None,
        **kwargs: Any,
    ):
        self._mlir_program = mlir_program
        self._mlir_schedule = mlir_schedule
        self._target = target
        self._config = config
        self.dump_file = kwargs.get("dump_file")

    def dump_ir(self, title: str):
        print(f"// -----// {title} //----- //", file=sys.stderr)
        print(str(self._mlir_program.mlir_module), file=sys.stderr)

    def mlir_insert_transform_pass(self) -> None:
        insert_transform_pass = MlirProgramInsertTransformPass(
            mlir_program=self._mlir_program,
            mlir_schedule=self._mlir_schedule,
            concluding_passes=self._config.concluding_passes,
            always_vectorize=self._config.always_vectorize,
            vectors_size=self._config.vectors_size,
        )
        insert_transform_pass.run()
        if self._config.print_source_ir:
            self.dump_ir("IR Dump Before transform")

    def mlir_apply_transform_pass(self) -> None:
        apply_transform_pass = MlirProgramApplyTransformPass(
            mlir_program=self._mlir_program,
        )
        apply_transform_pass.run()
        if self._config.print_transformed_ir:
            self.dump_ir("IR Dump After transform")

    def _save_temp(self, fname: str, content: Any) -> None:
        if not self._config.save_temps:
            return
        os.makedirs(self._config.save_temps_dir, exist_ok=True)
        with open(f"{self._config.save_temps_dir}/{fname}", "w") as outf:
            outf.write(str(content))

    def compile(self) -> None:
        save_temp = self._save_temp
        save_temps_dir = self._config.save_temps_dir
        dump_file = self.dump_file
        temp_dir = None
        if dump_file is None:
            temp_dir = tempfile.mkdtemp()
            dump_file = f"{temp_dir}/payload"
        if self._config.save_temps:
            assert self.dump_file is not None, "TODO: save_temp requires dump_file"
            dump_tmp_dir = Path(save_temps_dir)
            os.makedirs(save_temps_dir, exist_ok=True)
        else:
            dump_tmp_dir = Path(dump_file).parent

        dump_base = Path(dump_file).name
        dump_tmp_file = f"{dump_tmp_dir}/{dump_base}"
        ir_dump_file = f"{dump_tmp_file}.ir"
        bc_dump_file = f"{dump_tmp_file}.bc"
        obj_dump_file = f"{dump_tmp_file}.o"
        exe_c_file = f"{dump_tmp_file}.main.c"
        so_dump_file = f"{dump_file}.so"
        exe_dump_file = f"{dump_file}.out"
        src_ir_dump_file = f"{dump_base}.mlir"
        mlir_btrn_dump_file = f"{dump_base}.before_trn.mlir"
        mlir_atrn_dump_file = f"{dump_base}.after_trn.mlir"

        save_temp(src_ir_dump_file, self._mlir_program.mlir_module)

        self.mlir_insert_transform_pass()
        save_temp(mlir_btrn_dump_file, self._mlir_program.mlir_module)

        self.mlir_apply_transform_pass()
        save_temp(mlir_atrn_dump_file, self._mlir_program.mlir_module)

        self._target.generate_code_for_target(self._mlir_program, dump_file=dump_file)
