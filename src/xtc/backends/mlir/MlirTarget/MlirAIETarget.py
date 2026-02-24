#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import Any, cast
import os
import sys
import tempfile
from pathlib import Path

from xtc.targets.accelerator.aie import AIEModule
import xtc.itf as itf
from xtc.itf.graph import Graph

from .MlirTarget import MlirTarget
from ..MlirConfig import MlirConfig
from ..MlirProgram import RawMlirProgram

from mlir.passmanager import PassManager
from mlir.ir import OpResult, FunctionType, Context
from mlir.dialects import transform, func

from mlir_sdist.extras.run_aie import compile_for_aie
from mlir_sdist.extras.config import init_config as init_sdist_config
from mlir_sdist.extras.run_aie import AIEModuleWrapper


__all__ = ["MlirAIETarget"]


class MlirAIETarget(MlirTarget):
    """Amd AIE Target

    This target implements the lowering and code generation
    for the Amd AIE architecture, using the Mlir-AIE backend.
    """

    def __init__(self, config: MlirConfig):
        super().__init__(config)
        # config.required_extensions.append("sdist")
        self._mlir_aie_backend = MlirAIEBackend(config)
        self._wrapper: AIEModuleWrapper | None = None
        init_sdist_config()

    @override
    def name(self) -> str:
        return "aie"

    @override
    def arch(self) -> str:
        return "aie2"

    @override
    def generate_code_for_target(
        self,
        mlir_program: RawMlirProgram,  # Will be modified in place
        **kwargs: Any,
    ) -> None:
        save_temp = self._save_temp
        save_temps_dir = self._config.save_temps_dir
        temp_dir = None
        dump_file = kwargs.get("dump_file", None)
        if dump_file is None:
            temp_dir = tempfile.mkdtemp()
            dump_file = f"{temp_dir}/payload"
        if self._config.save_temps:
            assert dump_file is not None, "TODO: save_temp requires dump_file"
            dump_tmp_dir = Path(save_temps_dir)
            os.makedirs(save_temps_dir, exist_ok=True)
        else:
            dump_tmp_dir = Path(dump_file).parent
        dump_base = Path(dump_file).name

        dump_tmp_file = f"{dump_tmp_dir}/{dump_base}"
        mlir_atrn_dump_file = f"{dump_base}.after_trn.mlir"
        mlir_baie_dump_file = f"{dump_base}.before_aie.mlir"

        # extract the function_type of the first function of the mlir_module
        mlir_module = mlir_program.mlir_module
        # try to get the first function op and its type
        function_type = None
        for op in mlir_module.body.operations:
            if isinstance(op, func.FuncOp):
                function_type = cast(func.FuncOp, op).type
                break
        if function_type is None:
            raise RuntimeError(
                "Failed to extract the function_type from the first function in the MLIR module."
            )
        assert isinstance(function_type, FunctionType)

        # Lower to MLIR with aie dialect
        save_temp(mlir_atrn_dump_file, mlir_program.mlir_module)
        self._mlir_to_aie_pass(mlir_program)

        # Run MLIR aie backend
        with open(mlir_baie_dump_file, "w") as outf:
            outf.write(str(mlir_program.mlir_module))

        self._wrapper = self._mlir_aie_backend.run_lowering(
            mlir_before_aie_dump_file=mlir_baie_dump_file,
            function_type=function_type,
            mlir_context=mlir_program.mlir_context,
        )

        # Remove intermediate files if needed
        if not self._config.save_temps:
            os.remove(mlir_baie_dump_file)

    @override
    def create_module(
        self,
        name: str,
        payload_name: str,
        file_name: str,
        file_type: str,
        graph: Graph | None = None,
        **kwargs: Any,
    ) -> itf.comp.Module:
        assert self._wrapper is not None, "wrapper must be set"
        return AIEModule(name, self._wrapper, payload_name, graph, **kwargs)

    @override
    def custom_vectorize(self) -> bool:
        return True

    @override
    def apply_custom_vectorize(self, handle: OpResult) -> None:
        transform.AnnotateOp(handle, "xtc.request_vectorization")

    def dump_ir(self, mlir_program: RawMlirProgram, title: str):
        print(f"// -----// {title} //----- //", file=sys.stderr)
        print(str(mlir_program.mlir_module), file=sys.stderr)

    def _mlir_to_aie_pass(self, mlir_program: RawMlirProgram):
        to_aie_pass = MlirProgramToMlirAIEPass(
            mlir_program=mlir_program,
        )
        to_aie_pass.run()
        if self._config.print_lowered_ir:
            self.dump_ir(mlir_program, "IR Dump After MLIR Opt")

    def _save_temp(self, fname: str, content: Any) -> None:
        if not self._config.save_temps:
            return
        os.makedirs(self._config.save_temps_dir, exist_ok=True)
        with open(f"{self._config.save_temps_dir}/{fname}", "w") as outf:
            outf.write(str(content))


class MlirProgramToMlirAIEPass:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
    ) -> None:
        self._mlir_program = mlir_program

    def _lowering_pipeline(self) -> list[str]:
        pipeline = [
            "cse",
            "sccp",
        ]
        if "sdist" in self._mlir_program.mlir_extensions:
            pipeline += [
                "convert-linalg-to-loops",
                "cse",
                "sdist-lower-distribution",
                "cse",
                "sdist-group-transfers",
                "cse",
                'convert-sdist-to-aie{device="NPU1Col1"}',  # TODO: make this configurable
                "cse",
                "convert-sdist-utils-to-aie",
                "cse",
                "canonicalize",
                "cse",
            ]
        return pipeline

    def run(self) -> None:
        self._mlir_program.mlir_context.allow_unregistered_dialects = True
        pm = PassManager(context=self._mlir_program.mlir_context)
        pm.enable_verifier(False)
        for opt in self._lowering_pipeline():
            pm.add(opt)  # type: ignore # no attribte add?
        pm.run(self._mlir_program.mlir_module.operation)
        print(self._mlir_program.mlir_module)
        self._mlir_program.mlir_context.allow_unregistered_dialects = False


class MlirAIEBackend:
    def __init__(self, config: MlirConfig):
        self._config = config
        try:
            import aie
        except ImportError:
            raise ImportError("aie is not installed but is required for aie target")
        self._aie_path = aie.__path__[0]

    def run_lowering(
        self,
        mlir_before_aie_dump_file: str,
        function_type: FunctionType,
        mlir_context: Context,
    ) -> AIEModuleWrapper:
        with mlir_context as ctx:
            # Compile and JIT MLIR module
            compiled = compile_for_aie(
                mlir_before_aie_dump_file, function_type, enable_trace=False
            )
            return compiled
