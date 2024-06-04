#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os
import numpy as np
from typing import Any

import utils
from evaluator import Evaluator, Executor
from ndarray import NDArray

from JIROps import Operation, Operators

from xdsl.printer import Printer
from xdsl.dialects.builtin import ModuleOp, StringAttr
from jir.util import get_host_target_triple
from jir.parser import JIRParser, JIRFormatter
from jir.transform.util.index import JIRFunctionDimensionIndex
from jir.context import JIRFunctionContext
from jir.backend.target import JIRBackendTargetProperties
from jir.backend.xdsl.translator import JIR2XDSLFunctionTranslator
from jir.backend.xdsl.computation import JIRComputationFunctionCallProviderForXDSL
from jir.backend.xdsl.benchmark import jir2xdsl_generate_benchmark_routine
from jir.backend.xdsl.compiler import (
    MLIRLowering,
    MLIR2LLVMConversion,
    LLVMSharedLibraryCompiler,
    PolygeistCompiler,
)
from jir.backend.util.merge_mlir_modules import merge_mlir_modules_by_content
from jir.transform.primitives.canonicalize import canonicalize
from jir.transform.command import (
    JIRTransformCommand,
    JIRWritebackBufferCommandClass,
    JIRDistributeCommandClass,
    JIRFuseCommandClass,
    JIRInterchangeCommandClass,
    JIRSplitLoopIterationDimensionCommandClass,
    JIRSubdimCommandClass,
    JIRTileCommandClass,
    JIRUpdateLoopPropsCommandClass,
    JIRWrapLoopCommandClass,
    JIRDropLoopCommandClass,
    JIRComplementaryCommandClass,
    JIRCanonicalizeCommandClass,
)


__all__ = [
    "Implementer",
]

COMMANDS = [
    JIRWritebackBufferCommandClass,
    JIRComplementaryCommandClass,
    JIRDistributeCommandClass,
    JIRFuseCommandClass,
    JIRInterchangeCommandClass,
    JIRSplitLoopIterationDimensionCommandClass,
    JIRSubdimCommandClass,
    JIRTileCommandClass,
    JIRUpdateLoopPropsCommandClass,
    JIRWrapLoopCommandClass,
    JIRDropLoopCommandClass,
    JIRCanonicalizeCommandClass,
]

COMMAND_INDEX = {cmd.command: cmd for cmd in COMMANDS}


class TransformAdaptor:
    def __init__(
        self,
        source_op: Operation,
        dims: dict[str, int],
    ) -> None:
        self.source_op = source_op
        self.dims = dims
        self.axes_map = {
            k: v for k, v in zip(dims.keys(), source_op.args_names[: len(self.dims)])
        }
        self.reset()

    def reset(self) -> None:
        self.tiled = {}
        self.vectorized = []
        self.parallelized = []
        self.unrolled = {}
        self.order = []

    def tile(self, axis: str, tiles: dict[str, int]) -> None:
        self.tiled[axis] = tiles

    def vectorize(self, axes: list[str]) -> None:
        self.vectorized = axes

    def parallelize(self, axes: list[str]) -> None:
        self.parallelized = axes

    def unroll(self, axes_unroll: dict[str, int]) -> None:
        self.unrolled = axes_unroll

    def interchange(self, axes_order: list[str]) -> None:
        self.order = axes_order

    def _generate_tiles_cmds(self) -> list[str]:
        if not self.tiled:
            return []
        dims = self._get_tiles_dims()
        cmds = []
        for axis, tiles in self.tiled.items():
            dim_names = [f"{axis}{idx}" for idx in range(1 + len(tiles.keys()))]
            assert len(dim_names) == 2, f"for now only 1 level tiling supported"
            if dims[dim_names[-1]] == 1:
                continue
            names = [self.axes_map[axis]] + list(tiles.keys())
            subs = " ".join(dim_names)
            cmds.extend(
                [
                    f"subdim parent={self.axes_map[axis]} sub=[{subs}]",
                    f"compl dim={dim_names[0]} other={dim_names[1]}",
                ]
            )
            for idx in range(len(names) - 1):
                tile_cmd = f"tile target={names[idx]} tile={dim_names[idx + 1]} inner={names[idx + 1]}"
                cmds.append(tile_cmd)
        return cmds

    def _get_tiles_dims(self) -> dict[str, int]:
        tiles_dims = {f"{ax}": size for ax, size in self.dims.items()}
        for axis, tiles in self.tiled.items():
            for tile, size in tiles.items():
                tiles_dims[tile] = size
        return tiles_dims

    def _get_transform_dims(self) -> dict[str, int]:
        tiles_dims = {}
        for axis, tiles in self.tiled.items():
            dim = self.dims[axis]
            last = f"{axis}0"
            assert len(tiles.items()) == 1, f"for now only 1 level tiling supported"
            for tile, size in tiles.items():
                if size == 1:
                    break
                assert dim % size == 0
                dim = dim // size
                tiles_dims[last] = dim
                tiles_dims[tile] = size
                last = tile
        return tiles_dims

    def _generate_vector_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        cmds = [
            f"update_props target={self.axes_map.get(axis, axis)} vector={dims[axis]}"
            for axis in self.vectorized
            if dims[axis] != 1
        ]
        return cmds

    def _generate_unroll_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        cmds = [
            f"update_props target={self.axes_map.get(axis, axis)} unroll={size}"
            for axis, size in self.unrolled.items()
            if dims[axis] != 1
        ]
        return cmds

    def _generate_parallel_cmds(self) -> list[str]:
        dims = self._get_tiles_dims()
        cmds = [
            f"update_props target={self.axes_map.get(axis, axis)} parallel"
            for axis in self.parallelized
            if dims[axis] != 1
        ]
        return cmds

    def _generate_interchange_cmds(self) -> list[str]:
        def generate_inter(current, order):
            inter = []
            assert len(current) == len(order), f"len mismatch {current} and {order}"
            for idx in range(len(order)):
                tgt_idx = current.index(order[idx])
                while tgt_idx != idx:
                    inter.append(current[tgt_idx - 1])
                    current[tgt_idx] = current[tgt_idx - 1]
                    current[tgt_idx - 1] = order[idx]
                    tgt_idx -= 1
            assert current == order
            return inter

        if not self.order:
            return []
        current_order = list(self.dims.keys())
        for axis, tiles in self.tiled.items():
            idx = current_order.index(axis)
            for tile in tiles.keys():
                idx += 1
                current_order.insert(idx, tile)
        dims = self._get_tiles_dims()
        current_order = [axis for axis in current_order if dims[axis] != 1]
        target_order = [axis for axis in self.order if dims[axis] != 1]
        inter = generate_inter(current_order, target_order)
        cmds = [f"interchange target={self.axes_map.get(axis, axis)}" for axis in inter]
        return cmds

    def generate_transform(self) -> tuple[str, dict[str, int]]:
        cmds = []
        transform_dims = self._get_transform_dims()
        tiles_cmds = self._generate_tiles_cmds()
        interchange_cmds = self._generate_interchange_cmds()
        vector_cmds = self._generate_vector_cmds()
        unroll_cmds = self._generate_unroll_cmds()
        parallel_cmds = self._generate_parallel_cmds()
        cmds = [
            *tiles_cmds,
            *interchange_cmds,
            *vector_cmds,
            *unroll_cmds,
            *parallel_cmds,
        ]
        if cmds:
            cmds.append("canonicalize")
        return cmds, transform_dims


class Implementer:
    def __init__(
        self,
        source_op: Operation,
        dims: dict[str, int],
        jir_install_dir: str,
        geist_install_dir: str,
        **kwargs,
    ) -> None:
        self.source_op = source_op
        self.args = self.source_op.args
        self.jir_install_dir = jir_install_dir
        self.geist_install_dir = geist_install_dir
        self.dims = dims
        self.jir_dims = {
            k: v
            for k, v in zip(
                source_op.args_names[: len(dims)], source_op.args[: len(dims)]
            )
        }
        self._op_function_mlir = None
        self._transformed_jir_function = None
        self._transformed_jir_dims = None
        self.op_function, self.jir_function, self.payload_name = (
            self.source_op.generate()
        )
        self.transformer = TransformAdaptor(source_op, self.dims)
        self._target_triple = kwargs.get("target_triple") or get_host_target_triple(
            self.jir_install_dir
        )
        self._target_arch = kwargs.get("target_arch") or "native"
        self._save_temps = kwargs.get("save_temps", False)
        self._save_temps_dir = kwargs.get("save_temps_dir") or "./save_temps_dir"

    def _get_op_function_mlir(self) -> str:
        if self._op_function_mlir is not None:
            return self._op_function_mlir
        polygeist_compiler = PolygeistCompiler(f"{self.geist_install_dir}/bin/cgeist")
        self._op_function_mlir = polygeist_compiler(self.op_function)
        return self._op_function_mlir

    def _generate_module_for(self, ctx: JIRFunctionContext) -> ModuleOp:
        computations = JIRComputationFunctionCallProviderForXDSL()
        function_translator = JIR2XDSLFunctionTranslator(
            computations, JIRBackendTargetProperties(vector_size=4)
        )
        fn = function_translator(ctx.function, function_ctx=ctx)
        module_attr = dict()
        module_attr["llvm.target_triple"] = StringAttr(self._target_triple)
        return ModuleOp(
            [fn, *computations.function_declarations], attributes=module_attr
        )

    def _save_temp(self, fname: str, content: str) -> None:
        if not self._save_temps:
            return
        os.makedirs(self._save_temps_dir, exist_ok=True)
        with open(f"{self._save_temps_dir}/{fname}", "w") as outf:
            outf.write(content)

    def default_transform(self) -> None:
        transform_cmds, transform_dims = self.transformer.generate_transform()
        transform_sequence = "".join([f"{t};\n" for t in transform_cmds])
        self.transform(transform_sequence, transform_dims)

    def transform(
        self, transform_sequence: str, transform_dims: dict[str, int]
    ) -> None:
        self._save_temp("implementer.jir", self.jir_function)
        self._save_temp("implementer.tjir", transform_sequence)
        dims = {**self.jir_dims, **transform_dims}
        self._save_temp(
            "implementer.dims",
            "".join([f"{dim}={size}\n" for dim, size in dims.items()]),
        )
        parser = JIRParser()
        formatter = JIRFormatter()
        function = parser(self.jir_function)
        transform_sequence = parser.parse_transform_sequence(transform_sequence)
        for cmd in transform_sequence:
            if cmd.command not in COMMAND_INDEX:
                raise RuntimeError(f"Unknown command {cmd.command}")
            function = COMMAND_INDEX[cmd.command].run(cmd, function)
        function = canonicalize(function)
        self._save_temp("implementer.transformed.jir", str(function))
        self._transformed_jir_function = function
        self._transformed_jir_dims = dims

    def reset_schedule(self) -> None:
        self._transformed_jir_function = None
        self._transformed_jir_dims = None
        self.transformer.reset()

    def tile(self, axis: str, tiles: dict[str, int]) -> None:
        self.transformer.tile(axis, tiles)

    def vectorize(self, axes: list[str]) -> None:
        self.transformer.vectorize(axes)

    def parallelize(self, axes: list[str]) -> None:
        self.transformer.parallelize(axes)

    def unroll(self, axes_unroll: dict[str, int]) -> None:
        self.transformer.unroll(axes_unroll)

    def interchange(self, axes_order: list[str]) -> None:
        self.transformer.interchange(axes_order)

    def compile_jir_module(self) -> Any:
        if self._transformed_jir_function is None:
            self.default_transform()
        fn = self._transformed_jir_function
        dims = self._transformed_jir_dims
        index = JIRFunctionDimensionIndex()
        ctx = JIRFunctionContext(fn)
        index(fn)
        for dimension, size in self._transformed_jir_dims.items():
            ctx.define_dimension(index[dimension], int(size))
        if not ctx.well_defined:
            raise RuntimeError("Some ctx dimensions are missing")
        module = self._generate_module_for(ctx)
        return module

    def compile(
        self, dump_file=None, debug=False, shared_lib=False, executable=False, **kwargs
    ) -> None:
        assert not executable, "TODO: executable output not implemented"
        assert shared_lib, "TODO: shared_lib mandatory"
        module = self.compile_jir_module()
        mlir_lowering = MLIRLowering(f"{self.jir_install_dir}/bin/mlir-opt")
        mlir2llvm = MLIR2LLVMConversion(f"{self.jir_install_dir}/bin/mlir-translate")
        llvm_compiler = LLVMSharedLibraryCompiler(
            f"{self.jir_install_dir}/bin/clang",
            f"{self.jir_install_dir}/lib",
            self._target_triple,
            self._target_arch,
        )
        computation_primitives = self._get_op_function_mlir()
        computation_module = str(
            merge_mlir_modules_by_content(str(module), computation_primitives)
        )
        self._save_temp("implementer.merged.mlir", computation_module)
        lowered_computation_module = mlir_lowering(computation_module)
        self._save_temp("implementer.lowered.mlir", computation_module)
        llvm_computation_module = mlir2llvm(lowered_computation_module)
        self._save_temp("implementer.lowered.ll", computation_module)
        compiled_computation_module = llvm_compiler(llvm_computation_module)
        if dump_file is not None:
            library_path = f"{dump_file}.so"
            with open(library_path, "wb") as out:
                out.write(compiled_computation_module)

    def load_and_evaluate(
        self,
        dll,
        sym,
        repeat=1,
        min_repeat_ms=0,
        number=1,
        validate=False,
        parameters=None,
    ):
        results, code, error = self.load_and_eval(
            dll,
            sym,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            number=number,
            validate=validate,
            init_zero=True,  # TODO: for now init is external
            parameters=parameters,
        )
        if code == 0:
            return min(results)
        else:
            return error

    def load_and_eval(
        self,
        dll,
        sym,
        repeat=1,
        min_repeat_ms=0,
        number=1,
        validate=False,
        init_zero=False,
        parameters=None,
    ):
        libpath = os.path.abspath(dll)
        with utils.LibLoader(libpath) as lib:
            func = getattr(lib, sym)
            inputs_spec = self.np_inputs_spec()
            outputs_spec = self.np_outputs_spec()
            out_init = np.zeros if init_zero else np.empty
            if parameters is None:
                inputs = [utils.np_init(**spec) for spec in inputs_spec]
                outputs = [out_init(**spec) for spec in outputs_spec]
                parameters = (
                    [NDArray(inp) for inp in inputs],
                    [NDArray(out) for out in outputs],
                )
            if validate:
                ref_inputs = [inp.numpy() for inp in parameters[0]]
                ref_outputs = [np.empty(**spec) for spec in outputs_spec]
                self.reference_impl(*ref_inputs, *ref_outputs)
                exec_func = Executor(func)
                test_outputs = [NDArray(out_init(**spec)) for spec in outputs_spec]
                exec_func(*parameters[0], *test_outputs)
                for out_ref, out in zip(
                    ref_outputs, [out.numpy() for out in test_outputs]
                ):
                    if not np.allclose(out_ref, out):
                        return [], 1, "Error in validation: outputs differ"
            eval_func = Evaluator(
                func, repeat=repeat, min_repeat_ms=min_repeat_ms, number=number
            )
            results = eval_func(*parameters[0], *parameters[1])
        return np.array(results), 0, ""

    def np_inputs_spec(self):
        operator = self.source_op.operator
        return [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                operator.inputs_dims(*self.args), operator.inputs_types(*self.args)
            )
        ]

    def np_outputs_spec(self):
        operator = self.source_op.operator
        return [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                operator.outputs_dims(*self.args), operator.outputs_types(*self.args)
            )
        ]

    def reference_impl(self, *operands):
        self.source_op.operator.reference_impl(*operands)
