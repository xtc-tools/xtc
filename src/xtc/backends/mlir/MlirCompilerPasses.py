#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from mlir.dialects import transform
from mlir.dialects.transform import (
    NamedSequenceOp,
    structured,
    vector,
    get_parent_op,
)
from mlir.dialects.transform.structured import structured_match
from mlir.dialects.transform.loop import loop_unroll
from mlir.ir import (
    Location,
    InsertionPoint,
    UnitAttr,
    OpResult,
)
from mlir.passmanager import PassManager

from xtc.utils.ext_tools import (
    transform_opts,
    lowering_opts,
)

from .MlirProgram import RawMlirProgram
from .MlirScheduler import MlirSchedule, MlirNodeSchedule


class MlirProgramInsertTransformPass:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
        mlir_schedule: MlirSchedule | None = None,
        concluding_passes: list[str] = [],
        always_vectorize: bool = True,
    ) -> None:
        self._mlir_program = mlir_program
        self._mlir_schedule = mlir_schedule
        self._loc = Location.unknown(self._mlir_program.mlir_context)
        self._concluding_passes = concluding_passes
        self._always_vectorize = always_vectorize
        self._named_sequence: NamedSequenceOp | None = None
        self._nodes_schedules = (
            self._mlir_schedule.schedule_impl if self._mlir_schedule is not None else []
        )

    def run(self) -> None:
        if self._mlir_schedule is None:
            return
        with (
            InsertionPoint(self._mlir_program.mlir_module.body),
            self._mlir_program.mlir_context,
            self._loc,
        ):
            self._mlir_program.mlir_module.operation.attributes[
                "transform.with_named_sequence"
            ] = UnitAttr.get()
            self._named_sequence = NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.readonly": UnitAttr.get()}],
            )
        with (
            InsertionPoint.at_block_begin(self._named_sequence.body),
            self._mlir_program.mlir_context,
            self._loc,
        ):
            if len(self._nodes_schedules) > 0:
                self._implement()
            else:
                transform.YieldOp([])

    def _generate_vectorization(self, handle: OpResult) -> OpResult:
        if self._always_vectorize or self._needs_vectorization():
            handle = structured.VectorizeChildrenAndApplyPatternsOp(handle)
            with InsertionPoint(transform.ApplyPatternsOp(handle).patterns):
                vector.ApplyLowerOuterProductPatternsOp()
                vector.ApplyLowerContractionPatternsOp()
        return handle

    def _needs_vectorization(self) -> bool:
        for schedule in self._nodes_schedules:
            if self._node_needs_vectorization(schedule):
                return True
        return False

    def _generate_tiling(self) -> OpResult:
        assert self._named_sequence is not None
        handle = None
        for schedule in self._nodes_schedules:
            match0 = structured_match(
                results_=transform.AnyOpType.get(),
                target=self._named_sequence.bodyTarget,
                op_attrs={schedule.node_ident: UnitAttr.get()},
            )
            handle = self._generate_node_tiling(match0, schedule)
        assert handle, "At least 1 operation should have been processed"
        return handle

    def _implement(self) -> None:
        assert self._named_sequence is not None
        with (
            InsertionPoint.at_block_begin(self._named_sequence.body),
            self._mlir_program.mlir_context,
            self._loc,
        ):
            handle = self._generate_tiling()
            handle = get_parent_op(
                transform.AnyOpType.get(),
                handle,
                isolated_from_above=True,
            )
            handle = self._generate_vectorization(handle)
            for p in self._concluding_passes:
                handle = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(), handle, pass_name=p
                )
            transform.YieldOp([])

    def _node_needs_vectorization(self, schedule: MlirNodeSchedule) -> bool:
        return len(schedule.vectorization) > 0

    def _generate_node_tiling(
        self, handle: OpResult, schedule: MlirNodeSchedule
    ) -> OpResult:
        # Produce the sequence of commands needed for the tiling
        tiling_arrays: dict[str, list[int]] = {}
        deepest_tiling = max(schedule.tiles.values(), key=len)
        depth_deepest_tiling = len(deepest_tiling)
        for tile_level in range(depth_deepest_tiling):
            for index_of_dim, (_, tiles) in enumerate(schedule.tiles.items()):
                # This dimension is not tiled at this level.
                if tile_level >= len(tiles):
                    continue

                # Create the array describing the tiling of this
                # dimension. If I have a (x,y,z) nest and I want
                # to tile the y dimension with a tile size of 16,
                # the resulting array is [0,16,0].
                tile_dim_name = list(tiles.keys())[tile_level]
                tiling_array = [
                    tiles[tile_dim_name] if i == index_of_dim else 0
                    for i in range(len(schedule.tiles))
                ]
                tiling_arrays[tile_dim_name] = tiling_array
        # Reorder the tiling according to permutation.
        tiling_arrays = {p: tiling_arrays[p] for p in schedule.permutation}
        # Materialize loops
        op_to_tile = handle
        all_loops = {}
        for tile_name, tiling_array in tiling_arrays.items():
            # Useless to materialize a loop which will be vectorized
            if tile_name in schedule.vectorization:
                break
            # Generate the tiling itself
            if tile_name in schedule.parallelization:
                tiling_command = structured.TileUsingForallOp(
                    op_to_tile, tile_sizes=tiling_array
                )
            else:
                tiling_command = structured.TileUsingForOp(
                    op_to_tile, sizes=tiling_array
                )
            # Annotate the resulting loop if successfully generated
            if len(tiling_command.results) > 1:
                generated_loop = tiling_command.results[1]
                transform.AnnotateOp(
                    generated_loop, f"{schedule.node_ident}{tile_name}"
                )
                all_loops[tile_name] = generated_loop
            #
            op_to_tile = tiling_command.results[0]

        # TODO: LLVM metadata instead of transform unroll may
        # ultimately put less pressure on opt/llc front-end
        # https://llvm.org/docs/LangRef.html#llvm-loop
        for dim in reversed(schedule.permutation):
            if dim in schedule.unrolling and dim in all_loops:
                loop_unroll(all_loops[dim], schedule.unrolling[dim])

        # The resulting operation is either the outermost loop or
        # the initial (not tiled) handle
        if all_loops:
            handle_after_tiling = next(iter(all_loops.values()))
        else:
            handle_after_tiling = handle

        # Stamp the resulting operation
        for s in schedule.loop_stamps:
            transform.AnnotateOp(handle_after_tiling, s)
        return handle_after_tiling


class MlirProgramApplyTransformPass:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
    ) -> None:
        self._mlir_program = mlir_program

    def run(self) -> None:
        transform_op = [op for op in self._mlir_program.mlir_module.body.operations][-1]
        transform = isinstance(transform_op, NamedSequenceOp)
        if not transform:
            return
        pm = PassManager(context=self._mlir_program.mlir_context)
        for opt in transform_opts:
            pm.add(opt)  # type: ignore # no attribte add?
        pm.run(self._mlir_program.mlir_module)
        transform_op = [op for op in self._mlir_program.mlir_module.body.operations][-1]
        assert isinstance(transform_op, NamedSequenceOp)
        transform_op.erase()


class MlirProgramToLLVMDialectPass:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
    ) -> None:
        self._mlir_program = mlir_program

    def run(self) -> None:
        pm = PassManager(context=self._mlir_program.mlir_context)
        for opt in lowering_opts:
            pm.add(opt)  # type: ignore # no attribte add?
        pm.run(self._mlir_program.mlir_module)
