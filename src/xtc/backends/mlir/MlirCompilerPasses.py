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
from mlir.dialects.transform.structured import (
    TileUsingForallOp,
    TileUsingForOp,
    VectorizeOp,
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

_VECTO_SEQ_NAME = "_vecto"


class MlirProgramInsertTransformPass:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
        mlir_schedule: MlirSchedule | None = None,
        concluding_passes: list[str] = [],
        always_vectorize: bool = True,
        vectors_size: int | None = None,
    ) -> None:
        self._mlir_program = mlir_program
        self._mlir_schedule = mlir_schedule
        self._loc = Location.unknown(self._mlir_program.mlir_context)
        self._concluding_passes = concluding_passes
        self._always_vectorize = always_vectorize
        self._vectors_size = vectors_size
        self._vecto_sequence: NamedSequenceOp | None = None
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

            self._vecto_sequence = NamedSequenceOp(
                _VECTO_SEQ_NAME,
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.consumed": UnitAttr.get()}],
            )

            self._named_sequence = NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.readonly": UnitAttr.get()}],
            )
        with (
            InsertionPoint.at_block_begin(self._vecto_sequence.body),
            self._mlir_program.mlir_context,
            self._loc,
        ):
            VectorizeOp(self._vecto_sequence.bodyTarget)
            transform.YieldOp([])
        with (
            InsertionPoint.at_block_begin(self._named_sequence.body),
            self._mlir_program.mlir_context,
            self._loc,
        ):
            if len(self._nodes_schedules) > 0:
                self._implement()
            else:
                transform.YieldOp([])

    def _generate_tiling(self) -> OpResult:
        assert self._named_sequence is not None
        handle = None
        for schedule in self._nodes_schedules:
            handle = structured_match(
                results_=transform.AnyOpType.get(),
                target=self._named_sequence.bodyTarget,
                op_attrs={schedule.node_ident: UnitAttr.get()},
            )
            if schedule.permutation:
                handle = self._generate_node_schedule(
                    handle, schedule, list(schedule.permutation)[0]
                )
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
            if self._vectors_size:
                handle = get_parent_op(
                    transform.AnyOpType.get(),
                    handle,
                    isolated_from_above=True,
                )
                affine_code = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    handle,
                    pass_name="convert-linalg-to-affine-loops",
                )
                handle = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    affine_code,
                    pass_name="affine-super-vectorize",
                    options=f"virtual-vector-size={self._vectors_size}",
                )

            for p in self._concluding_passes:
                handle = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(), handle, pass_name=p
                )
            transform.YieldOp([])

    def _generate_node_schedule(
        self,
        handle: OpResult,
        schedule: MlirNodeSchedule,
        root: str,
    ) -> OpResult:
        permutation = schedule.permutation[root]
        handle = self._generate_node_tiling(handle, permutation, schedule, root)
        # Stamp the resulting operation
        for s in schedule.loop_stamps:
            transform.AnnotateOp(handle, s)

        return handle

    def _generate_node_tiling(
        self,
        handle: OpResult,
        permutation: list[str],
        schedule: MlirNodeSchedule,
        root: str,
    ) -> OpResult:
        # Produce the sequence of commands needed for the tiling
        tiling_vectors = self._generate_tiling_vectors(root, schedule, permutation)
        #
        all_splits = {
            key: value
            for splits in schedule.splits.values()
            for key, value in splits.items()
        }
        previous_scar = {key: 0 for key in all_splits}
        split_to_dimension = {
            split: dimension
            for dimension, splits in schedule.splits.items()
            for split in splits
        }
        split_keys = list(all_splits)
        scar_of_split = {
            split_keys[i]: all_splits[split_keys[i + 1]]
            for i in range(len(split_keys) - 1)
        }
        # Materialize loops
        all_loops: dict[str, OpResult] = {}
        target_op = handle
        for axis_name in permutation:
            #
            if axis_name in scar_of_split:
                chunk_size = scar_of_split[axis_name] - previous_scar[axis_name]
                dim_to_split = split_to_dimension[axis_name]
                split_command = structured.SplitOp(
                    target=target_op,
                    dimension=schedule.dims.index(dim_to_split),
                    chunk_sizes=chunk_size,
                )
                left_loop_op = self._generate_node_schedule(
                    handle=split_command.results[0], schedule=schedule, root=axis_name
                )
                all_loops[axis_name] = left_loop_op
                target_op = split_command.results[1]
                continue
            # Catch the last chunk of the axis
            elif axis_name in split_to_dimension and axis_name != root:
                right_loop_op = self._generate_node_schedule(
                    handle=target_op, schedule=schedule, root=axis_name
                )
                all_loops[axis_name] = right_loop_op
                target_op = right_loop_op
                continue

            if axis_name in schedule.vectorization:
                continue

            # Generate the tiling itself
            elif axis_name in tiling_vectors:
                tiling_vector = tiling_vectors[axis_name]
                if axis_name in schedule.parallelization:
                    tiling_command = TileUsingForallOp(
                        target_op, tile_sizes=tiling_vector
                    )
                else:
                    tiling_command = TileUsingForOp(target_op, sizes=tiling_vector)
                # Extract the results
                target_op = tiling_command.results[0]
                if len(tiling_command.results) == 2:
                    new_loop = tiling_command.results[-1]
                    all_loops[axis_name] = new_loop
                    # Annotate the resulting loop if successfully generated
                    transform.AnnotateOp(new_loop, axis_name)

        if (
            set(permutation) & set(schedule.vectorization)
            and self._vectors_size is None
        ):
            transform.IncludeOp(
                results_=[],
                target=_VECTO_SEQ_NAME,
                failure_propagation_mode=2,
                operands_=[target_op],
            )
        # The resulting operation is either the outermost loop or
        # the initial (not tiled) handle
        if all_loops:
            handle_after_tiling = next(iter(all_loops.values()))
        else:
            handle_after_tiling = handle

        parent_op = get_parent_op(
            transform.AnyOpType.get(),
            handle_after_tiling,
            isolated_from_above=True,
        )
        if schedule.vectorization or self._always_vectorize:
            with InsertionPoint(transform.ApplyPatternsOp(parent_op).patterns):
                vector.ApplyVectorReductionToContractPatternsOp()
                vector.ApplyTransferPermutationPatternsOp()
            with InsertionPoint(transform.ApplyPatternsOp(parent_op).patterns):
                vector.ApplyLowerOuterProductPatternsOp()
                vector.ApplyLowerContractionPatternsOp()

        # TODO: LLVM metadata instead of transform unroll may
        # ultimately put less pressure on opt/llc front-end
        # https://llvm.org/docs/LangRef.html#llvm-loop
        for dim_name in reversed(permutation):
            if (
                dim_name in schedule.unrolling
                and not dim_name in schedule.vectorization
            ):
                assert self._named_sequence is not None
                handle = structured_match(
                    results_=transform.AnyOpType.get(),
                    target=parent_op,
                    op_attrs={dim_name: UnitAttr.get()},
                )
                loop_unroll(all_loops[dim_name], schedule.unrolling[dim_name])

        return handle_after_tiling

    def _generate_tiling_vectors(
        self, root: str, schedule: MlirNodeSchedule, permutation: list[str]
    ) -> dict[str, list[int]]:
        tiles = {}
        for axis, axis_tiles in schedule.tiles.items():
            tiles_names = [f"{root}/{axis}"] + list(axis_tiles.keys())
            tiles_sizes = [0] + list(axis_tiles.values())

            # Shift the list in order to provide a description of the tiling
            # instructions rather than a description of the (resulting) tiles
            # to the Transform dialect
            i = 0
            while i < len(tiles_sizes):
                # When zero: shift the list and erase the zero
                if tiles_sizes[i] == 0:
                    for j in range(i, len(tiles_sizes) - 1):
                        tiles_sizes[j] = tiles_sizes[j + 1]
                    # Put a 1 at the end
                    tiles_sizes[-1] = 1
                # When not zero: increment i (next)
                else:
                    i += 1
            #
            tiles[axis] = dict(zip(tiles_names, tiles_sizes))

        vector_size = len(schedule.dims)
        index_of_dim = {d: i for i, d in enumerate(schedule.dims)}
        # Handle splitted dimensions
        for dim, splits_of_dim in schedule.splits.items():
            for s in splits_of_dim:
                index_of_dim[s] = index_of_dim[dim]
        # Handle tiled dimensions
        size_of_tile: dict[str, int] = {}
        for dim, tiles_of_dim in tiles.items():
            size_of_tile = size_of_tile | tiles_of_dim
            for t in tiles_of_dim:
                index_of_dim[t] = index_of_dim[dim]
        # Build the tiling vectors
        tiling_vectors: dict[str, list[int]] = {}
        for d in permutation:
            if d in size_of_tile:
                tiling_vector = [
                    size_of_tile[d] if i == index_of_dim[d] else 0
                    for i in range(vector_size)
                ]
                tiling_vectors[d] = tiling_vector

        return tiling_vectors


class MlirProgramApplyTransformPass:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
    ) -> None:
        self._mlir_program = mlir_program

    def run(self) -> None:
        transform_op = [op for op in self._mlir_program.mlir_module.body.operations][-1]
        transform = isinstance(transform_op, NamedSequenceOp)
        assert transform
        pm = PassManager(context=self._mlir_program.mlir_context)
        for opt in transform_opts:
            pm.add(opt)  # type: ignore # no attribte add?
        pm.run(self._mlir_program.mlir_module)

        while True:
            transform_op = [
                op for op in self._mlir_program.mlir_module.body.operations
            ][-1]
            if isinstance(transform_op, NamedSequenceOp):
                transform_op.erase()
            else:
                break


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
