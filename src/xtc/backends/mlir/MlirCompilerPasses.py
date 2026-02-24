#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from dataclasses import dataclass
from mlir.dialects import transform
from mlir.dialects.transform import (
    NamedSequenceOp,
    structured,
    vector,
    memref,
    get_parent_op,
)
from mlir.dialects.transform.structured import (
    TileUsingForallOp,
    TileUsingForOp,
    VectorizeOp,
)
from mlir.dialects.transform.structured import structured_match
from mlir.dialects.transform.loop import loop_unroll
from mlir.dialects.transform import SplitHandleOp
from mlir.ir import (
    Location,
    InsertionPoint,
    UnitAttr,
    OpResult,
)
from mlir.passmanager import PassManager

# Import SDist if available
try:
    from mlir_sdist.dialects.transform import sdist as sdist_transform
except ImportError:
    sdist_transform = None
    pass

from xtc.utils.ext_tools import transform_opts

from .MlirProgram import RawMlirProgram
from .MlirScheduler import MlirSchedule, MlirNodeSchedule
from .MlirTarget import MlirTarget

_VECTO_SEQ_NAME = "_vecto"
_SUPER_VECTORIZE_SEQ_NAME = "_super_vectorize"


@dataclass
class SchedulingState:
    all_loops: dict[str, OpResult]
    handle: OpResult


@dataclass
class SplitState:
    # Explicit, flattened split maps
    all_splits_by_loop: dict[str, int]
    # loop -> dim
    loop_dim_by_split: dict[str, str]
    # For each loop to split, store the next (todo) cutting point
    prev_split_size: dict[str, int]
    # For each loop to split, store the previous (done) cutting point
    next_split_size: dict[str, int]
    #
    root: str

    def __init__(self, splits: dict[str, dict[str, int]], root: str):
        self.all_splits_by_loop: dict[str, int] = {
            loop: cut
            for sub_splits in splits.values()
            for loop, cut in sub_splits.items()
        }
        self.loop_dim_by_split: dict[str, str] = {
            loop: dim for dim, splits in splits.items() for loop in splits
        }
        self.prev_split_size = {loop: 0 for loop in self.all_splits_by_loop}
        loops_to_split: list[str] = list(self.all_splits_by_loop)
        self.next_split_size: dict[str, int] = {
            loops_to_split[i]: self.all_splits_by_loop[loops_to_split[i + 1]]
            for i in range(len(loops_to_split) - 1)
        }
        self.root = root

    def chunk_size(self, loop_name: str) -> int:
        return self.next_split_size[loop_name] - self.prev_split_size[loop_name]

    def must_be_splitted(self, loop_name: str) -> bool:
        return loop_name in self.next_split_size

    def if_last_chunk(self, loop_name: str) -> bool:
        return loop_name in self.loop_dim_by_split and loop_name != self.root

    def move_forward(self, loop_name: str):
        offset = self.chunk_size(loop_name)
        self.prev_split_size[loop_name] += offset
        self.next_split_size[loop_name] += offset


class MlirProgramInsertTransformPass:
    def __init__(
        self,
        mlir_program: RawMlirProgram,
        target: MlirTarget,
        mlir_schedule: MlirSchedule | None = None,
        concluding_passes: list[str] = [],
        always_vectorize: bool = True,
        vectors_size: int | None = None,
    ) -> None:
        self._mlir_program = mlir_program
        self._target = target
        self._mlir_schedule = mlir_schedule
        self._loc = Location.unknown(self._mlir_program.mlir_context)
        self._concluding_passes = concluding_passes
        self._always_vectorize = always_vectorize
        self._vectors_size = vectors_size
        self._super_vectorize = self._vectors_size is not None
        self._vecto_sequence: NamedSequenceOp | None = None
        self._super_vectorize_sequence: NamedSequenceOp | None = None
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
            if self._super_vectorize:
                self._super_vectorize_sequence = NamedSequenceOp(
                    _SUPER_VECTORIZE_SEQ_NAME,
                    [transform.AnyOpType.get()],
                    [transform.AnyOpType.get()],
                    arg_attrs=[{"transform.consumed": UnitAttr.get()}],
                )
            self._named_sequence = NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.readonly": UnitAttr.get()}],
            )
        assert self._vecto_sequence is not None
        assert self._named_sequence is not None
        with (
            InsertionPoint.at_block_begin(self._vecto_sequence.body),
            self._mlir_program.mlir_context,
            self._loc,
        ):
            VectorizeOp(self._vecto_sequence.bodyTarget)
            transform.YieldOp([])
        if self._super_vectorize:
            assert self._super_vectorize_sequence is not None
            with (
                InsertionPoint.at_block_begin(self._super_vectorize_sequence.body),
                self._mlir_program.mlir_context,
                self._loc,
            ):
                result = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    self._super_vectorize_sequence.bodyTarget,
                    pass_name="affine-super-vectorize",
                    options={"virtual-vector-size": self._vectors_size},
                )
                transform.YieldOp([result])
        with (
            InsertionPoint.at_block_begin(self._named_sequence.body),
            self._mlir_program.mlir_context,
            self._loc,
        ):
            if len(self._nodes_schedules) > 0:
                self._implement()
            else:
                transform.YieldOp([])

    def _generate_scheduling(self) -> OpResult:
        assert self._named_sequence is not None
        handle = None
        for schedule in self._nodes_schedules:
            self._create_sdist_meshes(schedule)
            handle = structured_match(
                results_=transform.AnyOpType.get(),
                target=self._named_sequence.bodyTarget,
                op_attrs={schedule.node_ident: UnitAttr.get()},
            )
            if schedule.permutation:
                scheduling_state = self._generate_node_scheduling(
                    schedule=schedule,
                    root=list(schedule.permutation)[0],
                    handle=handle,
                )
                if schedule.vectorization or self._always_vectorize:
                    self._post_vectorize(scheduling_state)
                handle = scheduling_state.handle
        assert handle, "At least 1 operation should have been processed"
        return handle

    def _create_sdist_meshes(self, schedule: MlirNodeSchedule) -> None:
        assert self._named_sequence is not None
        with (
            InsertionPoint.at_block_begin(self._named_sequence.body),
            self._mlir_program.mlir_context,
            self._loc,
        ):
            if len(schedule.memory_mesh) > 0:
                mesh = [
                    {axis_name: axis_size}
                    for axis_name, axis_size in schedule.memory_mesh.items()
                ]
                assert sdist_transform is not None
                sdist_transform.SDistCreateMemoryMeshOp(
                    self._named_sequence.bodyTarget,
                    name="memory_mesh",
                    mesh=mesh,
                )
            if len(schedule.processor_mesh) > 0:
                mesh = [
                    {axis_name: axis_size}
                    for axis_name, axis_size in schedule.processor_mesh.items()
                ]
                assert sdist_transform is not None
                sdist_transform.SDistCreateProcessorMeshOp(
                    self._named_sequence.bodyTarget,
                    name="processor_mesh",
                    memory_mesh="memory_mesh",
                    mesh=mesh,
                )

    def _implement(self) -> None:
        assert self._named_sequence is not None
        with (
            InsertionPoint.at_block_begin(self._named_sequence.body),
            self._mlir_program.mlir_context,
            self._loc,
        ):
            handle = self._generate_scheduling()
            if self._super_vectorize:
                handle = get_parent_op(
                    transform.AnyOpType.get(),
                    handle,
                    isolated_from_above=True,
                )
                handle = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(),
                    handle,
                    pass_name="convert-linalg-to-affine-loops",
                )
                handle = transform.IncludeOp(
                    results_=[transform.AnyOpType.get()],
                    target=_SUPER_VECTORIZE_SEQ_NAME,
                    failure_propagation_mode=2,
                    operands_=[handle],
                )

            for p in self._concluding_passes:
                handle = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(), handle, pass_name=p
                )
            transform.YieldOp([])

    def _generate_node_scheduling(
        self,
        schedule: MlirNodeSchedule,
        root: str,
        handle: OpResult,
    ) -> SchedulingState:
        sched_state = SchedulingState({}, handle)
        split_state = SplitState(schedule.splits, root)
        tiles_sizes_by_loops = self._generate_tiling_insns(schedule)
        permutation = schedule.permutation[root]
        if not permutation:
            return sched_state

        # Materialize the loops
        for loop_name in permutation:
            # Manage the splits
            if split_state.must_be_splitted(loop_name):
                self._split_section(loop_name, split_state, sched_state, schedule)
                continue
            elif split_state.if_last_chunk(loop_name):
                self._recursive_scheduling(
                    schedule=schedule, root=loop_name, sched_state=sched_state
                )
                continue

            # Bufferization
            if loop_name in schedule.distributed_buffers.keys():
                self._distribute_buffer(
                    loop_name=loop_name,
                    schedule=schedule,
                    sched_state=sched_state,
                )
            if loop_name in schedule.packed_buffers.keys():
                self._pack_buffer(
                    loop_name=loop_name,
                    schedule=schedule,
                    sched_state=sched_state,
                )

            # Manage the strip-mining
            if loop_name in schedule.vectorization:
                self._vectorize(sched_state)
                break
            elif loop_name in tiles_sizes_by_loops:
                self._strip_mine(
                    loop_name=loop_name,
                    tiling_vector=tiles_sizes_by_loops[loop_name],
                    schedule=schedule,
                    sched_state=sched_state,
                )

        # For now on, the focus is on the outermost loop
        if sched_state.all_loops:
            sched_state.handle = next(iter(sched_state.all_loops.values()))

        # Unrolling
        if schedule.unrolling:
            self._unroll(permutation, schedule, sched_state)

        return sched_state

    def _generate_tiling_insns(
        self, schedule: MlirNodeSchedule
    ) -> dict[str, list[int]]:
        tiles_sizes_by_loops: dict[str, list[int]] = {}
        state_of_tiling: dict[str, int] = {dim: 1 for dim in schedule.dims}
        candidate_state_of_tiling = state_of_tiling.copy()
        previous_root = ""
        for loc_root, permutation in reversed(schedule.permutation.items()):
            if len(loc_root) == len(previous_root):
                # Reset the view on the state of tiling (we are jumping into
                # a split of the same loop)
                candidate_state_of_tiling = state_of_tiling.copy()
            else:
                # Update the state of tiling
                state_of_tiling = candidate_state_of_tiling.copy()

            for loop in reversed(permutation):
                # The loop needs to be base or tile
                if not (schedule.is_tile(loop) or schedule.is_base(loop)):
                    continue

                # Fetch the dimension knowledge
                dim_of_loop = schedule.dim_of_tile(loop)
                index_of_dim = schedule.index_of_dim(dim_of_loop)

                # Build the strip size
                size_of_tile = schedule.size_of_tile(loop)
                strip_size = candidate_state_of_tiling[dim_of_loop]
                if schedule.is_tile(loop):
                    assert size_of_tile
                    candidate_state_of_tiling[dim_of_loop] = size_of_tile

                # Build the tiling instruction vector
                tiling_vector = [0] * len(schedule.dims)
                tiling_vector[index_of_dim] = strip_size
                tiles_sizes_by_loops[loop] = tiling_vector

            previous_root = loc_root

        return tiles_sizes_by_loops

    def _split_section(
        self,
        loop_name: str,
        split_state: SplitState,
        sched_state: SchedulingState,
        schedule: MlirNodeSchedule,
    ):
        dim_to_split = split_state.loop_dim_by_split[loop_name]
        split_handle = structured.SplitOp(
            target=sched_state.handle,
            dimension=schedule.dims.index(dim_to_split),
            chunk_sizes=split_state.chunk_size(loop_name),
        )
        split_command = SplitHandleOp(
            results_=[transform.AnyOpType.get(), transform.AnyOpType.get()],
            handle=split_handle,
        )
        sched_state.handle = split_command.results[0]
        self._recursive_scheduling(
            schedule=schedule, root=loop_name, sched_state=sched_state
        )
        sched_state.handle = split_command.results[1]
        split_state.move_forward(loop_name)

    def _recursive_scheduling(
        self, schedule: MlirNodeSchedule, root: str, sched_state: SchedulingState
    ):
        inner_sched_state = self._generate_node_scheduling(
            schedule=schedule, root=root, handle=sched_state.handle
        )
        sched_state.all_loops.update(inner_sched_state.all_loops)
        sched_state.handle = inner_sched_state.handle

    def _strip_mine(
        self,
        loop_name: str,
        tiling_vector: list[int],
        schedule: MlirNodeSchedule,
        sched_state: SchedulingState,
    ) -> OpResult:
        if loop_name in schedule.parallelization:
            tiling_command = TileUsingForallOp(
                sched_state.handle, tile_sizes=tiling_vector
            )
        else:
            tiling_command = TileUsingForOp(sched_state.handle, sizes=tiling_vector)
        # Extract the results
        sched_state.handle = tiling_command.results[0]
        assert len(tiling_command.results) == 2
        new_loop = tiling_command.results[-1]
        sched_state.all_loops[loop_name] = new_loop
        # Annotate the resulting loop if successfully generated
        transform.AnnotateOp(new_loop, loop_name)

        return new_loop

    def _vectorize(self, sched_state: SchedulingState):
        if self._vectors_size is not None:
            return

        if self._target.custom_vectorize():
            self._target.apply_custom_vectorize(sched_state.handle)
        else:
            transform.IncludeOp(
                results_=[],
                target=_VECTO_SEQ_NAME,
                failure_propagation_mode=2,
                operands_=[sched_state.handle],
            )

    def _post_vectorize(self, sched_state: SchedulingState):
        if self._vectors_size is not None:
            return

        parent_op = get_parent_op(
            transform.AnyOpType.get(),
            sched_state.handle,
            isolated_from_above=True,
        )
        with InsertionPoint(transform.ApplyPatternsOp(parent_op).patterns):
            vector.ApplyVectorReductionToContractPatternsOp()
            vector.ApplyTransferPermutationPatternsOp()
        with InsertionPoint(transform.ApplyPatternsOp(parent_op).patterns):
            vector.ApplyLowerOuterProductPatternsOp()
            vector.ApplyLowerContractionPatternsOp()

    def _unroll(
        self,
        permutation: list[str],
        schedule: MlirNodeSchedule,
        sched_state: SchedulingState,
    ):
        # TODO: LLVM metadata instead of transform unroll may
        # ultimately put less pressure on opt/llc front-end
        # https://llvm.org/docs/LangRef.html#llvm-loop
        for dim_name in reversed(permutation):
            if (
                dim_name in schedule.unrolling
                and not dim_name in schedule.vectorization
            ):
                assert self._named_sequence is not None
                loop_unroll(
                    sched_state.all_loops[dim_name], schedule.unrolling[dim_name]
                )
            if dim_name in schedule.distribution.keys():
                assert self._named_sequence is not None
                assert sdist_transform is not None
                sdist_transform.SDistDistributeLoopOp(
                    target=sched_state.all_loops[dim_name],
                    mesh="processor_mesh",
                    axis=schedule.distribution[dim_name],
                )

    def _distribute_buffer(
        self,
        loop_name: str,
        schedule: MlirNodeSchedule,
        sched_state: SchedulingState,
    ):
        # TODO multiple buffers
        assert sdist_transform is not None
        sdist_transform.SDistDistributeBufferAtOp(
            target=sched_state.handle,
            mesh="memory_mesh",
            input_idx=schedule.distributed_buffers[loop_name]["input_idx"],
            axes=schedule.distributed_buffers[loop_name]["memory_axes"],
        )

    def _pack_buffer(
        self,
        loop_name: str,
        schedule: MlirNodeSchedule,
        sched_state: SchedulingState,
    ):
        with InsertionPoint(transform.ApplyPatternsOp(sched_state.handle).patterns):
            memref.ApplyFoldMemrefAliasOpsPatternsOp()
        for input_idx in schedule.packed_buffers[loop_name]:
            if "sdist" in self._mlir_program.mlir_extensions:
                assert sdist_transform is not None
                sdist_transform.SDistLocalBufferAtOp(
                    target=sched_state.handle,
                    input_idx=input_idx,
                )


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
        pm.run(self._mlir_program.mlir_module.operation)

        while True:
            transform_op = [
                op for op in self._mlir_program.mlir_module.body.operations
            ][-1]
            if isinstance(transform_op, NamedSequenceOp):
                transform_op.erase()
            else:
                break
