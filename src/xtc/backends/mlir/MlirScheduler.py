#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override

from xtc.itf.schd.scheduler import DEFAULT_ROOT
from xtc.schedules.loop_nest import LoopNest, LoopNestNode, LoopInfo, SplitOrigin
import xtc.itf as itf
import xtc.backends.mlir as backend

from .MlirNodeScheduler import MlirNodeScheduler, MlirNodeSchedule

__all__ = [
    "MlirScheduler",
    "MlirSchedule",
]


class MlirScheduler(itf.schd.Scheduler):
    def __init__(
        self,
        backend: "backend.MlirBackend",
        nodes_schedulers: list["MlirScheduler"] | None = None,
        nodes: list[str] | None = None,
        default_node: str | None = None,
    ) -> None:
        from .MlirGraphBackend import MlirGraphBackend
        from .MlirNodeBackend import MlirNodeBackend

        self._backend = backend
        # MLIR extensions
        self.mlir_extensions: dict[str, bool] = {}
        if isinstance(backend, MlirGraphBackend):
            if nodes_schedulers is None:
                if nodes is None:
                    self._nodes_schedulers = [
                        MlirScheduler(node_impl) for node_impl in backend.nodes.values()
                    ]
                else:
                    self._nodes_schedulers = [
                        MlirScheduler(backend.nodes[name]) for name in nodes
                    ]
            else:
                self._nodes_schedulers = nodes_schedulers

            # Take last node or find default_node
            if default_node is None:
                candidate_nodes = self._nodes_schedulers
            else:
                candidate_nodes = [
                    scheduler
                    for scheduler in self._nodes_schedulers
                    if scheduler._current_scheduler.node_name == default_node
                ]

            self._node_scheduler: MlirNodeScheduler | None = (
                candidate_nodes[-1]._current_scheduler
                if len(candidate_nodes) > 0
                else None
            )
        else:
            assert isinstance(backend, MlirNodeBackend)
            assert nodes_schedulers is None
            assert nodes is None
            assert default_node is None
            self._nodes_schedulers = [self]
            self._node_scheduler = MlirNodeScheduler(
                node_name=backend.payload_name,
                node_ident=backend.op_id_attribute,
                dims=list(backend.dims),
                loop_stamps=backend.loop_stamps,
            )

    @property
    def _current_scheduler(self) -> MlirNodeScheduler:
        assert self._node_scheduler is not None
        return self._node_scheduler

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    def _require_extension(self, extension: str, weak: bool = False):
        if (
            extension in self.mlir_extensions
            and self.mlir_extensions[extension] == True
        ):
            return
        self.mlir_extensions[extension] = weak

    @override
    def schedule(self) -> itf.schd.Schedule:
        from .MlirGraphBackend import MlirGraphBackend
        from .MlirNodeBackend import MlirNodeBackend

        if isinstance(self._backend, MlirGraphBackend):
            nodes_schedules = [
                scheduler._current_scheduler.mlir_node_schedule()
                for scheduler in self._nodes_schedulers
            ]
        else:
            assert isinstance(self._backend, MlirNodeBackend)
            nodes_schedules = [self._current_scheduler.mlir_node_schedule()]
        return MlirSchedule(
            scheduler=self,
            nodes_schedules=nodes_schedules,
            mlir_extensions=self.mlir_extensions,
        )

    @override
    def set_dims(self, dims: list[str]) -> None:
        self._current_scheduler.set_dims(dims)

    @override
    def split(
        self, dim: str, segments: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None:
        self._current_scheduler.split(dim, segments, root=root)

    @override
    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        self._current_scheduler.tile(dim, tiles, root=root)

    @override
    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT) -> None:
        self._current_scheduler.interchange(permutation, root=root)

    @override
    def buffer_at(
        self, axis: str, mtype: str | None = None, root: str = DEFAULT_ROOT
    ) -> None:
        assert mtype is None or mtype == "global"
        # TODO: not implemented for now
        pass

    @override
    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        root: str = DEFAULT_ROOT,
    ) -> None:
        # The current implemntation exclusively rely on SDist, but upstream
        # transform dialect may be used for some cases.
        assert mtype is None or mtype == "global" or mtype == "local"
        if pad:
            return  # TODO: not implemented for now
        if mtype is None or mtype == "global":
            self._require_extension("sdist", weak=True)
        else:
            self._require_extension("sdist")
        self._current_scheduler.pack_at(axis, input_idx, mtype, pad, root=root)

    @override
    def vectorize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        self._current_scheduler.vectorize(axes, root=root)

    @override
    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        self._current_scheduler.parallelize(axes, root=root)

    @override
    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        self._current_scheduler.unroll(unrolls, root=root)

    @override
    def fuse_producer_at(
        self, axis: str, input_idx: int, root: str = DEFAULT_ROOT
    ) -> None:
        # TODO: not implemented for now
        pass

    @override
    def define_memory_mesh(self, axes: dict[str, int]) -> None:
        self._require_extension("sdist")
        self._current_scheduler.define_memory_mesh(axes)

    @override
    def define_processor_mesh(self, axes: dict[str, int]) -> None:
        self._require_extension("sdist")
        self._current_scheduler.define_processor_mesh(axes)

    @override
    def distribute(
        self, axis: str, processor_axis: str, root: str = DEFAULT_ROOT
    ) -> None:
        self._require_extension("sdist")
        self._current_scheduler.distribute(axis, processor_axis, root=root)

    @override
    def distributed_buffer_at(
        self,
        axis: str,
        input_idx: int,
        memory_axes: list[str],
        root: str = DEFAULT_ROOT,
    ) -> None:
        self._require_extension("sdist")
        self._current_scheduler.distributed_buffer_at(
            axis, input_idx, memory_axes, root=root
        )

    @override
    def get_loop_nest(self) -> LoopNest:
        node_sched = self._current_scheduler
        dims = node_sched.dims[:]

        loop_nest = LoopNest(abstract_dims=dims)
        root_node = loop_nest.build_root_node(node_sched.node_name)

        # Assign splits to root_node first
        for axis, axis_splits in node_sched.splits.items():
            root_node.splits[axis] = dict(axis_splits)

        # Build mapper to get splits_info
        mapper = LoopInfo.build_from_node(root_node)

        def populate_node(node: LoopNestNode, perm: list[str]) -> None:
            """Populate node with data for loops in its permutation."""
            node.interchange = list(perm)
            perm_set = set(perm)
            for axis, axis_tiles in node_sched.tiles.items():
                for tile_name, size in axis_tiles.items():
                    if tile_name in perm_set:
                        if axis not in node.tiles:
                            node.tiles[axis] = {}
                        node.tiles[axis][tile_name] = size
            node.vectorize = [v for v in node_sched.vectorization if v in perm_set]
            node.parallelize = [p for p in node_sched.parallelization if p in perm_set]
            node.unroll = {
                k: v for k, v in node_sched.unrolling.items() if k in perm_set
            }

        # Process each root in permutation
        for root, perm in node_sched.permutation.items():
            if root in mapper.splits_info:
                # This root is a split - create child node
                axis, start, end = mapper.splits_info[root]
                child = LoopNestNode(
                    root=root,
                    tiles={d: {} for d in dims},
                    split_origin=SplitOrigin(axis=axis, start=start, end=end),
                )
                populate_node(child, perm)
                root_node.add_child(child)
            else:
                # This is the main root
                populate_node(root_node, perm)

        return loop_nest


class MlirSchedule(itf.schd.Schedule):
    def __init__(
        self,
        scheduler: MlirScheduler,
        nodes_schedules: list[MlirNodeSchedule],
        mlir_extensions: dict[str, bool],
    ) -> None:
        self._scheduler = scheduler
        self._nodes_schedules = nodes_schedules
        self._mlir_extensions = mlir_extensions

    @property
    def schedule_impl(self) -> list[MlirNodeSchedule]:
        return self._nodes_schedules

    @property
    @override
    def scheduler(self) -> itf.schd.Scheduler:
        return self._scheduler

    @property
    def mlir_extensions(self) -> dict[str, bool]:
        return self._mlir_extensions

    @override
    def __str__(self) -> str:
        return str(self._nodes_schedules)
