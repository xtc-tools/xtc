#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import TYPE_CHECKING
from typing_extensions import override

import xtc.itf as itf
from xtc.itf.schd.scheduler import DEFAULT_ROOT
from xtc.schedules.loop_nest import LoopNest

if TYPE_CHECKING:
    from .IREEBackend import IREEBackend

__all__ = [
    "IREEScheduler",
    "IREESchedule",
]

# SIMD vector width (in f32 elements) used when a vectorized axis has no
# associated tile size (e.g. an untiled base dimension).
_VECTOR_WIDTH = 4

# Upper bound on the vector tile width. The width normally follows the XTC tile
# size of the vectorized axis (so semantics match the MLIR backend), but beyond
# ~32 IREE either narrows the vectors itself or rejects the program with a
# "large vector sizes" error. 32 stays within that limit even when two axes are
# vectorized (32x32 f32 = 4 KiB).
_MAX_VECTOR_WIDTH = 32


class IREENodeSchedule:
    """Scheduling primitives recorded for a single graph node.

    XTC's per-dimension tiling / vectorization / parallelization vocabulary is
    accumulated here and translated lazily, by :meth:`compilation_info`, into an
    ``iree_codegen.compilation_info`` attribute that IREE's own CPU code
    generation pipeline consumes.
    """

    def __init__(self, op_id: str, dims: list[str], kinds: list[str]) -> None:
        self.op_id = op_id
        self.dims = dims[:]
        self.kinds = kinds[:]
        # Per dimension, the tile sizes and axis names requested, outer to inner.
        self.tiles: dict[str, list[int]] = {}
        self.tile_names: dict[str, list[str]] = {}
        self.vectorization: list[str] = []
        self.parallelization: list[str] = []

    def set_dims(self, dims: list[str]) -> None:
        assert len(dims) == len(self.dims), (
            f"set_dims expects {len(self.dims)} dims, got {len(dims)}"
        )
        self.dims = dims[:]
        self.tiles = {}
        self.tile_names = {}

    def tile(self, dim: str, tiles: dict[str, int]) -> None:
        assert dim in self.dims, f"unknown dimension {dim!r}"
        self.tiles.setdefault(dim, [])
        self.tile_names.setdefault(dim, [])
        self.tiles[dim].extend(tiles.values())
        self.tile_names[dim].extend(tiles.keys())

    def vectorize(self, axes: list[str]) -> None:
        self.vectorization.extend(axes)

    def parallelize(self, axes: list[str]) -> None:
        self.parallelization.extend(axes)

    def _parallel_dims(self) -> list[str]:
        return [d for d, k in zip(self.dims, self.kinds) if k == "P"]

    def _reduction_dims(self) -> list[str]:
        return [d for d, k in zip(self.dims, self.kinds) if k == "R"]

    def _dim_of_axis(self, axis: str) -> str | None:
        if axis in self.dims:
            return axis
        for dim, names in self.tile_names.items():
            if axis in names:
                return dim
        return None

    def _axis_tile_size(self, axis: str) -> int | None:
        for dim, names in self.tile_names.items():
            if axis in names:
                return self.tiles[dim][names.index(axis)]
        return None

    def _parallelized_dims(self) -> set[str]:
        """Parallel dimensions explicitly targeted by ``parallelize``."""
        parallel = set(self._parallel_dims())
        dims: set[str] = set()
        for axis in self.parallelization:
            dim = self._dim_of_axis(axis)
            if dim is not None and dim in parallel:
                dims.add(dim)
        return dims

    def _vector_widths(self) -> dict[str, int]:
        """Map each vectorized dimension (parallel or reduction) to its width."""
        widths: dict[str, int] = {}
        for axis in self.vectorization:
            dim = self._dim_of_axis(axis)
            if dim is None:
                raise ValueError(
                    f"IREE backend: vectorized axis {axis!r} does not map to a "
                    f"known dimension"
                )
            size = self._axis_tile_size(axis)
            if size is None:
                # Untiled base dimension: extent unknown here, use a safe width.
                size = _VECTOR_WIDTH
            if size > _MAX_VECTOR_WIDTH:
                raise NotImplementedError(
                    f"IREE backend: vector width {size} of axis {axis!r} exceeds "
                    f"the IREE vector-size limit ({_MAX_VECTOR_WIDTH}); tile the "
                    f"axis to a smaller vector size"
                )
            widths[dim] = max(widths.get(dim, 0), size)
        return widths

    def _vectorized_axis(self, d: str) -> str | None:
        """The innermost tile axis of ``d``, when it is the one vectorized."""
        names = self.tile_names.get(d, [])
        vectorized = [n for n in names if n in self.vectorization]
        if not vectorized:
            return None
        if names and vectorized == [names[-1]]:
            return names[-1]
        raise NotImplementedError(
            f"IREE backend: only the innermost tile level of dimension {d!r} may "
            f"be vectorized; got {vectorized} (innermost is {names[-1]!r})"
        )

    def _loop_levels(self, d: str) -> list[int]:
        """Tile levels of ``d`` that stay loop tiles.

        The innermost level is dropped when it is the vectorized register tile
        (it moves to a ``vector_*`` level instead).
        """
        levels = self.tiles.get(d, [])
        if self._vectorized_axis(d) is not None:
            return levels[:-1]
        return levels[:]

    def _parallel_tiles(self, d: str, distributed: set[str]) -> tuple[int, int]:
        """Return ``(distribution_tile, cache_parallel_tile)`` for parallel ``d``."""
        levels = self.tiles.get(d, [])
        if not levels:
            return 0, 0
        loop = self._loop_levels(d)
        if d in distributed:
            if len(loop) > 2:
                raise NotImplementedError(
                    f"IREE backend: dimension {d!r} has too many loop tile levels "
                    f"{loop} for distribution + cache_parallel; keep at most two "
                    f"(plus a vectorized innermost level)"
                )
            if not loop:
                # Only the vectorized level: distribute in vector-sized chunks.
                return levels[-1], 0
            return loop[0], (loop[1] if len(loop) > 1 else 0)
        # Not parallelized: no distribution level, only cache_parallel.
        if len(loop) > 1:
            raise NotImplementedError(
                f"IREE backend: dimension {d!r} is tiled with levels {loop} but "
                f"not parallelized; only one sequential (cache_parallel) level is "
                f"available — parallelize it or keep a single tile level"
            )
        return 0, (loop[0] if loop else 0)

    def compilation_info(self) -> str | None:
        """Build the ``iree_codegen.compilation_info`` attribute, or None.

        XTC loop tiling is carried to IREE through the CPU-specific
        ``iree_cpu.lowering_config``, which exposes named tiling levels:

        - ``distribution``: outermost (workgroup) tile of the parallel
          dimensions. IREE spreads these across threads, so this level is the
          only source of task parallelism and is driven solely by
          ``parallelize``. With no ``parallelize`` the dispatch runs as a single
          workgroup: the level is omitted under ``CPUDefault``, but the expert
          pipeline requires it, so an explicit all-zero ``distribution`` is
          emitted when vectorizing;
        - ``cache_parallel``: sequential parallel-dim tiling, the middle loop
          tile of a parallelized dim, or the (single) tile of a parallel dim
          that was tiled but not parallelized;
        - ``cache_reduction``: the loop tile of each reduction dimension, unless
          that tile is vectorized (then it moves to ``vector_reduction``);
        - ``vector_common_parallel`` / ``vector_reduction``: SIMD vector tiles,
          emitted only when ``vectorize`` is requested (paired with the
          ``CPUDoubleTilingExpert`` pipeline).

        Returns None only for an empty schedule (no tiling at all); an
        unmappable schedule raises instead of silently degrading.
        """
        if not self.tiles:
            return None

        parallel = self._parallel_dims()
        reduction = self._reduction_dims()

        fields: list[str] = []

        def add_field(label: str, values: list[int], force: bool = False) -> None:
            """Append a named tiling level, skipping an all-zero one unless forced."""
            if force or any(values):
                fields.append(label + " = [" + ", ".join(map(str, values)) + "]")

        # Split each parallel dim's tile levels across distribution / cache_parallel
        # (non-parallel dims contribute 0 to both).
        parallel_set = set(parallel)
        distributed = self._parallelized_dims()
        dist_values: list[int] = []
        cache_values: list[int] = []
        for d in self.dims:
            dist, cache = (
                self._parallel_tiles(d, distributed) if d in parallel_set else (0, 0)
            )
            dist_values.append(dist)
            cache_values.append(cache)

        vector_widths = self._vector_widths()
        expert = bool(vector_widths)

        # The CPUDoubleTilingExpert pipeline requires a distribution level to anchor
        # on, even all-zero (single workgroup).
        add_field("distribution", dist_values, force=expert)
        add_field("cache_parallel", cache_values)

        def cache_reduction_tile(d: str) -> int:
            if d not in reduction or d not in self.tiles:
                return 0
            if len(self.tiles[d]) > 1:
                raise NotImplementedError(
                    f"IREE backend: cannot map deeper reduction tile levels "
                    f"{self.tiles[d][1:]} of dimension {d!r}; keep a single "
                    f"reduction tile level"
                )
            # A vectorized reduction tile moves to vector_reduction, leaving no
            # cache_reduction loop level.
            loop = self._loop_levels(d)
            return loop[0] if loop else 0

        add_field("cache_reduction", [cache_reduction_tile(d) for d in self.dims])

        # SIMD vectorization: emit explicit vector levels and switch to the
        # expert pipeline so IREE actually vectorizes the requested axes.
        pipeline = "CPUDefault"
        if expert:
            pipeline = "CPUDoubleTilingExpert"
            add_field(
                "vector_common_parallel",
                [
                    vector_widths.get(d, 0) if d in parallel_set else 0
                    for d in self.dims
                ],
            )
            # Reduction dims: vectorized ones take their tile width, others stay
            # scalar (width 1), matching MLIR vectorizing the whole inner region.
            add_field(
                "vector_reduction",
                [vector_widths.get(d, 1) if d in reduction else 0 for d in self.dims],
            )

        lowering_config = "#iree_cpu.lowering_config<" + ", ".join(fields) + ">"
        translation_info = f"#iree_codegen.translation_info<pipeline = {pipeline}>"
        return (
            "#iree_codegen.compilation_info<"
            f"lowering_config = {lowering_config}, "
            f"translation_info = {translation_info}>"
        )


class IREESchedule(itf.schd.Schedule):
    """Immutable capture of the IREE scheduling decisions for a graph."""

    def __init__(self, scheduler: "IREEScheduler", nodes: list[IREENodeSchedule]):
        self._scheduler = scheduler
        self._nodes = nodes

    @property
    @override
    def scheduler(self) -> itf.schd.Scheduler:
        return self._scheduler

    def lowering_configs(self) -> dict[str, str]:
        """Map each annotated op id to its ``compilation_info`` attribute."""
        configs: dict[str, str] = {}
        for node in self._nodes:
            info = node.compilation_info()
            if info is not None:
                configs[node.op_id] = info
        return configs

    @property
    def parallelized(self) -> bool:
        """Whether the schedule distributes any dimension across threads."""
        return any(node._parallelized_dims() for node in self._nodes)

    @override
    def __str__(self) -> str:
        return str(self.lowering_configs())


class IREEScheduler(itf.schd.Scheduler):
    """Scheduler mapping XTC primitives onto IREE ``compilation_info``."""

    def __init__(self, backend: "IREEBackend", **kwargs: object) -> None:
        self._backend = backend
        self._nodes: dict[str, IREENodeSchedule] = {
            name: IREENodeSchedule(info["op_id"], info["dims"], info["kinds"])
            for name, info in backend.nodes_info.items()
        }
        # Default to the last node, mirroring the MLIR scheduler behavior.
        self._current: IREENodeSchedule | None = (
            list(self._nodes.values())[-1] if self._nodes else None
        )

    @property
    def _node(self) -> IREENodeSchedule:
        assert self._current is not None, "no schedulable node"
        return self._current

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    @override
    def schedule(self) -> itf.schd.Schedule:
        return IREESchedule(self, list(self._nodes.values()))

    @override
    def set_dims(self, dims: list[str]) -> None:
        self._node.set_dims(dims)

    @override
    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        self._node.tile(dim, tiles)

    @override
    def vectorize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        self._node.vectorize(axes)

    @override
    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        self._node.parallelize(axes)

    @override
    def split(
        self, dim: str, segments: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None:
        raise NotImplementedError("IREE backend does not support split()")

    @override
    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT) -> None:
        raise NotImplementedError("IREE backend does not support interchange()")

    @override
    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        raise NotImplementedError("IREE backend does not support unroll()")

    @override
    def buffer_at(
        self, axis: str, mtype: str | None = None, root: str = DEFAULT_ROOT
    ) -> None:
        raise NotImplementedError(
            "IREE backend does not support buffer_at() (bufferized by IREE)"
        )

    @override
    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        root: str = DEFAULT_ROOT,
    ) -> None:
        raise NotImplementedError(
            "IREE backend does not support pack_at() (bufferized by IREE)"
        )

    @override
    def fuse_producer_at(
        self, axis: str, input_idx: int, root: str = DEFAULT_ROOT
    ) -> None:
        raise NotImplementedError("IREE backend does not support fuse_producer_at()")

    @override
    def define_memory_mesh(self, axes: dict[str, int]) -> None:
        raise NotImplementedError("IREE backend does not support memory meshes")

    @override
    def define_processor_mesh(self, axes: dict[str, int]) -> None:
        raise NotImplementedError("IREE backend does not support processor meshes")

    @override
    def distribute(
        self, axis: str, processor_axis: str, root: str = DEFAULT_ROOT
    ) -> None:
        raise NotImplementedError("IREE backend does not support explicit distribution")

    @override
    def distributed_buffer_at(
        self,
        axis: str,
        input_idx: int,
        memory_axes: list[str],
        root: str = DEFAULT_ROOT,
    ) -> None:
        raise NotImplementedError("IREE backend does not support distributed buffers")

    @override
    def get_loop_nest(self) -> LoopNest:
        raise NotImplementedError("IREE backend does not expose a loop nest")
