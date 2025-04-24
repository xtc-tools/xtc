#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import abstractmethod
from typing import TypeAlias, Any
from typing_extensions import override
from collections.abc import Sequence, Mapping, Iterator
import itertools
import numpy as np

from xtc.itf.graph import Graph
from xtc.itf.schd import Scheduler
from xtc.itf.search import Sample, Strategy
from xtc.utils.math import (
    factors_to_sizes,
    factors_enumeration,
)


__all__ = [
    "Strategies",
]

VecSample: TypeAlias = list[int]


class BaseStrategy(Strategy):
    """Base abstract class for implementing the strategies in this file.

    All strategies in this file define the search space as set of samples
    which are 1-D int vectors of type VecSample.
    """

    def __init__(
        self,
        graph: Graph,
        vec_size: int = 16,
        max_unroll: int = 256,
        threads: int = 1,
    ) -> None:
        self._graph = graph
        self._vec_size = vec_size
        self._max_unroll = max_unroll
        self._threads = threads
        # Schedule output operation
        self._op = graph.outputs_nodes[0].operation
        self._stats: dict[str, int] = {}

    @property
    @override
    def graph(self) -> Graph:
        return self._graph

    @override
    def generate(self, scheduler: Scheduler, sample: Sample) -> None:
        # Ensure sample is valid list kind
        in_x = list(sample)
        self._generate(scheduler, in_x)

    @override
    def exhaustive(self) -> Iterator[VecSample]:
        return self._exhaustive()

    @override
    def default_schedule(self, opt_level: int = 2) -> VecSample:
        return self._default_schedule()

    @abstractmethod
    def _generate(self, sch: Scheduler, in_x: list[int]) -> None: ...

    @abstractmethod
    def _exhaustive(self) -> Iterator[VecSample]: ...

    @abstractmethod
    def _default_schedule(self, opt_level: int = 2) -> list[int]: ...

    @property
    def stats(self) -> Mapping[str, int]:
        return self._stats

    def _constant_sizes(self) -> Mapping[str, int]:
        sizes = {a: v for a, v in self._op.dims.items() if isinstance(v, int)}
        return sizes

    def _constant_p_sizes(self) -> Mapping[str, int]:
        p_axes = self._op.dims_kind("P")
        sizes = {
            a: v for a, v in self._op.dims.items() if isinstance(v, int) and a in p_axes
        }
        return sizes

    def _constant_r_sizes(self) -> Mapping[str, int]:
        r_axes = self._op.dims_kind("R")
        sizes = {
            a: v for a, v in self._op.dims.items() if isinstance(v, int) and a in r_axes
        }
        return sizes

    def _iter_product(
        self, args: Sequence[Sequence[Sequence[int]]], stat: str = ""
    ) -> Iterator[VecSample]:
        if stat:
            self._stats[stat] = 0
        for x in itertools.product(*args):
            self._stats[stat] += 1
            yield list(itertools.chain(*x))

    def _vector_axis(self) -> str | None:
        p_dims = list(self._op.dims_kind("P"))
        return p_dims[-1] if p_dims else None

    def _filter_unroll(
        self,
        indexes: list[int],
        v_index: int | None,
        samples: Iterator[VecSample],
        stat: str = "",
    ) -> Iterator[VecSample]:
        # Filter inner n_axes unrolled tiles if > max_unroll
        # assuming inner is vectorized
        if stat:
            self._stats[stat] = 0
        for x in samples:
            inners = np.array(x)[indexes]
            inner_unroll = np.prod(inners)
            vec_size = min(x[v_index] if v_index is not None else 1, self._vec_size)
            if inner_unroll / vec_size <= self._max_unroll:
                self._stats[stat] += 1
                yield x

    def _default_schedule_inner_level(self, opt_level: int = 2) -> Mapping[str, int]:
        sizes = self._constant_sizes()
        p_sizes = self._constant_p_sizes()
        schedule = {axis: 1 for axis in sizes}
        if opt_level >= 3:
            vsize = self._vec_size
            vaxis = self._vector_axis()
            prev_axis = None
            prev_axes = [axis for axis in p_sizes if axis != vaxis]
            if len(prev_axes) >= 1:
                prev_axis = prev_axes[-1]
            if vaxis and p_sizes[vaxis] >= vsize and p_sizes[vaxis] % vsize == 0:
                schedule[vaxis] = vsize
            prev_unroll = 2  # TODO: IPC?
            if (
                prev_axis
                and p_sizes[prev_axis] >= prev_unroll
                and p_sizes[prev_axis] % prev_unroll == 0
            ):
                schedule[prev_axis] = prev_unroll
        return schedule


class Strategy_T1(BaseStrategy):
    """Strategy for 1-level tiling.

    Given P-axes: p1,...pn and R-axes: r1,...rn.
    We define the outer P-axis as O, and the remaining P-axis as P
    and all R-axes as R.
    The ordering of tiles is: ORPORP
    The generated sample is the tile factors for the inner ORP
    in the initial axes order.
    All the inner ORP axes are unrolled.
    The innermost axis of P is vectorized.
    The O axis is parallelized.

    For instance on a matmul(i, j, k):
    - sample [2, 16, 3] we have a schedule:
      - order: i, k, j, i1, k1, j1
      - unroll(i1, k1, j1)
      - vector(j1)
      - parallel(i)

    TODO:
    - this may be generalized to N-level tiling
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _generate(self, sch: Scheduler, in_x: list[int]) -> None:
        p_axes = self._op.dims_kind("P")
        r_axes = self._op.dims_kind("R")
        sizes = self._constant_sizes()
        assert len(in_x) == len(sizes)
        tilings = {}
        N = 1  # Tile level for P
        for axis, factors in [
            (axis, in_x[idx * N : (idx + 1) * N]) for idx, axis in enumerate(sizes)
        ]:
            tilings[axis] = {
                f"{axis}{idx + 1}": size
                for idx, size in enumerate(factors_to_sizes(list(factors)))
            }
        axes_order = [*p_axes[:1], *r_axes, *p_axes[1:]]
        for t in range(N):
            axes_order.extend(
                [
                    *[f"{axis}{t + 1}" for axis in p_axes[:1]],
                    *[f"{axis}{t + 1}" for axis in r_axes],
                    *[f"{axis}{t + 1}" for axis in p_axes[1:]],
                ]
            )
        parallel_axes = []
        if self._threads > 1:
            parallel_axes = axes_order[:1]
        vector_axes = axes_order[-1:] if tilings else []
        unroll_axes = {
            axis: tilings[axis[:-1]][axis]
            for axis in reversed(axes_order[-len(sizes) :])
        }
        for axis, tiling in tilings.items():
            sch.tile(axis, tiling)
        sch.interchange(axes_order)
        sch.parallelize(parallel_axes)
        sch.vectorize(vector_axes)
        sch.unroll(unroll_axes)

    @override
    def _exhaustive(self) -> Iterator[VecSample]:
        sizes = self._constant_sizes()
        N = 1  # Tile level for P
        tiles = [factors_enumeration(size, N) for size in sizes.values()]
        all_samples = self._iter_product(tiles, stat="all")
        indexes = [a * N for a in range(len(sizes))]
        vaxis = self._vector_axis()
        v_index = list(sizes.keys()).index(vaxis) * N if vaxis else None
        filtered_samples = self._filter_unroll(
            indexes, v_index, all_samples, stat="filtered"
        )
        return filtered_samples

    @override
    def _default_schedule(self, opt_level: int = 2) -> list[int]:
        sizes = self._constant_sizes()
        inner_sizes = self._default_schedule_inner_level(opt_level)
        N = 1  # Tile level for P
        schedule = [1] * (N - 1) * len(sizes)
        schedule.extend([inner_sizes.get(axis, 1) for axis in sizes])
        return schedule


class Strategy_PRP(BaseStrategy):
    """Strategy for PRP tiling.

    Given P-axes: p1,...pn and R-axes: r1,...rn.
    We define all P-axes as P and all R-axes as R.
    The ordering of tiles is: PRP
    The generated sample is the tile factors for the inner P
    in the initial axes order.
    All the inner P axes are unrolled.
    The innermost axis of P vectorized.
    The outermost axis of P parallelized.

    For instance on a matmul(i, j, k):
    - sample [2, 16] we have a schedule:
      - order: i, j, k, i1, j1
      - unroll(i1, j1)
      - vector(j1)
      - parallel(i)

    TODO:
    - this may be generalized to N-level tiling
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _generate(self, sch: Scheduler, in_x: list[int]) -> None:
        p_axes = self._op.dims_kind("P")
        r_axes = self._op.dims_kind("R")
        p_sizes = self._constant_p_sizes()
        assert len(in_x) == len(p_sizes)
        tilings = {}
        N = 1  # Tile level for P
        for axis, factors in [
            (axis, in_x[idx * N : (idx + 1) * N]) for idx, axis in enumerate(p_sizes)
        ]:
            tilings[axis] = {
                f"{axis}{idx + 1}": size
                for idx, size in enumerate(factors_to_sizes(list(factors)))
            }
        axes_order = [
            *p_axes,
            *r_axes,
            *[f"{axis}{n * N + 1}" for n in range(N) for axis in p_sizes],
        ]
        parallel_axes = []
        if self._threads > 1:
            parallel_axes = axes_order[:1]
        vector_axes = axes_order[-1:] if tilings else []
        unroll_axes = {
            inner: size
            for inner, size in [
                list(sizes.items())[-1] for sizes in reversed(tilings.values())
            ]
        }
        for axis, tiling in tilings.items():
            sch.tile(axis, tiling)
        sch.interchange(axes_order)
        sch.parallelize(parallel_axes)
        sch.vectorize(vector_axes)
        sch.unroll(unroll_axes)

    @override
    def _exhaustive(self) -> Iterator[VecSample]:
        p_sizes = self._constant_p_sizes()
        N = 1  # Tile level for P
        tiles = [factors_enumeration(size, N) for size in p_sizes.values()]
        all_samples = self._iter_product(tiles, stat="all")
        indexes = [a * N for a in range(len(p_sizes))]
        vaxis = self._vector_axis()
        v_index = list(p_sizes.keys()).index(vaxis) * N if vaxis else None
        filtered_samples = self._filter_unroll(
            indexes, v_index, all_samples, stat="filtered"
        )
        return filtered_samples

    @override
    def _default_schedule(self, opt_level: int = 2) -> list[int]:
        p_sizes = self._constant_p_sizes()
        inner_sizes = self._default_schedule_inner_level(opt_level)
        N = 1  # Tile level for P
        schedule = [1] * (N - 1) * len(p_sizes)
        schedule.extend([inner_sizes.get(axis, 1) for axis in p_sizes])
        return schedule


class Strategy_P1(BaseStrategy):
    """Strategy for 1-level tiling with permutation.

    Given all axes: a1,...,an as T.
    The ordering of tiles is: T, perm(T)
    Where perm(T) is a permutation of the axes.
    The generated sample is the tile factors for the inner T
    in the initial axis order, plus the permutation index.
    All the inner perm(T) axes are unrolled.
    The innermost axis of perm(T) is vectorized if it's
    the initial last parallel axis.
    Up to 2 outermost axis are parallelized if parallel.

    TODO: The implementation is limited to matmult like ops.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

        # TODO: for now limited to 3 axes i, j, k (i.e. matmul)
        # no need to be matmul specific as soon as
        # we have axes names
        # actually PPRPRP -> i j i1 j1 k i2 j2 k1 i3 j3
        # where the input vector is: i1 i2 i3 j1 j2 j3 k1
        assert tuple(self._op.dims) == ("i", "j", "k")
        assert tuple(self._op.dims_kind("P")) == ("i", "j")
        assert tuple(self._op.dims_kind("R")) == ("k",)

        # Precompute permutations of the inner tiles
        self._permutations = list(itertools.permutations(["i1", "j1", "k1"]))
        # Precompute permutation values for which j1 is inner (i.e. vectorizable)
        self._valid_vector_idx = [
            idx for idx, perm in enumerate(self._permutations) if perm[-1] == "j1"
        ]

    @override
    def _generate(self, sch: Scheduler, in_x: list[int]) -> None:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        ti = in_x[0]
        tj = in_x[1]
        tk = in_x[2]
        order = in_x[3]
        permutations = list(self._permutations[order])
        tiles = {"i1": ti, "j1": tj, "k1": tk}
        axes_order = ["i", "j", "k"] + permutations
        vector_axes = [axes_order[-1]] if axes_order[-1] == "j1" else []
        parallel_axes = []
        if self._threads > 1:
            parallel_axes += [axes_order[0]] if axes_order[0] in ["i", "j"] else []
            parallel_axes += [axes_order[1]] if axes_order[1] in ["i", "j"] else []
        unroll_axes = {axis: tiles[axis] for axis in permutations[::-1]}
        sch.tile("i", {"i1": ti})
        sch.tile("j", {"j1": tj})
        sch.tile("k", {"k1": tk})
        sch.interchange(axes_order)
        sch.parallelize(parallel_axes)
        sch.vectorize(vector_axes)
        sch.unroll(unroll_axes)

    @override
    def _filter_unroll(
        self,
        indexes: list[int],
        v_index: int | None,
        samples: Iterator[VecSample],
        stat: str = "",
    ) -> Iterator[VecSample]:
        # Filter inner n_axes unrolled tiles if > max_unroll
        # assuming inner is vectorized
        if stat:
            self._stats[stat] = 0
        for x in samples:
            inners = np.array(x)[indexes]
            inner_unroll = np.prod(inners)
            # Specific for checking valid vector permutations
            vsize = x[v_index] if v_index is not None else 1
            vsize = vsize if x[-1] in self._valid_vector_idx else 1
            vec_size = min(vsize, self._vec_size)
            if inner_unroll / vec_size <= self._max_unroll:
                self._stats[stat] += 1
                yield x

    @override
    def _exhaustive(self) -> Iterator[VecSample]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        i, j, k = self._constant_sizes().values()
        tiles_i = factors_enumeration(i, 1)
        tiles_j = factors_enumeration(j, 1)
        tiles_k = factors_enumeration(k, 1)
        orders = [[x] for x in range(len(self._permutations))]
        all_samples = self._iter_product(
            [tiles_i, tiles_j, tiles_k, orders], stat="all"
        )
        v_index = 1  # index of j1
        indexes = [0, 1, 2]  # indexs of i1, j1, k1
        filtered_samples = self._filter_unroll(
            indexes, v_index, all_samples, stat="filtered"
        )
        return filtered_samples

    @override
    def _default_schedule(self, opt_level: int = 2) -> list[int]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        i, j, k = i, j, k = self._constant_sizes().values()
        schedule = [1, 1, 1, 0]
        if opt_level >= 2:
            schedule = [1, 1, 1, 1]
        if opt_level >= 3:
            jtile = self._vec_size
            itile = 2  # TODO: IPC?
            ktile = 1
            idiv = i >= itile and i % itile == 0
            jdiv = j >= jtile and j % jtile == 0
            kdiv = k >= ktile and k % ktile == 0
            if idiv and jdiv and kdiv:
                schedule = [itile, jtile, ktile, 1]
        return schedule


class Strategy_P1v(Strategy_P1):
    """Strategy for 1-level tiling with permutation and vectorization.

    Same as Strategy_P1, but space is constraint to vectorized inner axis.

    TODO: The implementation is limited to matmult like ops.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _exhaustive(self) -> Iterator[VecSample]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        samples = super()._exhaustive()
        # Keep only vectorized dims, i.e. good permutation and j1 >= VEC_SIZE
        vidx = 1  # index of j1
        for x in samples:
            if x[-1] in self._valid_vector_idx and x[vidx] >= self._vec_size:
                yield x


class Strategy_PPRPRP(BaseStrategy):
    """Strategy for Ansor like tiling with.

    Given P-axes: p1,...,pn and R-axies: r1, ..., rn
    We define all P-axes as P and all R-axes as R.
    The ordering of tiles is: PPRPRP
    The generated sample is the tile factors for each axes
    in the initial axes order.
    All the inner RP axes are unrolled.
    The innermost axis of P vectorized.
    The outermost axis of P parallelized.

    Refer also to Strategy_PRP, a simplified version for examples.

    TODO: The implementation is limited to matmult like ops.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

        # TODO: for now limited to 3 axes i, j, k (i.e. matmul)
        # no need to be matmul specific as soon as
        # we have axes names
        # actually PPRPRP -> i j i1 j1 k i2 j2 k1 i3 j3
        # where the input vector is: i1 i2 i3 j1 j2 j3 k1
        assert tuple(self._op.dims) == ("i", "j", "k")
        assert tuple(self._op.dims_kind("P")) == ("i", "j")
        assert tuple(self._op.dims_kind("R")) == ("k",)

    @override
    def _generate(self, sch: Scheduler, in_x: list[int]) -> None:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        tiles_i = factors_to_sizes(list(in_x[0:3]))
        tiles_j = factors_to_sizes(list(in_x[3:6]))
        tiles_k = factors_to_sizes(list(in_x[6:7]))
        tiles_i_dict = {f"i{i + 1}": v for i, v in enumerate(tiles_i)}
        tiles_j_dict = {f"j{i + 1}": v for i, v in enumerate(tiles_j)}
        tiles_k_dict = {f"k{i + 1}": v for i, v in enumerate(tiles_k)}
        axes_order = ["i", "j", "i1", "j1", "k", "i2", "j2", "k1", "i3", "j3"]
        vector_axes = axes_order[-1:]
        parallel_axes = []
        if self._threads > 1:
            parallel_axes = axes_order[:2]
        unroll_axes = {"j3": tiles_j[-1], "i3": tiles_i[-1], "k1": tiles_k[-1]}
        sch.tile("i", tiles_i_dict)
        sch.tile("j", tiles_j_dict)
        sch.tile("k", tiles_k_dict)
        sch.interchange(axes_order)
        sch.parallelize(parallel_axes)
        sch.vectorize(vector_axes)
        sch.unroll(unroll_axes)

    @override
    def _exhaustive(self) -> Iterator[VecSample]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        i, j, k = self._constant_sizes().values()
        tiles_i = factors_enumeration(i, 3)
        tiles_j = factors_enumeration(j, 3)
        tiles_k = factors_enumeration(k, 1)
        all_samples = self._iter_product([tiles_i, tiles_j, tiles_k], stat="all")
        v_index = 5  # index of j3
        indexes = [2, 5, 6]  # indexs of i3, j3, k1
        filtered_samples = self._filter_unroll(
            indexes, v_index, all_samples, stat="filtered"
        )
        return filtered_samples

    @override
    def _default_schedule(self, opt_level: int = 2) -> list[int]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        i, j, k = self._constant_sizes().values()

        def sched_o2():
            jtile = self._vec_size
            itile = 2
            ktile = 1
            idiv = i >= itile and i % itile == 0
            jdiv = j >= jtile and j % jtile == 0
            kdiv = k >= ktile and k % ktile == 0
            if idiv and jdiv and kdiv:
                return [1, 1, itile, 1, 1, jtile, ktile]
            return None

        def sched_o3():
            jtile = self._vec_size * 4
            itile = 4
            ktiles = factors_enumeration(k, 1)
            ktile = [x[0] for x in ktiles if x[0] <= 16][-1]
            idiv = i >= itile and i % itile == 0
            jdiv = j >= jtile and j % jtile == 0
            kdiv = k >= ktile and k % ktile == 0
            if idiv and jdiv and kdiv:
                return [i // itile, 1, itile, 1, j // jtile, jtile, ktile]
            return None

        schedule = [1, 1, 1, 1, 1, 1, 1]
        if opt_level >= 2:
            o2 = sched_o2()
            if o2:
                schedule = o2
        if opt_level >= 3:
            o3 = sched_o3()
            if o3:
                schedule = o3
        return schedule


class Strategy_PPRPRPv(Strategy_PPRPRP):
    """Strategy for Ansor like tiling with space vectorization.

    Same as Strategy_PPRPRP, but with an additional constraint for the
    space to have the inner axis vectorized.

    TODO: The implementation is limited to matmult like ops.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _exhaustive(self) -> Iterator[VecSample]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        samples = super()._exhaustive()
        # Keep only vectorized dims, i.e. j3 >= VEC_SIZE
        vidx = 5  # index of j3
        for x in samples:
            if x[vidx] >= self._vec_size:
                yield x


class Strategy_PPRPRPvr(Strategy_PPRPRP):
    """Strategy for Ansor like tiling with space constraints.

    Same as Strategy_PPRPRP, but with an additional constraints on the
    space:
    - vectorized inner axis
    - inner P vector size lower than machine vector registers
    - inner RP bytes size lower than machine L1
    - inner PRP bytes size lower than machine L2

    TODO: The implementation is limited to matmult like ops.
    """

    # TODO: should go into some machine description
    _MAX_VREG = 32
    _MAX_L1_ELTS = 32 * 1024 / 4  # where 4 is float size (TODO)
    _MAX_L2_ELTS = 1024 * 1024 / 4  # where 4 is float size (TODO)

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _exhaustive(self) -> Iterator[VecSample]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        samples = super()._exhaustive()
        for x in samples:
            # Keep only vectorized dims, i.e. j3 >= VEC_SIZE
            if not x[5] >= self._vec_size:
                continue
            # Keep only inner i*j <= VEC_SIZE * MAX_REG
            if not x[2] * x[5] <= self._vec_size * self._MAX_VREG:
                continue
            # Keep only inner k*i*j <= MAX_L1_ELTS
            if not x[2] * x[5] * x[6] <= self._MAX_L1_ELTS:
                continue
            # Keep only inner 2 k*i*j <= MAX_L2_ELTS
            if not x[1] * x[2] * x[4] * x[5] * x[6] <= self._MAX_L2_ELTS:
                continue
            yield x


class Strategy_PPWRPRP(Strategy_PPRPRP):
    """Strategy for Ansor like tiling and write buffer.

    Same as Strategy_PPRPRP, but with an additional write buffer.
    The scheduler is PPWPRPR where W is the location of the
    allocated write buffer in the tiling.

    TODO: The implementation is limited to matmult like ops.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _generate(self, sch: Scheduler, in_x: list[int]) -> None:
        if in_x[-1] != 0:
            sch.buffer_at("j1", "write")

    @override
    def _exhaustive(self) -> Iterator[VecSample]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        samples = super()._exhaustive()
        for x in samples:
            yield x + [0]
            yield x + [1]

    @override
    def _default_schedule(self, opt_level: int = 2) -> list[int]:
        schedule = super()._default_schedule(opt_level)
        wc = 1 if opt_level >= 2 else 0
        return schedule + [wc]


class Strategy_PPWRPRPv(Strategy_PPWRPRP, Strategy_PPRPRPv):
    """Strategy for Ansor like tiling and write buffer with space vectorization.

    Same as Strategy_PPWRPRP, but with an additional constraint for the
    space to have the inner axis vectorized as in strategy PPRPRPV.

    TODO: The implementation is limited to matmult like ops.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _exhaustive(self) -> Iterator[VecSample]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        samples = Strategy_PPRPRPv.exhaustive(self)
        for x in samples:
            yield x + [0]
            yield x + [1]


class Strategy_PPWRPRPvr(Strategy_PPWRPRP, Strategy_PPRPRPvr):
    """Strategy for Ansor like tiling and write buffer with space constraints.

    Same as Strategy_PPWRPRP, but with additional constraints for the
    space as in strategy PPRPRPvr.

    TODO: The implementation is limited to matmult like ops.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _exhaustive(self) -> Iterator[VecSample]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        samples = Strategy_PPRPRPvr.exhaustive(self)
        for x in samples:
            yield x + [0]
            yield x + [1]


class Strategies:
    @classmethod
    def names(cls) -> Sequence[str]:
        return list(cls._map.keys())

    @classmethod
    def from_name(cls, name: str) -> type[BaseStrategy]:
        if name not in cls._map:
            raise ValueError(f"unknown strategy name: {name}")
        return cls._map[name]

    _map = {
        "tile_t1": Strategy_T1,
        "tile_p1": Strategy_P1,
        "tile_p1_v": Strategy_P1v,
        "tile_prp": Strategy_PRP,
        "tile_pprprp": Strategy_PPRPRP,
        "tile_pprprp_v": Strategy_PPRPRPv,
        "tile_pprprp_vr": Strategy_PPRPRPvr,
        "tile_ppwrprp": Strategy_PPWRPRP,
        "tile_ppwrprp_v": Strategy_PPWRPRPv,
        "tile_ppwrprp_vr": Strategy_PPWRPRPvr,
    }
