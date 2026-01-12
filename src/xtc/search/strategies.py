#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import abstractmethod
from typing import TypeAlias, Any
from typing_extensions import override
from collections.abc import Sequence, Mapping, Iterator, Generator
import itertools
import numpy as np

from xvs.properties import constraints_from_str, hypergraph, Context
from xvs.strategy import (
    execute_dynamic,
    execute_static,
    solve_with_z3,
    pretty_print_methods,
)
from xtc.itf.graph import Graph
from xtc.itf.schd import Scheduler
from xtc.itf.schd.scheduler import DEFAULT_ROOT
from xtc.itf.search import Sample, Strategy
from xtc.schedules.descript_extend import DescriptExtend
from xtc.utils.math import (
    factors_to_sizes,
    factors_enumeration,
)
from xtc.utils.algorithms import (
    sample_uniques,
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
        sample_names: list[str],
        vec_size: int = 16,
        max_unroll: int = 256,
        threads: int = 1,
        max_parallelize: int = 1,
        **kwargs: Any,
    ) -> None:
        self._graph = graph
        self._sample_names = sample_names
        self._vec_size = vec_size
        self._max_unroll = max_unroll
        self._threads = threads
        # Schedule output operation
        self._op = graph.outputs_nodes[0].operation
        self._stats: dict[str, int] = {}
        self._parallelize = self._threads > 1
        self._max_parallelize = max_parallelize
        self._vectorize = self._vec_size > 1
        self._unroll = self._max_unroll != 0
        # TODO: should go into some machine description
        self._arch_vreg_num = kwargs.get("vreg_num", 32)
        self._arch_l1_size = kwargs.get("l1_size", 32 * 1024)
        self._arch_l2_size = kwargs.get("l2_size", 1024 * 1024)

    @property
    @override
    def graph(self) -> Graph:
        return self._graph

    @property
    @override
    def sample_names(self) -> list[str]:
        return self._sample_names

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
        assert opt_level >= 0
        return self._default_schedule(opt_level)

    @override
    def sample(self, num: int, seed: int | None = 0) -> Iterator[VecSample]:
        assert num > 0
        assert seed is None or seed >= 0
        return self._sample(num, seed)

    @override
    def dict_to_sample(self, sample: dict[str, Any]) -> Sample:
        return list(sample.values())

    @override
    def sample_to_dict(self, sample: Sample) -> dict[str, int]:
        return dict(zip(self.sample_names, sample))

    @abstractmethod
    def _generate(self, sch: Scheduler, in_x: list[int]) -> None: ...

    @abstractmethod
    def _default_schedule(self, opt_level: int) -> list[int]: ...

    def _exhaustive(self) -> Iterator[VecSample]:
        inds = self._independents()
        samples = self._iter_product(inds, stat="all")
        filtered = self._filter(samples)
        return filtered

    @property
    def stats(self) -> Mapping[str, int]:
        return self._stats

    def _constant_sizes(self) -> Mapping[str, int]:
        sizes = {a: v for a, v in self._op.dims.items() if isinstance(v, int)}
        return sizes

    def _iter_product(
        self, args: Sequence[Sequence[Sequence[int]]], stat: str | None = None
    ) -> Iterator[VecSample]:
        if stat:
            self._stats[stat] = 0
        for x in itertools.product(*args):
            if stat:
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
        stat: str | None = None,
    ) -> Iterator[VecSample]:
        # Filter inner n_axes unrolled tiles if > max_unroll
        # assuming inner is vectorized
        if stat:
            self._stats[stat] = 0
        if self._max_unroll < 0:
            yield from samples
            return
        for x in samples:
            inners = np.array(x)[indexes]
            inner_unroll = np.prod(inners)
            vec_size = min(x[v_index] if v_index is not None else 1, self._vec_size)
            if inner_unroll / vec_size <= self._max_unroll:
                if stat:
                    self._stats[stat] += 1
                yield x

    def _sample_product(
        self, inds: list[list[list[int]]], num: int, rng: np.random.Generator
    ) -> list[VecSample]:
        draw = np.hstack(
            [
                np.array(var, dtype="int")[rng.integers(len(var), size=num)]
                for var in inds
            ]
        )
        return draw.tolist()

    def _sample(self, num: int, seed: int | None = 0) -> Iterator[VecSample]:
        rng = np.random.default_rng(seed=seed)
        inds = self._independents()

        def draw(num: int) -> Generator[tuple[int, ...]]:
            samples = self._sample_product(inds, num, rng)
            filtered = self._filter(iter(samples))
            return (tuple(sample) for sample in filtered)

        samples = sample_uniques(draw, num)
        return iter(list(sample) for sample in samples[:num])

    @abstractmethod
    def _independents(self) -> list[list[list[int]]]: ...

    @abstractmethod
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]: ...


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

    TODO: The implementation is limited to matmult like ops,
          refactor with PRTScheme with `U` item when implemented
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, ["ti", "tj", "tk", "order"], **kwargs)

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
        sch.tile("i", {"i1": ti}, root=".")
        sch.tile("j", {"j1": tj}, root=".")
        sch.tile("k", {"k1": tk}, root=".")
        sch.interchange(axes_order, root=".")
        sch.parallelize(parallel_axes, root=".")
        sch.vectorize(vector_axes, root=".")
        sch.unroll(unroll_axes, root=".")

    @override
    def _filter_unroll(
        self,
        indexes: list[int],
        v_index: int | None,
        samples: Iterator[VecSample],
        stat: str | None = None,
    ) -> Iterator[VecSample]:
        # Filter inner n_axes unrolled tiles if > max_unroll
        # assuming inner is vectorized
        if stat:
            self._stats[stat] = 0
        if self._max_unroll < 0:
            yield from samples
            return
        for x in samples:
            inners = np.array(x)[indexes]
            inner_unroll = np.prod(inners)
            # Specific for checking valid vector permutations
            vsize = x[v_index] if v_index is not None else 1
            vsize = vsize if x[-1] in self._valid_vector_idx else 1
            vec_size = min(vsize, self._vec_size)
            if inner_unroll / vec_size <= self._max_unroll:
                if stat:
                    self._stats[stat] += 1
                yield x

    @override
    def _independents(self) -> list[list[list[int]]]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        i, j, k = self._constant_sizes().values()
        tiles_i = factors_enumeration(i, 1)
        tiles_j = factors_enumeration(j, 1)
        tiles_k = factors_enumeration(k, 1)
        orders = [[x] for x in range(len(self._permutations))]
        return [tiles_i, tiles_j, tiles_k, orders]

    @override
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]:
        v_index = 1  # index of j1
        indexes = [0, 1, 2]  # indexs of i1, j1, k1
        samples = self._filter_unroll(indexes, v_index, samples, stat="filtered")
        return samples

    @override
    def _default_schedule(self, opt_level: int) -> list[int]:
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

    TODO: The implementation is limited to matmult like ops,
          refactor with PRTScheme with `U` item when implemented
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]:
        samples = super()._filter(samples)
        # Keep only vectorized dims, i.e. good permutation and j1 >= VEC_SIZE
        vidx = 1  # index of j1
        for x in samples:
            if x[-1] in self._valid_vector_idx and x[vidx] >= self._vec_size:
                yield x


class BaseStrategyPRTScheme(BaseStrategy):
    """Base Strategy for PR... tiling sequences.

    Given P-axes: p1,...pn and R-axes: r1,...rn.
    We define several tile reodering item as:
    - `P`: all P-axes in order
    - `R`: all R-axes in order
    - `T`: all axes in order
    - `U`: all axes in random order, TODO: not implemented yet
    - `O`: first P-axis then R-axes, then remaining P-axis

    We also define dditional items which can be mixed with tile items:
    - `W`: add an output write buffer at this level

    Then we can define scheme as a sequence of items, for instance:
    - `PRP`: P-axes, R-axes then a for all P-axes
    - `TT`: all axes, then a tile for all axes
    - `ORP`: all axes in O order, then tile for all R-axes, then tile for all P-axes
    - `PWRP`: P-axes, a write buffer, then R-axes then inner P-axes tiles

    In addition by default:
    - inner axis, if parallel is vectorized
    - inner tile for all tiled axes are unrolled
    - outer axis, if parallel is parallelized

    For instance on a matmul(i, j, k) with scheme `OO`:
    - axes order will be: `i, k, j, i1, k1, j1`
    - axis j1 is vectorized
    - axes j1, k1, i1 are unrolled
    - axis i is parallelized

    """

    def __init__(self, graph: Graph, scheme: str, **kwargs: Any) -> None:
        super().__init__(graph, [], **kwargs)
        self._scheme = scheme
        self._axes = list(self._op.dims)
        self._sizes = self._constant_sizes()
        self._p_axes = list(self._op.dims_kind("P"))
        self._r_axes = list(self._op.dims_kind("R"))
        (
            self._order,
            self._tiles_levels,
            self._base_axis,
            self._item_map,
            self._w_axes,
        ) = self._init_order_tiles()
        self._tiles_idx = self._init_tiles_idx()
        self._x_size = sum([level - 1 for level in self._tiles_levels.values()])
        self._parallel = self._init_parallel()
        self._unrolled = self._init_unrolled()
        self._vectorized = self._init_vectorized()
        self._sample_names = self._init_sample_names()
        self._out_dtype = self._op.outputs_types[0].dtype
        assert len(self._vectorized) <= 1

    def _init_order_tiles(
        self,
    ) -> tuple[
        list[str], dict[str, int], dict[str, str], dict[int, list[str]], list[str]
    ]:
        base_axis = {}
        item_map = {}
        order = []
        w_axes = []
        levels = {axis: 0 for axis in self._axes}
        prev_axis = ""
        for item_idx, item in enumerate(self._scheme):
            axis_lst: list[str] = []
            if item == "P":
                axis_lst = self._p_axes
            elif item == "R":
                axis_lst = self._r_axes
            elif item == "T":
                axis_lst = self._axes
            elif item == "O":
                # Item O is outer parallel first, reductions, remaining parallels
                axis_lst = [*self._p_axes[:1], *self._r_axes, *self._p_axes[1:]]
            else:
                assert item == "W"
                axis_lst = []
                assert prev_axis
                w_axes.append(prev_axis)
            item_axes = []
            for axis in axis_lst:
                tiled_axis = f"{axis}" if levels[axis] == 0 else f"{axis}{levels[axis]}"
                item_axes.append(tiled_axis)
                levels[axis] += 1
                base_axis[tiled_axis] = axis
            order.extend(item_axes)
            item_map[item_idx] = item_axes
            prev_axis = item_axes[-1] if item_axes else ""
        return order, levels, base_axis, item_map, w_axes

    def _init_tiles_idx(self) -> dict[str, int]:
        tiles_idx = {}
        idx = 0
        for axis in self._axes:
            levels = self._tiles_levels[axis]
            assert levels >= 1, f"no tile level specified for axis: {axis}"
            tiles_idx[axis] = idx
            for level in range(levels - 1):
                tiled_axis = f"{axis}{level + 1}"
                tiles_idx[tiled_axis] = idx
                idx += 1
        return tiles_idx

    def _init_unrolled(self) -> list[str]:
        if not self._unroll:
            return []
        max_unroll_num = 0
        if (
            self._scheme[-1:] == "T"
            or self._scheme[-1:] == "O"
            or self._scheme[-2:] == "PR"
            or self._scheme[-2:] == "RP"
        ):
            max_unroll_num = len(self._axes)
        elif self._scheme[-1:] == "P":
            max_unroll_num = len(self._p_axes)
        elif self._scheme[-1:] == "R":
            max_unroll_num = len(self._r_axes)
        unrolled = []
        # Ensure that unrolled axes are only
        # inner tiled axis (i.e. not base axis)
        for idx, axis in enumerate(self._order[::-1]):
            if idx >= max_unroll_num:
                break
            if axis == self._base_axis[axis]:
                break
            unrolled.append(axis)
        return unrolled

    def _init_vectorized(self) -> list[str]:
        if not self._vectorize:
            return []
        max_vectorized = 1
        vectorized = []
        # Ensure that the vectorized axis is the
        # inner axis, is tiled and is parallel
        for idx, axis in enumerate(self._order[::-1]):
            if idx >= max_vectorized:
                break
            if axis == self._base_axis[axis]:
                break
            if self._base_axis[axis] != self._vector_axis():
                break
            vectorized.append(axis)
        return vectorized

    def _init_parallel(self) -> list[str]:
        if not self._parallelize:
            return []
        parallel = []
        # Ensure that the parallel axes are the outer axes
        # (i.e. not sub tiles) and are parallel
        for idx, axis in enumerate(self._order):
            if self._max_parallelize >= 0 and idx >= self._max_parallelize:
                break
            if self._base_axis[axis] != axis:
                break
            if self._base_axis[axis] not in self._p_axes:
                break
            parallel.append(axis)
        return parallel

    def _init_sample_names(self) -> list[str]:
        x_names = [f"x{i}" for i in range(self._x_size)]
        w_names = [f"w{i}" for i in range(len(self._w_axes))]
        return x_names + w_names

    def _vector_index(self) -> int | None:
        indexes = [self._tiles_idx[axis] for axis in self._vectorized]
        return indexes[0] if indexes else None

    def _unrolled_indexes(self) -> list[int]:
        indexes = [self._tiles_idx[axis] for axis in self._unrolled]
        return indexes

    def _item_indexes(self, item_idx: int) -> list[int]:
        indexes = [self._tiles_idx[axis] for axis in self._item_map[item_idx]]
        return indexes

    def _inner_items_indexes(self, item_idx: int) -> list[int]:
        if item_idx < 0:
            item_idx = len(self._scheme) + item_idx
        indexes = []
        for idx in range(item_idx, len(self._scheme)):
            indexes.extend(self._item_indexes(idx))
        return indexes

    def _filter_vectorized(
        self, samples: Iterator[VecSample], stat: str | None = None
    ) -> Iterator[VecSample]:
        # Filter vectorized samples, i.e. vector tile >= VEC_SIZE
        if stat:
            self._stats[stat] = 0
        vaxis = self._vectorized[0] if self._vectorized else ""
        vsize = self._vec_size
        base_size = self._sizes[self._base_axis[vaxis]] if vaxis else 1
        if not vaxis or base_size <= vsize:
            yield from samples
            return
        vidx = self._tiles_idx[vaxis]
        for x in samples:
            if x[vidx] >= vsize:
                if stat:
                    self._stats[stat] += 1
                yield x

    def _filter_reg_num(
        self, samples: Iterator[VecSample], inner_items: int, stat: str | None = None
    ) -> Iterator[VecSample]:
        # Filter inner items <= VEC_SIZE * REG_NUM
        if stat:
            self._stats[stat] = 0
        indexes = self._inner_items_indexes(-inner_items)
        max_vreg = self._arch_vreg_num
        for x in samples:
            elts = np.prod(np.array(x)[indexes])
            if not elts <= self._vec_size * max_vreg:
                continue
            if stat:
                self._stats[stat] += 1
            yield x

    def _filter_l1_size(
        self, samples: Iterator[VecSample], inner_items: int, stat: str | None = None
    ) -> Iterator[VecSample]:
        # Filter inner items <= MAX_L1_ELTS
        if stat:
            self._stats[stat] = 0
        indexes = self._inner_items_indexes(-inner_items)
        max_l1_elts = self._arch_l1_size / np.dtype(self._out_dtype).itemsize
        for x in samples:
            elts = np.prod(np.array(x)[indexes])
            if not elts <= max_l1_elts:
                continue
            if stat:
                self._stats[stat] += 1
            yield x

    def _filter_l2_size(
        self, samples: Iterator[VecSample], inner_items: int, stat: str | None = None
    ) -> Iterator[VecSample]:
        # Filter inner items <= MAX_L1_ELTS
        if stat:
            self._stats[stat] = 0
        indexes = self._inner_items_indexes(-inner_items)
        max_l2_elts = self._arch_l2_size / np.dtype(self._out_dtype).itemsize
        for x in samples:
            elts = np.prod(np.array(x)[indexes])
            if not elts <= max_l2_elts:
                continue
            if stat:
                self._stats[stat] += 1
            yield x

    @override
    def _independents(self) -> list[list[list[int]]]:
        tiles = [
            factors_enumeration(self._sizes[axis], level - 1)
            for axis, level in self._tiles_levels.items()
        ]
        w_bools = [[[0], [1]]] * len(self._w_axes)
        return tiles + w_bools

    @override
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]:
        indexes = self._unrolled_indexes()
        v_index = self._vector_index()
        return self._filter_unroll(indexes, v_index, samples, stat="filtered")

    @override
    def _generate(self, sch: Scheduler, in_x: list[int]) -> None:
        tilings = {}
        for axis in self._axes:
            idx = self._tiles_idx[axis]
            nfactors = self._tiles_levels[axis] - 1
            factors = in_x[idx : idx + nfactors]
            sizes = factors_to_sizes(factors)
            tilings[axis] = {f"{axis}{idx + 1}": size for idx, size in enumerate(sizes)}
        buffers = [
            axis
            for idx, axis in enumerate(self._w_axes)
            if in_x[-len(self._w_axes) + idx] != 0
        ]
        axes_order = self._order
        parallel_axes = self._parallel
        vector_axes = self._vectorized
        unroll_axes = {axis: in_x[self._tiles_idx[axis]] for axis in self._unrolled}
        for axis, tiling in tilings.items():
            sch.tile(axis, tiling)
        sch.interchange(axes_order)
        sch.parallelize(parallel_axes)
        sch.vectorize(vector_axes)
        sch.unroll(unroll_axes)
        for axis in buffers:
            sch.buffer_at(axis)

    def _check_divisibility_from_tiles(self, tiles: dict[str, int]) -> bool:
        divisible = all(
            [
                (
                    tile > 0
                    and self._sizes[self._base_axis[axis]] >= tile
                    and self._sizes[self._base_axis[axis]] % tile == 0
                )
                for axis, tile in tiles.items()
            ]
        )
        return divisible

    def _generate_from_tiles(self, tiles: dict[str, int]) -> list[int] | None:
        schedule = [1] * self._x_size
        for axis, tile in tiles.items():
            schedule[self._tiles_idx[axis]] = tile
        return schedule

    def _get_max_axis_factor(self, axis: str, max_factor: int) -> int:
        base_size = self._sizes[self._base_axis[axis]]
        factors = factors_enumeration(base_size, 1)
        factor = [x[0] for x in factors if x[0] <= max_factor][-1]
        return factor

    @override
    def _default_schedule(self, opt_level: int) -> list[int]:
        def base_sched(v_unroll: int, p_unroll: int, r_unroll: int):
            # get last vectorized (vectorize order is reversed)
            vaxes = self._vectorized[:1]
            # get last 2 parallel (unroll order is reversed)
            paxes = [
                axis for axis in self._unrolled if self._base_axis[axis] in self._p_axes
            ][:2]
            # get last reduction (unroll order is reversed)
            raxes = [
                axis for axis in self._unrolled if self._base_axis[axis] in self._r_axes
            ][:1]
            vtiles = {
                axis: self._get_max_axis_factor(axis, max(p_unroll, v_unroll))
                for axis in vaxes
            }
            ptiles = {
                axis: self._get_max_axis_factor(axis, p_unroll)
                for axis in paxes
                if axis not in vtiles
            }
            rtiles = {axis: self._get_max_axis_factor(axis, r_unroll) for axis in raxes}
            tiles = {**ptiles, **rtiles, **vtiles}
            if not self._check_divisibility_from_tiles(tiles):
                return None
            schedule = self._generate_from_tiles(tiles)
            return schedule

        def sched_o2():
            return base_sched(self._vec_size, 2, 1)

        def sched_o3():
            return base_sched(self._vec_size, 4, 16)

        tiles_sched = [1] * self._x_size
        w_sched = [0] * len(self._w_axes)
        if opt_level >= 2:
            if len(w_sched) > 0:
                w_sched[-1] = 1
            o2 = sched_o2()
            if o2:
                tiles_sched = o2
        if opt_level >= 3:
            w_sched = [1] * len(self._w_axes)
            o3 = sched_o3()
            if o3:
                tiles_sched = o3
        return tiles_sched + w_sched


class Strategy_OO(BaseStrategyPRTScheme):
    """Strategy for 1-level tiling in O order.

    For instance on a matmul(i, j, k):
    - tiles order: i, k, j, i1, k1, j1

    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, scheme="OO", **kwargs)


class Strategy_PRP(BaseStrategyPRTScheme):
    """Strategy for PRP tiling.

    For instance on a matmul(i, j, k):
    - tiles order: i, j, k, i1, j1

    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, scheme="PRP", **kwargs)


class Strategy_PPRPRP(BaseStrategyPRTScheme):
    """Strategy for Ansor like tiling with.

    For instance on a matmul(i, j, k):
    - tiles order: i, j, i1, j1, k, i2, j2, k1, i3, j3

    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, scheme="PPRPRP", **kwargs)


class Strategy_PPRPRPv(Strategy_PPRPRP):
    """Strategy for Ansor like tiling with vectorization.

    For instance on a matmul(i, j, k):
    - tiles order: i, j, i1, j1, k, i2, j2, k1, i3, j3
    - filter on: j3 >= VEC_SIZE

    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]:
        samples = super()._filter(samples)
        samples = self._filter_vectorized(samples, "filtered_vec")
        return samples


class Strategy_PPRPRPvr(Strategy_PPRPRP):
    """Strategy for Ansor like tiling with space constraints.

    For instance on a matmul(i, j, k):
    - tiles order: i, j, i1, j1, k, i2, j2, k1, i3, j3
    - filter on:
      - j3 >= VEC_SIZE
      - inner P (i3*j3) vector size lower than number of vector registers
      - inner RP (k1*i3*j3) bytes size lower than machine L1
      - inner PRP (i2*j2*k1*i3*j3) bytes size lower than machine L2

    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]:
        samples = super()._filter(samples)
        samples = self._filter_vectorized(samples, "filtered_vec")
        samples = self._filter_reg_num(samples, 1, "filtered_reg")
        samples = self._filter_l1_size(samples, 2, "filtered_l1")
        samples = self._filter_l2_size(samples, 3, "filtered_l2")
        return samples


class Strategy_PPWRPRP(BaseStrategyPRTScheme):
    """Strategy for Ansor like tiling and write buffer.

    Same as Strategy_PPRPRP, but with an additional write buffer.
    The scheduler is PPWPRPR where W is the location of the
    allocated write buffer in the tiling.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, scheme="PPWRPRP", **kwargs)


class Strategy_PPWRPRPv(Strategy_PPWRPRP):
    """Strategy for Ansor like tiling and write buffer with space vectorization.

    Same as Strategy_PPWRPRP, but with an additional constraint for the
    space to have the inner axis vectorized as in strategy PPRPRPV.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]:
        samples = super()._filter(samples)
        samples = self._filter_vectorized(samples, "filtered_vec")
        return samples


class Strategy_PPWRPRPvr(Strategy_PPWRPRP):
    """Strategy for Ansor like tiling and write buffer with space constraints.

    Same as Strategy_PPWRPRP, but with additional constraints for the
    space as in strategy PPRPRPvr.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]:
        samples = super()._filter(samples)
        samples = self._filter_vectorized(samples, "filtered_vec")
        samples = self._filter_reg_num(samples, 1, "filtered_reg")
        samples = self._filter_l1_size(samples, 2, "filtered_l1")
        samples = self._filter_l2_size(samples, 3, "filtered_l2")
        return samples


class Strategy_GOTO(BaseStrategy):
    """Strategy for Goto tiling with vectorisation.

    Each tiling parameter is free when exploring exhausive samples.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(
            graph,
            ["iL2", "iR", "jL3", "jR", "kL1", "unroll_k", "pack_B", "pack_A"],
            **kwargs,
        )

    @override
    def _generate(self, sch: Scheduler, in_x: list[int]) -> None:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        iL2, iR, jL3, jR, kR1, unroll_k, pack_B, pack_A = in_x[:8]
        axes_order = ["j", "k", "i", "j1", "i1", "k1", "i2", "j2"]
        vector_axes = ["j2"]
        parallel_axes = []
        unroll_axes = {"i2": iR, "k1": unroll_k}
        if self._threads > 1:
            parallel_axes.append("j")
        if pack_B:
            sch.pack_at("k", 1, pad=True)
        if pack_A:
            sch.pack_at("i", 0, pad=True)
        sch.tile("i", {"i1": iR * iL2, "i2": iR}, root=".")
        sch.tile("j", {"j1": jR * jL3, "j2": jR}, root=".")
        sch.tile("k", {"k1": kR1}, root=".")
        sch.interchange(axes_order, root=".")
        sch.parallelize(parallel_axes, root=".")
        sch.vectorize(vector_axes, root=".")
        sch.unroll(unroll_axes, root=".")

    @override
    def _independents(self) -> list[list[list[int]]]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        i, j, k = self._constant_sizes().values()
        tiles_i = factors_enumeration(i, 2)
        tiles_j = factors_enumeration(j, 2)
        tiles_k = factors_enumeration(k, 2)
        boolean = [[0], [1]]
        return [tiles_i, tiles_j, tiles_k, boolean, boolean]

    @override
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]:
        v_index = 3
        indexes = [1, 3, 5]
        samples = self._filter_unroll(indexes, v_index, samples, stat="filtered")
        return samples

    @override
    def _default_schedule(self, opt_level: int) -> list[int]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        i, j, k = i, j, k = self._constant_sizes().values()
        schedule = [1, 1, 1, 1, 1, 0, 0, 0]
        if opt_level >= 3:
            jtile = self._vec_size
            itile = 2  # TODO: IPC?
            ktile = 1
            idiv = i >= itile and i % itile == 0
            jdiv = j >= jtile and j % jtile == 0
            kdiv = k >= ktile and k % ktile == 0
            if idiv and jdiv and kdiv:
                schedule = [1, itile, 1, jtile, ktile, 2, 1, 1]
        return schedule


class Strategy_GOTO_R(Strategy_GOTO):
    """
    Goto Strategy with a reduced search space.
    """

    @override
    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(graph, **kwargs)

    @override
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]:
        samples = super()._filter(samples)
        # Only keep tiles samples with pertinent tiles sizes and small unrolls on k
        reg_size = self._vec_size * self._arch_vreg_num
        for x in samples:
            io, ii, jo, ji, _, unroll_k = x[:6]
            reg_tile = ii * ji
            if (
                reg_tile < reg_size // 2
                and ii < 4
                and unroll_k < 4
                and 128 > io > 1
                and 128 > jo > 1
            ):
                yield x


class Strategy_Descript(Strategy):
    def __init__(
        self,
        graph: Graph,
        spec: dict[str, dict] | str,
        constraints: list[str] = [],
        partial_tiles: bool = False,
        partial_unrolls: bool = False,
        initialize: bool = True,
    ) -> None:
        self._graph = graph
        self._op = graph.outputs_nodes[0].operation
        self._stats: dict[str, int] = {}
        self._axes = list(self._op.dims)
        self._sizes = self._constant_sizes()
        descript = DescriptExtend(
            abstract_axis=self._axes,
            abstract_axis_sizes=dict(self._sizes),
            abstract_matrix=["A", "B", "C"],
            partial_tiles=partial_tiles,
            partial_unrolls=partial_unrolls,
        )
        self._descript = descript
        self._initialized = False
        input_constraints = constraints
        self._flat_schedules, self._sample_names, constraints, axes, orders = (
            descript.flatten_schedule(node_name=DEFAULT_ROOT, spec=spec)
        )
        for a, v in self._sizes.items():
            for i, s in enumerate(constraints):
                assert isinstance(s, str)
                constraints[i] = s.replace(f"[{a}]", str(v))
        self._axes_names = {}
        for a, v in axes.items():
            self._axes_names[a] = v
        self._orders: dict[str, list] = {}
        order_constraints: list[str] = []
        for a, v in orders.items():
            assert isinstance(v, dict)
            permutation = list(itertools.permutations(v))
            a_holder = f"order_{a}"
            self._orders[a_holder] = permutation
            order_constraints.append(f"{a_holder} in {set(range(len(permutation)))}")
        self._constraints = constraints + input_constraints + order_constraints
        self._constraints.sort()
        if initialize:
            self._initialize()

    def _initialize(self):
        if self._initialized:
            return
        max_enum = int(1 + np.log2(max(self._sizes.values())))
        context = Context()
        constraints, self.constrants = constraints_from_str(
            self._constraints, context=context
        )
        properties, constraints = hypergraph(
            constraints, max_enum=max_enum, context=context
        )
        methods = solve_with_z3(list(context.variables.keys()), properties, constraints)
        enumerations = execute_static(methods, properties, constraints)
        self._context = context
        self._properties = properties
        self._z3_constraints = constraints
        self._methods = methods
        self._enumerations = enumerations
        self._initialized = True

    @property
    @override
    def graph(self) -> Graph:
        return self._graph

    @override
    def generate(self, scheduler: Scheduler, sample: Sample) -> None:
        descript = self._descript
        flat_schedules = self._flat_schedules
        for a, p in self._orders.items():
            if a in sample:
                if isinstance(sample[a], int):
                    sample[a] = p[sample[a]]
        flat_schedules = descript.apply_sample(
            flat_schedules=flat_schedules, sample=sample
        )
        descript.apply_scheduler(flat_schedules, scheduler)

    @override
    def sample(self, num: int, seed: int | None = 0) -> Iterator[Sample]:
        samples = sample_uniques(self._sample_once_tuple, num)
        for x in samples:
            yield dict(zip(self.sample_names, x))

    def sample_once(self, num: int) -> Iterator[Sample]:
        self._initialize()
        draw = execute_dynamic(
            self._methods,
            self._properties,
            self._z3_constraints,
            self._enumerations,
            k=num,
        )
        return draw

    def pretty_print_methods(self, tab: str = "\t"):
        self._initialize()
        pretty_print_methods(
            self._methods, self._properties, self._z3_constraints, tab=tab
        )

    def _sample_once_tuple(self, num: int) -> Iterator[tuple]:
        draw = self.sample_once(num)
        for d in draw:
            yield tuple([d[x] for x in self.sample_names])

    @override
    def exhaustive(self) -> Iterator[Sample]:
        return self.sample(1000)

    @override
    def default_schedule(self, opt_level: int = 2) -> Sample:
        while True:
            for x in self.sample(1):
                if x:
                    return x

    def _constant_sizes(self) -> Mapping[str, int]:
        sizes = {a: v for a, v in self._op.dims.items() if isinstance(v, int)}
        return sizes

    @override
    def dict_to_sample(self, sample: dict[str, Any]) -> Sample:
        return sample

    @override
    def sample_to_dict(self, sample: Sample) -> dict[str, int]:
        return sample

    @property
    @override
    def sample_names(self) -> list[str]:
        return self._sample_names


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
        "tile_p1": Strategy_P1,
        "tile_p1_v": Strategy_P1v,
        "tile_oo": Strategy_OO,
        "tile_prp": Strategy_PRP,
        "tile_pprprp": Strategy_PPRPRP,
        "tile_pprprp_v": Strategy_PPRPRPv,
        "tile_pprprp_vr": Strategy_PPRPRPvr,
        "tile_ppwrprp": Strategy_PPWRPRP,
        "tile_ppwrprp_v": Strategy_PPWRPRPv,
        "tile_ppwrprp_vr": Strategy_PPWRPRPvr,
        "tile_goto": Strategy_GOTO,
        "tile_goto_r": Strategy_GOTO_R,
    }
