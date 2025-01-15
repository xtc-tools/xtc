#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os
import sys
import tempfile
import subprocess
from io import StringIO
from typing import Any
from pathlib import Path
from copy import deepcopy
import numpy as np

import utils
from ndarray import NDArray
from TVMOps import Operation, Operators
from evaluator import Evaluator, Executor

__all__ = [
    "Implementer",
    "Scheduler",
    "Schedule",
]

objdump_bin = "objdump"

objdump_opts = ["-d", "--no-addresses", "--no-show-raw-insn", "--visualize-jumps"]

objdump_color_opts = [
    "--visualize-jumps=color",
    "--disassembler-color=on",
]


class Schedule:
    def __init__(self, scheduler: "Scheduler") -> None:
        self.scheduler = deepcopy(scheduler)

    def get_schedule_impl(self) -> str:
        io = StringIO()
        self.scheduler._dump_tvm_schedule(outf=io)
        schedule_str = io.getvalue()
        return schedule_str

    def __str__(self) -> str:
        return self.get_schedule_impl()


class Scheduler:
    def __init__(self, impl: "Implementer") -> None:
        self.dims = {**impl.dims}
        self.parallel_dims = [*impl.parallel_dims]
        self.tiles = {k: {k: v} for k, v in self.dims.items()}
        self.permutation = []
        self.vectorization = []
        self.parallelization = []
        self.unrolling = {}
        self.write_caches = []
        self._update_loops()

    def implement(self) -> Schedule:
        return Schedule(scheduler=self)

    def _update_loops(self):
        loops = dict()
        parallels = []
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for k, v in self.tiles.items():
                if tile_level >= len(v):
                    continue
                dim_name = list(v.keys())[tile_level]
                loops[dim_name] = v[dim_name]
                if k in self.parallel_dims:
                    parallels.append(dim_name)
        self._working_dims = loops
        self._working_parallel_dims = parallels
        self._working_permutation = list(self._working_dims.keys())
        self._parallel_axes = [k for k in self.dims.keys() if k in self.parallel_dims]
        self._reduction_axes = [
            k for k in self.dims.keys() if k not in self.parallel_dims
        ]

    def tile(
        self,
        dim: str,
        tiles: dict[str, int],
    ):
        ndims = list(tiles.keys())
        tiles_sizes = list(tiles.values())

        assert len(ndims) == len(tiles_sizes)

        previous_tile_size = self.dims[dim]
        for ts in tiles_sizes:
            assert previous_tile_size % ts == 0
            previous_tile_size = ts

        dims = [dim] + ndims
        sizes = [self.dims[dim]] + tiles_sizes
        for d, s in zip(dims, sizes):
            self.tiles[dim][d] = s
        self._update_loops()

    def interchange(self, permutation: list[str]):
        self.permutation = permutation

    def buffer_at(self, axis: str, typ: str):
        assert typ == "write"
        self.write_caches.append(axis)

    def vectorize(self, vectorization: list[str]):
        for p in vectorization:
            assert p in self._working_parallel_dims
        self.vectorization = vectorization

    def parallelize(self, parallelization: list[str]):
        for p in parallelization:
            assert p in self._working_parallel_dims
        self.parallelization = parallelization

    def unroll(self, unrolling: dict[str, int]):
        self.unrolling = unrolling

    def _full_order(self) -> list[str]:
        permutation = self.permutation + self._working_permutation
        permutation = list(dict.fromkeys(permutation))
        return permutation

    def _full_tilings(self) -> dict[str, tuple[str, int]]:
        order = self._full_order()
        tilings = {}
        for dim, tiles in self.tiles.items():
            t_axes = [dim] + list(tiles.keys())
            t_sizes = list(tiles.values())
            tilings[dim] = (dim, "", self.dims[dim])
            for idx in range(1, len(tiles)):
                tilings[t_axes[idx + 1]] = (dim, t_axes[idx], t_sizes[idx])
        tilings = {axis: tilings[axis] for axis in order}
        return tilings

    def _write_buffer_tiling(self, axis: str, tilings):
        child = {
            parent: (dim, axis, factor)
            for axis, (dim, parent, factor) in tilings.items()
        }
        tiles_axis = list(tilings)
        tiles = list(tilings.items())
        tile_idx = tiles_axis.index(axis)
        outer_tiles = dict(list(tilings.items())[: tile_idx + 1])
        inner_tiles = dict(list(tilings.items())[tile_idx + 1 :])
        for axis, (dim, parent, factor) in list(outer_tiles.items()):
            factor = child.get(axis, ("", "", 1))[2]
            outer_tiles[f"{dim}_"] = (dim, axis, factor)
        dims = set()
        for axis, (dim, parent, factor) in list(inner_tiles.items()):
            if dim not in dims:
                dims.add(dim)
                parent = ""
            inner_tiles[axis] = (dim, parent, factor)
        return outer_tiles, inner_tiles

    def _full_write_buffers(self) -> list[str]:
        tilings = self._full_tilings()
        reorder_idx = {axis: idx for idx, axis in enumerate(tilings)}
        write_axis = sorted(self.write_caches, key=lambda axis: reorder_idx[axis])
        buffer_tilings = {}
        last_idx = 0
        out = ("O", "", "")
        tiling = tilings
        for idx, axis in enumerate(write_axis):
            outer_tiling, inner_tiling = self._write_buffer_tiling(axis, tiling)
            buffer_tilings[out] = outer_tiling
            out = (f"O_W{idx}", out[0], axis)
            tiling = inner_tiling
        buffer_tilings[out] = tiling
        return buffer_tilings

    def _emit_assign_axis(self, sch: str, tens: str, outf) -> list[str]:
        if self._parallel_axes:
            print(f"{', '.join(self._parallel_axes)}, = {tens}.op.axis", file=outf)
        if self._reduction_axes:
            print(
                f"{', '.join(self._reduction_axes)}, = {tens}.op.reduce_axis", file=outf
            )

    def _emit_assign_tilings(
        self, sch: str, tens: str, tilings: dict[str, tuple[str, int]], outf
    ) -> list[str]:
        for axis, (dim, parent, factor) in tilings.items():
            if not parent:
                if axis != dim:
                    print(f"{axis} = {dim}", file=outf)
                continue
            print(
                f"{parent}, {axis} = {sch}[{tens}].split({parent}, factor={factor})",
                file=outf,
            )
        print(f"{sch}[{tens}].reorder({', '.join(tilings)})", file=outf)

    def _dump_tvm_schedule(self, obj="obj", sch="sch", outf=sys.stdout):
        tilings = self._full_write_buffers()
        for tens, parent, _ in tilings:
            if not parent:
                print(f"{tens} = {obj}[-1]", file=outf)
            else:
                print(f'{tens} = {sch}.cache_write({parent}, "local")', file=outf)
        for idx, ((tens, parent, axis), tilings) in enumerate(tilings.items()):
            if parent:
                print(f"{sch}[{tens}].compute_at({sch}[{parent}], {axis})", file=outf)
            self._emit_assign_axis(sch, tens, outf)
            self._emit_assign_tilings(sch, tens, tilings, outf)
            for axis, unroll in self.unrolling.items():
                if axis in tilings:
                    print(f"{sch}[{tens}].unroll({axis})", file=outf)
            for axis in self.vectorization:
                if axis in tilings:
                    print(f"{sch}[{tens}].vectorize({axis})", file=outf)
            if self.parallelization:
                if self.parallelization[0] in tilings:
                    if len(self.parallelization) > 1:
                        print(
                            f"{self.parallelization[0]} = {sch}[{tens}].fuse({', '.join(self.parallelization)})",
                            file=outf,
                        )
                    print(
                        f"{sch}[{tens}].parallel({self.parallelization[0]})", file=outf
                    )

    def dump_schedule(self, obj=None, outf=sys.stdout):
        if obj is None:
            obj = "sch"
        for dim, tiles in self.tiles.items():
            t_tiles = {k: v for i, (k, v) in enumerate(tiles.items()) if i >= 1}
            print(f"{obj}.tile('{dim}', {t_tiles})", file=outf)
        print(f"{obj}.interchange({self.permutation})", file=outf)
        print(f"{obj}.vectorize({self.vectorization})", file=outf)
        print(f"{obj}.unroll({self.unrolling})", file=outf)
        print(f"{obj}.parallelize({self.parallelization})", file=outf)

    def get_schedule_str(self, obj=None) -> str:
        io = StringIO()
        self.dump_schedule(outf=io)
        return io.getvalue()

    def __str__(self) -> str:
        return self.get_scheduler_state_str()


class Implementer:
    def __init__(
        self,
        source_op: Operation,
        dims: dict[str, int],
        parallel_dims: list[str],
    ):
        self.op = source_op
        self.dims = dims
        self.parallel_dims = parallel_dims
        self.payload_name = self.op.name

    def get_scheduler(self) -> Scheduler:
        return Scheduler(self)

    def compile(
        self,
        schedule: Schedule,
        dump_file: str = None,
        shared_lib: bool = False,
        executable: bool = False,
        **kwargs,
    ) -> None:
        print_source_ir = kwargs.get("print_source_ir", False)
        print_transformed_ir = kwargs.get("print_transformed_ir", False)
        print_assembly = kwargs.get("print_assembly", False)
        color = kwargs.get("color", False)
        operation = self.op.generate()
        if print_source_ir:
            sch = self.op.schedule(operation)
            print(self.op.lower(operation, sch))
            sys.stdout.flush()
        schedule_impl = schedule.get_schedule_impl()
        if print_transformed_ir:
            print(schedule_impl)
            sys.stdout.flush()
        sch = self.op.schedule(operation, schedule_impl)
        if print_transformed_ir:
            print(self.op.lower(operation, sch))
            sys.stdout.flush()
        built = self.op.build(operation, sch)
        if print_assembly:
            with tempfile.TemporaryDirectory() as tdir:
                soname = f"{tdir}/built.so"
                fname = f"{self.op.operator.name}_compute_"
                built.export_library(soname)
                cmd_disassembler = (
                    [objdump_bin] + [soname] + objdump_opts + [f"--disassemble={fname}"]
                )
                if color:
                    cmd_disassembler += objdump_color_opts
                print("Running", " ".join(cmd_disassembler))
                subprocess.run(cmd_disassembler, text=True)
        if dump_file is not None:
            assert not executable, f"executable generation not supported yet for TVM"
            if shared_lib:
                built.export_library(f"{dump_file}.so")

    def load_and_evaluate(
        self,
        dll,
        sym,
        repeat=1,
        min_repeat_ms=0,
        number=1,
        validate=False,
        init_zero=False,
        parameters=None,
        reference=None,
    ):
        results, code, error = self.load_and_eval(
            dll,
            sym,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            number=number,
            validate=validate,
            init_zero=init_zero,
            parameters=parameters,
            reference=reference,
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
        reference=None,
    ):
        results, code, error = self.run_eval_dll(
            dll,
            sym,
            repeat=repeat,
            number=number,
            min_repeat_ms=min_repeat_ms,
            validate=validate,
            init_zero=init_zero,
            parameters=parameters,
            reference=reference,
        )
        return results, code, error

    def run_eval_dll(
        self,
        dll,
        sym,
        repeat=1,
        min_repeat_ms=0,
        number=1,
        validate=False,
        init_zero=False,
        parameters=None,
        reference=None,
    ):
        dll = os.path.abspath(dll)
        with utils.LibLoader(dll) as lib:
            func = getattr(lib, sym)
            assert func is not None, f"Cannot find {sym} in lib {dll}"
            func.packed = True
            if parameters is None:
                inputs_spec = self.np_inputs_spec()
                outputs_spec = self.np_outputs_spec()
                out_init = np.zeros if init_zero else np.empty
                inputs = [utils.np_init(**spec) for spec in inputs_spec]
                outputs = [out_init(**spec) for spec in outputs_spec]
                parameters = (
                    [NDArray(inp) for inp in inputs],
                    [NDArray(out) for out in outputs],
                )
            if validate:
                ref_inputs = [inp.numpy() for inp in parameters[0]]
                ref_outputs = [
                    np.empty(shape=out.shape, dtype=out.dtype) for out in parameters[1]
                ]
                if reference is None:
                    reference = self.reference_impl
                reference(*ref_inputs, *ref_outputs)
                exec_func = Executor(func)
                exec_func(*parameters[0], *parameters[1])
                for out_ref, out in zip(
                    ref_outputs, [out.numpy() for out in parameters[1]]
                ):
                    if not np.allclose(out_ref, out):
                        return [], 1, "Error in validation: outputs differ"
            eval_func = Evaluator(
                func, repeat=repeat, min_repeat_ms=min_repeat_ms, number=number
            )
            results = eval_func(*parameters[0], *parameters[1])
        return results, 0, ""

    def evaluate(
        self, scheduler: Scheduler, compiler_args: dict = {}, evaluate_args: dict = {}
    ) -> float | str:
        with tempfile.TemporaryDirectory() as dirname:
            libpath = Path(dirname) / "payload_{self.payload_name}"
            self.compile(
                scheduler,
                dump_file=libpath,
                shared_lib=True,
                **compiler_args,
            )
            result = self.load_and_evaluate(
                f"{libpath}.so",
                self.payload_name,
                validate=True,
                **evaluate_args,
            )
        return result

    def np_inputs_spec(self):
        return self.op.np_inputs_spec()

    def np_outputs_spec(self):
        return self.op.np_outputs_spec()

    def reference_impl(self, *args):
        self.op.reference_impl(*args)


def _test_generate_tiling_1():
    dims = {"i": 256, "j": 256, "k": 512}
    parallel_dims = ["i", "j"]
    op = Operation(Operators.matmul, (*list(dims.values()), "float32"))
    imp_args = dict(source_op=op, dims=dims, parallel_dims=parallel_dims)
    imp = Implementer(**imp_args)
    sch = imp.get_scheduler()
    sch.tile("i", {"i1": 4})
    sch.tile("j", {"j1": 64})
    sch.tile("k", {"k1": 8})
    sch.interchange(["i", "j", "k", "k1", "i1", "j1"])
    sch.vectorize(["j1"])
    # TODO: parallel runtime not supported yet
    # sch.parallelize(['i', 'j'])
    sch.unroll({"j1": 64, "i1": 4, "k1": 8})
    return imp, sch


def _test_generate_tiling_2():
    dims = {"i": 256, "j": 256, "k": 512}
    parallel_dims = ["i", "j"]
    op = Operation(Operators.matmul, (*list(dims.values()), "float32"))
    imp_args = dict(source_op=op, dims=dims, parallel_dims=parallel_dims)
    imp = Implementer(**imp_args)
    sch = imp.get_scheduler()
    sch.tile("i", {"i1": 128, "i2": 4})
    sch.tile("j", {"j1": 128, "j2": 64})
    sch.tile("k", {"k1": 8})
    sch.interchange(["i", "j", "k", "i1", "j1", "k1", "i2", "j2"])
    sch.vectorize(["j2"])
    # TODO: parallel runtime not supported yet
    # sch.parallelize(['i', 'j'])
    sch.unroll({"j2": 64, "i2": 4, "k1": 8})
    return imp, sch


def _test_self_schedule(imp, sch):
    print("Raw schedule 1")
    sch.dump_schedule()
    schedule_str = sch.get_schedule_str(obj="sch")
    sch2 = imp.get_scheduler()
    exec(schedule_str, {"sch": sch2}, {})
    print("Raw schedule 2")
    sch2.dump_schedule()
    schedule_str2 = sch2.get_schedule_str(obj="sch")
    assert schedule_str == schedule_str2, f"self dump schedule not equal"


def test_self_schedule():
    imp, sch = _test_generate_tiling_1()
    _test_self_schedule(imp, sch)
    imp, sch = _test_generate_tiling_2()
    _test_self_schedule(imp, sch)


def _test_tvm_schedule(imp, sch):
    print("Raw TVM schedule 1")
    sch._dump_tvm_schedule()


def test_tvm_schedule():
    imp, imp_args = _test_generate_tiling_1()
    _test_tvm_schedule(imp, imp_args)
    imp, imp_args = _test_generate_tiling_2()
    _test_tvm_schedule(imp, imp_args)


def _test_tvm_evaluate(imp, sch):
    schedule = sch.implement()
    result = imp.evaluate(
        schedule,
        compiler_args=dict(
            print_transformed_ir=True,
            print_assembly=True,
            print_source_ir=True,
        ),
    )
    assert isinstance(result, float), f"evaluation error: {result}"
    print(f"Execution time: {result * 1000:.3f} msecs")


def test_tvm_evaluate():
    imp, imp_args = _test_generate_tiling_1()
    _test_tvm_evaluate(imp, imp_args)
    imp, imp_args = _test_generate_tiling_2()
    _test_tvm_evaluate(imp, imp_args)


if __name__ == "__main__":
    test_self_schedule()
    test_tvm_schedule()
    test_tvm_evaluate()
