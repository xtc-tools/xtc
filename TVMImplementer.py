#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import sys
import tempfile
import subprocess
from io import StringIO
from typing import Any
from utils import LazyImport
from TVMOps import Operation, Operators

__all__ = [
    "Implementer",
]

objdump_bin = "objdump"

objdump_opts = ["-d", "--no-addresses", "--no-show-raw-insn", "--visualize-jumps"]

objdump_color_opts = [
    "--visualize-jumps=color",
    "--disassembler-color=on",
]


class Implementer:
    def __init__(
        self, source_op: Operation, dims: dict[str, int], parallel_dims: list[str]
    ):
        self.op = source_op
        self.dims = dims
        self.parallel_dims = parallel_dims
        self.tiles = {k: {k: v} for k, v in self.dims.items()}
        self.permutation = []
        self.vectorization = []
        self.parallelization = []
        self.unrolling = {}
        self._update_loops()

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
        self.working_dims = loops
        self.working_parallel_dims = parallels
        self.permutation = list(self.working_dims.keys())

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

    def vectorize(self, vectorization: list[str]):
        for p in vectorization:
            assert p in self.working_parallel_dims
        self.vectorization = vectorization

    def parallelize(self, parallelization: list[str]):
        for p in parallelization:
            assert p in self.working_parallel_dims
        self.parallelization = parallelization

    def unroll(self, unrolling: dict[str, int]):
        self.unrolling = unrolling

    def evaluate(self, **kwargs):
        print_source_ir = kwargs.get("print_source_ir", False)
        print_transformed_ir = kwargs.get("print_transformed_ir", False)
        print_assembly = kwargs.get("print_assembly", False)
        color = kwargs.get("color", False)
        self.op.generate()
        if print_source_ir:
            self.op.schedule()
            print(self.op.lower())
            sys.stdout.flush()
        io = StringIO()
        self.dump_tvm_schedule(outf=io)
        schedule_str = io.getvalue()
        self.op.schedule(schedule_str)
        if print_transformed_ir:
            print(self.op.lower())
            sys.stdout.flush()
        self.op.build()
        if print_assembly:
            with tempfile.TemporaryDirectory() as tdir:
                soname = f"{tdir}/built.so"
                fname = f"{self.op.operator.name}_compute_"
                self.op.built.export_library(soname)
                cmd_disassembler = (
                    [objdump_bin] + [soname] + objdump_opts + [f"--disassemble={fname}"]
                )
                if color:
                    cmd_disassembler += objdump_color_opts
                print("Running", " ".join(cmd_disassembler))
                subprocess.run(cmd_disassembler, text=True)
        time = self.op.run()
        return time

    def dump_schedule(self, obj=None, outf=sys.stdout):
        if obj is None:
            obj = "imp"
            clsname = self.__class__.__name__
            print(
                f"{obj} = {clsname}(source_op={repr(self.op)}, dims={self.dims}, parallel_dims={self.parallel_dims})",
                file=outf,
            )
        for dim, tiles in self.tiles.items():
            t_tiles = {k: v for i, (k, v) in enumerate(tiles.items()) if i >= 1}
            print(f"{obj}.tile('{dim}', {t_tiles})", file=outf)
        print(f"{obj}.interchange({self.permutation})", file=outf)
        print(f"{obj}.vectorize({self.vectorization})", file=outf)
        print(f"{obj}.unroll({self.unrolling})", file=outf)
        print(f"{obj}.parallelize({self.parallelization})", file=outf)

    def dump_tvm_schedule(self, obj="obj", sch="sch", outf=sys.stdout):
        print(f"O = {obj}[-1]", file=outf)
        parallel_axes = [k for k in self.dims.keys() if k in self.parallel_dims]
        reduction_axes = [k for k in self.dims.keys() if k not in self.parallel_dims]
        print(f"{', '.join(parallel_axes)}, = O.op.axis", file=outf)
        if reduction_axes:
            print(f"{', '.join(reduction_axes)}, = O.op.reduce_axis", file=outf)
        for dim, tiles in self.tiles.items():
            t_sizes = list(tiles.values())
            t_axes = [dim] + list(tiles.keys())
            for idx in range(1, len(tiles)):
                print(
                    f"{t_axes[idx]}, {t_axes[idx + 1]} = {sch}[O].split({t_axes[idx]}, factor={t_sizes[idx]})",
                    file=outf,
                )
        print(f"{sch}[O].reorder({', '.join(self.permutation)})", file=outf)
        for axis in self.vectorization:
            print(f"{sch}[O].vectorize({axis})", file=outf)
        for axis, unroll in self.unrolling.items():
            print(f"{sch}[O].unroll({axis})", file=outf)
        if self.parallelization:
            if len(self.parallelization) > 1:
                print(
                    f"{self.parallelization[0]} = {sch}[O].fuse({', '.join(self.parallelization)})",
                    file=outf,
                )
            print(f"{sch}[O].parallel({self.parallelization[0]})", file=outf)


def _test_generate_tiling_1():
    dims = {"i": 256, "j": 256, "k": 512}
    parallel_dims = ["i", "j"]
    op = Operation(Operators.matmul, (*list(dims.values()), "float32"))
    imp_args = dict(source_op=op, dims=dims, parallel_dims=parallel_dims)
    imp = Implementer(**imp_args)
    imp.tile("i", {"i1": 4})
    imp.tile("j", {"j1": 64})
    imp.tile("k", {"k1": 8})
    imp.interchange(["i", "j", "k", "k1", "i1", "j1"])
    imp.vectorize(["j1"])
    imp.parallelize(["i", "j"])
    imp.unroll({"i1": 4, "k1": 8})
    return imp, imp_args


def _test_generate_tiling_2():
    dims = {"i": 256, "j": 256, "k": 512}
    parallel_dims = ["i", "j"]
    op = Operation(Operators.matmul, (*list(dims.values()), "float32"))
    imp_args = dict(source_op=op, dims=dims, parallel_dims=parallel_dims)
    imp = Implementer(**imp_args)
    imp.tile("i", {"i1": 128, "i2": 4})
    imp.tile("j", {"j1": 128, "j2": 64})
    imp.tile("k", {"k1": 8})
    imp.interchange(["i", "j", "k", "i1", "j1", "k1", "i2", "j2"])
    imp.vectorize(["j2"])
    imp.parallelize(["i", "j"])
    imp.unroll({"i2": 4, "k1": 8})
    return imp, imp_args


def _test_self_schedule(imp, imp_args):
    print("Raw schedule 1")
    imp.dump_schedule()
    io = StringIO()
    imp.dump_schedule(obj="imp", outf=io)
    imp2 = Implementer(**imp_args)
    exec(io.getvalue(), {"imp": imp2}, {})
    print("Raw schedule 2")
    imp2.dump_schedule()
    io2 = StringIO()
    imp2.dump_schedule(obj="imp", outf=io2)
    assert io.getvalue() == io2.getvalue(), f"self dump schedule not equal"


def test_self_schedule():
    imp, imp_args = _test_generate_tiling_1()
    _test_self_schedule(imp, imp_args)
    imp, imp_args = _test_generate_tiling_2()
    _test_self_schedule(imp, imp_args)


def _test_tvm_schedule(imp, imp_args):
    print("Raw TVM schedule 1")
    imp.dump_tvm_schedule()


def test_tvm_schedule():
    imp, imp_args = _test_generate_tiling_1()
    _test_tvm_schedule(imp, imp_args)
    imp, imp_args = _test_generate_tiling_2()
    _test_tvm_schedule(imp, imp_args)


def _test_tvm_evaluate(imp, imp_args):
    time = imp.evaluate()
    print(f"Execution time: {time} secs")


def test_tvm_evaluate():
    imp, imp_args = _test_generate_tiling_1()
    _test_tvm_evaluate(imp, imp_args)
    imp, imp_args = _test_generate_tiling_2()
    _test_tvm_evaluate(imp, imp_args)


if __name__ == "__main__":
    test_self_schedule()
    test_tvm_schedule()
    test_tvm_evaluate()
