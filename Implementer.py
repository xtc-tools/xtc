#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC
import subprocess

from xdsl.parser import Parser
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.ir import Block, Region, MLContext, Operation
from xdsl.dialects.builtin import (
    ModuleOp,
    DenseIntOrFPElementsAttr,
    TensorType,
    MemRefType,
    f32,
    f64,
)
from xdsl.dialects import func, arith, linalg

import transform

transform_opt = "transform-interpreter"
transform_opts = [
    f"--{transform_opt}",
    "--func-bufferize",
    "--buffer-deallocation",
    "--convert-linalg-to-loops",
]

lowering_opts = [
    # "--allow-unregistered-dialect",
    # "--mlir-print-op-generic",
    # "--canonicalize",
    # "--test-transform-dialect-erase-schedule",
    "--test-transform-dialect-erase-schedule",
    "--convert-scf-to-cf",
    "--canonicalize",
    "--convert-vector-to-llvm=enable-x86vector",
    "--test-lower-to-llvm",
]

mliropt_opts = transform_opts + lowering_opts

obj_dump_file = "/tmp/dump.o"

mlirrunner_opts = [
    "-e",
    "main",
    "--entry-point-result=void",
    "--O3",
]

objdump_bin = "objdump"

objdump_opts = ["-d", "--no-addresses", "--no-show-raw-insn", "--visualize-jumps"]

objdump_color_opts = [
    "--visualize-jumps=color",
    "--disassembler-color=on",
]


class Implementer:
    count = 0

    def __init__(
        self,
        mlir_install_dir: str,
        source_op: Operation,
        dims: dict[str, int],
        parallel_dims: list[str],
    ):
        #
        self.payload_name = f"payload{Implementer.count}"
        Implementer.count += 1
        #
        self.mliropt = f"{mlir_install_dir}/bin/mlir-opt"
        self.cmd_run_mlir = [
            f"{mlir_install_dir}/bin/mlir-cpu-runner",
            f"-shared-libs={mlir_install_dir}/lib/libmlir_runner_utils.so",
            f"-shared-libs={mlir_install_dir}/lib/libmlir_c_runner_utils.so",
        ] + mlirrunner_opts
        self.cmd_disassembler = (
            [objdump_bin] + objdump_opts + [f"--disassemble={self.payload_name}"]
        )
        #
        self.source_op = source_op
        # TODO: should be computed
        self.dims = dims
        self.parallel_dims = parallel_dims
        self.tiles = {k: {k: 1} for k, v in self.dims.items()}
        self.permutation = self.get_default_interchange()
        self.vectorization = []
        self.parallelization = []
        self.unrolling = dict([])

    def payload(self):
        # Fetch data
        operands = self.source_op.operands
        inputs = self.source_op.inputs
        inputs_types = [o.type for o in inputs]
        results_types = [r.type for r in self.source_op.results]
        #
        payload = Block(arg_types=inputs_types)
        outputs = self.outputs_init()
        outputs_vars = []
        for o in outputs:
            outputs_vars += o.results
        concrete_operands = list(payload.args) + outputs_vars
        value_mapper = {o: p for o, p in zip(operands, concrete_operands)}

        new_op = self.source_op.clone(value_mapper=value_mapper)
        payload.add_ops(outputs + [new_op, func.Return(new_op)])
        payload_func = func.FuncOp.from_region(
            self.payload_name, inputs_types, results_types, Region(payload)
        )
        return payload_func

    def inputs_init(self):
        inputs_types = [o.type for o in self.source_op.inputs]
        return [
            arith.Constant(
                DenseIntOrFPElementsAttr.tensor_from_list(
                    [1.0], ty.get_element_type(), ty.get_shape()
                )
            )
            for ty in inputs_types
        ]

    def outputs_init(self):
        outputs_types = [o.type for o in self.source_op.outputs]
        return [
            arith.Constant(
                DenseIntOrFPElementsAttr.tensor_from_list(
                    [0.0], ty.get_element_type(), ty.get_shape()
                )
            )
            for ty in outputs_types
        ]

    def uniquely_match(self):
        dims = self.dims.values()

        sym_name, input_var, seq_sig = transform.get_seq_signature(
            input_consumed=False,
            has_output=True,
        )

        res_var, global_match_sig = transform.get_match_sig(input_var)
        bb_input_var, bb_header = transform.get_bb_header()

        match_dims = transform.get_match_dims(bb_input_var, dims)

        match_opname = transform.get_match_op_name(bb_input_var, self.source_op.name)

        tmyield = transform.get_match_structured_terminator(bb_input_var)

        tyield = transform.get_terminator(result=res_var)

        lines = (
            [
                seq_sig,
                "{",
                global_match_sig,
                "{",
                bb_header,
            ]
            + match_dims
            + [match_opname, tmyield, "}", tyield, "}"]
        )

        return sym_name, "\n".join(lines)

    def materialize_schedule(self):
        sym_name, input_var, seq_sig = transform.get_seq_signature(
            input_consumed=True,
            has_output=False,
        )

        dims_vectors = {}

        # Build the transform vectors corresponding to each tiling instruction
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for dim_to_tile, (k, v) in enumerate(self.tiles.items()):
                if tile_level >= len(v):
                    continue
                dim_name = list(v.keys())[tile_level]
                # TODO depends on matmul
                if dim_name in self.parallelization:
                    vsize = len(self.dims) - (len(self.dims) - len(self.parallel_dims))
                else:
                    vsize = len(self.dims)
                #
                dims_vectors[dim_name] = [
                    v[dim_name] if i == dim_to_tile else 0
                    for i in range(len(self.tiles))
                ]

        # Reorder the vectors (according to the permutation)
        dims_vectors = dict([(p, dims_vectors[p]) for p in self.permutation])

        # Actually produce the tiling (and vectorization) instructions
        loop_nest = dict()
        current_state = input_var
        tiling_instrs = []
        vect_instrs = []
        for dim, dims_vector in dims_vectors.items():
            if dim in self.vectorization:
                break
            new_state, new_loop, new_instr = transform.produce_tiling_instr(
                current_state=current_state,
                dims_vector=dims_vector,
                parallel=dim in self.parallelization,
            )

            tiling_instrs.append(new_instr)
            loop_nest[dim] = new_loop
            current_state = new_state

        parent, parent_instr = transform.get_parent(current_state)
        tiling_instrs.append(parent_instr)
        tiling_instrs += transform.tiling_apply_patterns(parent)

        if self.vectorization:
            vect_instrs += [transform.get_vectorize(current_state)]
            # vectorized,vectorize = transform.get_vectorize_children(parent)
            # vect_instrs.append(vectorize)
            vect_instrs += transform.vector_apply_patterns(parent)
        else:
            scalarized, scalarization = transform.get_scalarize(current_state)
            vect_instrs.append(scalarization)
            current_state = scalarized
            # tiling_instrs.append(transform.get_vectorize(current_state))

        # Produce the unrolling instructions (prevent unrolling of
        # "single tiles" : automatically performed
        unroll_instrs = [
            transform.get_unroll(loop=loop_nest[dim], factor=factor)
            for dim, factor in self.unrolling.items()
            if factor > 1
        ]

        lines = (
            [seq_sig, "{"]
            + tiling_instrs
            + vect_instrs
            + unroll_instrs
            + [transform.get_terminator(), "}"]
        )
        return sym_name, "\n".join(lines)

    def glue(self):
        # Fetch data
        operands = self.source_op.operands
        operands_types = [o.type for o in operands]
        results_types = [r.type for r in self.source_op.results]
        # External functions
        ext_rtclock = func.FuncOp.external("rtclock", [], [f64])
        ext_printF64 = func.FuncOp.external("printF64", [f64], [])
        # Build the payload function
        payload_func = self.payload()
        # Build the main function
        inputs = self.inputs_init()
        rtclock_call1 = func.Call(ext_rtclock.sym_name.data, [], [f64])
        outputs = self.outputs_init()
        payload_call = func.Call(payload_func.sym_name.data, inputs, results_types)
        rtclock_call2 = func.Call(ext_rtclock.sym_name.data, [], [f64])
        elapsed = arith.Subf(rtclock_call2, rtclock_call1)
        print_elapsed = func.Call(ext_printF64.sym_name.data, [elapsed], [])
        main = Block()
        main.add_ops(
            inputs
            + [
                rtclock_call1,
                payload_call,
                rtclock_call2,
                elapsed,
                print_elapsed,
                func.Return(),
            ]
        )
        main_func = func.FuncOp.from_region("main", [], [], Region(main))
        # Glue the module
        # mod = ModuleOp([ext_rtclock,ext_printF64,payload_func,main_func])
        # str_mod = str(mod)
        str_mod = "\n".join(
            [str(tl) for tl in [ext_rtclock, ext_printF64, payload_func, main_func]]
        )
        match_sym_name, str_trans_match = self.uniquely_match()

        sched_sym_name, str_trans_sched = self.materialize_schedule()

        main_name, str_trans_main = transform.build_main(
            [(match_sym_name, sched_sym_name)]
        )
        str_glued = (
            "module attributes {transform.with_named_sequence} {"
            + "\n"
            + str_mod
            + "\n"
            + str_trans_sched
            + "\n"
            + str_trans_match
            + "\n"
            + str_trans_main
            + "\n"
            + "}"
        )

        return str_glued

    def evaluate(
        self,
        print_source_ir=False,
        print_transformed_ir=False,
        print_ir_after=[],
        print_ir_before=[],
        print_assembly=False,
        color=False,
    ):
        str_module = self.glue()

        compile_extra_opts = []
        run_extra_opts = []
        disassemble_extra_opts = []

        if print_source_ir:
            zero_opt = mliropt_opts[0].replace("--", "")
            compile_extra_opts.append(f"--mlir-print-ir-before={zero_opt}")
        if print_transformed_ir:
            zero_lowering_opt = lowering_opts[0].replace("--", "")
            compile_extra_opts.append(f"--mlir-print-ir-before={zero_lowering_opt}")

        compile_extra_opts += [f"--mlir-print-ir-after={p}" for p in print_ir_after]
        compile_extra_opts += [f"--mlir-print-ir-before={p}" for p in print_ir_before]

        if print_assembly:
            run_extra_opts += [
                "--dump-object-file",
                f"--object-filename={obj_dump_file}",
            ]
            disassemble_extra_opts += [obj_dump_file]

        if color:
            disassemble_extra_opts += objdump_color_opts

        module_llvm = subprocess.run(
            [self.mliropt] + mliropt_opts + compile_extra_opts,
            input=str_module,
            stdout=subprocess.PIPE,
            text=True,
        )
        result = subprocess.run(
            self.cmd_run_mlir + run_extra_opts,
            input=module_llvm.stdout,
            stdout=subprocess.PIPE,
            text=True,
        )
        if print_assembly:
            subprocess.run(self.cmd_disassembler + disassemble_extra_opts, text=True)

        return result.stdout

    def loops(self):
        loops = dict()
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for k, v in self.tiles.items():
                if tile_level >= len(v):
                    continue
                dim_name = list(v.keys())[tile_level]
                loops[dim_name] = v[dim_name]
        return loops

    def get_default_interchange(self):
        return list(self.loops().keys())

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
        sizes = tiles_sizes + [1]
        for d, s in zip(dims, sizes):
            self.tiles[dim][d] = s
        self.permutation = self.get_default_interchange()

    def interchange(self, permutation: list[str]):
        self.permutation = permutation

    def vectorize(self, vectorization: list[str]):
        self.vectorization = vectorization

    def parallelize(self, parallelization: list[str]):
        for p in parallelization:
            assert p in self.parallel_dims
        self.parallelization = parallelization

    def unroll(self, unrolling: dict[str, int]):
        self.unrolling = unrolling
