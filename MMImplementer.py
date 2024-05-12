#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC
import subprocess
import numpy

from mlir.ir import *
import mlir
from mlir.dialects import arith, builtin, func, linalg, tensor

from PerfectlyNestedImplementer import PerfectlyNestedImplementer
import transform


class MMImplementer(PerfectlyNestedImplementer):
    def __init__(
        self,
        mlir_install_dir: str,
        dims: dict[str, int],
        parallel_dims: list[str],
        reduction_dims: list[str],
    ):
        super().__init__(mlir_install_dir, dims, parallel_dims, reduction_dims)

        self.ctx = Context()
        self.elt_type = F32Type.get(context=self.ctx)
        self.loc = Location.unknown(self.ctx)
        self.module = builtin.ModuleOp(loc=self.loc)

    def initialize_tensor(self, shape, value):
        tensor_type = RankedTensorType.get(
            shape=shape, element_type=self.elt_type, loc=self.loc
        )
        elt = arith.ConstantOp(
            value=FloatAttr.get(self.elt_type, value, loc=self.loc),
            result=self.elt_type,
            loc=self.loc,
        ).result
        # TODO strange bug
        with self.loc as loc:
            empty = tensor.EmptyOp(shape, self.elt_type)
            return linalg.fill(elt, outs=[empty])

    # return tensor.SplatOp(tensor_type,elt)

    def build_rtclock(self):
        f64 = F64Type.get(context=self.ctx)
        with InsertionPoint.at_block_begin(self.module.body):
            frtclock = func.FuncOp(
                name="rtclock",
                type=FunctionType.get(inputs=[], results=[f64]),
                visibility="private",
                loc=self.loc,
            )
        return frtclock

    def build_printF64(self):
        f64 = F64Type.get(context=self.ctx)
        with InsertionPoint.at_block_begin(self.module.body):
            fprint = func.FuncOp(
                name="printF64",
                type=FunctionType.get(inputs=[f64], results=[]),
                visibility="private",
                loc=self.loc,
            )
        return fprint

    def payload(self):
        i = self.dims[self.parallel_dims[0]]
        j = self.dims[self.parallel_dims[1]]
        k = self.dims[self.reduction_dims[0]]
        A_tensor_type = RankedTensorType.get(
            shape=(i, k),
            element_type=self.elt_type,
            loc=self.loc,
        )
        B_tensor_type = RankedTensorType.get(
            shape=(k, j),
            element_type=self.elt_type,
            loc=self.loc,
        )
        C_tensor_type = RankedTensorType.get(
            shape=(i, j),
            element_type=self.elt_type,
            loc=self.loc,
        )
        with InsertionPoint.at_block_begin(self.module.body):
            f = func.FuncOp(
                name=self.payload_name,
                type=FunctionType.get(
                    inputs=[A_tensor_type, B_tensor_type, C_tensor_type],
                    results=[C_tensor_type],
                ),
                loc=self.loc,
            )
        entry_block = f.add_entry_block(arg_locs=[self.loc, self.loc, self.loc])
        with InsertionPoint(entry_block):
            A = f.entry_block.arguments[0]
            B = f.entry_block.arguments[1]
            C_init = f.entry_block.arguments[2]
            # TODO strange bug
            with self.loc as loc:
                matmul = linalg.matmul(A, B, outs=[C_init])
            func.ReturnOp([matmul], loc=self.loc)
        return f

    def uniquely_match(self):
        dims = self.dims.values()

        sym_name, input_var, seq_sig = transform.get_seq_signature(
            input_consumed=False,
            has_output=True,
        )

        res_var, global_match_sig = transform.get_match_sig(input_var)
        bb_input_var, bb_header = transform.get_bb_header()

        match_dims = transform.get_match_dims(bb_input_var, dims)

        match_opname = transform.get_match_op_name(bb_input_var, "linalg.matmul")

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

    def init_payload(self):
        i = self.dims[self.parallel_dims[0]]
        j = self.dims[self.parallel_dims[1]]
        C_tensor_type = RankedTensorType.get(
            shape=(i, j), element_type=self.elt_type, loc=self.loc
        )
        with InsertionPoint.at_block_begin(self.module.body):
            init_func = func.FuncOp(
                name=self.init_payload_name,
                type=FunctionType.get(inputs=[], results=[C_tensor_type]),
                loc=self.loc,
            )
        with InsertionPoint(init_func.add_entry_block()):
            C = self.initialize_tensor((i, j), 0.0)
            func.ReturnOp([C], loc=self.loc)
        return init_func

    def main(self, frtclock, fprint, fmatmul, init_payload):
        #
        i = self.dims[self.parallel_dims[0]]
        j = self.dims[self.parallel_dims[1]]
        k = self.dims[self.reduction_dims[0]]
        A_tensor_type = RankedTensorType.get(
            shape=(i, k),
            element_type=self.elt_type,
            loc=self.loc,
        )
        B_tensor_type = RankedTensorType.get(
            shape=(k, j),
            element_type=self.elt_type,
            loc=self.loc,
        )
        with InsertionPoint.at_block_begin(self.module.body):
            fmain = func.FuncOp(
                name="main",
                type=FunctionType.get(inputs=[], results=[]),
                loc=self.loc,
            )
        with InsertionPoint(fmain.add_entry_block()):
            #
            A = self.initialize_tensor(shape=(i, k), value=numpy.random.random())
            #
            B = self.initialize_tensor(shape=(k, j), value=numpy.random.random())
            #
            callrtclock1 = func.CallOp(frtclock, [], loc=self.loc)
            C_init = func.CallOp(init_payload, [], loc=self.loc)
            C = func.CallOp(fmatmul, [A, B, C_init], loc=self.loc)
            callrtclock2 = func.CallOp(frtclock, [], loc=self.loc)
            time = arith.SubFOp(callrtclock2, callrtclock1, loc=self.loc)
            func.CallOp(fprint, [time], loc=self.loc)
            func.ReturnOp([], loc=self.loc)

        return fmain
