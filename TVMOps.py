#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import utils
import os
import numpy as np
from evaluator import Evaluator, Executor

__all__ = [
    "Operation",
    "Operators",
    "Operator",
    "OperatorMatmul",
]

import tvm
import tvm.te as te


class Operation:
    def __init__(self, operator, args):
        self.operator = operator
        self.args = args
        self.tgt = tvm.target.Target(target="llvm -mcpu=skylake-avx512")
        self.dev = tvm.device(self.tgt.kind.name, 0)
        self.params = None
        self.sch = None
        self.built = None

    def generate(self):
        self.params = self.operator.generate_op(*self.args)

    def schedule(self, schedule=""):
        self.sch = te.create_schedule(self.params[-1].op)
        if schedule:
            exec(schedule, {"sch": self.sch, "obj": self.params}, {})

    def build(self):
        self.built = tvm.build(self.sch, self.params, self.tgt, name=self.operator.name)

    def load_module(self, dll):
        self.params = None
        self.sch = None
        self.built = tvm.runtime.load_module(dll)
        assert self.built.implements_function(self.operator.name), (
            f"Loaded module does not correspond to operator"
        )

    def run_eval(self, repeat=1, min_repeat_ms=0, number=1, validate=False):
        evaluator = self.built.time_evaluator(
            self.built.entry_name,
            self.dev,
            min_repeat_ms=min_repeat_ms,
            repeat=repeat,
            number=number,
        )
        inputs_spec = self.np_inputs_spec()
        outputs_spec = self.np_outputs_spec()
        inputs = [utils.np_init(**spec) for spec in inputs_spec]
        outputs = [np.empty(**spec) for spec in outputs_spec]
        tvm_inputs = [tvm.nd.array(inp) for inp in inputs]
        tvm_outputs = [tvm.nd.array(out) for out in outputs]
        if validate:
            ref_outputs = [o.copy() for o in outputs]
            self.reference_impl(*inputs, *ref_outputs)
            self.built(*tvm_inputs, *tvm_outputs)
            for ref_out, out in zip(ref_outputs, tvm_outputs):
                if not np.allclose(ref_out, out.numpy()):
                    return [], 1, "Error in validation: outputs differ"
        results = evaluator(*tvm_inputs, *tvm_outputs).results
        return results, 0, ""

    def run_eval_dll(
        self,
        dll,
        sym,
        repeat=1,
        min_repeat_ms=0,
        number=1,
        validate=False,
        parameters=None,
    ):
        dll = os.path.abspath(dll)
        with utils.LibLoader(dll) as lib:
            func = getattr(lib, "matmul")
            assert func is not None, f"Cannot find {sym} in lib {dll}"
            func.packed = True
            inputs_spec = self.np_inputs_spec()
            outputs_spec = self.np_outputs_spec()
            if parameters is None:
                inputs = [utils.np_init(**spec) for spec in inputs_spec]
                outputs = [np.empty(**spec) for spec in outputs_spec]
                parameters = (
                    [NDArray(inp) for inp in inputs],
                    [NDArray(out) for out in outputs],
                )
            if validate:
                ref_inputs = [inp.numpy() for inp in parameters[0]]
                ref_outputs = [np.empty(**spec) for spec in outputs_spec]
                self.reference_impl(*ref_inputs, *ref_outputs)
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

    def run(self):
        results, code, error = self.run_eval(repeat=1, min_repeat_ms=0, number=1)
        return min(results) if code == 0 else error

    def lower(self):
        return tvm.lower(self.sch, self.params, simple_mode=True)

    def np_inputs_spec(self):
        inputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                self.operator.inputs_dims(*self.args),
                self.operator.inputs_types(*self.args),
            )
        ]
        return inputs_spec

    def np_outputs_spec(self):
        outputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                self.operator.outputs_dims(*self.args),
                self.operator.outputs_types(*self.args),
            )
        ]
        return outputs_spec

    def reference_impl(self, *args):
        self.operator.reference_impl(*args)


class Operator:
    name = "undef"

    @staticmethod
    def generate_op(i, j, k, dtype):
        raise Exception("unimplemneted")


class OperatorMatmul(Operator):
    name = "matmul"

    @staticmethod
    def generate_op(i, j, k, dtype):
        A = te.placeholder((i, k), name="A")
        B = te.placeholder((k, j), name="B")

        k = te.reduce_axis((0, k), "k")
        O = te.compute(
            (i, j),
            lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
            attrs={"layout_free_placeholders": [B]},
            name="O",
        )
        return A, B, O

    @staticmethod
    def inputs_dims(i, j, k, dtype):
        return (i, k), (k, j)

    @staticmethod
    def inputs_types(i, j, k, dtype):
        return dtype, dtype

    @staticmethod
    def outputs_dims(i, j, k, dtype):
        return ((i, j),)

    @staticmethod
    def outputs_types(i, j, k, dtype):
        return (dtype,)

    @staticmethod
    def reference_impl(*args):
        np.matmul(args[0], args[1], out=args[2])


class Operators:
    matmul = OperatorMatmul


def test_matmul(tdir="."):
    operation = Operation(Operators.matmul, (256, 256, 512, "float32"))
    operation.generate()
    operation.schedule()
    operation.build()
    operation.built.export_library(f"{tdir}/tvm-payload.so")
    time = operation.run()
    print(f"Execution time {operation.built.entry_name}: {time} secs")


def test_load_and_run(tdir="."):
    modlib = tvm.runtime.load_module(f"{tdir}/tvm-payload.so")
    operation = Operation(Operators.matmul, (256, 256, 512, "float32"))
    operation.built = modlib
    time = operation.run()
    print(f"Execution time {operation.built.entry_name}: {time} secs")


def test_evaluator(tdir="."):
    from evaluator import Evaluator, Executor

    i, j, k, dtype = 256, 256, 512, "float32"
    repeat, min_repeat_ms, number = 5, 100, 1
    with utils.LibLoader(f"{tdir}/tvm-payload.so") as lib:
        func = getattr(lib, "matmul")
        func.packed = True
        inputs = [utils.np_init(shape=shape, dtype=dtype) for shape in [(i, k), (k, j)]]
        outputs = [np.empty(shape=shape, dtype=dtype) for shape in [(i, j)]]
        nd_inputs = [tvm.nd.array(inp) for inp in inputs]
        nd_outputs = [tvm.nd.array(out) for out in outputs]
        exec_func = Executor(func)
        exec_func(*nd_inputs, *nd_outputs)
        assert np.allclose(inputs[0] @ inputs[1], nd_outputs[0].numpy()), (
            f"Error in evaluation"
        )
        eval_func = Evaluator(
            func, repeat=repeat, min_repeat_ms=min_repeat_ms, number=number
        )
        results = eval_func(*nd_inputs, *nd_outputs)
    print(f"Execution time results: {results}")


if __name__ == "__main__":
    import tempfile

    tmpdir = "."
    with tempfile.TemporaryDirectory() as tdir:
        tdir = tmpdir or tdir
        test_matmul(tdir)
        test_load_and_run(tdir)
        test_evaluator(tdir)
