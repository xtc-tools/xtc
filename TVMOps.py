#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from utils import LazyImport

__all__ = [
    "Operation",
    "Operators",
    "Operator",
    "OperatorMatmul",
]

tvm = LazyImport("tvm")
te = LazyImport("tvm.te")
np = LazyImport("numpy")


class Operation:
    def __init__(self, operator, args):
        self.operator = operator
        self.args = args
        self.tgt = tvm.target.Target(target="llvm -mcpu=skylake-avx512")
        self.dev = tvm.device(self.tgt.kind.name, 0)
        self.params = None
        self.sch = None
        self.built = None

    def generate(self, schedule=""):
        self.params = self.operator.generate_op(*self.args)

    def schedule(self, schedule=""):
        self.sch = te.create_schedule(self.params[-1].op)
        if schedule:
            exec(schedule, {"sch": self.sch, "obj": self.params}, {})

    def build(self):
        self.built = tvm.build(self.sch, self.params, self.tgt, name=self.operator.name)

    def run(self):
        evaluator = self.built.time_evaluator(
            self.built.entry_name, self.dev, min_repeat_ms=0, repeat=1, number=1
        )
        params = []
        for shape, dtype in zip(
            self.operator.inputs_dims(*self.args),
            self.operator.inputs_types(*self.args),
        ):
            params.append(np.random.uniform(size=shape).astype(dtype))
        for shape, dtype in zip(
            self.operator.outputs_dims(*self.args),
            self.operator.outputs_types(*self.args),
        ):
            params.append(np.random.uniform(size=shape).astype(dtype))
        tvm_params = [tvm.nd.array(t) for t in params]
        results = evaluator(*tvm_params).results
        return min(results)

    def lower(self):
        return tvm.lower(self.sch, self.params, simple_mode=True)


class Operator:
    name = "undef"

    @staticmethod
    def generate_op(i, j, k, dtype):
        raise Excception("unimplemneted")


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


class Operators:
    matmul = OperatorMatmul


def test_matmul():
    operation = Operation(Operators.matmul, (256, 256, 512, "float32"))
    operation.generate()
    operation.schedule()
    operation.build()
    time = operation.run()
    print(f"Execution time: {time} secs")


if __name__ == "__main__":
    test_matmul()
