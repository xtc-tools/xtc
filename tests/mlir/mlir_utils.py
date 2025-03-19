
import numpy as np

from xtc.utils.numpy import (
    np_init,
)
from xtc.runtimes.types.ndarray import NDArray

def requires_mlir(*arg):
    import pytest
    def has_mlir():
        try:
            import mlir
            return True
        except:
            return False
    return pytest.mark.skipif(not has_mlir(), reason="requires MLIR")(*arg)


def matmul_graph(i, j, k, dtype, name):
    from xdsl.dialects import func, linalg, arith, builtin
    from xdsl.dialects.builtin import MemRefType, f32, f64, UnitAttr
    from xdsl.ir import Block, Region
    from xdsl.builder import ImplicitBuilder

    elt_type = {"float32": f32, "float64": f64}[dtype]
    ops_types = [MemRefType(elt_type, shape)
                 for shape in [[i,k], [k,j], [i,j]]]
    block = Block(arg_types=ops_types)
    with ImplicitBuilder(block):
        cst0 = arith.ConstantOp(builtin.FloatAttr(0, 32))
        fill = linalg.FillOp(
            res=(),
            inputs=(cst0.results[0],),
            outputs=(block.args[2],),
        )
        matmul = linalg.MatmulOp(
            res=(),
            inputs=(block.args[0], block.args[1]),
            outputs=(block.args[2],),
        )
        func.ReturnOp()
    fill.attributes["__xtc_id_fill__"] = UnitAttr()
    matmul.attributes["__xtc_id_matmul__"] = UnitAttr()
    region = Region([block])
    payload = func.FuncOp.from_region(
        name=name,
        input_types=ops_types,
        return_types=[],
        region=region,
    )
    return payload, [fill, matmul]

def matmul_impl(i, j, k, dtype, name):
    from xtc.backends.mlir.MlirNodeImplementer import MlirNodeImplementer
    from xtc.backends.mlir.MlirGraphImplementer import MlirGraphImplementer

    xdsl_func, (xdsl_fill_op, xdsl_matmul_op) = matmul_graph(i, j, k, dtype, name)
    fill = MlirNodeImplementer(
        payload_name="fill",
        source_op=xdsl_fill_op,
        dims=["i", "j"],
        no_alias=True,
        id=f"__xtc_id_fill__",
    )
    matmul = MlirNodeImplementer(
        payload_name="matmul",
        source_op=xdsl_matmul_op,
        dims=["i", "j", "k"],
        no_alias=True,
        id=f"__xtc_id_matmul__",
    )
    impl = MlirGraphImplementer(
        xdsl_func=xdsl_func,
        nodes=[fill, matmul],
    )
    return impl

def matmul_node_impl(i, j, k, dtype, name):
    from xdsl.dialects import linalg
    from xdsl.dialects.builtin import MemRefType, f32, f64, UnitAttr
    from xdsl.ir import Block
    from xdsl.builder import ImplicitBuilder
    from xtc.backends.mlir.MlirNodeImplementer import MlirNodeImplementer

    elt_type = {"float32": f32, "float64": f64}[dtype]
    ops_types = [MemRefType(elt_type, shape)
                 for shape in [[i,k], [k,j], [i,j]]]
    block = Block(arg_types=ops_types)
    with ImplicitBuilder(block):
        xdsl_matmul_op = linalg.MatmulOp(
            res=(),
            inputs=(block.args[0], block.args[1]),
            outputs=(block.args[2],),
        )
    xdsl_matmul_op.attributes["__xtc_id_matmul__"] = UnitAttr()
    impl = MlirNodeImplementer(
        payload_name=name,
        source_op=xdsl_matmul_op,
        dims=["i", "j", "k"],
        no_alias=True,
        id=f"__xtc_id_matmul__",
    )
    return impl

def get_matmul_params(i, j, k, dtype):
    inputs_shapes = ((i, k), (k, j))
    outputs_shapes = ((i, j),)
    inputs = [
        np_init(shape=shape, dtype=dtype)
        for shape in inputs_shapes
    ]
    outputs = [
        np.empty(shape=shape, dtype=dtype)
        for shape in outputs_shapes
    ]
    parameters = (
        [NDArray(inp) for inp in inputs],
        [NDArray(out) for out in outputs],
    )
    return parameters

def matmul_ref(a, b, c):
    return np.matmul(a, b, out=c)
