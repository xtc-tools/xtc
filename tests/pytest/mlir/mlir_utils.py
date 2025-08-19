
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


def matmul_impl(i, j, k, dtype, name):
    import xtc.graphs.xtc.op as O
    from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend

    a = O.tensor((i, k), dtype, name="A")
    b = O.tensor((k, j), dtype, name="B")

    with O.graph(name=name) as gb:
        O.matmul(a, b, name="C")

    return MlirGraphBackend(gb.graph, always_vectorize=False, no_alias=True)
