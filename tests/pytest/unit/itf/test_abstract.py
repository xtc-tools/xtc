
def test_abstract_imports():
    from xtc.itf.graph.graph import Graph
    from xtc.itf.graph.node import Node
    from xtc.itf.data.tensor import Tensor, TensorType
    from xtc.itf.operator.operator import Operator

    from xtc.itf.back.backend import Backend
    from xtc.itf.comp.compiler import Compiler
    from xtc.itf.schd.scheduler import Scheduler
    from xtc.itf.schd.schedule import Schedule
