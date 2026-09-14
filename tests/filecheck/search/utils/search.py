from itertools import islice

from xtc.itf.graph import Graph
from xtc.itf.back import Backend
from xtc.itf.search import Strategy

import xtc.graphs.xtc.op as O

def get_graph_matmul():
    I, J, K, dtype = 21, 32, 12, "float32"

    a = O.tensor((I, K), dtype)
    b = O.tensor((K, J), dtype)

    with O.graph(name="matmul") as gb:
        O.matmul(a, b)
    return gb.graph

def get_graph_conv2d():
    B, H, W, F, R, S, C, SH, SW, dtype = 2, 4, 4, 32, 7, 7, 3, 2, 2, "float32"
    IH, IW = H + R - 1, W + S - 1
    a = O.tensor((B, IH, IW, C), dtype)
    w = O.tensor((R, S, C, F), dtype)

    with O.graph(name="matmul") as gb:
        O.conv2d(a, w, stride=(SH, SW))
    return gb.graph

def get_backend(graph: Graph, backend: str = "mlir"):
    if backend == "mlir":
        from xtc.backends.mlir import Backend
    elif backend == "tvm":
        from xtc.backends.tvm import Backend
    else:
        assert backend == "jir"
        from xtc.backends.jir import Backend
    return Backend(graph)

def print_all_opt_schedules(backend: Backend, strategy: Strategy):
    for opt_level in range(4):
        sample = strategy.default_schedule(opt_level)
        print(f"schedule O{opt_level}: {sample}")
        sch = backend.get_scheduler()
        strategy.generate(sch, sample)
        print(sch.schedule())

def print_exhaustive_samples(backend: Backend, strategy: Strategy, num: int|None = None):
    generator = strategy.exhaustive()
    if num is not None:
        generator = islice(generator, num)
    sample = []
    for idx, sample in enumerate(generator):
        print(f"sample {idx}: {sample}")
    if hasattr(strategy, "stats"):
        print("stats", getattr(strategy, "stats"))
    sch = backend.get_scheduler()
    strategy.generate(sch, sample)
    print(sch.schedule())

def print_random_samples(backend: Backend, strategy: Strategy, num: int):
    generator = strategy.sample(num=num)
    sample = []
    for idx, sample in enumerate(generator):
        print(f"sample {idx}: {sample}")
    if hasattr(strategy, "stats"):
        print("stats", getattr(strategy, "stats"))
    sch = backend.get_scheduler()
    strategy.generate(sch, sample)
    print(sch.schedule())
