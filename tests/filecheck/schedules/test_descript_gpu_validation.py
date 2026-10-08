# REQUIRES: mlir-target=nvgpu
# RUN: not python %s --overlap 2>&1 | filecheck %s --check-prefix=CHECK-OVERLAP
# RUN: not python %s --no-block 2>&1 | filecheck %s --check-prefix=CHECK-NO-BLOCK
# RUN: not python %s --block-tile 2>&1 | filecheck %s --check-prefix=CHECK-BLOCK-TILE
# RUN: not python %s --thread-axis 2>&1 | filecheck %s --check-prefix=CHECK-THREAD-AXIS
# RUN: not python %s --block-not-outermost 2>&1 | filecheck %s --check-prefix=CHECK-OUTERMOST

import sys
import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.schedules.descript import descript_scheduler

I, J, K, dtype = 16, 32, 64, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph


def make_descript_scheduler(spec):
    impl = Backend(graph)
    sch = impl.get_scheduler()
    descript_scheduler(
        scheduler=sch,
        node_name="C",
        abstract_dims=["I", "J", "K"],
        spec=spec,
    )
    print("ok")

if "--overlap" in sys.argv:
    # The same loop mapped to two gpu primitives
    make_descript_scheduler(
        {
            "I": {"gpu_block": 0, "gpu_thread": 0},
            "J": {},
            "K": {},
        }
    )

# CHECK-OVERLAP: Loops I appear in both gpu_block and gpu_thread.

elif "--no-block" in sys.argv:
    # Threads need at least one block-mapped loop
    make_descript_scheduler(
        {
            "I": {"gpu_thread": 0},
            "J": {},
            "K": {},
        }
    )

# CHECK-NO-BLOCK: Need gpu_block to be specified for either gpu_thread or gpu_lane or gpu_warp.

elif "--block-tile" in sys.argv:
    # Blocks need to be mapped to base axes, not tiles
    make_descript_scheduler(
        {
            "I": {},
            "I#4": {"gpu_block": 0},
            "J": {},
            "J#8": {"gpu_thread": 0},
            "K": {},
        }
    )

# CHECK-BLOCK-TILE: Need gpu_block to be an axis and not a tile

elif "--thread-axis" in sys.argv:
    # Threads need to be mapped to tiles, not base axes
    make_descript_scheduler(
        {
            "I": {"gpu_block": 0},
            "J": {"gpu_thread": 0},
            "K": {},
        }
    )

# CHECK-THREAD-AXIS: gpu_thread need to be a tile

elif "--block-not-outermost" in sys.argv:
    # The block-mapped loops need to come first in the loop order
    make_descript_scheduler(
        {
            "K": {},
            "I": {"gpu_block": 0},
            "J": {"gpu_block": 1},
            "I#4": {"gpu_thread": 0},
            "J#8": {"gpu_thread": 1},
        }
    )

# CHECK-OUTERMOST: gpu_block needs to be in the most outermost loop
