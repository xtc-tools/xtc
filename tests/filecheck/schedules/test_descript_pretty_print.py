# RUN: python %s --simple 2>&1 | filecheck %s --check-prefix=CHECK-SIMPLE
# RUN: python %s --tiled 2>&1 | filecheck %s --check-prefix=CHECK-TILED
# RUN: python %s --vectorized 2>&1 | filecheck %s --check-prefix=CHECK-VECTORIZED
# RUN: python %s --full 2>&1 | filecheck %s --check-prefix=CHECK-FULL
# RUN: python %s --split 2>&1 | filecheck %s --check-prefix=CHECK-SPLIT
# RUN: python %s --buffer 2>&1 | filecheck %s --check-prefix=CHECK-BUFFER
# RUN: python %s --pack 2>&1 | filecheck %s --check-prefix=CHECK-PACK

import sys
from xtc.schedules.parsing import ScheduleParser
from xtc.schedules.descript import ScheduleInterpreter

parser = ScheduleParser()
abstract_axis = ["i", "j", "k"]
interpreter = ScheduleInterpreter(abstract_axis)

if "--simple" in sys.argv:
    spec = {"i": {}, "k": {}, "j": {}}
    ast = parser.parse(spec)
    loop_nest = interpreter.interpret(ast, root="C")
    print(loop_nest.root_node.pretty_print())

elif "--tiled" in sys.argv:
    spec = {"i": {}, "k": {}, "j": {}, "j#16": {}}
    ast = parser.parse(spec)
    loop_nest = interpreter.interpret(ast, root="C")
    print(loop_nest.root_node.pretty_print())

elif "--vectorized" in sys.argv:
    spec = {"i": {}, "k": {}, "j": {}, "j#16": {"vectorize": True}}
    ast = parser.parse(spec)
    loop_nest = interpreter.interpret(ast, root="C")
    print(loop_nest.root_node.pretty_print())

elif "--full" in sys.argv:
    spec = {
        "i": {"parallelize": True},
        "k": {},
        "j": {},
        "j#32": {},
        "j#16": {"vectorize": True, "unroll": 4},
    }
    ast = parser.parse(spec)
    loop_nest = interpreter.interpret(ast, root="C")
    print(loop_nest.root_node.pretty_print())

elif "--split" in sys.argv:
    spec = {
        "i": {},
        "j[:128]": {"k": {}, "k#32": {}},
        "j[128:]": {"k": {}, "k#16": {"vectorize": True}},
    }
    ast = parser.parse(spec)
    loop_nest = interpreter.interpret(ast, root="C")
    print(loop_nest.root_node.pretty_print())

elif "--buffer" in sys.argv:
    spec = {
        "i": {"parallelize": True},
        "k": {"buffer": "default"},
        "j": {"buffer": "shared"},
        "j#16": {"vectorize": True},
    }
    ast = parser.parse(spec)
    loop_nest = interpreter.interpret(ast, root="C")
    print(loop_nest.root_node.pretty_print())

elif "--pack" in sys.argv:
    spec = {
        "i": {"parallelize": True},
        "k": {"pack": (0, "default", False)},
        "j": {"pack": (1, "shared", True)},
        "j#16": {"vectorize": True},
    }
    ast = parser.parse(spec)
    loop_nest = interpreter.interpret(ast, root="C")
    print(loop_nest.root_node.pretty_print())

# CHECK-SIMPLE:      loop i
# CHECK-SIMPLE-NEXT:   loop k
# CHECK-SIMPLE-NEXT:     loop j
# CHECK-SIMPLE-NEXT:       ...

# CHECK-TILED:      loop i
# CHECK-TILED-NEXT:   loop k
# CHECK-TILED-NEXT:     loop j
# CHECK-TILED-NEXT:       tile(j, 16)
# CHECK-TILED-NEXT:         ...

# CHECK-VECTORIZED:      loop i
# CHECK-VECTORIZED-NEXT:   loop k
# CHECK-VECTORIZED-NEXT:     loop j
# CHECK-VECTORIZED-NEXT:       tile(j, 16)  // vectorized
# CHECK-VECTORIZED-NEXT:         ...

# CHECK-FULL:      loop i  // parallelized
# CHECK-FULL-NEXT:   loop k
# CHECK-FULL-NEXT:     loop j
# CHECK-FULL-NEXT:       tile(j, 32)
# CHECK-FULL-NEXT:         tile(j, 16)  // vectorized, unroll(4)
# CHECK-FULL-NEXT:           ...

# CHECK-SPLIT:      loop i
# CHECK-SPLIT-NEXT:   split(j, 0, 128)
# CHECK-SPLIT-NEXT:     loop j
# CHECK-SPLIT-NEXT:       loop k
# CHECK-SPLIT-NEXT:         tile(k, 32)
# CHECK-SPLIT-NEXT:           ...
# CHECK-SPLIT-NEXT:   split(j, 128, ...)
# CHECK-SPLIT-NEXT:     loop j
# CHECK-SPLIT-NEXT:       loop k
# CHECK-SPLIT-NEXT:         tile(k, 16)  // vectorized
# CHECK-SPLIT-NEXT:           ...

# CHECK-BUFFER:      loop i  // parallelized
# CHECK-BUFFER-NEXT:   loop k  // buffer
# CHECK-BUFFER-NEXT:     loop j  // buffer(shared)
# CHECK-BUFFER-NEXT:       tile(j, 16)  // vectorized
# CHECK-BUFFER-NEXT:         ...

# CHECK-PACK:      loop i  // parallelized
# CHECK-PACK-NEXT:   loop k  // pack(0)
# CHECK-PACK-NEXT:     loop j  // pack(1, shared, pad)
# CHECK-PACK-NEXT:       tile(j, 16)  // vectorized
# CHECK-PACK-NEXT:         ...
