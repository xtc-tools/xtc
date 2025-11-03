# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_jir

import xtc.graphs.xtc.op as O
from xtc.backends.jir import Backend
from xtc.schedules.descript import descript_scheduler

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
descript_scheduler(
    scheduler = sch,
    node_name = "C",
    abstract_axis = ["I","J","K"],
    spec = {
        "K": {},
        "I": {},
        "J": {},
        "I#2": {"unroll": None},
        "J#16": {"vectorize": None},
    }
)

sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_descript_jir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x512xfloat32
# CHECK-NEXT:    - %1 : 512x32xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 4x32xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  
# CHECK-NEXT:  function matmul
# CHECK-NEXT:    dimensions
# CHECK-NEXT:      I, J, K
# CHECK-NEXT:    buffers
# CHECK-NEXT:      A: <I, K> f32
# CHECK-NEXT:      B: <K, J> f32
# CHECK-NEXT:      O: <I, J> f32
# CHECK-NEXT:    {
# CHECK-NEXT:      I0: for i in I (O)
# CHECK-NEXT:        J0: for j in J (O)
# CHECK-NEXT:          op0_C(O)
# CHECK-NEXT:      II: for i in I (*)
# CHECK-NEXT:        JJ: for j in J (*)
# CHECK-NEXT:          KK: for reduction k in K (*)
# CHECK-NEXT:              op_C(O, A, B)
# CHECK-NEXT:    }
# CHECK-NEXT:  
# CHECK-NEXT:  commands:
# CHECK-NEXT:    - subdim parent=I sub=[I_1 I_1$]
# CHECK-NEXT:    - compl dim=I_1 other=I_1$
# CHECK-NEXT:    - tile target=II tile=I_1 inner=II_I0
# CHECK-NEXT:    - subdim parent=J sub=[J_1 J_1$]
# CHECK-NEXT:    - compl dim=J_1 other=J_1$
# CHECK-NEXT:    - tile target=JJ tile=J_1 inner=JJ_J0
# CHECK-NEXT:    - interchange target=JJ_J0
# CHECK-NEXT:    - interchange target=JJ
# CHECK-NEXT:    - interchange target=II_I0
# CHECK-NEXT:    - interchange target=II
# CHECK-NEXT:    - interchange target=II_I0
# CHECK-NEXT:    - update_props target=JJ_J0 vector=16
# CHECK-NEXT:    - update_props target=II_I0 unroll=2
# CHECK-NEXT:  
# CHECK-NEXT:  dims:
# CHECK-NEXT:    I: 4
# CHECK-NEXT:    I_1: 2
# CHECK-NEXT:    I_1$: 2
# CHECK-NEXT:    J: 32
# CHECK-NEXT:    J_1: 16
# CHECK-NEXT:    J_1$: 2
# CHECK-NEXT:    K: 512
# CHECK-NEXT:  
# CHECK-NEXT:  
# CHECK-NEXT:  function matmul
# CHECK-NEXT:    dimensions
# CHECK-NEXT:      I, J, K;
# CHECK-NEXT:      subdim(I) I_1, I_1$;
# CHECK-NEXT:      compl(I_1$) I_1;
# CHECK-NEXT:      subdim(J) J_1, J_1$;
# CHECK-NEXT:      compl(J_1$) J_1;
# CHECK-NEXT:    buffers
# CHECK-NEXT:      A: <I, K> f32;
# CHECK-NEXT:      B: <K, J> f32;
# CHECK-NEXT:      O: <I, J> f32;{
# CHECK-NEXT:    I0: for i in I (O: <J>) 
# CHECK-NEXT:      J0: for j in J (O: <>) 
# CHECK-NEXT:        op0_C(O /* [I=i, J=j] */)
# CHECK-NEXT:    KK: for reduction k in K (B: <J>, A: <I>) 
# CHECK-NEXT:      II: for i in I step I_1 (A: <I_1>, O: <I_1, J>) 
# CHECK-NEXT:        JJ: for j in J step J_1 (O: <I_1, J_1>, B: <J_1>) 
# CHECK-NEXT:          II_I0: for unroll(2) i_1 in I_1 (A: <>, O: <J_1>) 
# CHECK-NEXT:            JJ_J0: for vector(16) j_1 in J_1 (O: <>, B: <>) 
# CHECK-NEXT:              op_C(O /* [I=(i + i_1), J=(j + j_1)] */, A /* [I=(i + i_1), K=k] */, B /* [K=k, J=(j + j_1)] */)
# CHECK-NEXT:  }
# CHECK-NEXT:  CODE: 0
