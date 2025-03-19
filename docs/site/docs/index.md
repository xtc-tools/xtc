# Xdsl Transform Compiler

The XTC project, is a compiler framework that provides high-level scheduling specifications for linear algebra operations over a dataflow graph.

## About

The projet provides high-level syntax for applying transformations like:

- Tiling
- Loop interchange
- Vectorization
- Unrolling

It allow the integration of multiple backend and implements:

- MLIR backend (using MLIR linalg and transform dialects)
- TVM backend (using TVM Tensor IR and TVM Schedule APIs)
- JIR backend (using JIR intermediate representation and MLIR scf and affine dialects)

Core components:

- Abstract interfaces for tensors, graphs, nodes, and operators
- Implementations for different backends
- Schedulers for transformation management
- Compilers for code generation
- Evaluators for performance measurement
- Autotuner for scheduling space exploration

For details, get to the [Architecture Design](design/architecture.md) or [Reference API](reference/api.md).

## Links

Refer to the [Source Repository](https://gitlab.inria.fr/hpompoug/xdsl-transform).
