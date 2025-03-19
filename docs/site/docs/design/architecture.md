# Software Architecture Design

XTC stands for Xdsl Transform Compiler, a dataflow graph compiler
using domain specific (DSL) description for the operational semantic of the graph
and the transformations over the graph.

The compiler can have different backends which implement the same interfaces
and may use their own intermediate representations for operations and transformations
and target different hardwares and runtimes.

## Interface classes

Input classes:

- Tensor: an abstract representation of a multidimensional object
- Graph: an abstract representation of a dataflow graph over Tensor types
- Node: an abstract reprensentation of a node in the graph
- Operator: an abstract representation of the algebraic operation for a node

A Node is defined as a pure operation on input Tensor objects, resulting in output Tensor objects.

A Graph is a directed acyclic graph over Node objects with input Tensors and output Tensors.

An Operator defines the semantic of the operation for a given Node.

From given input Tensor dimensions and type, all node inputs, outputs and graph ouputs Tensor
dimensions and types can be inferred.

A Graph can be evaluated by interpretation or compiled.


Compiler classes:

- Backend: an abstract implementation of a Graph for some evaluation/compilation backend
- Scheduler: an abstract implementation of the backend scheduler
- Schedule: an abstract representation of the result of transformations from a scheduler
- Compiler: an abstract implementation of a compiler for a given backend and schedule
- Module: an abstract representation of an executable module

An Backend is constructed for an input Graph.

A Scheduler is constructed from a given Backend.

A Compiler is constructed from a given Backend.

A Scheduler is used to apply primitive scheduling operations over the Backend,
a resulting Schedule can be constructed.

A Compiler is used to compile the Backend implementation of the Graph given a generated Schedule,
a resulting Module can be constructed.

Evaluation Classes:

- Executor: an abstract implementation of a Module executor
- Evaluator: an abstract implementation of a Module performance evaluator

An Executor or an Evaluator can be given a Module to execute and report validity and
performannce metrics.

## Concrete implementations

Example of concrete implementations for Backend, Scheduler, Compiler provided by some different backends:

- MLIR backend: using mlir linalg and transform dialects passed to mlir/llvm toolchain
- TVM backend: using tvm tensor IR and schedule APIs passed to tvm backends 
- JIR backend: using the jir intermediate representation and transformation to output mlir scf and affine dialects passed to mlir/llvm toolchain

All backends above are able to generate Module in the form of shared objects for direct
excution and evaluation or usage in a larger application.
