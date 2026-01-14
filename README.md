# XTC

## Links

![logo.png](logo.png)

Refer to documentation at https://corse.gitlabpages.inria.fr/XTC

Refer to installable python packages at: https://gitlab.inria.fr/corse/xtc/-/packages

Refer to tutorials [here](docs/tutorials) and to additionnal developers documentation [here](docs/develop).

## Overview

XTC is a domain-specific dataflow graph compiler for linear algebra operations. It provides:
- **Operational DSL**: Define computation graphs with tensors and operators
- **Scheduling DSL**: High-level transformations (tiling, parallelization, vectorization, etc.)
- **Multiple backends**: MLIR (linalg + transform), TVM (Tensor IR), JIR (INRIA internal)
- **Autotuning**: Definition and exploration of the optimization space

## Build & Development

### System requirements

Debian-like distributions:
```bash
sudo apt install python3 python3-dev build-essential libomp5 binutils binutils-aarch64-linux-gnu binutils-x86-64-linux-gnu

# Optionally if using PMU counters on CPU for evaluation
sudo apt install libpfm4-dev 
sudo sysctl kernel.perf_event_paranoid=1
```

Fedora:
```bash
sudo dnf install python3 python3-devel libomp binutils binutils-aarch64-linux-gnu binutils-x86_64-linux-gnu

# For Fedora 40+
sudo dnf group install c-development development-tools # For Fedora 40+

# Optionally if using PMU counters on CPU for evaluation
sudo dnf install libpfm-devel
sudo sysctl kernel.perf_event_paranoid=1
```

### Installation

**Python**: 3.10 to 3.14 inclusive

```bash
python3 -m venv .venv && source .venv/bin/activate
pip3 install -e '.[dev]'
pip3 install -r mlir_requirements.txt  # Optional: MLIR backend
pip3 install -r tvm_requirements.txt   # Optional: TVM backend
make test                              # Run minimal unit tests
```

### Code quality

Code quality requirements:
- **Type annotations**: Strict pyright mode, full annotations required
- **Formatting**: Ruff (line length 88)
- **License headers**: BSD-3-Clause required on all source files
- All checks must pass before merge

Type checking:
```bash
make check-type          # Run both pyright and mypy
pyright                  # Run pyright only
mypy                     # Run mypy only
```

Formatting:
```bash
make format              # Apply all formatting (license + ruff)
make check-format        # Check formatting without modifying files
```

Testing structure:
- `tests/pytest/unit/`: Core interface unit tests
- `tests/pytest/{mlir,tvm}/`: Backend-specific tests
- `tests/filecheck/`: Lit+FileCheck functional tests for code generation

Global test commands:
```bash
make test                # Run minimal unit tests
make check               # Run ALL acceptance tests (required for contributions)
make check-pytest        # Run pytest suite only
make check-lit           # Run LIT tests for LLVM IR target
make check-lit-c         # Run LIT tests for C target
pytest tests/pytest/unit # Run specific test directory
```

Running individual tests:
```bash
# Single pytest file
pytest tests/pytest/unit/test_specific.py -v

# Single lit test
lit -v tests/filecheck/backends/specific_test.py

# C target for lit tests
XTC_MLIR_TARGET=c lit -v tests/filecheck/backends/specific_test.py
```

## Architecture

### Core Abstractions (src/xtc/itf/)

Abstract interfaces defining the compilation pipeline:
- `data/` - Tensor, DataType, ShapeType
- `operator/` - Linear algebra operator interface
- `graph/` - Graph, Node, Operation abstractions
- `back/` - Backend interface
- `schd/` - Scheduler and Schedule abstractions
- `comp/` - Compiler interface
- `exec/` - Executor and Evaluator interfaces
- `search/` - Search space exploration interface

### Backends (src/xtc/backends/)

Exposed backends:
- `mlir/` - MLIR backend using linalg + transform dialects
- `tvm/` - TVM backend using Tensor IR + Schedule APIs

XTC also supports multiple MLIR Targets for the code generation:
  - llvmir (default)
  - c

To force the use of a specific target, you can set the env variable XTC_MLIR_TARGET=<mlir-target>.

### Compilation Pipeline

1. User defines Graph with Tensors and Operators
2. Backend created from Graph
3. Scheduler applies transformations and produces Schedule
4. Compiler generates executable Module
5. Executor/Evaluator runs and measures performance

### CLI Tools (src/xtc/cli/)

- `mlir-loop` - High-level scheduling for MLIR linalg operators
- `mlir-backend` - MLIR backend wrapper
- `loop-explore` - Autotuning and space exploration
- `loop-display` - Visualization of exploration results

## AI assistants

To create the `CLAUDE.md` file required to use Claude Code: `make claude`
