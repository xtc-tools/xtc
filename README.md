![logo.png](logo.png)

# XTC

## Links

Refer to documentation at https://xtc-tools.github.io/xtc

Refer to tutorials [here](docs/tutorials) and to additionnal developers documentation [here](docs/develop).

## Overview

XTC is a domain-specific dataflow graph compiler for linear algebra operations. It provides:
- **Operational DSL**: Define computation graphs with tensors and operators
- **Scheduling DSL**: High-level transformations (tiling, parallelization, vectorization, etc.)
- **Multiple backends**: MLIR (linalg + transform), TVM (Tensor IR), JIR (INRIA internal)
- **Autotuning**: Definition and exploration of the optimization space

## Build & Development

### Installation

If needed, install uv following the instructions [here](https://docs.astral.sh/uv/).

Debian-like Linux distributions (Python: 3.10 to 3.14 inclusive):
```bash
sudo apt install python3 python3-dev build-essential libomp5 binutils binutils-aarch64-linux-gnu binutils-x86-64-linux-gnu
sudo apt install libpfm4-dev # Optional: interface to Linux perf counters
sudo sysctl kernel.perf_event_paranoid=1 # Optional: give access to hardware counters
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e '.[dev]'
uv pip install -r mlir_requirements.txt
uv pip install -r tvm_requirements.txt
make test
```

MacOs M1+ (Python: 3.10 to 3.12 inclusive, wheels only available for macOS Sequoia):
```bash
brew install libomp
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e '.[dev]'
uv pip install -r macos_mlir_requirements.txt
uv pip install -r macos_tvm_requirements.txt
make test
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
  - nvgpu

To force the use of a specific target, you can set the env variable `XTC_MLIR_TARGET=<mlir-target>`.

The MLIR backend can be extended using the SDist extension, which provides distribution primitives.
To install SDist, follow the instructions in 
[docs/develop/optional_backends.md](docs/develop/optional_backends.md), in the "MLIR development version" section.

Note that the nvgpu target requires a recent version of Cuda (tested with Cuda 13.0).
By default, it tries to find Cuda at */usr/local/cuda*, but it can be overridden with the CUDA_INSTALL_DIR env variable.
The performance counters can be accessed is the GPU has a compute capability >=7.5.

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
