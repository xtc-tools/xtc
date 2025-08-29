![logo.png](logo.png)

# Overview

XTC is a domain specific dataflow graph compiler featuring operational DSL, scheduling DSL, multiple backends and autotuning.

Refer to documentation at https://corse.gitlabpages.inria.fr/XTC

Refer to installable python packages at: https://gitlab.inria.fr/corse/xtc/-/packages

Roadmap:
+ Allow tensor-level specifications
+ Implement graph-level transformations (fusion, etc.).
+ Implement more node-level transformations (padding, packing, etc.).
+ Integrate with an ML front-end.

# XTC Framework

## Installation instructions

Ensure installation of minimal required dependencies on the distribution:

    sudo apt install python3 python3-dev build-essential libomp5 binutils binutils-aarch64-linux-gnu binutils-x86-64-linux-gnu
    sudo apt install libpfm4-dev  # Optionally if using PMU counters on CPU for evaluation

Setup a virtual python environment with python >= 3.10,
and install base requirements, for instance:

    python3 -m venv .venv
    source .venv/bin/activate

Install the package for development/testing with:

    pip install -e .[dev]


Then install the MLIR requirements and optionally TVM and JIT backend requirements
as described below.

Note: in order to use PMU counters on CPU, install `libpfm4-dev` as described above and
configure your system to access counters with: `sudo sysctl kernel.perf_event_paranoid=1`

### MLIR Backend Requirements

For the MLIR backend, install the python packages for MLIR dependencies
(maintained at https://gitlab.inria.fr/CORSE/mlir-bindings-wheels and https://gitlab.inria.fr/CORSE/xtc-mlir-bindings-wheels):

    pip install -r mlir_requirements.txt


Optionally, one can build the MLIR project as follow.

Ensure revision is earlier then the one specified iin `mlir-requirements.txt`.

Then execute, for instance:

    git clone git@github.com:llvm/llvm-project.git
    cd llvm-project
    git checkout v19.1.7

Compile MLIR/CLANG and the MLIR python bindings, for instance:

    sudo apt install pybind11-dev libxml2-dev
    pip install -r mlir/python/requirements.txt
    mkdir build
    cd build
    cmake -DLLVM_ENABLE_PROJECTS="clang;mlir"
    -DCMAKE_INSTALL_PREFIX=$HOME/install/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_ASM_COMPILER=clang \
    -DCMAKE_SHARED_LINKER_FLAGS="-fno-omit-frame-pointer" \
    -DCMAKE_EXE_LINKER_FLAGS="-fno-omit-frame-pointer" \
    ../llvm
    make -j4
    make install

Add the tools to your `PATH` and the python bindings to your `PYTHONPATH`:

    export PATH=$HOME/install/llvm
    export PYTHONPATH=$PYTHONPATH:$HOME/install/llvm/python_packages/mlir_core

Some features of XTC also require an out-of-tree project named XTC-MLIR.
It is installed automatically using the mlir_requirements.txt file.
For manual building and installation, please follow the README at https://gitlab.inria.fr/CORSE/xtc-mlir.
Note: The prebuilt XTC-MLIR package comes with its own version of the libLLVM.so

#### MLIR Targets

XTC supports multiple MLIR Targets for the code generation:
  - llvmir (default)
  - c

To force the use of a specific target, you can set the env variable XTC_MLIR_TARGET=<mlir-target>.

### TVM backend requirements

In order to use the tvm backend, install the python packages for TVM dependencies
(maintained at https://gitlab.inria.fr/CORSE/tvm-wheels):

    pip install -r tvm_requirements.txt


Note that, if compiling TVM v0.16+ from source instead of using these packages,
one should first apply the patch `patches/tvm-Bugfix-TIR-Fix-race-on-ComputationCache.patch`
which fix a race condition in TVM. This patch is included in the python package above.

### JIR backend requirements

In order to use the jir backend, install the python packages for JIR dependencies:

    pip install -r jir_requirements.txt

Note that JIR is currently an Inria internal project, in order to get access to the package
repository, put the following in yout `~/.netrc` file:

    machine gitlab.inria.fr login <gitlab_login> password <gitlab_token>

In order to get a gitlab token, get to https://gitlab.inria.fr/-/user_settings/personal_access_tokens
and add a new token with the `api` scope.

Optionally, one can use an alternative JIR build, refer to
https://gitlab.inria.fr/CORSE/jir for building JIR and dependent tools from sources.

## Test Installation

Validate installation by launching lit tests and pytest tests:

    lit tests/filecheck
    pytest tests


## Exploration

Use exploration script, for instance random 100 points for a simple matmul tiling strategy (3D tiling):

    loop-explore --debug --search random --trials 100 --output results.random.csv

Use exploration script, for instance on input data generated on some tvm search (3D tiling + permutations), 2054 points here:

    time -p loop-explore --debug --dims 256 256 512 --strategy tile4d --search data --data data/tvm_results.mm06.csv --output data/results.mm06-tile4d.csv
    ...
    2054/2054 [55:54,  1.63s/it]
    real 3356 secs

Use exhaustive search on a tiling strategy limited to tile4d + only vectorized tilings (450 points):

    # TVM backend
    time -p loop-explore --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backends tvm --output results.mm06-tile4dv-tvm.csv
    450/450 [24:04,  3.21s/it]
    real 1444.50

    # MLIR backend
    time -p loop-explore --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backends mlir --output results.mm06-tile4dv-mlir.csv
    450/450 [22:34<00:00,  3.01s/it]
    real 1355.98

    # JIR backend
    time -p loop-explore --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backends jir --output results.mm06-tile4dv-jir.csv
    450/450 [22:30<00:00,  3.00s/it]
    real 1352.37

Test a single tiling:

    # Dumps and execute MLIR tiling
    loop-explore --dump --debug --dims 256 256 512 --strategy tile4d --test 4 64 8 4
    ...
    INFO:__main__:Schedule: [4, 64, 8, 4]: time: 1.89 msecs, peak perf: 26.38%
    # Execute on all backends
    loop-explore --backends tvm mlir jir --debug --dims 256 256 512 --strategy tile4d --test 4 64 8 4
    ...
    INFO:__main__:Schedule: [4, 64, 8, 4]: time: 0.61 msecs, peak perf: 82.08%

## Display

Result of exploration and display in `data/results.mm06-tile7d-all.svg` were generated with:

    loop-explore --debug --dims 256 256 512 --backends tvm mlir jir --validate --strategy tile7d  --search random --trials 1000 --output data/results.mm06-tile7d-all.csv
    loop-display --title 'Tile7D tiling strategy on 1000 samples for 256x256x512 matmul' data/results.mm06-tile7d-all.csv:tvm:X:peak:tvm data/results.mm06-tile7d-all.csv:mlir:X:peak:mlir data/results.mm06-tile7d-all.csv:jir:X:peak:jir --output data/results.mm06-tile7d-all.svg

Comparative performance distribution on tile4dv tilings in `data/mlir_results.mm06-tile4dv-all.svg` were generated with:

    loop-explore --debug --dims 256 256 512 --backends tvm mlir jir --validate --strategy tile4dv  --search exhaustive --output data/results.mm06-tile4dv-all.csv
    loop-display --title "Tile4DV tiling strategy exhaustive for 256x256x512 vectorized matmul" data/results.mm06-tile4dv-all.csv:tvm:X:peak:tvm data/results.mm06-tile4dv-all.csv:mlir:X:peak:mlir data/results.mm06-tile4dv-all.csv:jir:X:peak:jir --output data/results.mm06-tile4dv-all.svg


# mlir-loop: high-level scheduling specifications in MLIR

The ```mlir-loop``` tool provides a high-level syntax for
controlling the scheduling of MLIR linear algebra (```linalg```)
operators. For now, it only applies at ```memref``` level
(not ```tensor```) and supports the following transformations:
+ Tiling
+ Loop interchange
+ Vectorization
+ Unrolling

See the code below. For the simplicity of the example, it is a
single operator function, but the tool accepts multiple operator
functions.

```
func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I" = {"parallelize"},
          "J",
            "K",
              "I#1" = {"unroll"},
                "K#8"= {"unroll"},
                  "J#64" = {"vectorize"}
      }
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
```

Under the hood, this declarative "loop" attributes dialect is
translated into the corresponding MLIR ```transform``` dialect
command sequence. Thus, ```mlir-loop``` transformations fully reuse
those implemented in ```mlir-opt```.

