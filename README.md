# Xdsl Transform

## Installation instructions

The previous version relied on XDSL (file Implementer.py). The current
one relies on upstream MLIR bindings.

### Install the right version of MLIR

Install dependencies (Debian):
```
sudo apt install pybind11-dev
```

Choose the commit for which xdsl is made (patches are in the directory
xdsl-transform/patches):
```
git clone git@github.com:llvm/llvm-project.git
cd llvm-project
git checkout 98e674c9f16d677d95c67bc130e267fae331e43c
git apply /path/to/each/patch
```

Compile MLIR/CLANG and the MLIR python bindings:
```
pip install -r mlir/python/requirements.txt
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS="clang;mlir" -DLLVM_BUILD_EXAMPLES=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/bin/llvm-xdsl -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_ASM_COMPILER=clang \
  -DCMAKE_SHARED_LINKER_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fno-omit-frame-pointer" ../llvm
make -j4
make install
```

Add the Python bindings to your PYTHONPATH:
```
export PYTHONPATH=$PYTHONPATH:$HOME/bin/llvm-xdsl/python_packages/mlir_core
```

## Install requirements

```
pip install -r requirements.txt
```

### TVM backend requirements

For using tvm backend, install TVM and do (on pinocchio use for instance TVM installed in `/opt/local/tvm/tvm-v0.16.0.rc0/`):
```
pip install -r tvm_requirements.txt
export PYTHONPATH=$PYTHONPATH:/path_to_tvm/python
```

Note that if compiling TVM version v0.16 from source, one should first
apply the patch `patches/tvm-Bugfix-TIR-Fix-race-on-ComputationCache.patch`
which fix a race condition in TVM.

### JIR backend requirements

For using jir backend, install JIR (ref to https://gitlab.inria/fr/jprotopo/jir.git) and set python path:
```
export PYTHONPATH=$PYTHONPATH:/path_to_jir
```

## Use it

+ Example in test.py
+ Just works with matmul (for now)

## Exploration

Use exploration script, for instance random 100 points for a simple matmul tiling strategy (3D tiling):

    ./explore.py --debug --search random --trials 100 --output results.random.csv

Use exploration script, for instance on input data generated on some tvm search (3D tiling + permutations), 2054 points here:

    time -p ./explore.py --debug --dims 256 256 512 --strategy tile4d --search data --data data/tvm_results.mm06.csv --output data/results.mm06-tile4d.csv
    ...
    2054/2054 [55:54,  1.63s/it]
    real 3356 secs

Use exhaustive search on a tiling strategy limited to tile4d + only vectorized tilings (450 points):

    # TVM backend
    time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backends tvm --output results.mm06-tile4dv-tvm.csv
    450/450 [24:04,  3.21s/it]
    real 1444.50

    # MLIR backend
    time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backends mlir --output results.mm06-tile4dv-mlir.csv
    450/450 [22:34<00:00,  3.01s/it]
    real 1355.98

    # JIR backend
    time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backends jir --output results.mm06-tile4dv-jir.csv
    450/450 [22:30<00:00,  3.00s/it]
    real 1352.37

Test a single tiling:

    # Dumps and execute MLIR tiling
    ./explore.py --dump --debug --dims 256 256 512 --strategy tile4d --test 4 64 8 4
    ...
    INFO:__main__:Schedule: [4, 64, 8, 4]: time: 1.89 msecs, peak perf: 26.38%
    # Execute on all backends
    ./explore.py --backends tvm mlir jir --debug --dims 256 256 512 --strategy tile4d --test 4 64 8 4
    ...
    INFO:__main__:Schedule: [4, 64, 8, 4]: time: 0.61 msecs, peak perf: 82.08%

## Display

Result of exploration and display in `data/results.mm06-tile7d-all.svg` were generated with:

    ./explore.py --debug --dims 256 256 512 --backends tvm mlir jir --validate --strategy tile7d  --search random --trials 1000 --output data/results.mm06-tile7d-all.csv
    ./display-results.py --title 'Tile7D tiling strategy on 1000 samples for 256x256x512 matmul' data/results.mm06-tile7d-all.csv:tvm:X:peak:tvm data/results.mm06-tile7d-all.csv:mlir:X:peak:mlir data/results.mm06-tile7d-all.csv:jir:X:peak:jir --output data/results.mm06-tile7d-all.svg
    
Comparative performance distribution on tile4dv tilings in `data/mlir_results.mm06-tile4dv-all.svg` were generated with:

    ./explore.py --debug --dims 256 256 512 --backends tvm mlir jir --validate --strategy tile4dv  --search exhaustive --output data/results.mm06-tile4dv-all.csv
    ./display-results.py --title "Tile4DV tiling strategy exhaustive for 256x256x512 vectorized matmul" data/results.mm06-tile4dv-all.csv:tvm:X:peak:tvm data/results.mm06-tile4dv-all.csv:mlir:X:peak:mlir data/results.mm06-tile4dv-all.csv:jir:X:peak:jir --output data/results.mm06-tile4dv-all.svg


## Notes

### Compile & link

After compilation using Python:
```
~/bin/llvm-xdsl/bin/clang /tmp/dump.bc ~/bin/llvm-xdsl/lib/libmlir_c_runner_utils.so -o /tmp/test
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hpompougnac/bin/llvm-xdsl/lib/
/tmp/test
```

### Scalar FMAs

The option ```--math-uplift-to-fma``` combines arith operations into ```math.fma``` if the flag ```fast``` is set, but how to generate the latter ?
