# Xdsl Transform

## Installation instructions

### Install xdsl

```
git clone git@github.com:xdslproject/xdsl.git
cd xdsl
pip install -e .
```

### Install the right version of MLIR

```
git clone git@github.com:llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/bin/llvm-xdsl -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_ASM_COMPILER=clang \
  -DCMAKE_SHARED_LINKER_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fno-omit-frame-pointer" ../llvm
make -j4
make install
```

## Install requirements

```
pip install -r requirements.txt
```


## Use it

+ Example in test.py
+ Just works with matmul (for now)

## Exploration

Use exploration script, for instance random 100 points for a simple matmul tiling strategy (3D tiling):

    ./explore.py --debug --search random --trials 100 --output results.random.csv

Use exploration script, for instance on input data generated on some tvm search (3D tiling + permutations):

    ./explore.py --debug --dims 256 256 512 --strategy tile4d --search data --data data/tvm_results.mm06.csv --output results.mm06.csv

## Issues

### Scalar FMAs

The option ```--math-uplift-to-fma``` combines arith operations into ```math.fma``` if the flag ```fast``` is set, but how to generate the latter ?
