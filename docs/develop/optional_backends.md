# Optional backends

## MLIR development version

Optionally, for using own MLIR development version, build the MLIR project as follow.

Ensure revision is compatible with the one specified iin `mlir-requirements.txt`.

Then execute, for instance:

    git clone git@github.com:llvm/llvm-project.git
    cd llvm-project
    git checkout v19.1.7

Compile MLIR/CLANG and the MLIR python bindings, for instance:

    sudo apt install pybind11-dev libxml2-dev
    pip install -r mlir/python/requirements.txt
    mkdir build
    cd build
    cmake -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DCMAKE_INSTALL_PREFIX=$HOME/install/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    ../llvm
    make -j4
    make install

Add the tools to your `PATH` and the python bindings to your `PYTHONPATH`:

    export PATH=$HOME/install/llvm/bin:$PATH
    export PYTHONPATH=$HOME/install/llvm/python_packages/mlir_core:$PYTHONPATH

Some features of XTC also require an out-of-tree project named XTC-MLIR.
It is installed automatically using the mlir_requirements.txt file.
For manual building and installation, please follow the README at https://gitlab.inria.fr/CORSE/xtc-mlir.
Note: The prebuilt XTC-MLIR package comes with its own version of the libLLVM.so

#### MLIR SDist extension

The SDist extension provides distribution primitives. To install SDist:

    pip uninstall mlir-python-bindings
    pip install -r sdist_requirements.txt

Note that SDist is currently an Inria internal project (cf JIR section on how to access).

## TVM development version

Note that, if compiling TVM v0.16+ from source instead of using our wheels,
one should first apply the patch `patches/tvm-Bugfix-TIR-Fix-race-on-ComputationCache.patch` which fix a race condition in TVM.

## JIR backend requirements

In order to use the jir backend, install the python packages for JIR dependencies:

    pip3 install -r jir_requirements.txt

Note that JIR is currently an Inria internal project, in order to get access to the package
repository, put the following in your `~/.netrc` file:

    machine gitlab.inria.fr login <gitlab_login> password <gitlab_token>

In order to get a gitlab token, get to https://gitlab.inria.fr/-/user_settings/personal_access_tokens
and add a new token with the `api` scope.

Optionally, one can use an alternative JIR build, refer to
https://gitlab.inria.fr/CORSE/jir for building JIR and dependent tools from sources.
