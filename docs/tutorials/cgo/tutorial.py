import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __():
    import sys
    from io import StringIO
    from contextlib import redirect_stderr
    import marimo as mo

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # XTC Tutorial - CGO 2026

    Welcome to the XTC tutorial! This interactive notebook will guide you through the fundamentals of performance engineering using XTC, a research platform for optimizing AI operators.

    By the end of this tutorial, you will understand how to:
    - Define computational graphs with XTC
    - Compile and evaluate operator performance
    - Apply high-level scheduling transformations using Descript
    - Explore the scheduling space to find optimal configurations
    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Installation

    Before starting, ensure you have XTC properly installed on your system.

    ### System Requirements

    XTC supports the following platforms:
    - **Linux x86_64** (recommended)
    - **MacOS M1+**

    ### System Dependencies

    Install the required system packages:

    **For Debian/Ubuntu:**
    ```bash
    sudo apt install python3 python3-dev build-essential libomp5 binutils binutils-aarch64-linux-gnu binutils-x86-64-linux-gnu
    sudo apt install libpfm4-dev  # Optional: for PMU counters on CPU
    ```

    **For Fedora:**
    ```bash
    sudo dnf install python3 python3-devel libomp binutils binutils-aarch64-linux-gnu binutils-x86_64-linux-gnu
    sudo dnf group install c-development development-tools
    sudo dnf install libpfm-devel
    ```

    ### Python Environment Setup

    Create and activate a virtual environment (Python version must be >=3.10 and <3.13):

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

    ### Installing XTC

    Install the XTC package for development/testing:

    ```bash
    pip3 install -e '.[dev]'
    ```

    ### Backend Requirements

    XTC supports multiple backends. Install the MLIR and and TVM ones:

    **MLIR Backend** (recommended for this tutorial):
    ```bash
    pip3 install -r mlir_requirements.txt
    ```

    **TVM Backend**:
    ```bash
    pip3 install -r tvm_requirements.txt
    ```

    ### PMU Counters (Optional)

    To use hardware performance counters for detailed profiling (on Linux), configure your system:

    ```bash
    sudo sysctl kernel.perf_event_paranoid=1
    ```

    ### Verify Installation

    Run the minimal test suite to verify your installation:

    ```bash
    make test
    ```

    If all tests pass, you're ready to proceed with the tutorial!
    """)
    return

if __name__ == "__main__":
    app.run()
