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

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Define Your First Graph with XTC

    In XTC, computations are represented as **dataflow graphs**. A graph consists of:
    - **Tensors**: Multi-dimensional arrays that hold data
    - **Operators**: Operations that transform tensors (e.g., matrix multiplication, convolution)
    - **Nodes**: Individual operations within the graph

    Let's start by creating a simple matrix multiplication graph. Matrix multiplication (matmul) computes $C = A \times B$ where:
    - $A$ is an $I \times K$ matrix
    - $B$ is a $K \times J$ matrix
    - $C$ is the resulting $I \times J$ matrix

    The code below demonstrates how to:
    1. Define input tensors with their shapes and data types
    2. Create a graph context and add the matmul operation
    3. Serialize the resulting graph.

    **Try modifying the dimensions or data type to see how the graph changes!**
    """)
    return

@app.cell
def _():
    def_editor = mo.ui.code_editor(
        value=
"""import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")
with O.graph(name="matmul") as gb:
   O.matmul(a, b, name="C")
graph = gb.graph
print(graph)
... # Compilation comes soon""",
        language="python",
        label=""
    )
    def_editor
    return def_editor,

@app.cell
def __(def_editor):
    
    _old_stdout = sys.stdout
    sys.stdout = _captured_output = StringIO()
    
    try:
        exec(def_editor.value)
        _output = _captured_output.getvalue()
    except Exception as e:
        _output = f"Error:\n{type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = _old_stdout
    
    mo.md(f"**Output:**\n```\n{_output}\n```")
    return _captured_output, _old_stdout, _output

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Practice:**

    Try modifying the dimensions or data type to see how the graph changes!
    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Compile, Evaluate and Schedule

    Now that we have a graph, we need to:
    1. **Create a Backend**: The backend handles code generation for a specific target (MLIR, TVM)
    2. **Get a Scheduler**: The scheduler allows us to apply transformations to optimize the code
    3. **Generate a Schedule**: Apply (or not) transformations and get a schedule object
    4. **Compile**: Generate executable code from the schedule
    5. **Evaluate**: Run the compiled code and measure performance

    The code below shows the complete workflow for compiling and evaluating a simple matmul operation without any optimizations.

    **Performance is measured as a percentage of the peak perf (the theoretical number of flops/seconds of the CPU), the baseline to compare against when we add optimizations!**
    """)
    return

@app.cell
def _():
    comp_editor = mo.ui.code_editor(
        value=
"""import xtc.graphs.xtc.op as O
import xtc.runtimes.host.runtime as rt
from xtc.backends.mlir import Backend  # Choose between mlir & tvm here

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")
with O.graph(name="matmul") as gb:
   O.matmul(a, b, name="C")
graph = gb.graph
impl = Backend(graph)

# Schedule
sch = impl.get_scheduler()
sch.set_dims(['i','j','k'])
...                                    # Schedule the operator here
sched = sch.schedule()

# Compile
comp = impl.get_compiler(
   shared_lib=True,
   dump_file="matmul_mlir",
   print_source_ir=False,              # Serialize the generated code here...
   print_transformed_ir=False,         # And/or here...
   print_lowered_ir=False,             # And/or here...
   print_assembly=False,               # And/or here !
)
code = StringIO()
with redirect_stderr(code):
   module = comp.compile(sched)

# Evaluate the generated code
evaluator = module.get_evaluator()
results, _, _ = evaluator.evaluate()
peak_flops = rt.evaluate_flops(dtype)
time_flops = (I*J*K) / min(results)
perf = time_flops / peak_flops * 100

print("perf: {:.2f}%".format(perf))
print(f"{code.getvalue()}")""",
        language="python",
        label=""
    )
    comp_editor
    return comp_editor,

@app.cell
def __(comp_editor):
    
    _old_stdout = sys.stdout
    sys.stdout = _captured_output = StringIO()
    
    try:
        exec(comp_editor.value)
        _output = _captured_output.getvalue()
    except Exception as e:
        _output = f"Error:\n{type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = _old_stdout
    
    mo.md(f"**Output:**\n```\n{_output}\n```")
    return _captured_output, _old_stdout, _output

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Practice:**
    1. *Serialize the IR.* The serialization of the generated assembly as well as the different IR levels is disabled for now. You should try enabling these different levels to see what the executed operator really looks like! If you're more comfortable with TVM than with MLIR, or simply curious to explore TVM, you can also replace MLIR with TVM as a backend.
    2. *Inspect the IR.* In your opinion, why is the performance so poor?
    3. *Transform the code.* You can start transforming the code using the primitives exposed by the scheduler:
       - `sch.tile("j", {"j1": 1})` creates an (useless) tile of size 1 along `j`.
       - `sch.vectorize(["j1"])` vectorizes the computation along the loop `j1`.
       - `sch.unroll({"j1":1})` unrolls the loop `j1` with an unroll factor of 1 (useless too).
       - `sch.interchange(["i", "k", "j", "j1"])` reorders the loops.
    4. Try to maximize the performance!
    """)
    return

if __name__ == "__main__":
    app.run()
