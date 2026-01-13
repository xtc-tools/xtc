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

    By the end of this notebook, you will understand how to:
    - Define computational graphs with XTC
    - Compile and evaluate operator performance
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
    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Practice.** The code below is editable and the output (i.e. the serialized graph) is dynamically computed.

    1. Try modifying the dimensions or data type (float32, float64) to see how the graph changes!
    2. Insteaf of using the pre-loaded function `matmul_graph`, craft a graph yourself, and add a ReLU activation after the matrix multiplication. Hint: `O.matmul()` returns a tensor that can be passed to another operator. The signature of ReLu is: `O.relu(inp, name)`.
    """)
    return

@app.cell
def _():
    def_editor = mo.ui.code_editor(
        value=
"""import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   \"\"\"Create a graph computing C = A @ B.

   Args:
      I: Number of rows in A and C.
      J: Number of columns in B and C.
      K: Shared dimension (columns of A, rows of B).
      dtype: Data type of the tensors (e.g., "float32", "float64").

   Returns:
      An XTCGraph containing the matmul operation.
   \"\"\"
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         C = O.matmul(a, b, name="C")
   return gb.graph

I, J, K, dtype = 4, 32, 512, "float32"
graph = matmul_graph(I=I,J=J,K=K,dtype=dtype)

print(graph)""",
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
    ## 3. Compile and Evaluate

    Now that we have a graph, let's compile it and measure its baseline performance — without any optimization.

    The compilation pipeline in XTC follows these steps:
    1. **Create a Backend**: The backend handles code generation for a specific target (MLIR, TVM)
    2. **Get a Scheduler**: Even without optimizations, we need a scheduler to define the loop structure
    3. **Compile**: Generate executable code
    4. **Evaluate**: Run the compiled code and measure performance

    ```python
    from xtc.backends.mlir import Backend

    # 1. Create a backend
    backend = Backend(graph)

    # 2. Get a scheduler (no transformations)
    sch = backend.get_scheduler()
    sch.set_dims(['i', 'j', 'k'])
    sched = sch.schedule()

    # 3. Compile
    comp = backend.get_compiler(shared_lib=True)
    module = comp.compile(sched)

    # 4. Evaluate
    evaluator = module.get_evaluator()
    results, _, _ = evaluator.evaluate()
    ```
    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Practice.** The code below compiles the matmul graph without any optimization. Use the radio buttons to select the backend (MLIR or TVM) and which IR to display. Performance is measured as a percentage of peak (the theoretical FLOP/s of the CPU).

    1. *Inspect the generated code.* Look at the Source IR, Transformed IR, Lowered IR, and Assembly. What do you notice?
    2. *Observe the performance.* In your opinion, why is it so poor? This is the baseline to compare against when we add optimizations in the next section!
    """)
    return

@app.cell
def _():
    compile_editor = mo.ui.code_editor(
        value=
'''# === Problem Definition ===
I, J, K, dtype = 4, 32, 512, "float32"
''',
        language="python",
        label=""
    )
    compile_output_radio = mo.ui.radio(
        options=["Source IR", "Transformed IR", "Lowered IR", "Assembly"],
        value="Assembly",
        label="Output options:"
    )
    compile_backend_radio = mo.ui.radio(
        options=["MLIR", "TVM"],
        value="MLIR",
        label="Backend:"
    )
    compile_editor
    return compile_editor, compile_output_radio, compile_backend_radio

@app.cell
def _(compile_editor, compile_output_radio, compile_backend_radio):
    import traceback as _traceback
    from io import StringIO as _StringIO
    from contextlib import redirect_stderr as _redirect_stderr, redirect_stdout as _redirect_stdout
    import xtc.graphs.xtc.op as _O
    from xtc.graphs.xtc.graph import XTCGraph as _XTCGraph
    import xtc.runtimes.host.runtime as _rt

    # Import backend based on radio selection
    if compile_backend_radio.value == "MLIR":
        from xtc.backends.mlir import Backend as _Backend
    else:
        from xtc.backends.tvm import Backend as _Backend

    # Define helper functions
    def _matmul_graph(I: int, J: int, K: int, dtype: str) -> _XTCGraph:
        a = _O.tensor((I, K), dtype, name="A")
        b = _O.tensor((K, J), dtype, name="B")
        with _O.graph(name="matmul") as gb:
            _O.matmul(a, b, name="C")
        return gb.graph

    # Parse configuration from editor
    _namespace = {}
    try:
        exec(compile_editor.value, _namespace)
    except Exception as e:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{_traceback.format_exc()}\n```"))

    _I = _namespace.get("I", 4)
    _J = _namespace.get("J", 32)
    _K = _namespace.get("K", 512)
    _dtype = _namespace.get("dtype", "float32")
    _print_source_ir = compile_output_radio.value == "Source IR"
    _print_transformed_ir = compile_output_radio.value == "Transformed IR"
    _print_lowered_ir = compile_output_radio.value == "Lowered IR"
    _print_assembly = compile_output_radio.value == "Assembly"

    # Check for unsupported option
    if _print_lowered_ir and compile_backend_radio.value == "TVM":
        mo.stop(True, mo.md("**Note:** The TVM backend does not support 'Lowered IR'. Please select another output option."))

    # Create graph and compile without optimizations
    _graph = _matmul_graph(_I, _J, _K, _dtype)
    _backend = _Backend(_graph)
    _scheduler = _backend.get_scheduler()
    _scheduler.set_dims(['i', 'j', 'k'])
    _sched = _scheduler.schedule()

    # Build compiler options
    _compiler_opts = {
        "dump_file": "test_mlir",
        "shared_lib": True,
        "print_source_ir": _print_source_ir,
        "print_transformed_ir": _print_transformed_ir,
        "print_assembly": _print_assembly
    }
    if compile_backend_radio.value == "MLIR":
        _compiler_opts["print_lowered_ir"] = _print_lowered_ir

    _comp = _backend.get_compiler(**_compiler_opts)
    _code_output_stderr = _StringIO()
    _code_output_stdout = _StringIO()
    with _redirect_stderr(_code_output_stderr), _redirect_stdout(_code_output_stdout):
        _module = _comp.compile(_sched)

    _code_output = _code_output_stderr.getvalue() + _code_output_stdout.getvalue()

    # Evaluate performance
    _peak_flops = _rt.evaluate_flops(_dtype)
    _evaluator = _module.get_evaluator()
    _results, _, _ = _evaluator.evaluate()
    _result = min(_results)
    _time_flops = (_I * _J * _K) / _result
    _perf = _time_flops / _peak_flops * 100

    # Build output
    _perf_display = mo.md(f"**Performance:** {_perf:.2f}% of peak")
    _radio_row = mo.hstack([compile_backend_radio, compile_output_radio], justify="start", gap=4)
    _code_content = mo.md(f"```asm\n{_code_output}\n```") if _code_output else mo.md("*No IR output.*")
    _code_accordion = mo.accordion({"Generated Code": _code_content})

    mo.vstack([_perf_display, _radio_row, _code_accordion])
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Optimize with Scheduling

    The baseline compilation produces correct but unoptimized code. To improve performance, XTC exposes a **scheduler** with imperative primitives that transform the loop nest:

    - `sch.tile("j", {"j1": 16})` creates a tile of size 16 along `j`. Tiling breaks a large loop into smaller chunks. This helps in two ways: (1) smaller chunks fit better in CPU cache, reducing slow memory accesses due to cache misses, and (2) choosing the right tile size (e.g., 8 for AVX with float32) allows the inner loop to map directly onto vector registers.
    - `sch.split("j", {"j1": 16})` splits `j` into two segments at position 16, creating `j` (iterations 0-15) and `j1` (iterations 16+). Splitting is useful for applying different transformations to different parts of a loop (e.g., vectorize the main part, handle the remainder separately).
    - `sch.vectorize(["j1"])` vectorizes the computation along the loop `j1`. Vectorization uses SIMD instructions to process multiple elements in parallel, significantly increasing throughput on modern CPUs.
    - `sch.unroll({"j1":1})` unrolls the loop `j1` with an unroll factor of 1 (useless too). Unrolling reduces loop overhead (fewer branches) and exposes more instruction-level parallelism to the hardware.
    - `sch.interchange(["i", "k", "j", "j1"])` reorders the loops. Interchange improves memory access patterns by ensuring stride-1 access (contiguous memory) rather than strided access, maximizing cache efficiency.

    ```python
    from xtc.backends.mlir import Backend

    backend = Backend(graph)
    sch = backend.get_scheduler()

    # Apply transformations
    sch.set_dims(['i', 'j', 'k'])
    sch.tile("j", {"j1": 16})
    sch.vectorize(["j1"])
    sched = sch.schedule()

    # Compile and evaluate
    comp = backend.get_compiler(shared_lib=True)
    module = comp.compile(sched)
    evaluator = module.get_evaluator()
    results, _, _ = evaluator.evaluate()
    ```
    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Practice.** The code below lets you define a schedule using imperative primitives. Use the radio buttons to select the backend and what IR to display. Compare the performance and generated code with the unoptimized version from the previous section.

    1. *Transform the code.* Start with simple transformations and build up.
    2. *Inspect the generated code.* How does the Assembly or IR differ from the baseline?
    3. *Try to maximize the performance!*
    """)
    return

@app.cell
def _():
    sched_editor = mo.ui.code_editor(
        value=
'''# === Problem Definition ===
I, J, K, dtype = 4, 32, 512, "float32"

# === Schedule (imperative) ===
def schedule(sch):
   sch.set_dims(['i','j','k'])
   # Add transformations here, e.g.:
   # sch.tile("j", {"j1": 16})
   # sch.vectorize(["j1"])
   # sch.interchange(["i", "k", "j", "j1"])
''',
        language="python",
        label=""
    )
    sched_output_radio = mo.ui.radio(
        options=["Source IR", "Transformed IR", "Lowered IR", "Assembly"],
        value="Assembly",
        label="Output options:"
    )
    sched_backend_radio = mo.ui.radio(
        options=["MLIR", "TVM"],
        value="MLIR",
        label="Backend:"
    )
    sched_editor
    return sched_editor, sched_output_radio, sched_backend_radio

@app.cell
def _(sched_editor, sched_output_radio, sched_backend_radio):
    import traceback as _traceback
    from io import StringIO as _StringIO
    from contextlib import redirect_stderr as _redirect_stderr, redirect_stdout as _redirect_stdout
    import xtc.graphs.xtc.op as _O
    from xtc.graphs.xtc.graph import XTCGraph as _XTCGraph
    import xtc.runtimes.host.runtime as _rt

    # Import backend based on radio selection
    if sched_backend_radio.value == "MLIR":
        from xtc.backends.mlir import Backend as _Backend
    else:
        from xtc.backends.tvm import Backend as _Backend

    # Define helper functions
    def _matmul_graph(I: int, J: int, K: int, dtype: str) -> _XTCGraph:
        a = _O.tensor((I, K), dtype, name="A")
        b = _O.tensor((K, J), dtype, name="B")
        with _O.graph(name="matmul") as gb:
            _O.matmul(a, b, name="C")
        return gb.graph

    # Parse configuration from editor
    _namespace = {}
    try:
        exec(sched_editor.value, _namespace)
    except Exception as e:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{_traceback.format_exc()}\n```"))

    _I = _namespace.get("I", 4)
    _J = _namespace.get("J", 32)
    _K = _namespace.get("K", 512)
    _dtype = _namespace.get("dtype", "float32")
    _schedule_fn = _namespace.get("schedule")
    _print_source_ir = sched_output_radio.value == "Source IR"
    _print_transformed_ir = sched_output_radio.value == "Transformed IR"
    _print_lowered_ir = sched_output_radio.value == "Lowered IR"
    _print_assembly = sched_output_radio.value == "Assembly"

    # Check for unsupported option
    if _print_lowered_ir and sched_backend_radio.value == "TVM":
        mo.stop(True, mo.md("**Note:** The TVM backend does not support 'Lowered IR'. Please select another output option."))

    # Create graph and apply schedule
    _graph = _matmul_graph(_I, _J, _K, _dtype)
    _backend = _Backend(_graph)
    _scheduler = _backend.get_scheduler()

    # Apply user-defined schedule function
    if _schedule_fn is not None:
        try:
            _schedule_fn(_scheduler)
        except Exception as e:
            mo.stop(True, mo.md(f"**Schedule error:**\n```\n{_traceback.format_exc()}\n```"))

    _sched = _scheduler.schedule()

    # Build compiler options (TVM doesn't support print_lowered_ir)
    _compiler_opts = {
        "dump_file": "test_mlir",
        "shared_lib": True,
        "print_source_ir": _print_source_ir,
        "print_transformed_ir": _print_transformed_ir,
        "print_assembly": _print_assembly
    }
    if sched_backend_radio.value == "MLIR":
        _compiler_opts["print_lowered_ir"] = _print_lowered_ir

    _comp = _backend.get_compiler(**_compiler_opts)
    _code_output_stderr = _StringIO()
    _code_output_stdout = _StringIO()
    with _redirect_stderr(_code_output_stderr), _redirect_stdout(_code_output_stdout):
        _module = _comp.compile(_sched)

    _code_output = _code_output_stderr.getvalue() + _code_output_stdout.getvalue()

    # Evaluate performance
    _peak_flops = _rt.evaluate_flops(_dtype)
    _evaluator = _module.get_evaluator()
    _results, _, _ = _evaluator.evaluate()
    _result = min(_results)
    _time_flops = (_I * _J * _K) / _result
    _perf = _time_flops / _peak_flops * 100

    # Build output
    _perf_display = mo.md(f"**Performance:** {_perf:.2f}% of peak")
    _radio_row = mo.hstack([sched_backend_radio, sched_output_radio], justify="start", gap=4)
    _code_content = mo.md(f"```asm\n{_code_output}\n```") if _code_output else mo.md("*No IR output.*")
    _code_accordion = mo.accordion({"Generated Code": _code_content})

    mo.vstack([_perf_display, _radio_row, _code_accordion])
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Define a schedule declaratively

    XTC allows you to describe the target loop structure using a Python dictionary. Instead of manually specifying each transformation step, you declare the desired final loop structure, and XTC automatically infers the sequence of transformations needed to achieve it.

    For example, the following dictionary:
    ```python
    {"i": {}, "k": {}, "j": {}, "j#16": {"vectorize": True}}
    ```

    Describes this loop structure:
    ```
    for i in ...
      for k in ...
        for j in ...
          for j1 in range(16):  // vectorized
    ```

    The dictionary keys define the loop order (outer to inner), `j#16` creates a tile of size 16 on `j`, and the `{"vectorize": True}` attribute marks that inner loop for vectorization.

    The declarative API supports several key optimizations:

    | Transformation      | Syntax                          |
    |---------------------|---------------------------------|
    | **Tiling**          | `"axis#size"`                   |
    | **Vectorization**   | `{"vectorize": True}`           |
    | **Parallelization** | `{"parallelize": True}`         |
    | **Unrolling**       | `{"unroll": factor}`            |
    | **Interchange**     | Key ordering in the dictionnary |

    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Practice.** The code below lets you define a schedule specification and see the generated assembly. Try replicating the good-enough schedule you discovered in the previous section!

    - Modify `schedule_spec` to change the loop structure
    - Set `print_assembly = True` to see the generated code
    """)
    return

@app.cell
def _():
    descript_editor = mo.ui.code_editor(
        value=
'''# === Problem Definition ===
I, J, K, dtype = 4, 32, 512, "float32"

# === Schedule Specification ===
schedule_spec = {
   "i": {},
   "j": {},
   "k": {}
}''',
        language="python",
        label=""
    )
    output_radio = mo.ui.radio(
        options=["Source IR", "Transformed IR", "Lowered IR", "Assembly"],
        value="Assembly",
        label="Output options:"
    )
    backend_radio = mo.ui.radio(
        options=["MLIR", "TVM"],
        value="MLIR",
        label="Backend:"
    )
    descript_editor
    return descript_editor, output_radio, backend_radio

@app.cell
def _(descript_editor, output_radio, backend_radio):
    import traceback as _traceback
    from io import StringIO as _StringIO
    from contextlib import redirect_stderr as _redirect_stderr, redirect_stdout as _redirect_stdout
    import xtc.graphs.xtc.op as _O
    from xtc.graphs.xtc.graph import XTCGraph as _XTCGraph
    from xtc.schedules.descript import descript_scheduler as _descript_scheduler
    import xtc.runtimes.host.runtime as _rt

    # Import backend based on radio selection
    if backend_radio.value == "MLIR":
        from xtc.backends.mlir import Backend as _Backend
    else:
        from xtc.backends.tvm import Backend as _Backend

    # Define helper functions
    def _matmul_graph(I: int, J: int, K: int, dtype: str) -> _XTCGraph:
        a = _O.tensor((I, K), dtype, name="A")
        b = _O.tensor((K, J), dtype, name="B")
        with _O.graph(name="matmul") as gb:
            _O.matmul(a, b, name="C")
        return gb.graph

    # Parse configuration from editor
    _namespace = {"matmul_graph": _matmul_graph}
    try:
        exec(descript_editor.value, _namespace)
    except Exception as e:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{_traceback.format_exc()}\n```"))

    _I = _namespace.get("I", 4)
    _J = _namespace.get("J", 32)
    _K = _namespace.get("K", 512)
    _dtype = _namespace.get("dtype", "float32")
    _spec = _namespace.get("schedule_spec", {})
    _print_source_ir = output_radio.value == "Source IR"
    _print_transformed_ir = output_radio.value == "Transformed IR"
    _print_lowered_ir = output_radio.value == "Lowered IR"
    _print_assembly = output_radio.value == "Assembly"

    # Check for unsupported option
    if _print_lowered_ir and backend_radio.value == "TVM":
        mo.stop(True, mo.md("**Note:** The TVM backend does not support 'Lowered IR'. Please select another output option."))

    # Create graph and apply schedule
    _graph = _matmul_graph(_I, _J, _K, _dtype)
    _backend = _Backend(_graph)
    _scheduler = _backend.get_scheduler()
    _descript_scheduler(
        scheduler=_scheduler,
        node_name="C",
        abstract_axis=["i", "j", "k"],
        spec=_spec
    )
    _schedule = _scheduler.schedule()

    # Build compiler options (TVM doesn't support print_lowered_ir)
    _compiler_opts = {
        "dump_file": "test_mlir",
        "shared_lib": True,
        "print_source_ir": _print_source_ir,
        "print_transformed_ir": _print_transformed_ir,
        "print_assembly": _print_assembly
    }
    if backend_radio.value == "MLIR":
        _compiler_opts["print_lowered_ir"] = _print_lowered_ir

    _comp = _backend.get_compiler(**_compiler_opts)
    _code_output_stderr = _StringIO()
    _code_output_stdout = _StringIO()
    with _redirect_stderr(_code_output_stderr), _redirect_stdout(_code_output_stdout):
        _module = _comp.compile(_schedule)

    _code_output = _code_output_stderr.getvalue() + _code_output_stdout.getvalue()

    # Evaluate performance
    _peak_flops = _rt.evaluate_flops(_dtype)
    _evaluator = _module.get_evaluator()
    _results, _, _ = _evaluator.evaluate()
    _result = min(_results)
    _time_flops = (_I * _J * _K) / _result
    _perf = _time_flops / _peak_flops * 100

    # Build output
    _perf_display = mo.md(f"**Performance:** {_perf:.2f}% of peak")
    _radio_row = mo.hstack([backend_radio, output_radio], justify="start", gap=4)
    _code_content = mo.md(f"```asm\n{_code_output}\n```") if _code_output else mo.md("*No IR output.*")
    _code_accordion = mo.accordion({"Generated Code": _code_content})

    mo.vstack([_perf_display, _radio_row, _code_accordion])

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Experimenting with Multiple Schedules

    Performance engineering is often about exploring different optimization strategies. Different schedules can have dramatically different performance depending on:
    - **Problem size**: Small matrices may not benefit from parallelization overhead
    - **Hardware**: Cache sizes, vector width, and core count affect optimal tiling
    - **Data layout**: Memory access patterns influence cache efficiency

    In this section, we'll write a simple loop to try several schedule configurations and compare their performance.
    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Practice.** The code below defines several schedule configurations using the declarative `descript` scheduler. You can:

    - **Add/modify configurations**: Try different tile sizes, loop orderings, or optimization combinations
    - **Change the acquisition function**: Add caching, error handling, or custom metrics
    - **Modify the exploration loop**: Add early stopping or custom filtering

    The `explore()` function must `yield` tuples of `(index, total, config_name, performance)` for real-time progress display.
    """)
    return

@app.cell
def _():
    explore_schedules_code = mo.ui.code_editor(
        value=
'''from xtc.backends.mlir import Backend
import xtc.runtimes.host.runtime as rt

# === Problem Definition ===
I, J, K, dtype = 256, 256, 512, "float32"
graph = matmul_graph(I=I, J=J, K=K, dtype=dtype)
peak_flops = rt.evaluate_flops(dtype)

# === Schedule Configurations ===
configurations = [
   {
      "name": "Baseline (no opts)",
      "spec": {"i": {}, "j": {}, "k": {}},
   },
   {
      "name": "Interchange j and k",
      "spec": {"i": {}, "k": {}, "j": {}},
   },
   {
      "name": "Vectorize along k",
      "spec": {"i": {}, "j": {}, "k": {"vectorize": True}},
   },
   {
      "name": "Tile j (64)",
      "spec": {"i": {}, "j": {}, "j#64": {}, "k": {}},
   },
   {
      "name": "Tile and unroll j",
      "spec": {"i": {}, "j": {}, "j#64": {"unroll": True}, "k": {}},
   },
]

# === Acquisition Function ===
def acquire(config):
   """Evaluate a single configuration and return its performance."""
   module, _ = apply_schedule(
         graph=graph,
         backend_cls=Backend,
         spec=config["spec"],
   )
   perf = evaluate(module, peak_flops, I*J*K)
   return perf

# === Exploration Loop ===
def explore():
   """Generator that yields (index, total, name, perf) for each evaluation."""
   total = len(configurations)

   for idx, config in enumerate(configurations):
         perf = acquire(config)
         yield (idx, total, config["name"], perf)

# === Metadata for Results Display ===
exploration_info = {
   "dims": f"{I}x{J}x{K}",
   "dtype": dtype,
}''',
        language="python",
        label=""
    )
    run_explore_button = mo.ui.run_button(label="Run exploration")
    mo.vstack([explore_schedules_code, run_explore_button])
    return explore_schedules_code, run_explore_button

@app.cell
def _(explore_schedules_code, run_explore_button):
    import traceback as _traceback
    from io import StringIO as _StringIO
    from contextlib import redirect_stderr as _redirect_stderr
    import xtc.graphs.xtc.op as _O
    from xtc.graphs.xtc.graph import XTCGraph as _XTCGraph
    from xtc.schedules.descript import descript_scheduler as _descript_scheduler

    mo.stop(not run_explore_button.value, mo.md("*Click 'Run exploration' to execute the code.*"))

    # Define helper functions
    def _matmul_graph(I: int, J: int, K: int, dtype: str) -> _XTCGraph:
        a = _O.tensor((I, K), dtype, name="A")
        b = _O.tensor((K, J), dtype, name="B")
        with _O.graph(name="matmul") as gb:
            _O.matmul(a, b, name="C")
        return gb.graph

    def _apply_schedule(graph, backend_cls, spec, print_source_ir=False,
                        print_transformed_ir=False, print_lowered_ir=False, print_assembly=False):
        backend = backend_cls(graph)
        scheduler = backend.get_scheduler()
        _descript_scheduler(
            scheduler=scheduler,
            node_name="C",
            abstract_axis=["i", "j", "k"],
            spec=spec
        )
        schedule = scheduler.schedule()
        comp = backend.get_compiler(
            dump_file="test_mlir",
            shared_lib=True,
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            print_lowered_ir=print_lowered_ir,
            print_assembly=print_assembly
        )
        code = _StringIO()
        with _redirect_stderr(code):
            module = comp.compile(schedule)
        return module, code.getvalue()

    def _compile(backend, sched, print_source_ir=False, print_transformed_ir=False,
                 print_lowered_ir=False, print_assembly=False):
        comp = backend.get_compiler(
            dump_file="test_mlir",
            shared_lib=True,
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            print_lowered_ir=print_lowered_ir,
            print_assembly=print_assembly
        )
        code = _StringIO()
        with _redirect_stderr(code):
            module = comp.compile(sched)
        return (module, code.getvalue())

    def _evaluate(module, peak_flops, nfmadds):
        evaluator = module.get_evaluator()
        results, _, _ = evaluator.evaluate()
        result = min(results)
        time_flops = nfmadds / result
        perf = time_flops / peak_flops * 100
        return perf

    # Execute the user's code with helper functions in namespace
    _namespace = {
        "matmul_graph": _matmul_graph,
        "apply_schedule": _apply_schedule,
        "compile": _compile,
        "evaluate": _evaluate,
        "StringIO": _StringIO,
        "redirect_stderr": _redirect_stderr,
    }
    try:
        exec(explore_schedules_code.value, _namespace)
    except Exception as e:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{_traceback.format_exc()}\n```"))

    _explore_fn = _namespace.get("explore")
    _info = _namespace.get("exploration_info", {})

    if _explore_fn is None:
        mo.stop(True, mo.md("**Error:** The code must define an `explore()` generator function."))

    # Run exploration with progress bar
    _results = []
    _best_name = None
    _best_perf = 0.0
    _total = None

    try:
        _generator = _explore_fn()
        _first_result = next(_generator, None)

        if _first_result is None:
            mo.stop(True, mo.md("**Error:** The `explore()` generator yielded no results."))

        _idx, _total, _name, _perf = _first_result
        _results.append({"name": _name, "perf": _perf})
        if _perf > _best_perf:
            _best_perf = _perf
            _best_name = _name

        with mo.status.progress_bar(total=_total, title="Exploring schedules...", remove_on_exit=False) as _progress:
            _progress.update(
                title=f"Exploring schedules... (best: {_best_perf:.1f}%)",
                subtitle=f"Config {_idx + 1}/{_total}: {_name} -> {_perf:.1f}%"
            )

            for _idx, _total, _name, _perf in _generator:
                _results.append({"name": _name, "perf": _perf})

                if _perf > _best_perf:
                    _best_perf = _perf
                    _best_name = _name

                _progress.update(
                    title=f"Exploring schedules... (best: {_best_perf:.1f}%)",
                    subtitle=f"Config {_idx + 1}/{_total}: {_name} -> {_perf:.1f}%"
                )

    except Exception as e:
        mo.stop(True, mo.md(f"**Exploration error:**\n```\n{_traceback.format_exc()}\n```"))

    # Build results summary
    _dims = _info.get("dims", "?")
    _dtype = _info.get("dtype", "?")
    _baseline_perf = _results[0]["perf"] if _results else 1.0

    _summary_lines = [
        f"### Descript Schedule Exploration Results",
        f"",
        f"- **Problem:** {_dims} matmul, {_dtype}",
        f"",
        f"**Total configurations evaluated:** {len(_results)}",
        f"",
        f"#### All configurations:",
        f"",
        f"| Rank | Configuration | Performance | vs Baseline |",
        f"|------|---------------|-------------|-------------|",
    ]

    _sorted_results = sorted(_results, key=lambda x: x["perf"], reverse=True)
    for _rank, _r in enumerate(_sorted_results, 1):
        _speedup = _r["perf"] / _baseline_perf if _baseline_perf > 0 else 0
        _summary_lines.append(f"| {_rank} | {_r['name']} | {_r['perf']:.2f}% | {_speedup:.2f}x |")

    _summary_lines.extend([
        f"",
        f"#### Best configuration: **{_best_name}** ({_best_perf:.2f}% of peak)",
    ])

    mo.md("\n".join(_summary_lines))

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Automated Schedule Search with Strategies

    In the previous section, you manually defined schedule configurations to explore. While this approach works for small experiments, the scheduling space for real-world operators is vast—thousands or even millions of valid configurations exist. Manually exploring this space is impractical.

    **XTC provides scheduling strategies** that automate the exploration of the scheduling space. Strategies:

    1. **Define a structured search space**: Instead of arbitrary combinations, strategies encode domain knowledge about effective tilings, loop orders, and optimizations.
    2. **Filter invalid configurations**: Strategies automatically prune configurations that violate hardware constraints (e.g., unroll factors too large, tiles that don't fit in cache).
    3. **Support exhaustive and random sampling**: You can enumerate all valid schedules or sample randomly for faster exploration.
    4. **Encode best practices**: Each strategy implements proven optimization patterns from the literature (e.g., Goto-style for BLAS, Ansor-style for auto-scheduling).

    ### Why Use Strategies?

    - **Efficiency**: A strategy like `Strategy_OO` reduces a matmul's search space from millions of arbitrary combinations to ~100-1000 valid, hardware-aware configurations.
    - **Reproducibility**: Strategies provide a systematic way to explore and compare optimizations.
    - **Automation**: Strategies integrate with XTC's search infrastructure to find optimal schedules automatically.
    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Available Strategies and Naming Convention

    Strategies are named following a pattern that describes the **tiling scheme**:

    | Strategy Name         | Description                                                                 | Use Case                              |
    |-----------------------|-----------------------------------------------------------------------------|---------------------------------------|
    | `tile_oo`             | One-level tiling in O order (outer P, R, inner P)                           | Simple exploration, good baseline     |
    | `tile_prp`            | P-R-P tiling (parallels, reductions, parallels)                             | Cache-friendly matmul                 |
    | `tile_p1` / `tile_p1_v` | One-level tiling with permutation (v = vectorization constrained)         | Exploring loop orderings              |
    | `tile_pprprp`         | Ansor-style multi-level tiling                                              | Advanced auto-tuning                  |
    | `tile_pprprp_v`       | Ansor-style with vectorization constraint                                   | Ensuring SIMD usage                   |
    | `tile_pprprp_vr`      | Ansor-style with register and cache constraints                             | Hardware-aware tuning                 |
    | `tile_ppwrprp`        | Ansor-style with write buffer                                               | Reducing memory traffic               |
    | `tile_goto` / `tile_goto_r` | Goto-style BLAS tiling (r = reduced search space)                     | High-performance GEMM                 |

    **Naming convention**:
    - `P` = Parallel axes (e.g., i, j in matmul)
    - `R` = Reduction axes (e.g., k in matmul)
    - `O` = Outer parallel first, then reductions, then remaining parallels
    - `W` = Write buffer insertion point
    - `_v` suffix = Vectorization constraint (inner axis must be vectorizable)
    - `_vr` suffix = Vectorization + register/cache constraints

    ### How to Use a Strategy

    ```python
    from xtc.search.strategies import Strategy_OO

    # Create a strategy for your graph
    strategy = Strategy_OO(graph, max_unroll=256, vec_size=16)

    # Get all valid schedules (exhaustive)
    for sample in strategy.exhaustive():
        print(sample)  # e.g., [1, 16, 1] for [i1_tile, j1_tile, k1_tile]

    # Or sample randomly
    for sample in strategy.sample(num=10, seed=42):
        print(sample)

    # Apply a sample to a scheduler
    strategy.generate(scheduler, sample)
    ```
    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Practice.** The code below demonstrates using `Strategy_OO` to automatically explore the scheduling space. The code defines an `explore()` generator function that you can fully customize:

    - **Change the strategy**: Try `Strategy_PRP`, `Strategy_PPRPRP`, or `Strategy_GOTO`
    - **Modify the search loop**: Add early stopping, custom filtering, or logging
    - **Customize the acquisition function**: The `acquire()` function evaluates a sample - you can add caching, error handling, or custom metrics

    The `explore()` function must `yield` tuples of `(index, total, sample, performance)` for real-time progress display.
    """)
    return

@app.cell
def _():
    strategy_editor = mo.ui.code_editor(
        value=
'''from xtc.backends.mlir import Backend
from xtc.search.strategies import Strategy_OO
import xtc.runtimes.host.runtime as rt

# === Problem Definition ===
I, J, K, dtype = 64, 128, 256, "float32"
graph = matmul_graph(I=I, J=J, K=K, dtype=dtype)
peak_flops = rt.evaluate_flops(dtype)

# === Strategy Selection ===
# Available: Strategy_OO, Strategy_PRP, Strategy_PPRPRP, Strategy_GOTO, etc.
strategy = Strategy_OO(graph, max_unroll=64, vec_size=16)

# === Acquisition Function ===
def acquire(sample):
   """Evaluate a single sample and return its performance."""
   impl = Backend(graph)
   scheduler = impl.get_scheduler()
   strategy.generate(scheduler, sample)
   schedule = scheduler.schedule()
   module, _ = compile(backend=impl, sched=schedule)
   perf = evaluate(module, peak_flops, I*J*K)
   return perf

# === Exploration Loop ===
def explore():
   """Generator that yields (index, total, sample, perf) for each evaluation."""
   # Choose search mode: exhaustive() or sample(num=N, seed=S)
   samples = list(strategy.exhaustive())
   # samples = list(strategy.sample(num=30, seed=42))  # Alternative: random sampling

   total = len(samples)

   for idx, sample in enumerate(samples):
         perf = acquire(sample)
         yield (idx, total, sample, perf)

         # Optional: early stopping if we find a good enough solution
         # if perf > 90.0:
         #     break

# === Metadata for Results Display ===
exploration_info = {
   "dims": f"{I}x{J}x{K}",
   "dtype": dtype,
   "strategy": strategy.__class__.__name__,
   "strategy_params": f"max_unroll=64, vec_size=16",
   "sample_names": strategy.sample_names,
   "stats_fn": lambda: dict(strategy.stats),
   "sample_to_dict_fn": strategy.sample_to_dict,
}''',
        language="python",
        label=""
    )
    run_strategy_button = mo.ui.run_button(label="Run strategy exploration")
    mo.vstack([strategy_editor, run_strategy_button])
    return strategy_editor, run_strategy_button

@app.cell
def _(strategy_editor, run_strategy_button):
    import traceback as _traceback
    from io import StringIO as _StringIO
    from contextlib import redirect_stderr as _redirect_stderr
    import xtc.graphs.xtc.op as _O
    from xtc.graphs.xtc.graph import XTCGraph as _XTCGraph
    from xtc.itf.comp import Module as _Module
    from xtc.itf.schd import Schedule as _Schedule

    mo.stop(not run_strategy_button.value, mo.md("*Click 'Run strategy exploration' to execute the code.*"))

    # Define helper functions that will be available in the editor's namespace
    def _matmul_graph(I: int, J: int, K: int, dtype: str) -> _XTCGraph:
        a = _O.tensor((I, K), dtype, name="A")
        b = _O.tensor((K, J), dtype, name="B")
        with _O.graph(name="matmul") as gb:
            _O.matmul(a, b, name="C")
        return gb.graph

    def _compile(backend, sched, print_source_ir=False, print_transformed_ir=False,
                 print_lowered_ir=False, print_assembly=False):
        comp = backend.get_compiler(
            dump_file="test_mlir",
            shared_lib=True,
            print_source_ir=print_source_ir,
            print_transformed_ir=print_transformed_ir,
            print_lowered_ir=print_lowered_ir,
            print_assembly=print_assembly
        )
        code = _StringIO()
        with _redirect_stderr(code):
            module = comp.compile(sched)
        return (module, code.getvalue())

    def _evaluate(module, peak_flops, nfmadds):
        evaluator = module.get_evaluator()
        results, _, _ = evaluator.evaluate()
        result = min(results)
        time_flops = nfmadds / result
        perf = time_flops / peak_flops * 100
        return perf

    # Execute the user's code with helper functions in namespace
    _namespace = {
        "matmul_graph": _matmul_graph,
        "compile": _compile,
        "evaluate": _evaluate,
        "StringIO": _StringIO,
        "redirect_stderr": _redirect_stderr,
    }
    try:
        exec(strategy_editor.value, _namespace)
    except Exception as e:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{_traceback.format_exc()}\n```"))

    _explore_fn = _namespace.get("explore")
    _info = _namespace.get("exploration_info", {})

    if _explore_fn is None:
        mo.stop(True, mo.md("**Error:** The code must define an `explore()` generator function."))

    # Run exploration with progress bar
    _results = []
    _best_sample = None
    _best_perf = 0.0
    _total = None

    try:
        # First, peek at the generator to get total count
        _generator = _explore_fn()
        _first_result = next(_generator, None)

        if _first_result is None:
            mo.stop(True, mo.md("**Error:** The `explore()` generator yielded no results."))

        _idx, _total, _sample, _perf = _first_result
        _results.append({"sample": _sample, "perf": _perf})
        _best_perf = _perf
        _best_sample = _sample

        with mo.status.progress_bar(total=_total, title="Exploring schedules...", remove_on_exit=False) as _progress:
            _progress.update(
                title=f"Exploring schedules... (best: {_best_perf:.1f}%)",
                subtitle=f"Sample {_idx + 1}/{_total}: {_sample} -> {_perf:.1f}%"
            )

            for _idx, _total, _sample, _perf in _generator:
                _results.append({"sample": _sample, "perf": _perf})

                if _perf > _best_perf:
                    _best_perf = _perf
                    _best_sample = _sample

                _progress.update(
                    title=f"Exploring schedules... (best: {_best_perf:.1f}%)",
                    subtitle=f"Sample {_idx + 1}/{_total}: {_sample} -> {_perf:.1f}%"
                )

    except Exception as e:
        mo.stop(True, mo.md(f"**Exploration error:**\n```\n{_traceback.format_exc()}\n```"))

    # Build results summary
    _dims = _info.get("dims", "?")
    _dtype = _info.get("dtype", "?")
    _strategy_name = _info.get("strategy", "?")
    _strategy_params = _info.get("strategy_params", "")
    _stats_fn = _info.get("stats_fn", lambda: {})

    _summary_lines = [
        f"### Strategy Exploration Results",
        f"",
        f"- **Problem name:** {_dims} matmul, {_dtype}",
        f"- **Strategy:** {_strategy_name} ({_strategy_params})",
        f"- **Search space stats:** {_stats_fn()}",
        f"",
        f"**Total configurations evaluated:** {len(_results)}",
        f"",
        f"#### Top 10 configurations:",
        f"",
        f"| Rank | Sample | Performance |",
        f"|------|--------|-------------|",
    ]

    _sorted_results = sorted(_results, key=lambda x: x["perf"], reverse=True)[:10]
    for _rank, _r in enumerate(_sorted_results, 1):
        _summary_lines.append(f"| {_rank} | `{_r['sample']}` | {_r['perf']:.2f}% |")

    _summary_lines.extend([
        f"",
        f"#### Best schedule found:",
        f"",
        f"- **Sample:** `{_best_sample}`",
        f"- **Performance:** {_best_perf:.2f}% of peak",
    ])

    mo.md("\n".join(_summary_lines))

if __name__ == "__main__":
    app.run()
