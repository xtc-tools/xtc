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
    Please [follow the README](https://github.com/xtc-tools/xtc/blob/main/README.md).
    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Define Your First Graph with XTC

    In XTC, computations are represented as **dataflow graphs**. A graph consists of:
    - **Nodes** representing operations each implementing a specific computation (Operators)
    - **Edges** representing data dependencies between operations (through Tensors)
    Where:
    - **Tensors** are multi-dimensional arrays that hold data
    - **Operators** are tensor operations (e.g., matrix multiplication, convolution)

    Let us start by creating a simple matrix multiplication graph. Matrix multiplication (matmul) computes $C = A \times B$ where:
    - $A$ is an $I \times K$ matrix
    - $B$ is a $K \times J$ matrix
    - $C$ is the resulting $I \times J$ matrix

    The code below demonstrates how to:
    1. Define input tensors with their shapes and data types
    2. Create a graph context and add a matmul operation
    3. Print the resulting graph in a serialized form
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

## Section 3 - Compile and Evaluate

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Compile and Evaluate

    Now that we have a graph, we can compile it and measure its baseline performance (without any optimization).

    The compilation pipeline in XTC follows these steps:
    1. **Create a Backend**: In XTC, the backend corresponds to an existing framework such as MLIR or TVM that, given a schedule, can generate the code for a specific target
    2. **Get a Scheduler**: In XTC, a scheduler is a builder that creates a schedule. Even without optimizations, we need a scheduler to get a default loop structure.
    3. **Compile**: Generate executable code
    4. **Evaluate**: Run the compiled code and measure performance
    """)
    return

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Practice.** The code below compiles the matmul graph without any optimization. Use the radio buttons to select the backend (MLIR or TVM) and which IR to display. Performance is measured as a percentage of peak (the theoretical FLOP/s of the CPU).

    1. *Inspect the generated code.* Look at the Source IR, Transformed IR, Lowered IR, and Assembly. What do you notice?
    2. *Observe the performance.* In your opinion, why is it so poor? This is the baseline to compare against when we will add optimizations in the next section!
    """)
    return

@app.cell
def _(compile_backend_radio, compile_output_radio):
    # Build editor content based on backend selection
    if compile_backend_radio.value == "MLIR":
        _backend_import = "from xtc.backends.mlir import Backend"
    else:
        _backend_import = "from xtc.backends.tvm import Backend"

    # Build print options based on output radio
    _print_opts = []
    if compile_output_radio.value == "Source IR":
        _print_opts.append("print_source_ir=True")
    elif compile_output_radio.value == "Transformed IR":
        _print_opts.append("print_transformed_ir=True")
    elif compile_output_radio.value == "Lowered IR":
        _print_opts.append("print_lowered_ir=True")
    elif compile_output_radio.value == "Assembly":
        _print_opts.append("print_assembly=True")
    _print_opts_str = ", ".join(_print_opts)

    compile_editor = mo.ui.code_editor(
        value=f'''import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph
{_backend_import}
import xtc.runtimes.host.runtime as rt

# Problem setup
I, J, K, dtype = 4, 32, 512, "float32"

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   """Create a graph computing C = A @ B."""
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         O.matmul(a, b, name="C")
   return gb.graph

graph = matmul_graph(I, J, K, dtype)
backend = Backend(graph)

# Compile (no transformations, just default loop structure)
scheduler = backend.get_scheduler()
scheduler.set_dims(['i','j','k'])
schedule = scheduler.schedule()

compiler = backend.get_compiler(dump_file="matmul", shared_lib=True, {_print_opts_str})
module = compiler.compile(schedule)

# Evaluate and display results
peak_flops = rt.evaluate_flops(dtype)
evaluator = module.get_evaluator()
results, _, _ = evaluator.evaluate()
perf = (I * J * K) / min(results) / peak_flops * 100

display_results(perf)
''',
        language="python",
        label=""
    )
    compile_editor
    return compile_editor,

@app.cell
def _():
    compile_backend_radio = mo.ui.radio(
        options=["MLIR", "TVM"],
        value="MLIR",
        label="Backend:"
    )
    return compile_backend_radio,

@app.cell
def _(compile_backend_radio):
    # Output options depend on backend (TVM doesn't support Lowered IR)
    if compile_backend_radio.value == "TVM":
        _output_options = ["Source IR", "Transformed IR", "Assembly"]
    else:
        _output_options = ["Source IR", "Transformed IR", "Lowered IR", "Assembly"]

    compile_output_radio = mo.ui.radio(
        options=_output_options,
        value="Assembly",
        label="Output options:"
    )
    # Display radios after the editor
    mo.hstack([compile_backend_radio, compile_output_radio], justify="start", gap=4)
    return compile_output_radio,

@app.cell
def _(compile_editor):
    import traceback as _traceback
    from io import StringIO as _StringIO
    from contextlib import redirect_stderr as _redirect_stderr, redirect_stdout as _redirect_stdout

    _captured = {"perf": 0.0}

    def _display_results(perf):
        _captured["perf"] = perf

    # Execute the editor code with display_results available
    _namespace = {"display_results": _display_results}
    _code_output_stderr = _StringIO()
    _code_output_stdout = _StringIO()
    try:
        with _redirect_stderr(_code_output_stderr), _redirect_stdout(_code_output_stdout):
            exec(compile_editor.value, _namespace)
    except Exception as e:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{_traceback.format_exc()}\n```"))

    _code_output = _code_output_stderr.getvalue() + _code_output_stdout.getvalue()

    # Build output
    _perf_display = mo.md(f"**Performance:** {_captured['perf']:.2f}% of peak")
    _code_content = mo.md(f"```asm\n{_code_output}\n```") if _code_output else mo.md("*No IR output.*")
    _code_accordion = mo.accordion({"Generated Code": _code_content})

    mo.vstack([_perf_display, _code_accordion])
    return


## Section 4 - Optimize with Scheduling

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Optimize with Scheduling

    The baseline compilation produces correct but unoptimized code. To improve performance, XTC exposes a **scheduler** with imperative primitives that transform the loop nest:

    - `sch.split("j", {"j1": 16})` splits `j` into two segments at position 16, creating `j` (iterations 0-15) and `j1` (iterations 16+). Splitting is useful for applying different transformations to different parts of a loop (e.g., vectorize the main part, handle the remainder separately).
    - `sch.interchange(["i", "k", "j"])` reorders the loops. Interchange improves memory access patterns by ensuring stride-1 access (contiguous memory) rather than strided access, maximizing cache efficiency. Along with primitive `sch.tile` (see below) this allows to actually tile the loop body.
    - `sch.tile("j", {"j1": 16})` breaks loop `j` with smaller chunks of size 16. This transformation originally called strip-mining can also be seen as a 1d-tiling (thus the name used by most compiler factory).
    - To perform actual tiling, `sch.tile` must be combined with `sch.interchange`. As an example `sch.tile("j", {"j1": 16})` followed by `sch.tile("k", {"k1": 16})` followed by `sch.interchange(["i", "j", "k", "j1", "k1"])` would create tiles of size $j_1\times k_1=16\times 16$.
    - `sch.vectorize(["j1"])` vectorizes the computation along the loop `j1`. Vectorization uses SIMD instructions to process multiple elements in parallel, significantly increasing throughput on modern CPUs.
    - `sch.unroll({"j1":1})` unrolls the loop `j1` with an unroll factor of 1 (useless too). Unrolling reduces loop overhead (fewer branches) and exposes more instruction-level parallelism to the hardware.
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
def _(sched_backend_radio, sched_output_radio):
    # Build editor content based on backend selection
    if sched_backend_radio.value == "MLIR":
        _backend_import = "from xtc.backends.mlir import Backend"
    else:
        _backend_import = "from xtc.backends.tvm import Backend"

    # Build print options based on output radio
    _print_opts = []
    if sched_output_radio.value == "Source IR":
        _print_opts.append("print_source_ir=True")
    elif sched_output_radio.value == "Transformed IR":
        _print_opts.append("print_transformed_ir=True")
    elif sched_output_radio.value == "Lowered IR":
        _print_opts.append("print_lowered_ir=True")
    elif sched_output_radio.value == "Assembly":
        _print_opts.append("print_assembly=True")
    _print_opts_str = ", ".join(_print_opts)

    sched_editor = mo.ui.code_editor(
        value=f'''import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph
{_backend_import}
import xtc.runtimes.host.runtime as rt

# Problem setup
I, J, K, dtype = 4, 32, 512, "float32"

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   """Create a graph computing C = A @ B."""
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         O.matmul(a, b, name="C")
   return gb.graph

graph = matmul_graph(I, J, K, dtype)
backend = Backend(graph)

# Schedule definition
def schedule(sch):
   """Apply transformations to the scheduler."""
   sch.set_dims(['i','j','k'])
   # Add transformations here, e.g.:
   # sch.tile("j", {{"j1": 16}})
   # sch.vectorize(["j1"])
   # sch.interchange(["i", "k", "j", "j1"])

# Compile
scheduler = backend.get_scheduler()
schedule(scheduler)
sched = scheduler.schedule()

compiler = backend.get_compiler(dump_file="matmul", shared_lib=True, {_print_opts_str})
module = compiler.compile(sched)

# Evaluate and display results
peak_flops = rt.evaluate_flops(dtype)
evaluator = module.get_evaluator()
results, _, _ = evaluator.evaluate()
perf = (I * J * K) / min(results) / peak_flops * 100

display_results(perf)
''',
        language="python",
        label=""
    )
    sched_editor
    return sched_editor,

@app.cell
def _():
    sched_backend_radio = mo.ui.radio(
        options=["MLIR", "TVM"],
        value="MLIR",
        label="Backend:"
    )
    return sched_backend_radio,

@app.cell
def _(sched_backend_radio):
    # Output options depend on backend (TVM doesn't support Lowered IR)
    if sched_backend_radio.value == "TVM":
        _output_options = ["Source IR", "Transformed IR", "Assembly"]
    else:
        _output_options = ["Source IR", "Transformed IR", "Lowered IR", "Assembly"]

    sched_output_radio = mo.ui.radio(
        options=_output_options,
        value="Assembly",
        label="Output options:"
    )
    # Display radios after the editor
    mo.hstack([sched_backend_radio, sched_output_radio], justify="start", gap=4)
    return sched_output_radio,

@app.cell
def _(sched_editor):
    import traceback as _traceback
    from io import StringIO as _StringIO
    from contextlib import redirect_stderr as _redirect_stderr, redirect_stdout as _redirect_stdout

    _captured = {"perf": 0.0}

    def _display_results(perf):
        _captured["perf"] = perf

    # Execute the editor code with display_results available
    _namespace = {"display_results": _display_results}
    _code_output_stderr = _StringIO()
    _code_output_stdout = _StringIO()
    try:
        with _redirect_stderr(_code_output_stderr), _redirect_stdout(_code_output_stdout):
            exec(sched_editor.value, _namespace)
    except Exception as e:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{_traceback.format_exc()}\n```"))

    _code_output = _code_output_stderr.getvalue() + _code_output_stdout.getvalue()

    # Build output
    _perf_display = mo.md(f"**Performance:** {_captured['perf']:.2f}% of peak")
    _code_content = mo.md(f"```asm\n{_code_output}\n```") if _code_output else mo.md("*No IR output.*")
    _code_accordion = mo.accordion({"Generated Code": _code_content})

    mo.vstack([_perf_display, _code_accordion])
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
def _(descript_backend_radio, descript_output_radio):
    # Build editor content based on backend selection
    if descript_backend_radio.value == "MLIR":
        _backend_import = "from xtc.backends.mlir import Backend"
    else:
        _backend_import = "from xtc.backends.tvm import Backend"

    # Build print options based on output radio
    _print_opts = []
    if descript_output_radio.value == "Source IR":
        _print_opts.append("print_source_ir=True")
    elif descript_output_radio.value == "Transformed IR":
        _print_opts.append("print_transformed_ir=True")
    elif descript_output_radio.value == "Lowered IR":
        _print_opts.append("print_lowered_ir=True")
    elif descript_output_radio.value == "Assembly":
        _print_opts.append("print_assembly=True")
    _print_opts_str = ", ".join(_print_opts)

    descript_editor = mo.ui.code_editor(
        value=f'''import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph
{_backend_import}
from xtc.schedules.descript import descript_scheduler
import xtc.runtimes.host.runtime as rt

# Problem setup
I, J, K, dtype = 4, 32, 512, "float32"

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   """Create a graph computing C = A @ B."""
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         O.matmul(a, b, name="C")
   return gb.graph

graph = matmul_graph(I, J, K, dtype)
backend = Backend(graph)

# Schedule specification
schedule_spec = {{
   "i": {{}},
   "j": {{}},
   "k": {{}}
}}

# Compile
scheduler = backend.get_scheduler()
descript_scheduler(
   scheduler=scheduler,
   node_name="C",
   abstract_axis=["i", "j", "k"],
   spec=schedule_spec
)
schedule = scheduler.schedule()

compiler = backend.get_compiler(dump_file="matmul", shared_lib=True, {_print_opts_str})
module = compiler.compile(schedule)

# Evaluate and display results
peak_flops = rt.evaluate_flops(dtype)
evaluator = module.get_evaluator()
results, _, _ = evaluator.evaluate()
perf = (I * J * K) / min(results) / peak_flops * 100

display_results(perf)
''',
        language="python",
        label=""
    )
    descript_editor
    return descript_editor,

@app.cell
def _():
    descript_backend_radio = mo.ui.radio(
        options=["MLIR", "TVM"],
        value="MLIR",
        label="Backend:"
    )
    return descript_backend_radio,

@app.cell
def _(descript_backend_radio):
    # Output options depend on backend (TVM doesn't support Lowered IR)
    if descript_backend_radio.value == "TVM":
        _output_options = ["Source IR", "Transformed IR", "Assembly"]
    else:
        _output_options = ["Source IR", "Transformed IR", "Lowered IR", "Assembly"]

    descript_output_radio = mo.ui.radio(
        options=_output_options,
        value="Assembly",
        label="Output options:"
    )
    # Display radios after the editor
    mo.hstack([descript_backend_radio, descript_output_radio], justify="start", gap=4)
    return descript_output_radio,

@app.cell
def _(descript_editor):
    import traceback as _traceback
    from io import StringIO as _StringIO
    from contextlib import redirect_stderr as _redirect_stderr, redirect_stdout as _redirect_stdout

    _captured = {"perf": 0.0}

    def _display_results(perf):
        _captured["perf"] = perf

    # Execute the editor code with display_results available
    _namespace = {"display_results": _display_results}
    _code_output_stderr = _StringIO()
    _code_output_stdout = _StringIO()
    try:
        with _redirect_stderr(_code_output_stderr), _redirect_stdout(_code_output_stdout):
            exec(descript_editor.value, _namespace)
    except Exception as e:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{_traceback.format_exc()}\n```"))

    _code_output = _code_output_stderr.getvalue() + _code_output_stdout.getvalue()

    # Build output
    _perf_display = mo.md(f"**Performance:** {_captured['perf']:.2f}% of peak")
    _code_content = mo.md(f"```asm\n{_code_output}\n```") if _code_output else mo.md("*No IR output.*")
    _code_accordion = mo.accordion({"Generated Code": _code_content})

    mo.vstack([_perf_display, _code_accordion])

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
'''import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph
from xtc.backends.tvm import Backend as TVM_Backend
from xtc.backends.mlir import Backend as MLIR_Backend
from xtc.schedules.descript import descript_scheduler
import xtc.runtimes.host.runtime as rt
from io import StringIO
from contextlib import redirect_stderr
import itertools

# Problem setup
I, J, K, dtype = 256, 256, 512, "float32"

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   """Create a graph computing C = A @ B."""
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         O.matmul(a, b, name="C")
   return gb.graph

graph = matmul_graph(I=I, J=J, K=K, dtype=dtype)
peak_flops = rt.evaluate_flops(dtype)

# Evaluation helpers
def apply_schedule(graph, backend_cls, spec):
   """Apply a declarative schedule specification and compile."""
   backend = backend_cls(graph)
   scheduler = backend.get_scheduler()
   descript_scheduler(
         scheduler=scheduler,
         node_name="C",
         abstract_axis=["i", "j", "k"],
         spec=spec
   )
   schedule = scheduler.schedule()
   comp = backend.get_compiler(dump_file="test_mlir", shared_lib=True)
   code = StringIO()
   with redirect_stderr(code):
         module = comp.compile(schedule)
   return module, code.getvalue()

def evaluate(module, peak_flops, nfmadds):
   """Evaluate module performance as percentage of peak."""
   evaluator = module.get_evaluator()
   results, _, _ = evaluator.evaluate()
   result = min(results)
   time_flops = nfmadds / result
   perf = time_flops / peak_flops * 100
   return perf

def acquire(i1: int, j1: int, backend):
   """Evaluate a single configuration and return its performance."""
   module, _ = apply_schedule(
         graph=graph,
         backend_cls=backend,
         spec={
               "i": {},
               "j": {},
               "k": {},
               f"i#{i1}": {"unroll": True},
               f"j#{j1}": {"vectorize": True}
         },
   )
   return evaluate(module, peak_flops, I*J*K)

# Exploration loop
def explore():
   """Generator that yields (index, total, name, perf) for each evaluation."""
   i1_range = range(1, 5)
   j1_range = [8, 16, 24, 32]
   configurations = list(itertools.product(i1_range, j1_range))
   total = len(configurations)

   for idx, (i1, j1) in enumerate(configurations):
         perf = acquire(i1=i1, j1=j1, backend=MLIR_Backend)
         yield (idx, total, f"mlir:{i1}x{j1}", perf)

# Run exploration (displays a progress bar and returns sorted results)
def get_info():
   return {"dims": f"{I}x{J}x{K}", "dtype": dtype}

results = run_exploration(explore(), get_info)
''',
        language="python",
        label=""
    )
    run_explore_button = mo.ui.run_button(label="Run exploration")
    mo.vstack([explore_schedules_code, run_explore_button])
    return explore_schedules_code, run_explore_button

@app.cell
def _(explore_schedules_code, run_explore_button):
    import traceback as _traceback

    mo.stop(not run_explore_button.value, mo.md("*Click 'Run exploration' to execute the code.*"))

    # Define run_exploration helper that will be available in the editor
    _captured_info = {}

    def _run_exploration(generator, get_info=None):
        """Run exploration with progress bar. Returns sorted results."""
        results = []
        best_name = None
        best_perf = 0.0

        first_result = next(generator, None)
        if first_result is None:
            return []

        idx, total, name, perf = first_result
        results.append({"name": name, "perf": perf})
        if perf > best_perf:
            best_perf = perf
            best_name = name

        with mo.status.progress_bar(total=total, title="Exploring schedules...", remove_on_exit=False) as progress:
            progress.update(
                title=f"Exploring schedules... (best: {best_perf:.1f}%)",
                subtitle=f"Config {idx + 1}/{total}: {name} -> {perf:.1f}%"
            )

            for idx, total, name, perf in generator:
                results.append({"name": name, "perf": perf})

                if perf > best_perf:
                    best_perf = perf
                    best_name = name

                progress.update(
                    title=f"Exploring schedules... (best: {best_perf:.1f}%)",
                    subtitle=f"Config {idx + 1}/{total}: {name} -> {perf:.1f}%"
                )

        # Call get_info after exploration to capture dynamic values
        if get_info:
            _captured_info.update(get_info())

        return sorted(results, key=lambda x: x["perf"], reverse=True)

    # Execute the user's code with run_exploration available
    _namespace = {"run_exploration": _run_exploration}
    try:
        exec(explore_schedules_code.value, _namespace)
    except Exception as e:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{_traceback.format_exc()}\n```"))

    _results = _namespace.get("results", [])

    # Build results summary
    if not _results:
        mo.stop(True, mo.md("**Error:** No results returned. Make sure to call `results = run_exploration(explore(), get_info)`"))

    _dims = _captured_info.get("dims", "?")
    _dtype = _captured_info.get("dtype", "?")
    _baseline_perf = _results[-1]["perf"] if _results else 1.0
    _best = _results[0]

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

    for _rank, _r in enumerate(_results, 1):
        _speedup = _r["perf"] / _baseline_perf if _baseline_perf > 0 else 0
        _summary_lines.append(f"| {_rank} | {_r['name']} | {_r['perf']:.2f}% | {_speedup:.2f}x |")

    _summary_lines.extend([
        f"",
        f"#### Best configuration: **{_best['name']}** ({_best['perf']:.2f}% of peak)",
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

    **Why Use Strategies?**

    - *Efficiency*: A strategy like `Strategy_OO` reduces a matmul's search space from millions of arbitrary combinations to ~100-1000 valid, hardware-aware configurations.
    - *Reproducibility*: Strategies provide a systematic way to explore and compare optimizations.
    - *Automation*: Strategies integrate with XTC's search infrastructure to find optimal schedules automatically.
    
    **Available strategies:**

    | Strategy Name                       | Description                                                       | Use Case                          |
    |-------------------------------------|-------------------------------------------------------------------|-----------------------------------|
    | `Strategy_OO`                       | One-level tiling in O order (outer P, R, inner P)                 | Simple exploration, good baseline |
    | `Strategy_PRP`                      | P-R-P tiling (parallels, reductions, parallels)                   | Cache-friendly matmul             |
    | `Strategy_P1` / `Strategy_P1_v`     | One-level tiling with permutation (v = vectorization constrained) | Exploring loop orderings          |
    | `Strategy_PPRPRP`                   | Ansor-style multi-level tiling                                    | Advanced auto-tuning              |
    | `Strategy_PPRPRPv`                  | Ansor-style with vectorization constraint                         | Ensuring SIMD usage               |
    | `Strategy_PPRPRP_vr`                | Ansor-style with register and cache constraints                   | Hardware-aware tuning             |
    | `Strategy_PPWRPRP`                  | Ansor-style with write buffer                                     | Reducing memory traffic           |
    | `Strategy_GOTO` / `Strategy_GOTO_R` | Goto-style BLAS tiling (r = reduced search space)                 | High-performance GEMM             |

    **Naming convention**:
    - `P` = Parallel axes (e.g., i, j in matmul)
    - `R` = Reduction axes (e.g., k in matmul)
    - `O` = Outer parallel first, then reductions, then remaining parallels
    - `W` = Write buffer insertion point
    - `_v` suffix = Vectorization constraint (inner axis must be vectorizable)
    - `_vr` suffix = Vectorization + register/cache constraints
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
'''import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph
from xtc.backends.mlir import Backend
from xtc.search.strategies import Strategy_OO
import xtc.runtimes.host.runtime as rt
from io import StringIO
from contextlib import redirect_stderr

# Problem setup
I, J, K, dtype = 64, 128, 256, "float32"

def matmul_graph(I: int, J: int, K: int, dtype: str) -> XTCGraph:
   """Create a graph computing C = A @ B."""
   a = O.tensor((I, K), dtype, name="A")
   b = O.tensor((K, J), dtype, name="B")
   with O.graph(name="matmul") as gb:
         O.matmul(a, b, name="C")
   return gb.graph

graph = matmul_graph(I=I, J=J, K=K, dtype=dtype)
peak_flops = rt.evaluate_flops(dtype)

# Strategy selection (try: Strategy_PRP, Strategy_PPRPRP, Strategy_GOTO, etc.)
strategy = Strategy_OO(graph, max_unroll=64, vec_size=16)

# Evaluation helpers
def compile_schedule(backend, sched):
   """Compile a schedule and return the module."""
   comp = backend.get_compiler(dump_file="test_mlir", shared_lib=True)
   code = StringIO()
   with redirect_stderr(code):
         module = comp.compile(sched)
   return module, code.getvalue()

def evaluate(module, peak_flops, nfmadds):
   """Evaluate module performance as percentage of peak."""
   evaluator = module.get_evaluator()
   results, _, _ = evaluator.evaluate()
   result = min(results)
   time_flops = nfmadds / result
   perf = time_flops / peak_flops * 100
   return perf

def acquire(sample):
   """Evaluate a single sample and return its performance."""
   impl = Backend(graph)
   scheduler = impl.get_scheduler()
   strategy.generate(scheduler, sample)
   schedule = scheduler.schedule()
   module, _ = compile_schedule(backend=impl, sched=schedule)
   return evaluate(module, peak_flops, I*J*K)

# Exploration loop
def explore():
   """Generator that yields (index, total, sample, perf) for each evaluation."""
   # Choose: strategy.exhaustive() or strategy.sample(num=N, seed=S)
   samples = list(strategy.exhaustive())
   total = len(samples)

   for idx, sample in enumerate(samples):
         perf = acquire(sample)
         yield (idx, total, sample, perf)

# Run exploration (displays a progress bar and returns sorted results)
def get_info():
   return {
         "dims": f"{I}x{J}x{K}",
         "dtype": dtype,
         "strategy": strategy.__class__.__name__,
         "strategy_params": "max_unroll=64, vec_size=16",
         "stats": dict(strategy.stats),
   }

results = run_exploration(explore(), get_info)
''',
        language="python",
        label=""
    )
    run_strategy_button = mo.ui.run_button(label="Run strategy exploration")
    mo.vstack([strategy_editor, run_strategy_button])
    return strategy_editor, run_strategy_button

@app.cell
def _(strategy_editor, run_strategy_button):
    import traceback as _traceback

    mo.stop(not run_strategy_button.value, mo.md("*Click 'Run strategy exploration' to execute the code.*"))

    # Define run_exploration helper that will be available in the editor
    _captured_info = {}

    def _run_exploration(generator, get_info=None):
        """Run exploration with progress bar. Returns sorted results."""
        results = []
        best_sample = None
        best_perf = 0.0

        first_result = next(generator, None)
        if first_result is None:
            return []

        idx, total, sample, perf = first_result
        results.append({"sample": sample, "perf": perf})
        if perf > best_perf:
            best_perf = perf
            best_sample = sample

        with mo.status.progress_bar(total=total, title="Exploring schedules...", remove_on_exit=False) as progress:
            progress.update(
                title=f"Exploring schedules... (best: {best_perf:.1f}%)",
                subtitle=f"Sample {idx + 1}/{total}: {sample} -> {perf:.1f}%"
            )

            for idx, total, sample, perf in generator:
                results.append({"sample": sample, "perf": perf})

                if perf > best_perf:
                    best_perf = perf
                    best_sample = sample

                progress.update(
                    title=f"Exploring schedules... (best: {best_perf:.1f}%)",
                    subtitle=f"Sample {idx + 1}/{total}: {sample} -> {perf:.1f}%"
                )

        # Call get_info after exploration to capture dynamic values
        if get_info:
            _captured_info.update(get_info())

        return sorted(results, key=lambda x: x["perf"], reverse=True)

    # Execute the user's code with run_exploration available
    _namespace = {"run_exploration": _run_exploration}
    try:
        exec(strategy_editor.value, _namespace)
    except Exception as e:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{_traceback.format_exc()}\n```"))

    _results = _namespace.get("results", [])

    # Build results summary
    if not _results:
        mo.stop(True, mo.md("**Error:** No results returned. Make sure to call `results = run_exploration(explore(), get_info)`"))

    _dims = _captured_info.get("dims", "?")
    _dtype = _captured_info.get("dtype", "?")
    _strategy_name = _captured_info.get("strategy", "?")
    _strategy_params = _captured_info.get("strategy_params", "")
    _stats = _captured_info.get("stats", {})
    _best = _results[0]

    _summary_lines = [
        f"### Strategy Exploration Results",
        f"",
        f"- **Problem name:** {_dims} matmul, {_dtype}",
        f"- **Strategy:** {_strategy_name} ({_strategy_params})",
        f"- **Search space stats:** {_stats}",
        f"",
        f"**Total configurations evaluated:** {len(_results)}",
        f"",
        f"#### Top 10 configurations:",
        f"",
        f"| Rank | Sample | Performance |",
        f"|------|--------|-------------|",
    ]

    for _rank, _r in enumerate(_results[:10], 1):
        _summary_lines.append(f"| {_rank} | `{_r['sample']}` | {_r['perf']:.2f}% |")

    _summary_lines.extend([
        f"",
        f"#### Best schedule found:",
        f"",
        f"- **Sample:** `{_best['sample']}`",
        f"- **Performance:** {_best['perf']:.2f}% of peak",
    ])

    mo.md("\n".join(_summary_lines))

if __name__ == "__main__":
    app.run()
