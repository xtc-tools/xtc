import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    # === Utility functions for the tutorial ===
    from io import StringIO
    from contextlib import redirect_stderr, redirect_stdout
    import traceback
    from typing import Any
    import queue
    import time
    import multiprocessing as mp
    import exec_utils
    import matplotlib.pyplot as plt

    def get_backend_import(backend_name: str) -> str:
        """Return the import statement for the selected backend."""
        if backend_name == "MLIR":
            return "from xtc.backends.mlir import Backend"
        else:
            return "from xtc.backends.tvm import Backend"

    def get_print_opts_str(output_option: str) -> str:
        """Return the compiler print options string for the selected output."""
        opts_map = {
            "Source IR": "print_source_ir=True",
            "Transformed IR": "print_transformed_ir=True",
            "Lowered IR": "print_lowered_ir=True",
            "Assembly": "print_assembly=True",
        }
        return opts_map.get(output_option, "")

    def get_print_opts_dict(output_option: str) -> dict:
        """Return the compiler print options as a dictionary."""
        opts_map = {
            "Source IR": {"print_source_ir": True},
            "Transformed IR": {"print_transformed_ir": True},
            "Lowered IR": {"print_lowered_ir": True},
            "Assembly": {"print_assembly": True},
        }
        return opts_map.get(output_option, {})

    def get_backend_class(backend_name: str):
        """Return the Backend class for the selected backend."""
        if backend_name == "MLIR":
            from xtc.backends.mlir import Backend
        else:
            from xtc.backends.tvm import Backend
        return Backend

    def get_output_options(backend_name: str) -> list:
        """Return available output options based on backend (TVM doesn't support Lowered IR)."""
        if backend_name == "TVM":
            return ["Source IR", "Transformed IR", "Assembly"]
        else:
            return ["Source IR", "Transformed IR", "Lowered IR", "Assembly"]

    def create_backend_radio(label: str = "Backend:"):
        """Create a radio button for backend selection."""
        return mo.ui.radio(options=["MLIR", "TVM"], value="MLIR", label=label)

    def create_output_radio(backend_name: str, label: str = "Output options:"):
        """Create a radio button for output options based on backend."""
        return mo.ui.radio(
            options=get_output_options(backend_name),
            value="Assembly",
            label=label
        )

    def execute_editor_code(editor_value: str, display_results_fn=None, initial_namespace=None):
        """
        Execute editor code with stdout/stderr capture.
        Returns (success, output, captured_data).
        - If display_results_fn is provided, it's injected as 'display_results' in the namespace.
        - If initial_namespace is provided, those values are injected before execution.
        - captured_data contains any data captured via display_results.
        """
        captured = {"perf": 0.0}

        def _display_results(perf):
            captured["perf"] = perf

        namespace = dict(initial_namespace) if initial_namespace else {}
        if display_results_fn is not None:
            namespace["display_results"] = _display_results

        code_stderr = StringIO()
        code_stdout = StringIO()

        try:
            with redirect_stderr(code_stderr), redirect_stdout(code_stdout):
                exec(editor_value, namespace)
            output = code_stderr.getvalue() + code_stdout.getvalue()
            return True, output, captured
        except Exception:
            return False, traceback.format_exc(), captured

    def render_editor_output(success: bool, output: str, captured: dict):
        """Render the output of an editor execution as marimo elements."""
        if not success:
            return mo.md(f"**Code error:**\n```\n{output}\n```")

        perf_display = mo.md(f"**Performance:** {captured['perf']:.2f}% of peak")
        code_content = mo.md(f"```asm\n{output}\n```") if output else mo.md("*No IR output.*")
        code_accordion = mo.accordion({"Generated Code": code_content})
        return mo.vstack([perf_display, code_accordion])

    def run_exploration(generator, get_info=None):
        """
        Run exploration with progress bar. Returns sorted results.
        Generator should yield (index, total, sample_or_name, perf) tuples.
        """
        results = []
        best_sample = None
        best_perf = 0.0
        captured_info = {}

        first_result = next(generator, None)
        if first_result is None:
            return [], {}

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

        if get_info:
            captured_info.update(get_info())

        sorted_results = sorted(results, key=lambda x: x["perf"], reverse=True)
        return sorted_results, captured_info, results

    def start_streaming_execution(
        *,
        code: str,
        out: Any,
        throttle_s: float = 0.5,
    ) -> Any:
        """
        Start a marimo Thread that launches a subprocess and streams its output.
        Cancellation: if the spawning cell is invalidated, thread.should_exit becomes True,
        and we terminate the subprocess.
        """
        def target():
            import marimo as mo
            thread = mo.current_thread()

            ctx = mp.get_context("spawn")
            out_q: "mp.Queue" = ctx.Queue()
            p = ctx.Process(
                target=exec_utils._child_exec,
                args=(code, out_q),
                daemon=True
            )
            p.start()

            buf: list[str] = []
            last = 0.0

            def render(force: bool = False):
                nonlocal last
                now = time.time()
                if force or (now - last) >= throttle_s:
                    out.replace(
                        mo.md(
                            f"**Output:**\n\n```text\n{''.join(buf)}\n```"
                        )
                    )
                    last = now

            try:
                out.replace(mo.md("**Output:**\n\n```text\n\n```"))
                render(force=True)

                while True:
                    # Cancel requested? (cell invalidated by Cancel click / rerun / interrupt)
                    if thread.should_exit:
                        buf.append("\n[Cancelled]\n")
                        render(force=True)
                        if p.is_alive():
                            p.terminate()
                            p.join(timeout=1)
                        break

                    # Process ended?
                    if not p.is_alive():
                        # Drain remaining queue chunks
                        while True:
                            try:
                                kind, payload = out_q.get_nowait()
                            except queue.Empty:
                                break
                            if kind == "chunk":
                                buf.append(payload)
                        render(force=True)
                        break

                    # Get output chunk (non-blocking-ish)
                    try:
                        kind, payload = out_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if kind == "chunk":
                        buf.append(payload)
                        render()
                    elif kind == "done":
                        # allow loop to observe process exit / drain remaining data
                        continue
            finally:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1)

        t = mo.Thread(target=target, daemon=True)
        t.start()
        return t

    return (
        create_backend_radio,
        create_output_radio,
        execute_editor_code,
        get_backend_class,
        get_print_opts_dict,
        plt,
        render_editor_output,
        run_exploration,
        start_streaming_execution,
        traceback,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Descript

    XTC allows you to descripte target loop structures for operators using a small DSL called `descript`. Instead of manually specifying each transformation step, you declare the desired final loop structure, and XTC automatically infers the sequence of transformations needed to achieve it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define a schedule declaratively

    `descript` can be used to describe a completed loop structure, with annotations marking optimisations.

    For example, the following loop structure:
    ```
    for i in ... // parallelized
      for k in ...
        for j in ...
          for i1 in range(8): // unrolled
            for j1 in range(16):  // vectorized
    ```

    Can be described as:
    ```python
    '''
    i: parallelize
    k:
    j:
    i#8: unroll
    j#16: vectorize
    '''
    ```

    The loop order follows the given structure (outer to inner), `j#16` creates a tile of size 16 on `j`, and the `vectorize` attribute marks that inner loop for vectorization. Similarly, `parallelize` and `unroll` mark their respective loops for parallelization and (full) unrolling.

    `descript` also has syntax for loop spliting.

    For example:
    ```
    for j in ...:
      for i in ...:
        for k in ...:
          for i1 in range(8):
            for j1 in range(8):
              for i2 in range(4):
          for i3 in range(8, 17):
            for j2 in range(8):
              for i4 in range(3):
    ```

    Can be described as:
    ```python
    j:
    i:
    k:
    i[:8]:
      j#8:
      i#4:
    i[8:16]:
      j#8:
      i#3:
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Practice.** The code below lets you define a schedule specification and see the generated assembly. Try replicating the good-enough schedule you discovered in the previous section!
    """)
    return


@app.cell
def _(mo):
    descript_editor = mo.ui.code_editor(
        value='''import xtc.graphs.xtc.op as O
    from xtc.graphs.xtc.graph import XTCGraph
    from xtc.schedules.descript import descript_scheduler
    from xtc.runtimes.host import HostRuntime

    rt = HostRuntime.get()

    # Problem setup
    I, J, K, dtype = 16, 32, 512, "float32"

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
    schedule_spec = """
    i: parallelize
    k:
    j:
    i#8: unroll
    j#16: vectorize
    """

    # Compile
    scheduler = backend.get_scheduler()
    descript_scheduler(
       scheduler=scheduler,
       node_name="C",
       abstract_dims=["i", "j", "k"],
       spec=schedule_spec
    )
    schedule = scheduler.schedule()

    compiler = backend.get_compiler(dump_file="matmul", shared_lib=True, **print_opts)
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
    return (descript_editor,)


@app.cell
def _(create_backend_radio):
    descript_backend_radio = create_backend_radio()
    return (descript_backend_radio,)


@app.cell
def _(create_output_radio, descript_backend_radio, mo):
    descript_output_radio = create_output_radio(descript_backend_radio.value)
    mo.hstack([descript_backend_radio, descript_output_radio], justify="start", gap=4)
    return (descript_output_radio,)


@app.cell
def _(
    descript_backend_radio,
    descript_editor,
    descript_output_radio,
    execute_editor_code,
    get_backend_class,
    get_print_opts_dict,
    mo,
    render_editor_output,
):
    _namespace = {
        "Backend": get_backend_class(descript_backend_radio.value),
        "print_opts": get_print_opts_dict(descript_output_radio.value),
    }
    _success, _output, _captured = execute_editor_code(descript_editor.value, display_results_fn=True, initial_namespace=_namespace)
    mo.stop(not _success, mo.md(f"**Code error:**\n```\n{_output}\n```"))
    render_editor_output(_success, _output, _captured)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experimenting with Multiple Schedules

    Performance engineering is often about exploring different optimization strategies. Different schedules can have dramatically different performance depending on:
    - **Problem size**: Small matrices may not benefit from parallelization overhead
    - **Hardware**: Cache sizes, vector width, and core count affect optimal tiling
    - **Data layout**: Memory access patterns influence cache efficiency

    In this section, we'll write a simple loop to try several schedule configurations and compare their performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `descript` can also be used to describe loop structures without specifying every tile size. This allows you to easily write a set of strategies to explore.

    For example:
    ```python
    i:
    j:
    k:
    j#16:
    k#32:
    i#i1: unroll
    j#j1: vectorize
    ```
    leaves the sizes of the inner kernel as variables to explore.

    On top of that, variables can also be used to explore configurations for optimizations, for instance: `i#i1: unroll=i_r` allows exploring different unroll sizes, and `j#j1: vectorize=j_v` allows toggling the vectorization.

    Finally, it is also possible to use additional constraints to limit the search space with expert intuitions. For example, limiting the size of the kernel to fit in the CPU registers: `j1+j1*i1 < 128`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Practice.** The code below defines several schedule configurations using our declarative scheduler. You can:

    - **Add/modify configurations**: Try different tile sizes, loop orderings, or optimization combinations. The pre-built search space (variable `configurations` in function `explore` and lines above) may be poorly designed...
    - **Change the acquisition function**: Add caching, error handling, or custom metrics
    - **Modify the exploration loop**: Add early stopping or custom filtering

    The `explore()` function must `yield` tuples of `(index, total, config_name, performance)` for real-time progress display.
    """)
    return


@app.cell
def _(mo):
    explore_schedules_intro = mo.ui.code_editor(
        value=
    '''import xtc.graphs.xtc.op as O
    from xtc.search.strategies import Strategy_Descript as Strategy
    from xtc.graphs.xtc.graph import XTCGraph
    from xtc.backends.tvm import Backend as TVM_Backend
    from xtc.backends.mlir import Backend as MLIR_Backend
    from xtc.runtimes.host import HostRuntime
    from io import StringIO
    from contextlib import redirect_stderr

    rt = HostRuntime.get()

    # Problem setup
    I, J, K, dtype = 16, 32, 512, "float32"

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
    def apply_schedule(strategy: Strategy, graph, backend_cls, sample):
       """Apply a declarative schedule specification and compile."""
       backend = backend_cls(graph)
       scheduler = backend.get_scheduler()
       strategy.generate(scheduler, sample)
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

    def acquire(strategy: Strategy, sample: dict, backend):
       """Evaluate a single configuration and return its performance."""
       module, _ = apply_schedule(strategy, graph, backend, sample)
       return evaluate(module, peak_flops, I*J*K)

    # Run exploration (displays a progress bar and returns sorted results)
    def get_info():
       return {"dims": f"{I}x{J}x{K}", "dtype": dtype}''',
        language="python",
        label=""
    )

    explore_schedules_code = mo.ui.code_editor(
        value=
    '''def explore():
       """Generator that yields (index, total, name, perf) for each evaluation."""
       schedule_spec = """
      i:
      j:
      k:
      j#16:
      k#32: unroll
      i#i1: unroll=i_u
      j#j1: vectorize=j_v
      constraints:
          - j1+j1*i1<128
          - i1*j1 > 16
       """
       strategy = Strategy(graph, schedule_spec, partial_unrolls=False, partial_tiles=False)
       backend = MLIR_Backend
       configurations = list(strategy.sample(1000))
       total = len(configurations)

       for idx, sample in enumerate(configurations):
        perf = acquire(strategy=strategy, sample=sample, backend=backend)
        yield (idx, total, f"{sample}", perf)

    results = run_exploration(explore(), get_info)
    ''',
        language="python",
        label=""
    )
    run_explore_button = mo.ui.run_button(label="Run exploration")
    mo.vstack([explore_schedules_intro, explore_schedules_code, run_explore_button])
    return explore_schedules_code, explore_schedules_intro, run_explore_button


@app.cell
def _(
    explore_schedules_code,
    explore_schedules_intro,
    mo,
    plt,
    run_exploration,
    run_explore_button,
    traceback,
):
    mo.stop(not run_explore_button.value, mo.md("*Click 'Run exploration' to execute the code.*"))

    # Execute the user's code with run_exploration available
    _namespace = {"run_exploration": run_exploration}
    try:
        exec(explore_schedules_intro.value, _namespace)
        exec(explore_schedules_code.value, _namespace)
    except Exception:
        mo.stop(True, mo.md(f"**Code error:**\n```\n{traceback.format_exc()}\n```"))

    _results, _captured_info, _unsorted_results = _namespace.get("results", ([], {}))

    # Build results summary
    if not _results:
        mo.stop(True, mo.md("**Error:** No results returned. Make sure to call `results = run_exploration(explore(), get_info)`"))

    _dims = _captured_info.get("dims", "?")
    _dtype = _captured_info.get("dtype", "?")
    _baseline_perf = _results[-1]["perf"] if _results else 1.0
    _best = _results[0]

    _summary_lines = [
        f"### Schedule Exploration Results",
        f"",
        f"- **Problem:** {_dims} matmul, {_dtype}",
        f"",
        f"**Total configurations evaluated:** {len(_results)}",
        f"",
        f"#### Top 10 configurations:",
        f"",
        f"| Rank | Configuration | Performance | vs Baseline |",
        f"|------|---------------|-------------|-------------|",
    ]

    for _rank, _r in enumerate(_results[:10], 1):
        _speedup = _r["perf"] / _baseline_perf if _baseline_perf > 0 else 0
        _summary_lines.append(f"| {_rank} | {_r['sample']} | {_r['perf']:.2f}% | {_speedup:.2f}x |")

    _summary_lines.extend([
        f"",
        f"#### Best configuration: **{_best['sample']}** ({_best['perf']:.2f}% of peak)",
    ])

    max_perf = _unsorted_results[0]["perf"]
    cumul = [max_perf]
    for d in _unsorted_results[1:]:
        max_perf = max(max_perf, d["perf"])
        if d["perf"] > max_perf:
            raise Exception("HEIN")
        cumul.append(max_perf)


    plt.plot(cumul)
    plt.xlabel('Number of samples')
    plt.ylabel('% of peak performance')
    plt.title('Cumulative best performance')
    plt.grid(True)

    ax = mo.ui.matplotlib(plt.gca())

    mo.vstack([mo.md("\n".join(_summary_lines)), ax])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sandbox

    This place is your playground. It is where you will achieve the greatest challenges,
    using XTC from scratch ;)
    """)
    return


@app.cell
def _(mo):
    sandbox_editor = mo.ui.code_editor(
        value=
    '''\
    # Implement your challenge here!
    print("Hello XTC!")''',
        language="python",
        label=""
    )
    run = mo.ui.run_button(label="Run sandbox")
    cancel = mo.ui.button(label="Stop execution", kind="danger")

    mo.vstack(
        [
            sandbox_editor,
            mo.hstack([run, cancel]),
        ]
    )
    return cancel, run, sandbox_editor


@app.cell
def _(cancel, mo, run, sandbox_editor, start_streaming_execution):
    # This output placeholder is what the background thread updates.
    out = mo.output
    out

    # If cancel is clicked, this cell reruns; that invalidates the previous run-cell,
    # making the old background thread's `should_exit` flip to True, and it will terminate.
    if cancel.value:
        out.replace(mo.md("Cancelled (if something was running, it will stop)."))
    else:
        mo.stop(not run.value, mo.md("*Click 'Run sandbox' to execute the code, and 'Stop execution' to cancel long runs.*"))
        # Start background execution. It will keep streaming until done or cancelled.
        start_streaming_execution(code=sandbox_editor.value, out=out, throttle_s=0.05)
    return


if __name__ == "__main__":
    app.run()
