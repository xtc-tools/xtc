import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

with app.setup:
    import os
    import sys
    import marimo as mo
    from sys import platform

    notebook_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(notebook_dir, "..", ".."))
    os.chdir(project_root)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hardware Performance Counters & Top-down Analysis

    Optimizing code requires understanding exactly how the CPU executes it. While theoretical complexity (Big O) is useful, modern CPUs are incredibly complex beasts with deep pipelines, multiple cache levels, and speculative execution.

    In this notebook, we will use **XTC** to compile a Matrix Multiplication and evaluate it using:
    1. **Raw PMU Counters** (Performance Monitoring Units) to count exact hardware events.
    2. **Top-down Microarchitecture Analysis (TMA)** to identify actual bottlenecks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Disclaimer and prerequisites:**
    > Results will vary depending on your hardware architecture (MacOS is currently not supported).
    > If your microarchitecture is not explicitly mapped by the internal C resolver, the system will gracefully fallback to the `perf stat` command-line tool.
    >
    > To make hardware counters available to userspace applications (ring 3), run this in your terminal:
    > ```bash
    > sudo sysctl kernel.perf_event_paranoid=-1
    > echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog
    > ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Defining the Workload & Schedule
    We define a medium-sized matrix multiplication (1024x2048x4096).

    **Use the code editor below** to modify the scheduling specification using the `descript` notation.

    **Interactive Workflow:**
    1. Adjust the sliders in the sidebar to see how the Cache Footprint updates instantly.
    2. Click the **"Compile & Evaluate"** button in the sidebar to apply the schedule and update the TMA metrics.
    """)
    return


@app.cell
def _(mo):
    # Interactive UI elements for the schedule
    tile_i_ui = mo.ui.slider(start=1, stop=64, step=1, value=3, label="Tile I (Rows)")
    tile_j_ui = mo.ui.slider(start=8, stop=512, step=8, value=24, label="Tile J (Cols)")
    unroll_ui = mo.ui.slider(start=1, stop=8, step=1, value=2, label="Unroll factor")
    return tile_i_ui, tile_j_ui, unroll_ui


@app.cell
def _(mo):
    _editor_code = '''import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.schedules.descript import descript_scheduler

    # Problem setup
    I, J, K, dtype = 1024, 2048, 4096, "float32"

    a = O.tensor((I, K), dtype, name="A")
    b = O.tensor((K, J), dtype, name="B")

    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="C")

    backend = Backend(gb.graph)

    # Schedule specification
    schedule_spec = {
        "j": {},
        "i": {},
        "k": {},
        f"j#{slider_j}": {"unroll": slider_unroll},
        f"i#{slider_i}": {"vectorize": True}
    }

    scheduler = backend.get_scheduler()
    descript_scheduler(
        scheduler=scheduler,
        node_name="C",
        abstract_dims=["j", "i", "k"],
        spec=schedule_spec
    )
    sched = scheduler.schedule()

    compiler = backend.get_compiler(dump_file="matmul_mlir", shared_lib=True)
    module = compiler.compile(sched)
    '''
    descript_editor = mo.ui.code_editor(
        value=_editor_code,
        language="python",
        label="Schedule & Compilation Sandbox"
    )
    return (descript_editor,)


@app.cell
def _(descript_editor):
    descript_editor
    return


@app.cell
def _(descript_editor, mo, tile_i_ui, tile_j_ui, unroll_ui):
    # State management to isolate heavy computation from slider movement
    compile_params, set_compile_params = mo.state(None)

    def _trigger_compile(_):
        set_compile_params({
            "code": descript_editor.value,
            "i": tile_i_ui.value,
            "j": tile_j_ui.value,
            "unroll": unroll_ui.value
        })

    compile_btn = mo.ui.button(
        label="Compile & Evaluate",
        kind="success",
        on_click=_trigger_compile
    )
    return compile_btn, compile_params, set_compile_params


@app.cell
def _(compile_params, mo):
    mo.stop(compile_params() is None, mo.md("*👈 Click **'Compile & Evaluate'** in the sidebar to start.*"))

    _params = compile_params()
    local_vars = {}

    global_vars = globals().copy()
    global_vars.update({
        "slider_i": _params["i"],
        "slider_j": _params["j"],
        "slider_unroll": _params["unroll"],
    })

    exec_error = None
    module = None

    try:
        exec(_params["code"], global_vars, local_vars)
        module = local_vars.get("module")
    except Exception as e:
        exec_error = str(e)

    return exec_error, module


@app.cell
def _(compile_btn, mo, os, sys, tile_i_ui, tile_j_ui, unroll_ui):
    caches_kb = {}
    if sys.platform == "linux" and os.path.exists("/sys/devices/system/cpu/cpu0/cache"):
        try:
            for i in range(4): # Usually index0 to index3 (L1d, L1i, L2, L3)
                path = f"/sys/devices/system/cpu/cpu0/cache/index{i}"
                if os.path.exists(path):
                    with open(f"{path}/level", "r") as f:
                        level = int(f.read().strip())
                    with open(f"{path}/type", "r") as f:
                        ctype = f.read().strip()
                    with open(f"{path}/size", "r") as f:
                        size_str = f.read().strip()

                    s_kb = 0
                    if size_str.endswith('K'): s_kb = int(size_str[:-1])
                    elif size_str.endswith('M'): s_kb = int(size_str[:-1]) * 1024
                    else: s_kb = int(size_str) // 1024

                    if level == 1 and ctype == "Data": caches_kb["L1d"] = s_kb
                    elif level == 2: caches_kb["L2"] = s_kb
                    elif level == 3: caches_kb["L3"] = s_kb
        except Exception:
            pass

    # Fallbacks in case reading failed
    if "L1d" not in caches_kb: caches_kb["L1d"] = 1
    if "L2" not in caches_kb: caches_kb["L2"] = 1
    if len(caches_kb) <= 2 and "L3" not in caches_kb: caches_kb["L3"] = 1

    _I, _J, _K = 1024, 2048, 4096
    b_size = 4 # float32
    # todo : check element size dynamicly

    def fmt(b):
        if b >= 1024**2: return f"{b / 1024**2:.1f} MiB"
        if b >= 1024: return f"{b / 1024:.1f} KiB"
        return f"{b} B"

    geo_md = f"""
    | Tensor | Total Size | 1 Row | 1 Col |
    |---|---|---|---|
    | **A** ({_I}×{_K}) | {fmt(_I*_K*b_size)} | {fmt(_K*b_size)} | {fmt(_I*b_size)} |
    | **B** ({_K}×{_J}) | {fmt(_K*_J*b_size)} | {fmt(_J*b_size)} | {fmt(_K*b_size)} |
    | **C** ({_I}×{_J}) | {fmt(_I*_J*b_size)} | {fmt(_J*b_size)} | {fmt(_I*b_size)} |
    """

    tile_c_bytes = tile_i_ui.value * tile_j_ui.value * b_size
    flops = 2 * _I * _J * _K
    gflops = flops / 1e9

    def format_cache(name, kb):
        mb = kb / 1024
        if kb >= 1024: return f"**{name}** : {mb:.1f} MiB"
        return f"**{name}** : {kb:.0f} KiB"

    cache_lines = [
        format_cache("L1 Data", caches_kb.get("L1d", 32)),
        format_cache("L2 Cache", caches_kb.get("L2", 1024))
    ]
    if "L3" in caches_kb:
        cache_lines.append(format_cache("L3 Cache", caches_kb["L3"]))

    cache_md = "<br>".join([f" {l}" for l in cache_lines])

    sidebar = mo.sidebar(
        mo.vstack([
            compile_btn,
            mo.md("---"),
            mo.md("### Schedule Controls"),
            tile_i_ui,
            tile_j_ui,
            unroll_ui,
            mo.md("---"),
            mo.md("### Problem Geometry"),
            mo.md(geo_md),
            mo.md(f"**Total Math:** `{gflops:.1f} GFLOPs`"),
            mo.md(f"**Current C-Tile (i×j):** `{fmt(tile_c_bytes)}`"),
            mo.md("---"),
            mo.md("### Your CPU Caches"),
            mo.md(cache_md),
            mo.md("<br><sub>*Compare row/col sizes to your caches to predict bottlenecks!*</sub>")
        ])
    )
    return (sidebar,)


@app.cell
def _(sidebar):
    sidebar
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Raw Hardware Counters (PMU)
    CPUs expose raw counters to track specific events. We can ask the CPU exactly how many cycles were spent or how many L1/L2 cache misses occurred.

    *Note: Raw event names are highly architecture-dependent. `libpfm4` helps translating them. The `perf list` command can show you available ones.*
    """)
    return


@app.cell
def _(exec_error, mo, module, platform):
    if module is None:
        pmu_ui = mo.md(f"**Compilation Error in Sandbox:**\n```python\n{exec_error}\n```")
    else:
        pmu_counters = ["cycles", "instructions"]

        if platform == "linux":
            pmu_counters += [
                "mem_load_retired.l1_miss",
                "mem_load_retired.l2_miss"
            ]

        evaluator_pmu = module.get_evaluator(
            validate=True,
            pmu_counters=pmu_counters,
        )
        results_pmu, code_pmu, error_pmu = evaluator_pmu.evaluate()

        _pmu_data = [{"Counter": c, "Value": int(v)} for c, v in zip(pmu_counters, results_pmu)]
        pmu_ui = mo.vstack([
            mo.md(f"**Execution Code:** `{code_pmu}`"),
            mo.ui.table(_pmu_data, label="Raw PMU Results")
        ])
    return evaluator_pmu, pmu_counters, pmu_ui, results_pmu


@app.cell
def _(pmu_ui):
    pmu_ui
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Top-down Microarchitecture Analysis (Level 1)
    Raw counters are hard to interpret: *Is 5 million cache misses bad? Does it actually stall the CPU?*

    To solve this, **TMA** (Top-down Analysis) groups all CPU pipeline slots into 4 distinct categories, summing up to 100%:
    *   🟢 **Retiring:** Good! The CPU is doing useful work (executing our math).
    *   🔴 **Bad Speculation:** The CPU guessed a branch wrong and had to flush its pipeline.
    *   🔵 **Frontend Bound:** The CPU is starved; it cannot fetch/decode instructions fast enough.
    *   🟣 **Backend Bound:** The CPU is waiting (usually for Memory or Execution Units) to finish the current instructions.
    """)
    return


@app.cell
def _(exec_error, mo, module, platform):
    if module is None:
        l1_ui = mo.md(f"**Compilation Error in Sandbox:**\n```python\n{exec_error}\n```")
    else:
        tma_l1_counters = ["TopdownL1"] if platform == "linux" else []
        evaluator_l1 = module.get_evaluator(validate=False, pmu_counters=tma_l1_counters)
        results_l1, code_l1, error_l1 = evaluator_l1.evaluate()

        _l1_labels = ["Retiring", "Bad Speculation", "Frontend Bound", "Backend Bound"]

        if tma_l1_counters and len(results_l1) > 0:
            if results_l1[0] < 0:
                 _l1_ui_table = mo.accordion({
                     "Fallback to `perf stat` output": mo.md(f"```text\n{error_l1}\n```")
                 })
            else:
                _l1_data = [{"Category": l, "Percentage (%)": round(v, 2)} for l, v in zip(_l1_labels, results_l1)]
                _l1_ui_table = mo.ui.table(_l1_data, label="Topdown L1 Results")
        else:
            _l1_ui_table = mo.md("*TMA L1 is not supported on this machine.*")

        l1_ui = mo.vstack([
            mo.md(f"**Execution Code:** `{code_l1}`"),
            _l1_ui_table
        ])
    return code_l1, error_l1, evaluator_l1, l1_ui, results_l1, tma_l1_counters


@app.cell
def _(l1_ui):
    l1_ui
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Drilling Down (Topdown Level 2)
    If our code is heavily **Backend Bound**, we need to know why! Is it because our math is too complex for the ALU/FPU (`Core Bound`), or are we constantly waiting for the RAM (`Memory Bound`)?

    TMA Level 2 splits the L1 categories further. The Backend Bound is split into:

    *   **Core Bound:** The pipeline is likely stalled due to a lack of available execution units (ALU/FPU) or data dependencies between instructions preventing out-of-order execution.
    *   **Memory Bound:** The pipeline is likely stalled due to demand load/store instructions. This represents the fraction of slots where Execution Units are starved because of non-completed in-flight memory demand loads, or occasionally when store buffers are completely full imposing backpressure.
    """)
    return


@app.cell
def _(exec_error, mo, module, platform):
    if module is None:
        l2_ui = mo.md(f"**Compilation Error in Sandbox:**\n```python\n{exec_error}\n```")
    else:
        tma_l2_counters = ["TopdownL2"] if platform == "linux" else []
        evaluator_l2 = module.get_evaluator(validate=False, pmu_counters=tma_l2_counters)
        results_l2, code_l2, error_l2 = evaluator_l2.evaluate()

        _l2_labels = [
            "🟢 Retiring: Light Ops",
            "🟢 Retiring: Heavy Ops",
            "🔴 Bad Spec: Machine Clears",
            "🔴 Bad Spec: Branch Mispredicts",
            "🔵 Frontend: Fetch Bandwidth",
            "🔵 Frontend: Fetch Latency",
            "🟣 Backend: Core Bound",
            "🟣 Backend: Memory Bound"
        ]

        if tma_l2_counters and len(results_l2) > 0:
            if results_l2[0] < 0:
                 _l2_ui_table = mo.vstack([
                     mo.callout(mo.md("⚠️ *Internal C resolver unsupported for L2. Showing `perf stat` output below.*"), kind="warn"),
                     mo.accordion({"Terminal Output from `perf`": mo.md(f"```text\n{error_l2}\n```")})
                 ])
            else:
                _l2_data = [{"Sub-Category": l, "Percentage (%)": round(v, 2)} for l, v in zip(_l2_labels, results_l2)]
                _l2_ui_table = mo.ui.table(_l2_data, label="Topdown L2 Results")
        else:
            _l2_ui_table = mo.md("*TMA L2 is not supported on this machine.*")

        l2_ui = mo.vstack([
            mo.md(f"**Execution Code:** `{code_l2}`"),
            _l2_ui_table
        ])
    return code_l2, error_l2, evaluator_l2, l2_ui, results_l2, tma_l2_counters


@app.cell
def _(l2_ui):
    l2_ui
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to fix Core Bound vs Memory Bound?

    Once you know where your bottleneck is, you can adapt your code or your MLIR schedule:

    ### If you are heavily Core Bound:
    Your CPU has all the data it needs in the caches, but it cannot crunch the numbers fast enough.
    *   **Vectorization:** Ensure your loops are properly vectorized so the CPU processes 8 or 16 floats per instruction (AVX2/AVX-512) instead of 1.
    *   **Instruction-Level Parallelism:** Increase the `unroll` factor. This breaks data dependency chains and allows the CPU's Out-Of-Order engine to execute multiple independent additions/multiplications at the exact same time.

    ### If you are heavily Memory Bound:
    Your CPU's execution units are starving because they are waiting hundreds of cycles for data to arrive from RAM.
    *   **Cache Locality:** Modify your `tile` sizes (Tiling/Blocking). The goal is to make sure a chunk of Matrix A and Matrix B fits perfectly inside the ultra-fast L1 or L2 cache before computing it.
    *   **Memory Access Pattern:** Ensure contiguous memory accesses. If your code jumps around memory (strided access), the hardware prefetcher cannot predict what to load next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Going deeper: Topdown L3, L4 and beyond...

    Topdown Level 2 is great, but it sometimes leaves us with more questions. If you are 60% *Memory Bound*, you might wonder: *"Am I bound by the L1 cache, the L3 cache, or the Main Memory (DRAM)?"*

    This is where Topdown Level 3 and Level 4 come in (often requiring specific `perf stat -M` metrics depending on your architecture).
    For the moment depending of your microarchitecture XTC only support up to L2 topdown, the linux perf tool will be needed to go futher.

    The **Memory Bound** category further splits into:
    1.  **L1 Bound (L3):** Data is not in registers, but found extremely quickly in L1 cache. Often caused by high memory latency per instruction or bank conflicts.
    2.  **L2 Bound (L3):** Data missed L1 but was found in L2.
    3.  **L3 Bound (L3):** Data missed L1/L2 but was found in L3.
    4.  **Ext. Memory Bound (L3):** Data missed ALL caches and had to be fetched from Main RAM. This is catastrophic for performance!
        *    *Splits into L4:* **Mem Bandwidth** (the memory bus is saturated) vs **Mem Latency** (waiting for the RAM chip to respond).
    5.  **Store Bound (L3):** The CPU's store buffers are full because it is writing too much data to memory too quickly.

    > *Try increasing the `tile` values in the code sandbox to something huge. You will see the **Memory Bound** spike because the data chunk no longer fits in the fast caches!*
    """)
    return


@app.cell
def _(mo):
    _sandbox_default = '''[
    "instructions",
    "branches",
    "branch-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses"
]'''
    sandbox_editor = mo.ui.code_editor(
        value=_sandbox_default,
        language="python",
    )

    sandbox_form = sandbox_editor.form(submit_button_label=" Run Custom Metrics")
    return sandbox_editor, sandbox_form


@app.cell
def _(mo, sandbox_form):
    mo.vstack([
        mo.md(r"""
        ---
        ## Sandbox: Experiment with your own metrics

        Curious about branch misses or raw L1 cache loads? Write your own list of PMU/TMA events below.

        *Temporary TMA metrics must be evaluate one at time without be mix up with PMU*

        *This cell will not run automatically. You must click the button to evaluate the kernel.*
        """),
        sandbox_form
    ])
    return


@app.cell
def _(mo, module, sandbox_form):
    mo.stop(sandbox_form.value is None, mo.md("*Click 'Run Custom Metrics' to see the results.*"))

    if module is None:
        sandbox_output = mo.md("**Module is not compiled.** Please fix the compilation sandbox above first.")
    else:
        import ast
        try:
            # Safely parse the user's string into a Python list
            custom_counters = ast.literal_eval(sandbox_form.value)
            if not isinstance(custom_counters, list):
                raise ValueError("Input must be a Python list.")

            evaluator_sb = module.get_evaluator(validate=False, pmu_counters=custom_counters)
            results_sb, code_sb, err_sb = evaluator_sb.evaluate()

            output_lines = [f"**Execution Code:** `{code_sb}`"]

            if err_sb:
                output_lines.append(f"**Terminal Output / Errors:**\n```text\n{err_sb}\n```")

            output_lines.append("**Raw Results:**\n```text")
            for c, v in zip(custom_counters, results_sb):
                output_lines.append(f"{c:35} : {v}")
            output_lines.append("```")

            sandbox_output = mo.md("\n".join(output_lines))

        except Exception as e:
            sandbox_output = mo.md(f"**Error parsing your list:** {e}")

    return custom_counters, evaluator_sb, output_lines, results_sb, sandbox_output


@app.cell
def _(sandbox_output):
    sandbox_output
    return


if __name__ == "__main__":
    app.run()
