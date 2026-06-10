import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

with app.setup:
    import os
    import sys
    import marimo as mo
    from sys import platform

#    notebook_dir = os.path.dirname(os.path.realpath(__file__))
#    project_root = os.path.abspath(os.path.join(notebook_dir, "..", ".."))
#    os.chdir(project_root)


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

    **Interactive Workflow:**
    1. Adjust the sliders in the sidebar to update the tile size.
    2. Click the evaluation buttons below each section to run the PMU/TMA counters on demand.
    """)
    return


@app.cell
def _(mo):
    # UI sliders
    tile_i_ui = mo.ui.slider(start=8, stop=1024, step=8, value=16, label="Tile I (Rows)")
    tile_j_ui = mo.ui.slider(start=8, stop=1024, step=8, value=16, label="Tile J (Cols)")
    unroll_ui = mo.ui.slider(start=1, stop=32, step=1, value=2, label="Unroll factor")
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
        "i": {},
        "k": {},
        "j": {},
        f"i#{slider_i}": {"unroll": slider_unroll},
        f"j#{slider_j}": {"vectorize": True}
    }

    # Compile
    scheduler = backend.get_scheduler()
    descript_scheduler(
        scheduler=scheduler,
        node_name="C",
        abstract_dims=["i", "j", "k"],
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
def _(descript_editor, tile_i_ui, tile_j_ui, unroll_ui):
    # Execute user code safely
    local_vars = {}

    # Inject slider values into execution context
    global_vars = globals().copy()
    global_vars.update({
        "slider_i": tile_i_ui.value,
        "slider_j": tile_j_ui.value,
        "slider_unroll": unroll_ui.value,
    })

    exec_error = None
    module = None
    I_val, J_val, K_val = 1024, 2048, 4096

    try:
        exec(descript_editor.value, global_vars, local_vars)
        module = local_vars.get("module")
        I_val = local_vars.get("I", 1024)
        J_val = local_vars.get("J", 2048)
        K_val = local_vars.get("K", 4096)
    except Exception as e:
        exec_error = str(e)
    return I_val, J_val, K_val, exec_error, module


@app.cell
def _(I_val, J_val, K_val, mo, os, sys, tile_i_ui, tile_j_ui, unroll_ui):
    # Fetch hardware cache sizes dynamically (sysfs)
    caches_kb = {}
    if sys.platform == "linux" and os.path.exists("/sys/devices/system/cpu/cpu0/cache"):
        try:
            for i in range(4):
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

    # Defaults if reading fails
    if "L1d" not in caches_kb: caches_kb["L1d"] = 1
    if "L2" not in caches_kb: caches_kb["L2"] = 1
    if len(caches_kb) <= 2 and "L3" not in caches_kb: caches_kb["L3"] = 1

    b_size = 4

    def fmt(b):
        if b >= 1024**2: return f"{b / 1024**2:.1f} MiB"
        if b >= 1024: return f"{b / 1024:.1f} KiB"
        return f"{b} B"

    geo_md = f"""
    | Tensor | Total Size | 1 Row | 1 Col |
    |---|---|---|---|
    | **A** ({I_val}×{K_val}) | {fmt(I_val*K_val*b_size)} | {fmt(K_val*b_size)} | {fmt(I_val*b_size)} |
    | **B** ({K_val}×{J_val}) | {fmt(K_val*J_val*b_size)} | {fmt(J_val*b_size)} | {fmt(K_val*b_size)} |
    | **C** ({I_val}×{J_val}) | {fmt(I_val*J_val*b_size)} | {fmt(J_val*b_size)} | {fmt(I_val*b_size)} |
    """

    tile_c_bytes = tile_i_ui.value * tile_j_ui.value * b_size
    flops = 2 * I_val * J_val * K_val
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
            mo.md("### CPU Caches"),
            mo.md("<br><sub>*Compare row/col sizes to your caches to predict bottlenecks!*</sub>"),
            mo.md(cache_md)
        ])
    )
    return (sidebar,)


@app.cell
def _(sidebar):
    sidebar
    return


@app.cell
def _(mo):
    # On-demand execution buttons
    btn_pmu = mo.ui.run_button(kind="info", label="Evaluate Raw PMUs")
    btn_l1 = mo.ui.run_button(kind="info", label="Evaluate Topdown L1")
    btn_l2 = mo.ui.run_button(kind="info", label="Evaluate Topdown L2")

    _sandbox_default = '[\n    "instructions",\n    "branches",\n    "branch-misses",\n    "L1-dcache-loads",\n    "L1-dcache-load-misses",\n    "TopdownL1",\n    "TopdownL2",\n    "TopdownL3"\n]'
    sandbox_editor = mo.ui.code_editor(value=_sandbox_default, language="python")
    btn_sandbox = mo.ui.run_button(kind="success", label="Run Custom Metrics")
    return btn_l1, btn_l2, btn_pmu, btn_sandbox, sandbox_editor


@app.cell(hide_code=True)
def _(btn_pmu, mo):
    mo.md(f"""
    ## 2. Raw Hardware Counters (PMU)
    CPUs expose raw counters to track specific events. We can ask the CPU exactly how many cycles were spent or how many L1/L2 cache misses occurred.

    *Note: Raw event names are highly architecture-dependent. `libpfm4` helps translating them. The `perf list` command can show you available ones.*

    {btn_pmu}
    """)


@app.cell
def _(btn_pmu, exec_error, mo, module):
    mo.stop(not btn_pmu.value, mo.md("*Click the button above to execute PMU Evaluation.*"))

    if module is None:
        pmu_ui = mo.md(f"**Compilation Error:**\n```python\n{exec_error}\n```")
    else:
        pmu_counters = ["cycles", "instructions", "cache_access", "cache_misses", "branches", "branches_misses"]

        evaluator_pmu = module.get_evaluator(validate=True, pmu_counters=pmu_counters)
        results_pmu, code_pmu, error_pmu = evaluator_pmu.evaluate()

        _pmu_data = [{"Counter": c, "Value": "Fallback needed" if v == -1.0 else str(int(v))} for c, v in zip(pmu_counters, results_pmu)]
        pmu_ui = mo.vstack([
            mo.md(f"**Execution Code:** `{code_pmu}`"),
            mo.ui.table(_pmu_data, label="Raw PMU Results")
        ])

    pmu_ui
    return


@app.cell(hide_code=True)
def _(btn_l1, mo):
    mo.md(f"""
    ## 3. Top-down Microarchitecture Analysis (Level 1)
    Raw counters are hard to interpret. To solve this, **TMA** (Top-down Analysis) groups all CPU pipeline slots into 4 distinct categories, summing up to 100%:

    To solve this, **TMA** (Top-down Analysis) groups all CPU pipeline slots into 4 distinct categories, summing up to 100%:
    *   🟢 **Retiring:** Good! The CPU is doing useful work (executing our math).
    *   🔴 **Bad Speculation:** The CPU guessed a branch wrong and had to flush its pipeline.
    *   🔵 **Frontend Bound:** The CPU is starved; it cannot fetch/decode instructions fast enough.
    *   🟣 **Backend Bound:** The CPU is waiting (usually for Memory or Execution Units) to finish the current instructions.

    {btn_l1}
    """)


@app.cell
def _(btn_l1, exec_error, mo, module, platform):
    mo.stop(not btn_l1.value, mo.md("*Click the button above to execute Topdown L1 Evaluation.*"))

    if module is None:
        l1_ui = mo.md(f"**Compilation Error in Sandbox:**\n```python\n{exec_error}\n```")
    else:
        tma_l1_counters = ["TopdownL1"] if platform == "linux" else []
        evaluator_l1 = module.get_evaluator(validate=False, pmu_counters=tma_l1_counters)
        results_l1, code_l1, error_l1 = evaluator_l1.evaluate()

        _l1_labels = ["🟢 Retiring", "🔴 Bad Speculation", "🔵 Frontend Bound", "🟣 Backend Bound"]

        if tma_l1_counters and len(results_l1) > 0:
            if results_l1[0] < 0:
                 _l1_ui_display = mo.accordion({"Fallback to `perf stat` output": mo.md(f"```text\n{error_l1}\n```")})
            else:
                _l1_data = [{"Category": l, "Percentage (%)": round(v, 2)} for l, v in zip(_l1_labels, results_l1)]
                _l1_ui_table = mo.ui.table(_l1_data, label="Topdown L1 Results")

                try:
                    import altair as alt

                    _chart = alt.Chart(alt.Data(values=_l1_data)).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta(field="Percentage (%)", type="quantitative"),
                        color=alt.Color(
                            field="Category",
                            type="nominal",
                            scale=alt.Scale(
                                domain=_l1_labels,
                                range=["#4ade80", "#ff4a4a", "#60a5fa", "#c084fc"] # Vert, Rouge, Bleu, Violet
                            ),
                            legend=alt.Legend(title="Bottlenecks")
                        ),
                        tooltip=["Category:N", "Percentage (%):Q"]
                    ).properties(width=300, height=300, title="TMA L1 Breakdown")

                    _l1_ui_display = mo.hstack([_l1_ui_table, mo.ui.altair_chart(_chart)], justify="start", gap=4)

                except ImportError:
                    _l1_ui_display = mo.vstack([
                        mo.md("*Tip: Install `altair` (`pip install altair`) to see the interactive pie chart!*"),
                        _l1_ui_table
                    ])
        else:
            _l1_ui_display = mo.md("*TMA L1 is not supported on this machine.*")

        l1_ui = mo.vstack([
            mo.md(f"**Execution Code:** `{code_l1}`"),
            _l1_ui_display
        ])

    l1_ui
    return evaluator_l1, l1_ui, results_l1, tma_l1_counters


@app.cell
def _(btn_l2, mo):
    mo.md(f"""
    ## 4. Drilling Down (Topdown Level 2)
    If our code is heavily **Backend Bound**, we need to know why. TMA Level 2 splits the L1 categories further:

    *   **Core Bound:** Lack of execution units (ALU/FPU) or data dependencies between instructions.
    *   **Memory Bound:** Execution Units are starved because of non-completed in-flight memory demand loads.

    {btn_l2}
    """)
    return


@app.cell
def _(btn_l2, exec_error, mo, module, platform):
    mo.stop(not btn_l2.value, mo.md("*Click the button above to execute Topdown L2 Evaluation.*"))

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
                 _l2_ui_display = mo.vstack([
                     mo.callout(mo.md("*Internal C resolver unsupported for L2. Showing `perf stat` output below.*"), kind="warn"),
                     mo.accordion({"Terminal Output from perf": mo.md(f"```text\n{error_l2}\n```")})
                 ])
            else:
                _l2_data = [{"Sub-Category": l, "Percentage (%)": round(v, 2)} for l, v in zip(_l2_labels, results_l2)]
                _l2_ui_table = mo.ui.table(_l2_data, label="Topdown L2 Results")

                try:
                    import altair as alt2

                    _chart = alt2.Chart(alt2.Data(values=_l2_data)).mark_arc(innerRadius=50).encode(
                        theta=alt2.Theta(field="Percentage (%)", type="quantitative"),
                        color=alt2.Color(
                            field="Sub-Category",
                            type="nominal",
                            scale=alt2.Scale(
                                domain=_l2_labels,
                                range=[
                                    "#4ade80", "#22c55e", # light green, deep green (Retiring)
                                    "#f87171", "#ef4444", # light red, deep red (Bad Spec)
                                    "#60a5fa", "#3b82f6", # light blue, deep blue (Frontend)
                                    "#c084fc", "#a855f7"  # light purple, deep purple (Backend)
                                ]
                            ),
                            legend=alt2.Legend(title="Sub-Bottlenecks")
                        ),
                        tooltip=["Sub-Category:N", "Percentage (%):Q"]
                    ).properties(width=350, height=300, title="TMA L2 Breakdown")

                    _l2_ui_display = mo.hstack([_l2_ui_table, mo.ui.altair_chart(_chart)], justify="start", gap=4)

                except ImportError:
                    _l2_ui_display = mo.vstack([
                        mo.md("*Tip: Install `altair` (`pip install altair`) to see the interactive pie chart!*"),
                        _l2_ui_table
                    ])
        else:
            _l2_ui_display = mo.md("*TMA L2 is not supported on this machine.*")

        l2_ui = mo.vstack([mo.md(f"**Execution Code:** `{code_l2}`"), _l2_ui_display])

    l2_ui
    return evaluator_l2, l2_ui, results_l2, tma_l2_counters


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to fix Core Bound vs Memory Bound?

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
def _(btn_sandbox, mo, sandbox_editor):
    mo.vstack([
        mo.md(r"""
        ---
        ## Sandbox: Experiment with your own metrics

        Curious about branch misses or raw L1 cache loads? Write your own list of PMU/TMA events below.

        > **/!\\ Important Hardware Limits:**

        > CPUs only has a few programmable counters (usually 4 or 8).

        > * If you ask for too many events, the CPU will run out of hardware counters. Linux will silently disable the overflowing ones.

        > * Overflowing events from somes TMA can be handle by a multi-passes mechanic in XTC.*

        > * **Symptom:** You will see mathematically aberrant data, such as `[-0.0, ..., -0.0]` or an absurdly high ones.If you see this, shorten your list!


        *This cell will not run automatically. You must click the button to evaluate the kernel.*
        """),
        sandbox_editor,
        btn_sandbox
    ])
    return


@app.cell
def _(mo):
    tma_support_content = mo.md(
        """
        | Microarchitecture | Supported TMA Levels | Execution Mode |
        |---|---|---|
        | **Intel Skylake / Cascade Lake** | `TopdownL1`, `TopdownL2`, `TopdownL3_Mem` | Native API *(Multi-pass)* |
        | **Intel Modern (Ice Lake+)**     | `TopdownL1`, `TopdownL2` | Native API *(Single-pass)* |
        | **AMD Zen 4**                    | `TopdownL1`, `TopdownL2` | Native API *(Multi-pass)* |
        | **Generic Linux (`perf` tool)**  | `TopdownL1` -> `TopdownL6` | System Fallback *(Multiplexed)* |


        *Note : TopdownL3_Mem return [L1 Bound, L2 Bound, L3 Bound, DRAM Bound, Store Bound]*

        *Metrics beyond L2 or unmapped architectures will automatically use the `perf` fallback.*
        """
    )

    supported_arch_msg = mo.accordion({
        "💡 View supported TMA architectures": tma_support_content
    })
    return (supported_arch_msg,)


@app.cell
def _(mo):
    tma_output_content = mo.md(
        """
        TMA hierarchically breaks down CPU bottlenecks. When the native C API evaluates `TopdownL1` or `TopdownL2`, it returns a raw array of percentages. Here is the exact index mapping:

        ### Level 1 (4 metrics)
        **Array Order:** `[Retiring, Bad Speculation, Frontend Bound, Backend Bound]`

        | Index | Metric | Description |
        |---|---|---|
        | `[0]` | 🟢 **Retiring** | Fraction of slots utilized by useful work (issued uops that get retired). |
        | `[1]` | 🔴 **Bad Speculation** | Fraction of slots wasted due to incorrect speculations. |
        | `[2]` | 🔵 **Frontend Bound** | Fraction of slots where the Frontend undersupplies the Backend. |
        | `[3]` | 🟣 **Backend Bound** | Fraction of slots where no uops are delivered due to a lack of Backend resources. |

        ### Level 2 (8 metrics)
        **Array Order:** `[Light Ops, Heavy Ops, Machine Clears, Branch Mispredicts, Fetch Bandwidth, Fetch Latency, Core Bound, Memory Bound]`

        | Index | Category | Sub-Metric |
        |---|---|---|
        | `[0]` | 🟢 `Retiring` | `light_operations` |
        | `[1]` | 🟢 `Retiring` | `heavy_operations` |
        | `[2]` | 🔴 `Bad Speculation` | `machine_clears` |
        | `[3]` | 🔴 `Bad Speculation` | `branch_mispredicts` |
        | `[4]` | 🔵 `Frontend Bound` | `fetch_bandwidth` |
        | `[5]` | 🔵 `Frontend Bound` | `fetch_latency` |
        | `[6]` | 🟣 `Backend Bound` | `core_bound` |
        | `[7]` | 🟣 `Backend Bound` | `memory_bound` |

        ### Level 3 (26 metrics)
        Deep dive into L2 categories. Key metrics returned in the array include:
        * **Memory Subsystem:** `l1_bound`, `l2_bound`, `l3_bound`, `dram_bound`, `store_bound`, `pmm_bound`
        * **Frontend Fetch:** `dsb` (decoded uop cache), `mite` (legacy decode), `icache_misses`, `itlb_misses`, `lcp`
        * **Execution & ALU:** `fp_arith`, `ports_utilization`, `divider`, `serializing_operation`
        * **Retiring Details:** `memory_operations`, `fused_instructions`, `microcode_sequencer`
        * **Branch Prediction:** `branch_resteers`, `other_mispredicts`, `other_nukes`

        ### Level 4 (32 metrics)
        Granular details on memory behavior and port utilization.
        * **Memory Bandwidth & Latency:** `mem_bandwidth`, `mem_latency`, `l1_hit_latency`, `l3_hit_latency`
        * **Memory Bottlenecks:** `split_loads`, `split_stores`, `4k_aliasing`, `dtlb_load`, `dtlb_store`, `false_sharing`
        * **Port Utilization:** `ports_utilized_0`, `ports_utilized_1`, `ports_utilized_2`, `ports_utilized_3m`
        * **Operations:** `fp_scalar`, `fp_vector`, `nop_instructions`, `x87_use`, `cisc`, `assists`

        ### Level 5 (15 metrics)
        Specific unit utilization and Translation Lookaside Buffer (TLB) deep dives.
        * **Operations:** `alu_op_utilization`, `load_op_utilization`, `store_op_utilization`
        * **Vector Widths:** `fp_vector_128b`, `fp_vector_256b`, `fp_vector_512b`
        * **STLB:** `load_stlb_hit/miss`, `store_stlb_hit/miss`
        * **Memory/Cache:** `local_mem`, `remote_mem`, `remote_cache`

        ### Level 6 (8 metrics)
        * **`port_0` to `port_7`:** Fraction of cycles the CPU dispatched uops on specific hardware execution ports (e.g., ALU, Loads, Stores, Branches).
        """
    )

    reminder_output_msg = mo.accordion({
        "💡 Output reminder & Topdown Metrics Dictionary": tma_output_content
    })
    return (reminder_output_msg,)


@app.cell
def _(reminder_output_msg):
    reminder_output_msg
    return


@app.cell
def _(supported_arch_msg):
    supported_arch_msg
    return


@app.cell
def _(btn_sandbox, mo, module, sandbox_editor):
    mo.stop(not btn_sandbox.value, mo.md("*Click 'Run Custom Metrics' to see the results.*"))

    if module is None:
        sandbox_output = mo.md("**Module is not compiled.** Please fix the compilation sandbox above first.")
    else:
        import ast
        try:
            custom_counters = ast.literal_eval(sandbox_editor.value)
            if not isinstance(custom_counters, list): raise ValueError("Input must be a Python list.")

            evaluator_sb = module.get_evaluator(validate=False, pmu_counters=custom_counters)
            results_sb, code_sb, err_sb = evaluator_sb.evaluate()

            output_lines = [f"**Execution Code:** `{code_sb}`\n"]


            output_lines.append("**Raw Results:**\n```text")


            DERIVED_METRICS_SIZES = {"TopdownL1": 4, "TopdownL2": 8, "TopdownL3": 26, "TopdownL4": 32, "TopdownL5": 15, "TopdownL6": 8}
            current_idx = 0

            for c in custom_counters:
                size = DERIVED_METRICS_SIZES.get(c, 1)
                chunk = results_sb[current_idx : current_idx + size]

                if size == 1:
                    output_lines.append(f"{c:35} : {int(chunk[0])}")
                else:
                    rounded_chunk = [round(x, 2) for x in chunk]
                    output_lines.append(f"{c:35} : {rounded_chunk}")
                current_idx += size

            output_lines.append("```")

            if err_sb:
                output_lines.append(f"**Terminal Output / Errors:**\n```text\n{err_sb}\n```")

            sandbox_output = mo.md("\n".join(output_lines))

        except Exception as e:
            sandbox_output = mo.md(f"**Error parsing your list:** {e}")

    sandbox_output
    return


if __name__ == "__main__":
    app.run()
