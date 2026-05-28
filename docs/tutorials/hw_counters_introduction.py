import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

with app.setup:
    import os
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
    > ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Defining the Workload & Schedule (Interactive)
    First, we define a medium-sized matrix multiplication (1024x2048x4096).

    **Play with the sliders below!** Modifying the tiling sizes or the unroll factor will dynamically recompile the MLIR code and update the hardware counters in real-time. You will see the bottlenecks shift!
    (default: i=3 j=24 unroll=2)

    """)
    return


@app.cell
def _(mo):
    # Interactive UI elements for the schedule
    tile_i_ui = mo.ui.slider(start=1, stop=16, step=1, value=3, label="Tile I (Rows)")
    tile_j_ui = mo.ui.slider(start=8, stop=128, step=8, value=24, label="Tile J (Cols)")
    unroll_ui = mo.ui.slider(start=1, stop=8, step=1, value=2, label="Unroll factor")

    schedule_ui = mo.hstack([tile_i_ui, tile_j_ui, unroll_ui])
    return schedule_ui, tile_i_ui, tile_j_ui, unroll_ui


@app.cell
def _(schedule_ui):
    schedule_ui
    return


@app.cell
def _(tile_i_ui, tile_j_ui, unroll_ui):
    import xtc.graphs.xtc.op as O
    from xtc.backends.mlir import Backend

    # 1024x2048x4096 is large enough to show memory bottlenecks,
    # but small enough to evaluate quickly.
    I, J, K, dtype = 1024, 2048, 4096, "float32"

    a = O.tensor((I, K), dtype, name="A")
    b = O.tensor((K, J), dtype, name="B")

    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="C")

    # Scheduling using interactive values
    impl = Backend(gb.graph)
    sch = impl.get_scheduler()
    sch.tile("i", {"i1": tile_i_ui.value})
    sch.tile("j", {"j1": tile_j_ui.value})
    sch.interchange(["k", "i", "j", "i1", "j1"])
    sch.vectorize(["j1"])
    sch.unroll({"i1": unroll_ui.value})
    sched = sch.schedule()

    # Compilation
    comp = impl.get_compiler(shared_lib=True, dump_file="matmul_mlir")
    module = comp.compile(sched)
    return I, J, K, a, b, comp, dtype, gb, impl, module, sch, sched


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Raw Hardware Counters (PMU)
    CPUs expose raw counters to track specific events. We can ask the CPU exactly how many cycles were spent or how many L1/L2 cache misses occurred.

    *Note: Raw event names are highly architecture-dependent. `libpfm4` helps translating them.*
    """)
    return


@app.cell
def _(module, mo, platform):
    pmu_counters = ["cycles", "instructions"]

    # Adding architecture-specific counters (Linux/x86 usually)
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

    # Formatting PMU results
    _pmu_data = [{"Counter": c, "Value": int(v)} for c, v in zip(pmu_counters, results_pmu)]

    pmu_ui = mo.vstack([
        mo.md(f"**Execution Code:** `{code_pmu}`"),
        mo.ui.table(_pmu_data, label="Raw PMU Results")
    ])
    return code_pmu, error_pmu, evaluator_pmu, pmu_counters, pmu_ui, results_pmu


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
def _(module, mo, platform):
    tma_l1_counters = ["TopdownL1"] if platform == "linux" else []

    evaluator_l1 = module.get_evaluator(
        validate=False,
        pmu_counters=tma_l1_counters,
    )

    results_l1, code_l1, error_l1 = evaluator_l1.evaluate()

    _l1_labels = ["Retiring", "Bad Speculation", "Frontend Bound", "Backend Bound"]

    if tma_l1_counters and len(results_l1) >= 4:
        _l1_data = [{"Category": l, "Percentage (%)": round(v, 2)} for l, v in zip(_l1_labels, results_l1)]
        _l1_ui_table = mo.ui.table(_l1_data, label="Topdown L1 Results")
    else:
        _l1_ui_table = mo.md("*TMA L1 is not supported or returned empty on this machine.*")

    l1_ui = mo.vstack([
        mo.md(f"**Execution Code:** `{code_l1}`"),
        _l1_ui_table,
        mo.md("> *For a Matrix Multiplication, you should expect a high **Backend Bound** (waiting for RAM/Caches) and a solid **Retiring** percentage.*")
    ])
    return code_l1, error_l1, evaluator_l1, l1_ui, results_l1, tma_l1_counters, tma_l1_counters


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
def _(module, mo, platform):
    tma_l2_counters = ["TopdownL2"] if platform == "linux" else []

    evaluator_l2 = module.get_evaluator(
        validate=False,
        pmu_counters=tma_l2_counters,
    )

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

    _fallback_msg = mo.md("")

    if tma_l2_counters and len(results_l2) >= 8:
        # Check if the fallback hit (returns -1 on index 0 when internal resolver fails)
        if results_l2[0] < 0:
             _l2_ui_table = mo.md("⚠️ *Internal C resolver unsupported for L2. System gracefully fell back to `perf stat` output in your terminal.*")
             _fallback_msg = mo.callout(
                 mo.md("### How to read the `perf stat` fallback terminal output\n"
                       "Since your CPU requires too many events for a single pass, `perf` multiplexes the hardware. "
                       "Look at your terminal: `perf` automatically computes the percentages next to raw values. "
                       "Look for indented metrics like `core_bound` and `memory_bound` underneath `backend_bound`."),
                 kind="warn"
             )
        else:
            _l2_data = [{"Sub-Category": l, "Percentage (%)": round(v, 2)} for l, v in zip(_l2_labels, results_l2)]
            _l2_ui_table = mo.ui.table(_l2_data, label="Topdown L2 Results")
    else:
        _l2_ui_table = mo.md("*TMA L2 is not supported or returned empty on this machine.*")

    l2_ui = mo.vstack([
        mo.md(f"**Execution Code:** `{code_l2}`"),
        _l2_ui_table,
        _fallback_msg
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
    *   **Action 1 (Vectorization):** Ensure your loops are properly vectorized so the CPU processes 8 or 16 floats per instruction (AVX2/AVX-512) instead of 1.
    *   **Action 2 (Instruction-Level Parallelism):** Increase the `Unroll` factor. This breaks data dependency chains and allows the CPU's Out-Of-Order engine to execute multiple independent additions/multiplications at the exact same time.

    ### If you are heavily Memory Bound:
    Your CPU's execution units are starving because they are waiting hundreds of cycles for data to arrive from RAM.
    *   **Action 1 (Cache Locality):** Modify your `Tile` sizes (Tiling/Blocking). The goal is to make sure a chunk of Matrix A and Matrix B fits perfectly inside the ultra-fast L1 or L2 cache before computing it.
    *   **Action 2 (Memory Access Pattern):** Ensure contiguous memory accesses. If your code jumps around memory (strided access), the hardware prefetcher cannot predict what to load next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Going deeper: Topdown L3, L4 and beyond...

    Topdown Level 2 is great, but it sometimes leaves us with more questions. If you are 60% *Memory Bound*, you might wonder: *"Am I bound by the L1 cache, the L3 cache, or the Main Memory (DRAM)?"*

    This is where Topdown Level 3 and Level 4 come in (often requiring specific `perf stat -M` metrics depending on your architecture).

    The **Memory Bound** category further splits into:
    1.  **L1 Bound (L3):** Data is not in registers, but found extremely quickly in L1 cache. Often caused by high memory latency per instruction or bank conflicts.
    2.  **L2 Bound (L3):** Data missed L1 but was found in L2.
    3.  **L3 Bound (L3):** Data missed L1/L2 but was found in L3.
    4.  **Ext. Memory Bound (L3):** Data missed ALL caches and had to be fetched from Main RAM. This is catastrophic for performance!
        *   ➡️ *Splits into L4:* **Mem Bandwidth** (the memory bus is saturated) vs **Mem Latency** (waiting for the RAM chip to respond).
    5.  **Store Bound (L3):** The CPU's store buffers are full because it is writing too much data to memory too quickly.

    > *Try increasing the Tile sizes in the first cell to something huge (e.g., Tile J = 128). You will see the **Memory Bound** spike because the data chunk no longer fits in the fast L1 cache!*
    """)
    return


if __name__ == "__main__":
    app.run()
