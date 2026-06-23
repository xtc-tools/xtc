"""
Schema architecture global (caches,front,back...) coloré en fonctions des métriques
Exemple de code de recuperation de compteur avec l'API XTC (une seule partie éditable?)
Tuilage imposé progressif avec le tuto (naif -> ... -> maxi tuilé opti)
Exemple de raté ? (4k aliasing)
Taille de tuile éditable -> possibilité de rentrer les chiffres
"""


import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

with app.setup:
    import os
    import sys
    import marimo as mo
    from sys import platform


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hardware Performance Counters & Top-down Analysis

    Optimizing code requires understanding exactly how the CPU executes it.

    In this notebook, we will use **XTC** to compile a Matrix Multiplication and evaluate it using hardware counters and Top-down Microarchitecture Analysis (TMA)
    """)
    return


@app.cell(hide_code=True)
def _(mo, platform):
    _warnings = []

    if platform == "darwin":
        _warnings.append(mo.callout(mo.md("**MacOS Detected:** Hardware counters are restricted. The notebook will run, but TMA metrics might be unavailable or require `sudo` privileges."), kind="danger"))

    _warnings.append(mo.callout(mo.md(r"""
    **Hardware Counters Prerequisites:**
    To allow userspace applications (ring 3) to read CPU PMU counters on Linux, please run the following commands in your terminal:
    ```bash
    sudo sysctl kernel.perf_event_paranoid=-1
    echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog
    ```

    *Note : Reloading this page may be required*
    """), kind="warn"))

    prerequisites_ui = mo.vstack(_warnings)
    return prerequisites_ui,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ---

        For a better experience, it's recommended to be familiar with the XTC's `descript` syntax and the general usage of XTC.

        ---
    """)
    return

# µpipe Sankey diagram
@app.cell(hide_code=True)
def _(mo):
    def plot_upipe(l1_res, l2_res=None, l3_res=None):
        """Generates a hierarchical Sankey diagram for TMA."""
        if not l1_res or len(l1_res) < 4 or l1_res[0] < 0:
            return mo.md("⚠️ **TMA Data not available.**")

        try:
            import plotly.graph_objects as go
        except ImportError:
            return mo.callout(mo.md(" **Please install `plotly` (`pip install plotly`) to visualize the Sankey diagram**"), kind="warn")

        try:
            # Color palettes (L1, L2, L3)
            c_ret = "#4ade80" # Green
            c_bad = "#f87171" # Red
            c_fe  = "#60a5fa" # Blue
            c_be  = "#c084fc" # Purple

            desc_l1 = {
                "Retiring": "Fraction of slots utilized by useful work.",
                "Bad Spec": "Fraction of slots wasted due to incorrect speculations.",
                "Frontend": "Fraction of slots where the Frontend undersupplies the Backend.",
                "Backend": "Fraction of slots where no uops are delivered due to a lack of Backend resources."
            }
            desc_l2 = {
                "Light Ops": "Retiring typical, single-uop instructions.",
                "Heavy Ops": "Retiring complex, multi-uop or microcoded instructions.",
                "Clears": "Wasted slots due to pipeline flushes.",
                "Branch Misp": "Wasted slots due to incorrect branch predictions.",
                "Fetch BW": "Frontend cannot decode or deliver enough instructions per cycle.",
                "Fetch Lat": "Frontend is completely starved and delivering nothing.",
                "Core Bound": "Execution stalled waiting for execution units (ALU/FPU).",
                "Mem Bound": "Execution stalled waiting for data from the memory (L1/L2/L3/RAM)."
            }
            desc_l3 = [
                "Stalls due to branch resteers.",
                "Cycles where Divider unit was active.",
                "Stalled on external memory (DRAM).",
                "Limited by Decoded Stream Buffer pipeline.",
                "Stalls due to switching from DSB to MITE.",
                "Instructions decoded into 2 or more uops.",
                "Floating-point (FP) operations executed.",
                "One uop representing multiple contiguous instructions.",
                "Stalls due to instruction cache misses.",
                "Stalls due to ITLB misses.",
                "Stalled without missing the L1D cache.",
                "Stalled due to L2 cache accesses.",
                "Stalled due to L3 cache accesses or Core contention.",
                "Stalls due to Length Changing Prefixes.",
                "Memory load or store uops retired.",
                "Uops fetched by the Microcode Sequencer.",
                "Limited by MITE pipeline (legacy decode).",
                "Stalls due to switching uop delivery to MS unit.",
                "Branch instructions that were not fused.",
                "Remaining light uops not covered by sibling nodes.",
                "Stalls due to other misprediction cases.",
                "Machine Clears not related to memory ordering.",
                "Stalled on accesses to Persistent Memory Modules.",
                "Limited due to execution ports saturation.",
                "Issue-pipeline stalled due to serializing operations.",
                "Stalled due to store memory accesses and RFO requests."
            ]

            nodes_label = []
            nodes_color = []
            nodes_desc = []
            links_source = []
            links_target = []
            links_value = []
            links_color = []

            def hex_to_rgba(hex_color, alpha=0.4):
                h = hex_color.lstrip('#')
                return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"

            def add_node(name, pct, color, desc=""):
                nodes_label.append(f"{name} ({pct:.1f}%)")
                nodes_color.append(color)
                nodes_desc.append(desc)
                return len(nodes_label) - 1

            def add_link(source, target, value, color):
                links_source.append(source)
                links_target.append(target)
                links_value.append(max(value, 0.1)) # 0.1 ensures 0% metrics remain visible as thin lines
                links_color.append(hex_to_rgba(color, 0.4))

            # TopdownL1
            n_ret = add_node("Retiring", l1_res[0], c_ret, desc_l1["Retiring"])
            n_bad = add_node("Bad Spec", l1_res[1], c_bad, desc_l1["Bad Spec"])
            n_fe  = add_node("Frontend", l1_res[2], c_fe,  desc_l1["Frontend"])
            n_be  = add_node("Backend",  l1_res[3], c_be,  desc_l1["Backend"])

            # TopdownL2, TopdownL3
            if l2_res and len(l2_res) >= 8 and l2_res[0] >= 0:
                def process_l2(l2_name, l2_idx, parent_node, color, desc, l3_mappings=None):
                    l2_val = l2_res[l2_idx]
                    n_l2 = add_node(l2_name, l2_val, color, desc)
                    add_link(parent_node, n_l2, l2_val, color)

                    if l3_res and l3_mappings and len(l3_res) > 0 and l3_res[0] >= 0:
                        for l3_name, l3_idx in l3_mappings:
                            if l3_idx < len(l3_res):
                                l3_val = l3_res[l3_idx]
                                desc_text = desc_l3[l3_idx] if len(desc_l3) > l3_idx else ""
                                n_l3 = add_node(l3_name, l3_val, color, desc_text)
                                add_link(n_l2, n_l3, l3_val, color)

                map_light = map_heavy = map_clears = map_misp = map_fbw = map_flat = map_core = map_mem = None

                if l3_res and len(l3_res) >= 26:
                    map_light = [("FP Arith", 6), ("Mem Ops", 14), ("Fused", 7), ("Non-Fused", 18), ("Other Light", 19)]
                    map_heavy = [("Few Uops", 5), ("Microcode", 15)]
                    map_clears = [("Other Nukes", 21)]
                    map_misp = [("Branch Rest.", 0), ("Other Misp.", 20)]
                    map_fbw = [("MITE", 16), ("DSB", 3), ("DSB Swit.", 4), ("LCP", 13)]
                    map_flat = [("ICache Miss", 8), ("ITLB Miss", 9), ("MS Switches", 17)]
                    map_core = [("Divider", 1), ("Ports Util.", 23), ("Serializing", 24)]
                    map_mem = [("L1 Bound", 10), ("L2 Bound", 11), ("L3 Bound", 12), ("DRAM Bound", 2), ("Store Bound", 25), ("PMM Bound", 22)]
                elif l3_res and len(l3_res) >= 5: # Support for TopdownL3_Mem
                    map_mem = [("L1 Bound", 0), ("L2 Bound", 1), ("L3 Bound", 2), ("DRAM Bound", 3), ("Store Bound", 4)]

                process_l2("Light Ops", 0, n_ret, c_ret, desc_l2["Light Ops"], map_light)
                process_l2("Heavy Ops", 1, n_ret, c_ret, desc_l2["Heavy Ops"], map_heavy)
                process_l2("Clears", 2, n_bad, c_bad, desc_l2["Clears"], map_clears)
                process_l2("Branch Misp", 3, n_bad, c_bad, desc_l2["Branch Misp"], map_misp)
                process_l2("Fetch BW", 4, n_fe, c_fe, desc_l2["Fetch BW"], map_fbw)
                process_l2("Fetch Lat", 5, n_fe, c_fe, desc_l2["Fetch Lat"], map_flat)
                process_l2("Core Bound", 6, n_be, c_be, desc_l2["Core Bound"], map_core)
                process_l2("Mem Bound", 7, n_be, c_be, desc_l2["Mem Bound"], map_mem)

            fig = go.Figure(data=[go.Sankey(
                node = dict(
                  pad = 15,
                  thickness = 20,
                  line = dict(color = "black", width = 0.5),
                  label = nodes_label,
                  color = nodes_color,
                  customdata = nodes_desc,
                  hovertemplate = "<b>%{label}</b><br>%{customdata}<extra></extra>"
                ),
                link = dict(
                  source = links_source,
                  target = links_target,
                  value = links_value,
                  color = links_color,
                  hovertemplate = "%{source.label} → %{target.label}<br>Value: %{value:.1f}%<extra></extra>"
                )
            )])

            fig.update_layout(
                title_text="Top-down Microarchitecture Pipeline µpipe",
                font_size=12,
                height=650 if (l3_res and len(l3_res) > 5) else 450,
                margin=dict(t=40, l=20, r=20, b=20)
            )

            return fig

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return mo.md(f"**Error generating Sankey diagram:**\n```python\n{error_trace}\n```")

    return plot_upipe,




@app.cell(hide_code=True)
def _(mo):
    btn_all = mo.ui.run_button(label="Run All Experiments", kind="success")

    run_all_ui = mo.vstack([
        mo.md("---"),
        mo.md("## Evaluate All Steps"),
        mo.md("Click the button below to compile and evaluate all steps at once."),
        btn_all
    ])

    run_all_ui
    return btn_all, run_all_ui



#@app.cell(hide_code=True)
#def _(mo):
#    mo.md(r"""
#    ---
#    ## Step 1: The Naive Implementation
#    Let's evaluate a standard, unoptimized Matrix Multiplication. No tiling, no vectorization, no unrolling.
#
#    ```
#    for i in 128
#      for j in 512
#        for k in 1024
#    ```
#
#    """)
#    return
#
#
#
#@app.cell
#def _(mo, platform):
#
#    import xtc.graphs.xtc.op as O
#    from xtc.backends.mlir import Backend
#    from xtc.schedules.descript import descript_scheduler
#
#    I, J, K, dtype = 128, 512, 1024, "float32"
#
#    a = O.tensor((I, K), dtype, name="A")
#    b = O.tensor((K, J), dtype, name="B")
#    with O.graph(name="matmul") as gb:
#        O.matmul(a, b, name="C")
#
#    backend = Backend(gb.graph)
#
#    # Basic Schedule based on UI inputs
#    schedule_spec = '''
#    i:
#    j:
#    k:
#    '''
#
#
#    scheduler = backend.get_scheduler()
#    descript_scheduler(
#        scheduler=scheduler,
#        node_name="C",
#        abstract_dims=["i", "j", "k"],
#        spec=schedule_spec
#    )
#
#    sched = scheduler.schedule()
#
#    compiler = backend.get_compiler(
#        dump_file="matmul_mlir",
#        print_source_ir=False,
#        print_transformed_ir=False,
#        print_assembly=False,
#        shared_lib=True
#    )
#
#    module = compiler.compile(sched)
#
#
#    # Evaluation
#    tma_counters = ["TopdownL1", "TopdownL2", "TopdownL3"] if platform == "linux" else []
#    evaluator = module.get_evaluator(validate=False, hw_counters=tma_counters)
#    res, code, err = evaluator.evaluate()
#
#    # Slicing results
#    ex1_l1_res = res[0:4] if len(res) >= 4 else None
#    ex1_l2_res = res[4:12] if len(res) >= 12 else None
#    ex1_l3_res = res[12:38] if len(res) >= 38 else None
#
#    print(f"[DEBUG] ex1 l1 : {ex1_l1_res}")
#    print(f"[DEBUG] ex1 l2 : {ex1_l2_res}")
#    print(f"[DEBUG] ex1 l3 : {ex1_l3_res}")
#
#
#    return I, J, K, a, b, backend, code, dtype, err, evaluator, gb, ex1_l1_res, ex1_l2_res, ex1_l3_res,module, res, schedule_spec, scheduler, tma_counters
#
#
#@app.cell
#def _(err, ex1_l1_res, ex1_l2_res, ex1_l3_res, mo, plot_upipe):
#    # Display the upipe diagram
#    if err:
#        ex1_output = mo.md(f"**Error:**\n```text\n{err}\n```")
#    else:
#        ex1_output = mo.vstack([
#            mo.md("**Microarchitecture Pipeline Bottlenecks:**"),
#            plot_upipe(ex1_l1_res, ex1_l2_res, ex1_l3_res)
#        ])
#
#    ex1_output
#    return ex1_output,





@app.cell(hide_code=True)
def _(mo, platform, plot_upipe):
    def run_experiment(schedule_spec):
        import xtc.graphs.xtc.op as O
        from xtc.backends.mlir import Backend
        from xtc.schedules.descript import descript_scheduler

        # Fixed matrix size for all tests
        I, J, K, dtype = 512, 512, 512, "float32"

        a = O.tensor((I, K), dtype, name="A")
        b = O.tensor((K, J), dtype, name="B")
        with O.graph(name="matmul") as gb:
            O.matmul(a, b, name="C")

        backend = Backend(gb.graph)
        scheduler = backend.get_scheduler()

        descript_scheduler(scheduler=scheduler, node_name="C", abstract_dims=["i", "j", "k"], spec=schedule_spec)

        compiler = backend.get_compiler(dump_file="matmul_mlir", shared_lib=True)
        module = compiler.compile(scheduler.schedule())

        tma_counters = ["TopdownL1", "TopdownL2", "TopdownL3"] if platform == "linux" else []
        evaluator = module.get_evaluator(validate=False, hw_counters=tma_counters)
        res, code, err = evaluator.evaluate()

        if err: return mo.md(f"**Error:**\n```text\n{err}\n```")

        l1_res = res[0:4] if len(res) >= 4 else None
        l2_res = res[4:12] if len(res) >= 12 else None
        l3_res = res[12:38] if len(res) >= 38 else None

        return mo.vstack([
            plot_upipe(l1_res, l2_res, l3_res),
            mo.accordion({"Click here to see the XTC schedule": mo.md(f"```\n{schedule_spec}\n```")})
        ])
    return run_experiment,


@app.cell(hide_code=True)
def _(mo):
    btn_ex1 = mo.ui.run_button(label="Run Step 1", kind="neutral")

    mo.md(f"""
    ---
    ## Step 1: Naive Implementation
    A standard, unoptimized Matrix Multiplication. The CPU executes instructions sequentially.

    ```python
    for i in 512:
      for j in 512:
        for k in 512:
          C[i, j] += A[i, k] * B[k, j]
    ```
    *Expectation: High Backend Bound (Memory latency).*

    {btn_ex1}
    """)
    return btn_ex1,


@app.cell(hide_code=True)
def _(btn_ex1, btn_all, mo, run_experiment):
    mo.stop(not (btn_ex1.value or btn_all.value))

    spec_1 = '''
i:
j:
k:
'''
    run_experiment(spec_1)
    return spec_1,


@app.cell(hide_code=True)
def _(mo):
    btn_ex2 = mo.ui.run_button(label="Run Step 2", kind="neutral")

    mo.md(f"""
    ---
    ## Step 2: Vectorization
    Processing multiple elements per instruction using SIMD (Single Instruction, Multiple Data).

    ```python
    for i in 512:
      for j_out in 32:
        for k in 512:
          for j_vec in 16: # Vectorized execution (e.g., AVX-512)
            C[i, ...] += A[i, k] * B[k, ...]
    ```
    *Expectation: Higher Retiring, but still bottlenecked by memory loads.*

    {btn_ex2}
    """)
    return btn_ex2,


@app.cell(hide_code=True)
def _(btn_ex2, btn_all, mo, run_experiment):
    mo.stop(not (btn_ex2.value or btn_all.value))

    spec_2 = '''
i:
j:
k:
j#16: vectorize
'''
    run_experiment(spec_2)
    return spec_2,


@app.cell(hide_code=True)
def _(mo):
    btn_ex3 = mo.ui.run_button(label="Run Step 3", kind="neutral")

    mo.md(f"""
    ---
    ## Step 3: Register Tiling (2x16)
    Unrolling the outer loop allows the CPU to process multiple vectors simultaneously, hiding instruction latency by keeping accumulators inside physical registers.

    ```python
    for i_out in 256:
      for j_out in 32:
        for k in 512:
          for i_unroll in 2: # Unrolled
            for j_vec in 16: # Vectorized
              C[...] += A[...] * B[...]
    ```
    *Expectation: Retiring improves, but the CPU still re-loads the accumulator `C` at every `k` iteration.*

    {btn_ex3}
    """)
    return btn_ex3,


@app.cell(hide_code=True)
def _(btn_ex3, btn_all, mo, run_experiment):
    mo.stop(not (btn_ex3.value or btn_all.value))

    spec_3 = '''
i:
j:
k:
i#2: unroll
j#16: vectorize
'''
    run_experiment(spec_3)
    return spec_3,


@app.cell(hide_code=True)
def _(mo):
    btn_ex4 = mo.ui.run_button(label="Run Step 4", kind="neutral")

    mo.md(f"""
    ---
    ## Step 4: Load/Store Hoisting
    By interchanging the `k` loop *outside* of the unrolled micro-kernel, we prevent the compiler from loading and storing the `C` matrix to L1 cache repeatedly. `C` stays in the vector registers during the entire `k` accumulation.

    ```python
    for i_out in 256:
      for j_out in 32:
        # Load C into registers once
        for k in 512:
          for i_unroll in 2:
            for j_vec in 16:
              C_reg[...] += A[...] * B[...]
        # Store C to memory once
    ```
    *Expectation: Massive drop in L1 Bound. The CPU becomes highly Core Bound (saturating ALUs).*

    {btn_ex4}
    """)
    return btn_ex4,


@app.cell(hide_code=True)
def _(btn_ex4, btn_all, mo, run_experiment):
    mo.stop(not (btn_ex4.value or btn_all.value))

    spec_4 = '''
        i:
        j:
        i#2: unroll
        j#16: vectorize
        k: vectorize
        '''
    run_experiment(spec_4)
    return spec_4,


@app.cell(hide_code=True)
def _(mo):
    btn_ex5 = mo.ui.run_button(label="Run Step 5", kind="neutral")

    mo.md(f"""
    ---
    ## Step 5: Cache Tiling
    Large matrices evict each other from L1/L2 caches. We tile the problem into 64x64 blocks to ensure data perfectly fits in the fast cache hierarchy.

    ```python
    for i_blk in 64, for j_blk in 64, for k_blk in 64: # Cache blocking
      for i_out, for j_out:
        for k_in in 64:
          for i_unroll in 2, for j_vec in 16:
            C_reg[...] += A[...] * B[...]
    ```
    *Expectation: Memory Bound drops significantly. Computation is now purely limited by execution units.*

    {btn_ex5}
    """)
    return btn_ex5,


@app.cell(hide_code=True)
def _(btn_ex5, btn_all, mo, run_experiment):
    mo.stop(not (btn_ex5.value or btn_all.value))

    spec_5 = '''
        i:
        j:
        k:
        i#64:
        j#64:
        k#64:
        i#2: unroll
        j#16: vectorize
        '''
    run_experiment(spec_5)
    return spec_5,


@app.cell(hide_code=True)
def _(mo):
    btn_ex6 = mo.ui.run_button(label="Run Step 6", kind="neutral")

    mo.md(f"""
    ---
    ## Step 6: Peak Optimization (3x16)
    We maximize physical register usage. A 3x16 tile uses 3 AVX-512 registers for `C`, maximizing Instruction-Level Parallelism without spilling to the L1 cache.

    ```python
    for i_blk in 64, for j_blk in 64, for k_blk in 64:
      for i_out, for j_out:
        for k_in in 64:
          for i_unroll in 3, for j_vec in 16: # Optimized Register Tile
            C_reg[...] += A[...] * B[...]
    ```
    *Expectation: Maximum Retiring percentage. Peak theoretical performance.*

    {btn_ex6}
    """)
    return btn_ex6,


@app.cell(hide_code=True)
def _(btn_ex6, btn_all, mo, run_experiment):
    mo.stop(not (btn_ex6.value or btn_all.value))

    spec_6 = '''
        i:
        j:
        k:
        i#64:
        j#64:
        k#64:
        i#3: unroll
        j#16: vectorize
        '''
    run_experiment(spec_6)
    return spec_6,

if __name__ == "__main__":
    app.run()




# TODO : Rajouter le calcul du peek-perf a chaque exp
