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


@app.cell(hide_code=True)
def _(mo, prerequisites_ui):
    mo.vstack([
        prerequisites_ui,
        mo.accordion({
            "View Supported TMA Architectures": mo.md("""
            | Microarchitecture | Supported TMA Levels | Execution Mode |
            |---|---|---|
            | **Intel Skylake / Cascade Lake** | `TopdownL1`, `TopdownL2`, `TopdownL3` | Native API |
            | **Intel Modern (Ice Lake+)**     | `TopdownL1`, `TopdownL2`, `TopdownL3` | Native API |
            | **AMD Zen 4**                    | `TopdownL1`, `TopdownL2` | Native API |
            | **ARM AArch64**                  | `TopdownL1 ?`, `TopdownL2 ?` | Native API |
            | **Generic Linux (`perf`)**       | `TopdownL1` to `TopdownL6` | System Fallback |
            
            *Unmapped architectures will automatically try to fallback to the `perf` tool using multiplexing.*
            """)
        })
    ])
    return

@app.cell(hide_code=True)
def _(mo):
    def plot_upipe(l1_res, l2_res=None, l3_res=None):
        """Generates a hierarchical Left-to-Right block diagram for TMA."""
        if not l1_res or len(l1_res) < 4 or l1_res[0] < 0:
            return mo.md("⚠️ **TMA Data not available.**")

        # Color palettes (L1, L2, L3)
        c_ret = ("#86efac", "#4ade80", "#22c55e")
        c_bad = ("#fca5a5", "#f87171", "#ef4444")
        c_fe  = ("#93c5fd", "#60a5fa", "#3b82f6")
        c_be  = ("#d8b4fe", "#c084fc", "#a855f7")

        desc_l1 = {
            "Retiring": "Fraction of slots utilized by useful work.",
            "Bad Spec": "Fraction of slots wasted due to incorrect speculations.",
            "Frontend Bound": "Fraction of slots where the Frontend undersupplies the Backend.",
            "Backend Bound": "Fraction of slots where no uops are delivered due to a lack of Backend resources."
        }
        desc_l2 = {
            "Light Ops": "Retiring typical, single-uop instructions.",
            "Heavy Ops": "Retiring complex, multi-uop or microcoded instructions.",
            "Clears": "Wasted slots due to pipeline flushes (e.g., memory ordering issues).",
            "Branch Misp": "Wasted slots due to incorrect branch predictions.",
            "Fetch BW": "Frontend cannot decode or deliver enough instructions per cycle.",
            "Fetch Lat": "Frontend is completely starved and delivering nothing.",
            "Core Bound": "Execution stalled waiting for execution units (ALU/FPU) or dependencies.",
            "Mem Bound": "Execution stalled waiting for data from the memory (L1/L2/L3/RAM)."
        }
        desc_l3 = [
            "Stalls due to branch resteers at execution stage.",
            "Cycles where the Divider unit was active.",
            "Stalled on external memory (DRAM) accesses.",
            "Limited by Decoded Stream Buffer (uop cache) pipeline.",
            "Stalls due to switching from DSB to MITE pipelines.",
            "Instructions decoded into 2 or more uops.",
            "Floating-point (FP) operations executed.",
            "One uop representing multiple contiguous instructions.",
            "Stalls due to instruction cache misses.",
            "Stalls due to Instruction TLB (ITLB) misses.",
            "Stalled without missing the L1 Data (L1D) cache.",
            "Stalled due to L2 cache accesses.",
            "Stalled due to L3 cache accesses or sibling Core contention.",
            "Stalls due to Length Changing Prefixes.",
            "Memory load or store uops retired.",
            "Uops fetched by the Microcode Sequencer (MS) unit.",
            "Limited by MITE pipeline (legacy decode pipeline).",
            "Stalls due to switching uop delivery to the MS unit.",
            "Branch instructions that were not fused.",
            "Remaining light uops not covered by other sibling nodes.",
            "Stalls due to other misprediction cases (non-retired branches).",
            "Machine Clears not related to memory ordering.",
            "Stalled on accesses to Persistent Memory Modules (Optane).",
            "Limited due to execution ports saturation (FPU/ALU contention).",
            "Issue-pipeline stalled due to serializing operations.",
            "Stalled due to store memory accesses and RFO requests."
        ]

        def make_block(name, pct_global, color, desc="", level=1, children_html=""):
            is_empty = pct_global < 0.1
            fill_pct = max(0.0, min(100.0, pct_global))
            
            # CSS Linear gradient for vertical progress bar
            bg_style = f"background: linear-gradient(to top, {color} {fill_pct}%, #f3f4f6 {fill_pct}%);"
            text_color = "#9ca3af" if is_empty else "#000"
            border_style = "1px dashed #d1d5db" if is_empty else "1px solid #d1d5db"
            
            safe_desc = desc.replace('"', '&quot;')
            tooltip = f"{name} ({pct_global:.1f}%)\n{safe_desc}"
            
            margin_b = "12px" if level == 1 else ("6px" if level == 2 else "3px")
            
            label_html = f'''
            <div style="width: 140px; flex-grow: 1; {bg_style} border: {border_style}; border-radius: 6px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 4px; text-align: center; color: {text_color}; cursor: help; box-shadow: 0 1px 3px rgba(0,0,0,0.1);" title="{tooltip}">
                <span style="font-size: 11px; font-family: sans-serif; font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; pointer-events: none;">{name}</span>
                <span style="font-size: 10px; font-family: monospace; background-color: rgba(255, 255, 255, 0.7); padding: 0 4px; border-radius: 4px; margin-top: 3px; pointer-events: none;">{pct_global:.1f}%</span>
            </div>
            '''

            if children_html:
                return f'''
                <div style="display: flex; flex-direction: row; width: 100%; margin-bottom: {margin_b}; align-items: stretch;">
                    <div style="display: flex; flex-direction: column; margin-right: 12px;">
                        {label_html}
                    </div>
                    <div style="display: flex; flex-direction: column; flex-grow: 1;">
                        {children_html}
                    </div>
                </div>
                '''
            else:
                return f'''
                <div style="display: flex; flex-direction: row; width: 100%; min-height: 36px; margin-bottom: {margin_b}; align-items: stretch;">
                    {label_html}
                </div>
                '''

        def get_l3(idx):
            return l3_res[idx] if l3_res and len(l3_res) > idx and l3_res[idx] >= 0 else 0.0

        def b_l3(name, idx, color):
            return make_block(name, get_l3(idx), color, desc=desc_l3[idx], level=3)

        ret_html = bad_html = fe_html = be_html = ""

        if l2_res and len(l2_res) >= 8 and l2_res[0] >= 0:
            l3_light_ops = l3_heavy_ops = l3_clears = l3_misp = l3_fetch_bw = l3_fetch_lat = l3_core_bnd = l3_mem_bnd = ""
            
            if l3_res and len(l3_res) >= 26 and l3_res[0] >= 0:
                l3_light_ops = b_l3("FP Arith", 6, c_ret[2]) + b_l3("Mem Ops", 14, c_ret[2]) + b_l3("Fused", 7, c_ret[2]) + b_l3("Non-Fused Br", 18, c_ret[2]) + b_l3("Other Light", 19, c_ret[2])
                l3_heavy_ops = b_l3("Few Uops", 5, c_ret[2]) + b_l3("Microcode", 15, c_ret[2])
                l3_clears    = b_l3("Other Nukes", 21, c_bad[2])
                l3_misp      = b_l3("Branch Rest.", 0, c_bad[2]) + b_l3("Other Misp.", 20, c_bad[2])
                l3_fetch_bw  = b_l3("MITE", 16, c_fe[2]) + b_l3("DSB", 3, c_fe[2]) + b_l3("DSB Swit.", 4, c_fe[2]) + b_l3("LCP", 13, c_fe[2])
                l3_fetch_lat = b_l3("ICache Miss", 8, c_fe[2]) + b_l3("ITLB Miss", 9, c_fe[2]) + b_l3("MS Switches", 17, c_fe[2])
                l3_core_bnd  = b_l3("Divider", 1, c_be[2]) + b_l3("Ports Util.", 23, c_be[2]) + b_l3("Serializing", 24, c_be[2])
                l3_mem_bnd   = b_l3("L1 Bound", 10, c_be[2]) + b_l3("L2 Bound", 11, c_be[2]) + b_l3("L3 Bound", 12, c_be[2]) + b_l3("DRAM Bound", 2, c_be[2]) + b_l3("Store Bound", 25, c_be[2]) + b_l3("PMM Bound", 22, c_be[2])

            ret_html = make_block("Light Ops", l2_res[0], c_ret[1], desc_l2["Light Ops"], 2, l3_light_ops) + make_block("Heavy Ops", l2_res[1], c_ret[1], desc_l2["Heavy Ops"], 2, l3_heavy_ops)
            bad_html = make_block("Clears", l2_res[2], c_bad[1], desc_l2["Clears"], 2, l3_clears) + make_block("Branch Misp", l2_res[3], c_bad[1], desc_l2["Branch Misp"], 2, l3_misp)
            fe_html  = make_block("Fetch BW", l2_res[4], c_fe[1], desc_l2["Fetch BW"], 2, l3_fetch_bw) + make_block("Fetch Lat", l2_res[5], c_fe[1], desc_l2["Fetch Lat"], 2, l3_fetch_lat)
            be_html  = make_block("Core Bound", l2_res[6], c_be[1], desc_l2["Core Bound"], 2, l3_core_bnd) + make_block("Mem Bound", l2_res[7], c_be[1], desc_l2["Mem Bound"], 2, l3_mem_bnd)

        l1_blocks = (
            make_block("Retiring", l1_res[0], c_ret[0], desc_l1["Retiring"], 1, ret_html) +
            make_block("Bad Spec", l1_res[1], c_bad[0], desc_l1["Bad Spec"], 1, bad_html) +
            make_block("Frontend Bound", l1_res[2], c_fe[0], desc_l1["Frontend Bound"], 1, fe_html) +
            make_block("Backend Bound", l1_res[3], c_be[0], desc_l1["Backend Bound"], 1, be_html)
        )

        return mo.Html(f"""
        <div style="display: flex; flex-direction: column; width: fit-content; max-width: 100%; overflow-x: auto; padding: 4px;">
            {l1_blocks}
        </div>
        """)

    return plot_upipe,
    
##################################################################################################
##########################################EX 1###############################################
##################################################################################################

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step 1: The Naive Implementation
    Let's evaluate a standard, unoptimized Matrix Multiplication. No tiling, no vectorization, no unrolling.

    ```
    for i in 128
      for j in 512
        for k in 1024
    ```

    """)
    return

    

@app.cell
def _(mo, platform):

    import xtc.graphs.xtc.op as O
    from xtc.backends.mlir import Backend
    from xtc.schedules.descript import descript_scheduler

    I, J, K, dtype = 128, 512, 1024, "float32"

    a = O.tensor((I, K), dtype, name="A")
    b = O.tensor((K, J), dtype, name="B")
    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="C")

    backend = Backend(gb.graph)

    # Basic Schedule based on UI inputs
    schedule_spec = '''
    i: 
    j:
    k:
    '''


    scheduler = backend.get_scheduler()
    descript_scheduler(
        scheduler=scheduler, 
        node_name="C", 
        abstract_dims=["i", "j", "k"], 
        spec=schedule_spec
    )

    sched = scheduler.schedule()

    compiler = backend.get_compiler(
        dump_file="matmul_mlir",
        print_source_ir=False,
        print_transformed_ir=False,
        print_assembly=False,
        shared_lib=True
    )
    
    module = compiler.compile(sched)

    
    # Evaluation
    tma_counters = ["TopdownL1", "TopdownL2", "TopdownL3"] if platform == "linux" else []
    evaluator = module.get_evaluator(validate=False, hw_counters=tma_counters)
    res, code, err = evaluator.evaluate()

    # Slicing results
    ex1_l1_res = res[0:4] if len(res) >= 4 else None
    ex1_l2_res = res[4:12] if len(res) >= 12 else None
    ex1_l3_res = res[12:38] if len(res) >= 38 else None

    print(f"[DEBUG] ex1 l1 : {ex1_l1_res}")
    print(f"[DEBUG] ex1 l2 : {ex1_l2_res}")
    print(f"[DEBUG] ex1 l3 : {ex1_l3_res}")


    return I, J, K, a, b, backend, code, dtype, err, evaluator, gb, ex1_l1_res, ex1_l2_res, ex1_l3_res,module, res, schedule_spec, scheduler, tma_counters


@app.cell
def _(err, ex1_l1_res, ex1_l2_res, ex1_l3_res, mo, plot_upipe):
    # Display the upipe diagram
    if err:
        ex1_output = mo.md(f"**Error:**\n```text\n{err}\n```")
    else:
        ex1_output = mo.vstack([
            mo.md("**Microarchitecture Pipeline Bottlenecks:**"),
            plot_upipe(ex1_l1_res, ex1_l2_res, ex1_l3_res)
        ])
    
    ex1_output
    return ex1_output,



##################################################################################################
##########################################EX 2###############################################
##################################################################################################

# todo : Exemples de plus en plus avancé d'optimisation avec visuel de compteurs



if __name__ == "__main__":
    app.run()






