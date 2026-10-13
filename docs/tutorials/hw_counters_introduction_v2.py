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

    In this Marimo, we will use **XTC** to compile a Matrix Multiplication and evaluate it using hardware counters and Top-down Microarchitecture Analysis (TMA)
    """)
    return


@app.cell(hide_code=True)
def _(mo, platform):
    _warnings = []

    if platform == "darwin":
        _warnings.append(mo.callout(mo.md("**MacOS Detected:** Hardware counters are restricted. The Marimo will run, but TMA metrics might be unavailable."), kind="danger"))

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
    prerequisites_ui
    return prerequisites_ui,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ---

        ### For a better experience, it's recommended to be familiar with the XTC's `descript` syntax and the general usage of XTC.

        ```bash
        marimo run docs/tutorials/xtc_101.py
        marimo run docs/tutorials/descript++_notebook.py
        ```
        ---
    """)
    return


@app.cell
def _(supported_arch_msg):
    supported_arch_msg
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



@app.cell(hide_code=True)
def _(mo, platform, plot_upipe):
    def run_experiment(schedule_spec):
        import xtc.graphs.xtc.op as O
        from xtc.backends.mlir import Backend
        from xtc.schedules.descript import descript_scheduler
        from xtc.runtimes.host import HostRuntime
        import subprocess

        # Fixed matrix size for all tests
        I, J, K, dtype = 512, 512, 512, "float32"

        a = O.tensor((I, K), dtype, name="A")
        b = O.tensor((K, J), dtype, name="B")
        with O.graph(name="matmul") as gb:
            O.matmul(a, b, name="C")

        backend = Backend(gb.graph)
        scheduler = backend.get_scheduler()

        descript_scheduler(
            scheduler=scheduler,
            node_name="C",
            abstract_dims=["i", "j", "k"],
            abstract_matrix=["A", "B", "C"],
            spec=schedule_spec
        )

        compiler = backend.get_compiler(
            dump_file="matmul_mlir",
            shared_lib=True,
            print_assembly=True
        )
        module = compiler.compile(scheduler.schedule())

        rt = HostRuntime.get()
        peak_flops = rt.evaluate_flops(dtype)

        # Evaluate pure execution time
        eval_time = module.get_evaluator(validate=False)
        time_results, _, _ = eval_time.evaluate()
        exec_time = min(time_results)

        # Calculate percentage of theoretical peak (MACs / Time / Peak_MACs)
        peak_perf_pct = (I * J * K) / exec_time / peak_flops * 100

        # Visual progress bar for Peak Perf
        bar_color = "#ef4444" if peak_perf_pct < 10 else ("#f59e0b" if peak_perf_pct < 50 else "#22c55e")
        fill_width = max(min(peak_perf_pct, 100), 2) # Ensure at least 2% so the text is visible

        perf_ui = mo.Html(f'''
        <div style="margin-bottom: 15px; width: 100%; max-width: 600px;">
            <strong>Computational Efficiency (Theoretical Peak):</strong>
            <div style="width: 100%; background-color: #f3f4f6; border-radius: 6px; margin-top: 6px; overflow: hidden; border: 1px solid #d1d5db; display: flex;">
                <div style="width: {fill_width}%; background-color: {bar_color}; color: #000; padding: 4px 0; text-align: center; font-size: 12px; font-family: monospace; font-weight: bold; white-space: nowrap; transition: width 0.5s ease-in-out;">
                    {peak_perf_pct:.1f}%
                </div>
            </div>
        </div>
        ''')



        tma_counters = ["TopdownL1", "TopdownL2", "TopdownL3"] if platform == "linux" else []
        evaluator = module.get_evaluator(validate=False, hw_counters=tma_counters)
        res, code, err = evaluator.evaluate()


        if err: return mo.md(f"**Error:**\n```text\n{err}\n```")

        l1_res = res[0:4] if len(res) >= 4 else None
        l2_res = res[4:12] if len(res) >= 12 else None
        l3_res = res[12:38] if len(res) >= 38 else None


        so_path = getattr(module, "file_name", None)
        if so_path:
            try:
                # Silently dump the assembly to a Python string
                asm_code = subprocess.check_output(["objdump", "--disassemble=matmul", so_path], universal_newlines=True)
            except Exception as e:
                asm_code = f"Failed to extract assembly: {e}"

        return mo.vstack([
            perf_ui,
            plot_upipe(l1_res, l2_res, l3_res),

            mo.accordion({
                "View XTC Schedule Source": mo.md(f"```yaml\n{schedule_spec}\n```"),
                "View Generated Assembly": mo.md(f"```x86asm\n{asm_code}\n```")
            })

        ])
    return run_experiment,


@app.cell(hide_code=True)
def _(mo):
    def _hint(meaning, causes, check, experiment, caution="", deeper=""):
        _paragraphs = [
            f"**Meaning:** {meaning}",
            f"**Possible causes:** {causes}",
            f"**Check:** {check}",
            f"**Try:** {experiment}",
        ]
        if caution:
            _paragraphs.append(f"**Keep in mind:** {caution}")
        if deeper:
            _paragraphs.append(f"**Optional deeper clues:** {deeper}")
        return mo.md("\n\n".join(_paragraphs))

    _memory = mo.vstack([
        mo.md(r"""
        **Memory Bound:** execution is waiting for loads or stores. Ask whether the
        schedule reuses data before it leaves a nearby cache, and whether it moves
        more data than necessary.

        A **cache line** is the unit transferred between caches (often 64 bytes).
        Contiguous accesses use more of each line than widely spaced accesses.
        **Latency** is the time to obtain data; **bandwidth** is how much data can
        be transferred per second. A schedule can suffer from either.

        For cache tiling, estimate the combined active data, not just one matrix:
        `4 * (Ti * Tk + Tk * Tj + Ti * Tj)` bytes for float32 A, B, and C tiles.
        Three 64×64 tiles total **48 KiB**, before packing buffers and other data.
        This is not guaranteed to fit in L1. Leave headroom: cache organization
        and other activity can cause eviction even below the advertised capacity.
        """),
        mo.accordion({
            "L1 Bound (`l1_bound`) is high — even nearby data causes stalls": _hint(
                "Load-related stalls occur without an L1 data-cache miss; this is not an L1 miss rate.",
                "A load feeds the next operation immediately; repeated accumulator loads; limited load throughput or store-to-load dependencies.",
                "In the assembly, is C loaded inside every k iteration? Are temporary values stored on the stack (register spills)? Are loads immediately used in a dependent chain?",
                "Keep C accumulators in registers (Step 4), reuse loaded operands, or add independent accumulators (Step 3). Reduce the register tile if it causes spills.",
                "A larger cache tile will not necessarily fix a dependency or load-throughput problem.",
                "`l1_latency_dependency`, `store_fwd_blk`, `split_loads`, and load-port metrics, if supported.",
            ),
            "L2 Bound (`l2_bound`) is high — data is not reused in L1": _hint(
                "Loads missing L1 and served by L2 contribute to waiting.",
                "The active tiles exceed useful L1 capacity; loop order delays reuse; strided accesses use only part of each cache line.",
                "Estimate the active A/B/C data and inspect which operand is reused by consecutive inner-loop iterations. Include any packed buffers.",
                "Reduce the cache tile or change loop order to reuse data sooner (Step 5). Try packing strided operands (Step 7), measuring the copying cost too.",
                "L2 hits are better than fetching from farther away. Minimize total runtime, not this percentage in isolation.",
                "`l2_hit_latency`, `dtlb_load`, and `split_loads` can help distinguish latency, address translation, and access-layout problems.",
            ),
            "L3 Bound (`l3_bound`) is high — data is not reused in closer caches": _hint(
                "Loads missing the closer caches and served by L3 contribute to waiting; shared-cache contention may also matter.",
                "Tiles exceed useful L1/L2 capacity; loop order delays reuse; large strides waste cache lines; other activity competes for cache resources.",
                "Consider A, B, and C together. How much data is accessed between two uses of the same tile? Is the machine running other memory-intensive work?",
                "Reduce cache-tile sizes or reorder loops to reuse a tile sooner (Step 5). Test packing for strided accesses (Step 7). Repeat on a less busy machine.",
                "This does not simply mean that L3 is too small. L3 hits are still preferable to DRAM accesses.",
                "`l3_hit_latency`, `contested_accesses`, and `data_sharing`, where available.",
            ),
            "DRAM Bound (`dram_bound`) is high — data comes from main memory": _hint(
                "Loads reaching main memory contribute to stalls.",
                "Little reuse before eviction; a large working set; strided accesses; insufficient independent loads to hide latency; competition for memory bandwidth.",
                "Does the loop repeatedly scan a matrix before reusing it? Is a cache line mostly unused? Compare smaller problem sizes and repeat measurements to assess sensitivity to cache residency.",
                "Use cache tiling and better loop order (Step 5), and test packing (Step 7). Reuse each loaded A/B value for more arithmetic rather than increasing data traffic.",
                "A high value alone cannot distinguish memory latency from bandwidth saturation. Warm-up and cache residency can change the result.",
                "`mem_latency` versus `mem_bandwidth`; `local_mem` / `remote_mem` on multi-socket NUMA machines.",
            ),
            "Store Bound (`store_bound`) is high — writing data holds up execution": _hint(
                "Store-related resources or obtaining ownership of cache lines contribute to stalls.",
                "C is written during every reduction iteration; register spills; scattered stores; packed buffers are copied too often.",
                "Count stores in the inner loop and inspect stack accesses. Check whether each output vector can be accumulated before writing it once.",
                "Hoist accumulator loads/stores out of the reduction (Step 4). Reduce an oversized register tile and avoid unnecessary packing copies.",
                "Stores may require fetching a cache line for ownership (RFO), so writing can also generate reads. Packing adds stores and is not always beneficial.",
                "`store_latency`, `split_stores`, `store_fwd_blk`, and `dtlb_store`. False sharing matters if multiple threads write the same cache line, not normally in a single-thread kernel.",
            ),
            "PMM Bound (`pmm_bound`) is reported — check hardware support first": _hint(
                "This metric concerns accesses to persistent-memory modules, such as Intel Optane, rather than ordinary DRAM.",
                "Data may reside in persistent memory on a supported system, or the metric may be unavailable or reconstructed differently on your CPU.",
                "Check whether the system actually has persistent memory and how this metric is collected. A zero or unavailable result is not evidence of a scheduling improvement.",
                "If applicable, increase data reuse and compare with allocation in ordinary DRAM. Otherwise, focus on the supported cache/DRAM metrics.",
                "This notebook's native Skylake implementation reports this metric as zero; it is not a meaningful diagnosis there.",
            ),
        }),
    ])

    _core = mo.vstack([
        mo.md(r"""
        **Core Bound:** execution is limited by computational resources or
        dependencies between instructions rather than primarily by fetching data.
        An **execution port** routes operations to units such as arithmetic or
        load/store units. Independent instructions can overlap; a dependent
        accumulator update must wait for its input.

        A fast matrix multiplication may legitimately become compute-bound.
        Check whether the bottleneck is useful SIMD arithmetic or avoidable overhead.
        """),
        mo.accordion({
            "Ports Util. (`ports_utilization`) is high — investigate resources and dependencies": _hint(
                "This points toward core execution limits, but is not a direct measurement of every port's occupancy. Some implementations estimate it from remaining core stalls.",
                "Instructions compete for execution resources; too many loads or address calculations; too few independent accumulator chains.",
                "Inspect assembly for SIMD/FMA instructions, repeated loads, address arithmetic, and spills. Does each FMA wait for the previous update of the same accumulator?",
                "Reuse loaded operands or modestly increase independent register accumulators (Steps 3 and 6). Sweep register-tile sizes rather than assuming that larger is better.",
                "High arithmetic-unit utilization can be desirable. More unrolling can instead cause spills or a larger instruction footprint; accept a change only if runtime improves.",
                "`ports_utilized_0/1/2/3m`, `alu/load/store_op_utilization`, and individual `port_*` metrics. Port assignments depend on the CPU.",
            ),
            "Divider (`divider`) activity is high — expensive division is present": _hint(
                "The divider is active; activity alone does not prove division is the main bottleneck.",
                "Integer division or remainder in indexing; floating-point division; setup or helper code included in the measurement.",
                "Look for division instructions in the measured kernel. Plain matrix multiplication should mostly use multiply/add, loads, stores, and loop bookkeeping.",
                "Test tile sizes or indexing that the compiler can simplify, and move invariant calculations outside inner loops where possible.",
                "Do not replace division with an approximate reciprocal without considering numerical correctness.",
            ),
            "Serializing (`serializing_operation`) is high — overlap is restricted": _hint(
                "Operations that restrict overlapping execution contribute to issue stalls.",
                "Fences, special serializing instructions, synchronization, or helper/instrumentation code in the measured region.",
                "Inspect assembly for fences and special instructions. Determine whether the result comes from the kernel or measurement/setup overhead.",
                "Repeat with enough kernel work to assess overhead, and keep unnecessary setup outside the timed region. Only remove synchronization if correctness is preserved.",
                "Ordinary dependence between successive FMAs is not the same as an explicitly serializing instruction.",
            ),
        }),
    ])

    _fetch_latency = mo.vstack([
        mo.md("**Fetch Latency:** instruction delivery stops for periods of time. Check code footprint and transitions between instruction-delivery mechanisms."),
        mo.accordion({
            "ICache Miss (`icache_misses`) is high — instructions are not nearby": _hint(
                "Instruction-cache misses contribute to frontend stalls. This cache holds code, not A/B/C data.",
                "Aggressive unrolling produces a large kernel; many code paths or helper calls increase the active code footprint.",
                "Compare assembly size across schedules. Does extra unrolling duplicate large blocks of instructions?",
                "Reduce unrolling or the register tile and compare runtime. Favor a compact, repeatedly executed inner loop.",
                "Cache tiling of matrix data does not directly solve instruction-cache misses.",
            ),
            "ITLB Miss (`itlb_misses`) is high — code-address translation is costly": _hint(
                "The instruction Translation Lookaside Buffer (ITLB), which caches code-address translations, misses.",
                "Code spans many pages; large generated kernels or scattered helper calls expand the instruction working set.",
                "Inspect code size and call structure, especially after increasing unrolling.",
                "Reduce code expansion and compare a smaller register tile or unroll factor.",
                "ITLB concerns instructions. Data packing may help data-TLB misses, but does not directly fix ITLB misses.",
                "`code_stlb_hit` / `code_stlb_miss` distinguish hits in a second-level translation cache from page-table walks.",
            ),
            "MS Switches (`ms_switches`) is high — instruction delivery changes to microcode": _hint(
                "Switching to the Microcode Sequencer (MS), which supplies operations for certain complex instructions, contributes to stalls.",
                "Complex instructions or exceptional cases interrupt the usual instruction-delivery path.",
                "Inspect assembly and compare Heavy Ops and Microcode. If supported, check assist metrics and whether unusual floating-point inputs trigger slow paths.",
                "Compare schedules that generate simpler hot-loop instructions. If assists are suspected, test representative normal inputs without changing the algorithm's correctness requirements.",
                "The switching cost and the amount of microcode-delivered work are different metrics.",
                "`assists` and `fp_assists`, where supported.",
            ),
        }),
    ])

    _fetch_bandwidth = mo.vstack([
        mo.md("**Fetch Bandwidth:** the frontend delivers instructions, but not fast enough. Too much loop/indexing overhead or expanded code can leave execution units underfed."),
        mo.accordion({
            "MITE / DSB (`mite`, `dsb`) is prominent — inspect instruction delivery": _hint(
                "MITE is the legacy instruction decoder; DSB is the Decoded Stream Buffer (a cache of already-decoded micro-operations, or uops).",
                "A hot loop may not fit the decoded-instruction cache well; its layout or instruction mix may constrain delivery; excessive unrolling may enlarge it.",
                "First confirm Fetch Bandwidth is high. Compare instruction count and code size across schedules, and inspect repeated loop/indexing work.",
                "Test moderate unrolling and SIMD to do more arithmetic per instruction, keeping the hot loop compact.",
                "Some implementations report MITE/DSB delivery activity rather than lost capacity. A large DSB or MITE value alone is not proof of a bottleneck.",
                "`decoder0_alone` can help identify legacy-decoder restrictions, where available.",
            ),
            "DSB Swit. (`dsb_switches`) is high — delivery sources keep changing": _hint(
                "Transitions between the decoded-uop cache and legacy decoder contribute to stalls.",
                "The hot loop does not remain well served by the uop cache; code size, layout, or branches cause delivery-source transitions.",
                "Compare code footprint and branch structure before and after unrolling. Look for a regression as the register tile grows.",
                "Reduce unrolling or simplify the hot loop, then measure again. Exact uop-cache capacity and layout rules are CPU-specific.",
            ),
            "LCP (`lcp`) is high — an instruction-encoding decode penalty appears": _hint(
                "Length-changing instruction prefixes cause legacy-decoder stalls on CPUs that report this metric.",
                "Particular instruction encodings in compiler-generated code; this is usually not a matrix-data locality issue.",
                "Inspect instruction encodings in the assembly and verify that the metric is supported on your CPU.",
                "Compare another schedule or compiler configuration that changes the hot-loop instruction mix. Prioritize larger, actionable bottlenecks first.",
                "This is a specialized diagnosis; students need not hand-edit assembly to proceed with the tutorial.",
            ),
        }),
    ])

    _branches = mo.vstack([
        mo.md("**Branch Mispredicts:** the CPU predicts which instructions come next incorrectly, then discards work. Regular matrix-multiplication loops are usually predictable, so a large result deserves investigation."),
        mo.accordion({
            "Branch Rest. (`branch_resteers`) is high — control flow redirects execution": _hint(
                "Execution is redirected and instruction fetching recovers. Exact attribution of this metric varies by implementation.",
                "Mispredicted branches, many short loops, or irregular boundary/remainder paths.",
                "Inspect branches in the assembly, including cleanup for dimensions not divisible by tile sizes. Compare with the parent Branch Mispredicts result.",
                "Compare tile sizes with and without remainder handling, or modestly unroll short hot loops to reduce branch overhead.",
                "Do not assume every redirect is an unpredictable data-dependent branch. Loop exits and measurement overhead can also contribute.",
                "`mispredicts_resteers` and `unknown_branches`, if available.",
            ),
            "Other Misp. (`other_mispredicts`) is high — speculation needs more evidence": _hint(
                "A residual speculation category captures cases not explained by the other reported components.",
                "Additional recovery cases or approximation/noise in reconstructed metrics.",
                "Repeat the measurement and inspect the parent category before assigning a specific cause. Check whether the result is stable across runs.",
                "Simplify hot-loop control flow or compare a schedule with fewer cleanup paths, then check both runtime and speculation metrics.",
                "A residual category is not a precise diagnosis; do not optimize an unstable percentage.",
            ),
        }),
    ])

    _clears = mo.vstack([
        mo.md("**Machine Clears:** the CPU restarts its pipeline for reasons other than an ordinary branch misprediction, such as memory-ordering recovery. This is not the same as a cache miss."),
        mo.accordion({
            "Other Nukes (`other_nukes`) is high — investigate pipeline restarts": _hint(
                "This reports remaining machine-clear cases after the implementation's other accounted causes; 'nuke' means a pipeline restart.",
                "Exceptional execution conditions, code modification, or residual/approximate accounting; the exact cause is CPU-specific.",
                "Repeat measurements, inspect the measured code, and consult CPU-specific events if the parent Machine Clears category remains substantial.",
                "Compare a simpler schedule to isolate when restarts appear. Treat this as a secondary diagnosis rather than immediately changing cache tiles.",
                "Memory-ordering clears belong to the parent investigation and are not necessarily explained by Other Nukes. Never remove required synchronization just to reduce this metric.",
                "`clears_resteers` and CPU-specific memory-ordering clear events, where available.",
            ),
        }),
    ])

    _light = mo.vstack([
        mo.md("**Light Ops:** ordinary completed operations dominate. They may implement useful arithmetic, but also loads, stores, branches, and address calculations. Retiring does not mean that every instruction advances the matrix multiplication efficiently."),
        mo.accordion({
            "Mem Ops (`memory_operations`) is high — many instructions move data": _hint(
                "Many retired operations are loads or stores. This is an instruction-mix metric, not itself a memory-stall metric.",
                "Repeated C loads/stores; limited reuse of A/B; register spills; packing copies.",
                "Inspect the inner-loop assembly: is C read and written every k iteration? Are values spilled to the stack? Interpret this together with Memory Bound.",
                "Keep accumulators in registers (Step 4), reuse operands, or shrink an oversized register tile. Include packing overhead when judging Step 7.",
                "Necessary, efficient loads can produce a high value. The objective is more useful arithmetic per byte and lower runtime, not zero memory operations.",
            ),
            "FP Arith (`fp_arith`) is prominent — verify how much arithmetic each instruction does": _hint(
                "Floating-point arithmetic is being completed; this is generally expected for matrix multiplication.",
                "Useful SIMD arithmetic, or scalar/narrow-vector operations that retire successfully but achieve little of the CPU's arithmetic peak.",
                "Inspect vector width and FMA use in assembly. Are there multiple independent accumulator chains? Check the actual SIMD capabilities of your CPU.",
                "Compare vectorization (Step 2) and register-tile sizes (Steps 3 and 6), using widths supported by the target.",
                "A high FP instruction share is not a FLOP count or proof of peak throughput. Wider vectors and FMAs do more arithmetic per instruction; frequency and dependencies also matter.",
                "`fp_scalar` / `fp_vector`, `fp_vector_128b/256b/512b`, and `x87_use`, if supported.",
            ),
            "Fused / Non-Fused (`fused_instructions`, `non_fused_branches`) is prominent": _hint(
                "These metrics concern instruction fusion, such as a comparison and branch handled together, and branches not fused that way.",
                "Loop-control instructions occupy a significant share of the work; the instruction sequence may or may not permit fusion on this CPU.",
                "Inspect comparisons and branches in the hot loop. Are very short loops doing little arithmetic per branch?",
                "Use SIMD or modest unrolling to amortize loop control over more useful work; compare runtime and code size.",
                "This fusion is not fused multiply-add (FMA). High fused-instruction activity is not inherently bad, and non-fused branches are not necessarily mispredicted.",
            ),
            "Other Light (`other_light_ops`) is high — bookkeeping may dominate": _hint(
                "Completed light operations are not explained by the listed sibling metrics.",
                "Address arithmetic, loop counters, moves, conversions, or residual accounting.",
                "Inspect how many non-arithmetic instructions occur per FMA in the hot loop. Check whether indexing expressions are recomputed unnecessarily.",
                "Compare SIMD, operand reuse, simpler indexing, and modest unrolling to amortize bookkeeping.",
                "This residual category alone does not identify an instruction. Confirm the hypothesis in assembly and runtime.",
                "`nop_instructions` and ALU-port metrics, where supported.",
            ),
        }),
    ])

    _heavy = mo.vstack([
        mo.md("**Heavy Ops:** instructions expand into multiple uops or use microcode. 'Heavy' does not automatically mean inefficient; investigate their cost and purpose in this kernel."),
        mo.accordion({
            "Few Uops (`few_uops_instructions`) is high — inspect multi-operation instructions": _hint(
                "This category represents instructions accounted as multiple micro-operations, although native reconstruction may be approximate.",
                "Complex instructions or an instruction mix that requires more internal work than its assembly instruction count suggests.",
                "Inspect the hot-loop instruction mix alongside Heavy Ops. A short assembly listing can still require many internal operations.",
                "Compare schedules that generate simpler indexing and arithmetic sequences, and retain the faster one.",
                "Do not treat this metric as an exact instruction count or replace a useful instruction merely because it expands into multiple uops.",
            ),
            "Microcode (`microcode_sequencer`) is high — complex or assisted execution": _hint(
                "The Microcode Sequencer supplies a substantial amount of internal operations.",
                "Complex instructions, helper code, or floating-point assists for exceptional input cases, depending on the CPU.",
                "Inspect the measured assembly, compare MS Switches, and check supported assist metrics. Plain matrix-multiplication FMAs should not normally require persistent assisted execution.",
                "Compare a simpler generated kernel and test whether representative input values change the result. Investigate correctness and numerical requirements before changing floating-point behavior.",
                "Microcode activity is not synonymous with a stall, and this percentage alone cannot identify the responsible instruction.",
                "`assists`, `fp_assists`, and `cisc`, where available.",
            ),
        }),
    ])

    mo.vstack([
        mo.md(r"""
        ---
        ## Why is my schedule slow? A hardware-counter troubleshooting guide

        A low percentage of peak performance tells you that your schedule has room
        for improvement, but not why. Hardware counters provide **clues, not a
        definitive diagnosis**. Results depend on the CPU, compiler, matrix sizes,
        cache residency, and other activity on the machine. The examples below
        suggest hypotheses to test, not percentages every computer should produce.

        **How to use this guide:**

        1. Start with a substantial **Level 1** category in the diagram, then inspect
           its **Level 2** children and the detailed metrics below. Use the largest
           stall category as a starting point, not as a verdict that the schedule is bad.
        2. Open **View Generated Assembly** to check whether the proposed cause
           actually appears in the compiled inner loop.
        3. Change **one scheduling decision at a time**, repeat measurements under
           similar conditions, and keep a change only if **execution time improves**.
           A smaller percentage can hide more absolute time in a slower program.
        4. Validate numerical correctness when experimenting (the sandbox enables
           validation). A faster schedule is only useful if it still computes the result.

        **Vocabulary:** the **frontend** fetches and decodes instructions; the
        **backend** executes them and accesses data. A **uop** is an internal
        micro-operation. **Retiring** means execution has completed and its result
        is committed, not that the instruction was necessary for an efficient algorithm.
        SIMD performs arithmetic on multiple elements per instruction; independent
        accumulator chains let the CPU overlap work.

        **Important distinctions:** Top-down **Levels 1, 2, and 3 are analysis depths**,
        not the L1/L2/L3 data caches. Retiring and instruction-delivery activity are
        not themselves wasted work. Formulas, denominators, and availability vary
        by architecture; reconstructed metrics and separate measurement passes need
        not form exact parent/child percentage partitions. Do not add all leaf values
        as if they were mutually exclusive, or use a universal "bad" threshold.
        Unsupported metrics are not evidence that a bottleneck is absent.

        Expand the symptom that matches your result. Titles include the diagram
        label and the raw metric name; optional deeper clues refer to counters that
        may be available separately, not extra counters automatically collected here.
        """),
        mo.accordion({
            "Backend Bound dominates — waiting for data or execution resources": mo.vstack([
                mo.md("**Backend Bound:** instructions cannot make progress because data or execution resources are unavailable. Distinguish Memory Bound from Core Bound before choosing between cache tiling and register/SIMD changes."),
                mo.accordion({
                    "Mem Bound (`memory_bound`) dominates — investigate data movement": _memory,
                    "Core Bound (`core_bound`) dominates — investigate resources and dependencies": _core,
                }),
            ]),
            "Frontend Bound dominates — instructions do not arrive fast enough": mo.vstack([
                mo.md("**Frontend Bound:** the execution machinery is underfed. A larger unrolled kernel can hurt instruction delivery even when it improves data reuse. Distinguish interruptions in delivery (latency) from insufficient delivery rate (bandwidth)."),
                mo.accordion({
                    "Fetch Lat (`fetch_latency`) is high — delivery pauses": _fetch_latency,
                    "Fetch BW (`fetch_bandwidth`) is high — delivery rate is insufficient": _fetch_bandwidth,
                }),
            ]),
            "Bad Speculation dominates — speculative work is discarded": mo.vstack([
                mo.md("**Bad Speculation:** the CPU starts work before knowing it is needed, then discards it or restarts execution. Separate incorrect branch predictions from other pipeline clears; changing cache tiles is not a general cure."),
                mo.accordion({
                    "Branch Misp (`branch_mispredicts`) is high — predictions fail": _branches,
                    "Clears (`machine_clears`) is high — the pipeline restarts": _clears,
                }),
            ]),
            "Retiring is high, but peak performance remains low — inspect completed work": mo.vstack([
                mo.md("**Retiring:** many execution slots complete operations. This is usually better than stalling, but scalar arithmetic, loads/stores, and bookkeeping also retire. A scalar instruction can retire efficiently while computing far fewer elements than SIMD. Check work per instruction and avoidable overhead, rather than trying to minimize Retiring."),
                mo.accordion({
                    "Light Ops (`light_operations`) dominates — inspect arithmetic versus overhead": _light,
                    "Heavy Ops (`heavy_operations`) dominates — inspect complex instructions": _heavy,
                }),
            ]),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    btn_ex1 = mo.ui.run_button(label="Run Step 1", kind="neutral")

    mo.md(f"""
    ---
    ## Step 1: Naive Implementation
    A standard, unoptimized Matrix Multiplication. The inner reduction has a dependent accumulator chain, and accesses to `B[k, j]` are strided for row-major data. The CPU can still overlap independent instructions; a scalar loop does not imply strictly sequential hardware execution.

    ```python
    for i in 512:
      for j in 512:
        for k in 512:
          C[i, j] += A[i, k] * B[k, j]
    ```
    *Hypothesis to test: Backend Bound may be substantial due to data-access costs or the dependent accumulation. Use Memory Bound versus Core Bound to distinguish them; the dominant cache level depends on your CPU.*

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
    A SIMD instruction can operate on 16 float32 elements with AVX-512; the supported width and instruction throughput depend on your CPU. The scalar loop over those elements is replaced by vector operations, but the reduction loop remains.

    ```python
    for i in range(512):
        for j in range(0, 512, 16): # Steps of 16
            for k in range(512):

                # Vectorized execution (e.g., AVX-512)
                # A single SIMD instruction computes 16 elements
                C[i, j : j+16] += A[i, k] * B[k, j : j+16]
    ```

    *Hypothesis to test: SIMD may improve runtime by doing more arithmetic per instruction, without necessarily increasing Retiring. Check whether the compiler still loads/stores `C` inside the reduction and whether Memory Bound or Core Bound remains substantial.*

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
    Unrolling the outer loop allows the CPU to process multiple vectors simultaneously. The compiler duplicates the instructions, breaking data dependency chains and keeping accumulators inside physical registers.

    ```python
    for i in range(0, 512, 2):      # Steps of 2
        for j in range(0, 512, 16): # Steps of 16
            for k in range(512):

                # Unrolled & Vectorized (2x16)
                # Two independent FMAs are exposed to the CPU per k-iteration
                C[i,     j : j+16] += A[i,     k] * B[k, j : j+16]
                C[i + 1, j : j+16] += A[i + 1, k] * B[k, j : j+16]
    ```

    *Hypothesis to test: Independent accumulators may hide arithmetic latency and improve runtime. Inspect the assembly to see whether `C` stays in registers or is repeatedly loaded/stored; Retiring need not increase.*

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
    I = 512
    J = 512
    K = 512
    # Global navigation (RAM)
    for i_out in range(I / 2):       # Steps of 2
        for j_out in range(J / 16):  # Steps of 16

            # Load hosting
            C_vec_0 = C[i,     j : j+16]
            C_vec_1 = C[i + 1, j : j+16]

            # Accumulation loop
            for k in range(K)

                # Vector Load (16 floats loaded at once)
                B_vec = B[k, j : j+16]

                # Unrolled & Vectorized Micro-kernel
                C_vec_0 += A[i, k]     * B_vec
                C_vec_1 += A[i + 1, k] * B_vec

            # Store hoisting
            C[i,     j : j+16] = C_vec_0
            C[i + 1, j : j+16] = C_vec_1
    ```

    *Hypothesis to test: Moving accumulator loads/stores out of the reduction may reduce memory-operation overhead and L1/store-related stalls. Confirm register residency in assembly; ordinary accumulator stores are not necessarily register spills.*

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

    Large active working sets can evict reused data from L1/L2 caches. We try 64x64 blocks to shorten the distance between uses. This size is a starting point, not a guarantee of fitting: three float32 64x64 tiles occupy 48 KiB before other data and buffers.


    ```python
    # Global navigation (RAM / L3 Cache)
    I = 512
    J = 512
    K = 512
    for i_out in range(I / 64):
        for j_out in range(J / 64):
            for k_out in range(K / 64):

                # Block navigation (L2 / L1 Cache)
                for i_in in range(64 / 2):       # Steps of 2
                    for j_in in range(64 / 16):  # Steps of 16

                        # Load hoisting
                        C_reg_0 = C[i,     j : j+16]
                        C_reg_1 = C[i + 1, j : j+16]

                        # Accumulation loop
                        for k_in in range(64):
                            B_vec = B[k, j : j+16] # Broadcasted or Vector loaded

                            # Unrolled micro-kernel
                            C_reg_0 += A[i, k]     * B_vec
                            C_reg_1 += A[i + 1, k] * B_vec

                        # Store hoisting
                        # C[i_out...i_in, j_out...j_in] = C_reg
                        C[i,     j : j+16] = C_reg_0
                        C[i + 1, j : j+16] = C_reg_1
    ```

    *Hypothesis to test: Better reuse may reduce stalls associated with farther caches or DRAM. Sweep tile sizes for your CPU and inspect the remaining bottleneck; tiling does not guarantee purely compute-bound execution.*

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
    We test another register tile. With AVX-512, a 3x16 float32 tile needs three vector accumulators for `C`, plus registers for operands and bookkeeping. More independent accumulators may improve overlap, but this is not necessarily the optimal tile for your CPU and compiler.

    ```python
    # Global navigation (RAM / L3 Cache)
    for i_out in range(I / 64):
        for j_out in range(J / 64):
            for k_out in range(K / 64):

                # Block navigation (L2 / L1 Cache)
                for i_in in range(64 / 3):       # Steps of 3
                    for j_in in range(64 / 16):  # Steps of 16

                        # Load hoisting
                        C_reg_0 = C[i,     j : j+16]
                        C_reg_1 = C[i + 1, j : j+16]
                        C_reg_2 = C[i + 2, j : j+16]

                        # Accumulation loop
                        for k_in in range(64):
                            B_vec = B[k, j : j+16] # Broadcasted or Vector loaded

                            # Micro-kernel (Physical Registers)
                            #for i_unroll in range(3):        # Unrolled 3x
                            #    for j_vec in range(16):      # Vectorized 16x
                            #       C_reg[...] += A[...] * B[...]

                            # Unrolled micro-kernel
                            C_reg_0 += A[i, k]     * B_vec
                            C_reg_1 += A[i + 1, k] * B_vec
                            C_reg_2 += A[i + 2, k] * B_vec

                        # Store hoisting
                        # C[i_out...i_in, j_out...j_in] = C_reg
                        C[i,     j : j+16] = C_reg_0
                        C[i + 1, j : j+16] = C_reg_1
                        C[i + 2, j : j+16] = C_reg_2
    ```

    *Hypothesis to test: The extra accumulator may improve arithmetic throughput. Compare runtime and inspect spills, dependencies, and code size; a larger register tile can also make the schedule slower.*

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



@app.cell(hide_code=True)
def _():
    btn_ex7 = mo.ui.run_button(label="Run Step 7", kind="neutral")

    mo.md(f"""
    ---
    ## Step 7: Data Packing (Memory Layout Optimization)
    Even with cache tiling, large strides can waste cache-line data and increase data-TLB (Translation Lookaside Buffer) pressure. A TLB caches virtual-to-physical address translations.
    By **packing** the data, the compiler copies blocks into contiguous temporary buffers before the inner loops. This can improve the micro-kernel's access pattern, but copying adds work and the benefit depends on layout and reuse.

    ```python
    # Global navigation (RAM / L3 Cache)
    for i_out in range(I / 64):
        for j_out in range(J / 64):
            for k_out in range(K / 64):

                # Data packing
                # Copy non-contiguous tiles into contiguous L2/L1 buffers
                A_pack = contiguous_copy(A[i_out*64 : i_out*64+64, k_out*64 : k_out*64+64])
                B_pack = contiguous_copy(B[k_out*64 : k_out*64+64, j_out*64 : j_out*64+64])

                # Block navigation (L2 / L1 Cache)
                for i_in in range(64 / 3):
                    for j_in in range(64 / 16):

                        # Load hoisting
                        C_vec_0 = C[i,     j : j+16]
                        C_vec_1 = C[i + 1, j : j+16]
                        C_vec_2 = C[i + 2, j : j+16]

                        # Accumulation loop
                        for k_in in range(64):
                            # Reading from contiguous packed buffer
                            B_vec = B_pack[k_in, j_in : j_in+16]

                            # Unrolled Micro-kernel
                            C_vec_0 += A_pack[i_in,     k_in] * B_vec
                            C_vec_1 += A_pack[i_in + 1, k_in] * B_vec
                            C_vec_2 += A_pack[i_in + 2, k_in] * B_vec

                        # Store hoisting
                        C[i,     j : j+16] = C_vec_0
                        C[i + 1, j : j+16] = C_vec_1
                        C[i + 2, j : j+16] = C_vec_2
    ```
    *Hypothesis to test: Packing may improve locality and reduce data-TLB pressure enough to offset copying costs. Compare total runtime and memory metrics. The default Level 3 diagram does not directly report data-TLB misses; ITLB Miss concerns code, not matrix data.*

    {btn_ex7}
    """)
    return


@app.cell(hide_code=True)
def _(btn_ex7, btn_all, mo, run_experiment):
    mo.stop(not (btn_ex7.value or btn_all.value))

    spec_7 = '''
        i:
        j:
        k:
        A: pack
        B: pack
        i#64:
        j#64:
        k#64:
        i#3: unroll
        j#16: vectorize
    '''
    run_experiment(spec_7)
    return



@app.cell(hide_code=True)
def _():
    _sandbox_code = """import xtc.graphs.xtc.op as O
    from xtc.backends.mlir import Backend
    from xtc.schedules.descript import descript_scheduler

    # 1. Problem setup
    I, J, K, dtype = 256, 256, 256, "float32"
    a = O.tensor((I, K), dtype, name="A")
    b = O.tensor((K, J), dtype, name="B")

    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="C")

    # 2. Scheduling
    backend = Backend(gb.graph)
    scheduler = backend.get_scheduler()

    schedule_spec = '''
        i:
        j:
        k:
        A: pack
        B: pack
        i#64:
        j#64:
        k#64:
        i#3: unroll
        j#16: vectorize
    '''

    descript_scheduler(
        scheduler=scheduler, 
        node_name="C", 
        abstract_dims=["i", "j", "k"], 
        abstract_matrix=["A", "B", "C"],
        spec=schedule_spec
    )
    sched = scheduler.schedule()

    # 3. Compilation
    compiler = backend.get_compiler(
        dump_file="sandbox_mlir", 
        shared_lib=True,
        print_assembly=False
    )
    module = compiler.compile(sched)

    # 4. Evaluation
    import sys
    hw_counters = ["Cycles","TopdownL1", "TopdownL2","TopdownL3_Mem"] if sys.platform == "linux" else []
    evaluator = module.get_evaluator(validate=True, hw_counters=hw_counters)

    print("Evaluating Schedule...")
    results, code, err = evaluator.evaluate()

    if err:
        print(f"Error:\\n{err}")
    else:
        print(f"Raw Results Array: {[round(x, 2) for x in results]}")
    """
    api_sandbox_editor = mo.ui.code_editor(
        value=_sandbox_code,
        language="python",
        label="XTC Full API Sandbox"
    )
    btn_api_sandbox = mo.ui.run_button(kind="neutral", label="Run Sandbox")

    sandbox_ui = mo.vstack([
        mo.md(r"""
        ---
        ## Complete API Sandbox

        Below is a fully editable XTC script. It demonstrates the complete API workflow: defining the computational graph, applying a schedule, compiling to MLIR, and running the hardware evaluation. 

        Feel free to modify the tensor sizes, the schedule specification, or the list of hardware counters. If `Topdown` metrics are requested, the microarchitecture pipeline diagram will automatically be rendered below the output.
        """),
        api_sandbox_editor,
        btn_api_sandbox
    ])
    return api_sandbox_editor, btn_api_sandbox, sandbox_ui


@app.cell
def _(sandbox_ui):
    sandbox_ui
    return


@app.cell
def _(mo):
    tma_support_content = mo.md(
        """
        | Microarchitecture | Supported TMA Levels | Execution Mode |
        |---|---|---|
        | **Intel Skylake / Cascade Lake** | `TopdownL1`, `TopdownL2`, `TopdownL3_Mem`, `TopdownL3` | Native API |
        | **Intel Modern (Ice Lake+)**     | `TopdownL1`, `TopdownL2`, `TopdownL3_Mem`, `TopdownL3` | Native API |
        | **AMD Zen 4**                    | `TopdownL1`, `TopdownL2` | Native API |
        | **ARM**                          | `TopdownL1 ?`, `TopdownL2 ?` | Native API (untested) |
        | **Generic Linux (`perf` tool)**  | `TopdownL1` to `TopdownL6` | System Fallback *(Multiplexed)* |

        *Note 1: TopdownL3_Mem returns [L1 Bound, L2 Bound, L3 Bound, DRAM Bound, Store Bound].*

        *Note 2: AMD architectures do not have native Topdown metrics. They are rebuilt using AMD-specific hardware events and pipeline width formulas.*
        """
    )

    amd_zen4_formulas = mo.md(
        """
        **Zen 4 Pipeline Width:** 6 slots per cycle.
        `Total Slots = Cycles * 6`

        **Level 1 Formulas:**
        *   **Frontend Bound:** `Frontend Stalls / Total Slots`
        *   **Retiring:** `Ops Retired / Total Slots`
        *   **Bad Speculation:** `(Ops Dispatched - Ops Retired) / Total Slots`
        *   **Backend Bound:** `100% - (Frontend Bound + Retiring + Bad Speculation)`

        **Level 2 Formulas:**
        *   **Fetch Latency:** `Fetch Latency Stalls / Total Slots`
        *   **Fetch Bandwidth:** `Frontend Bound - Fetch Latency`
        *   **Memory Bound:** `Backend Memory Stalls / Total Slots`
        *   **Core Bound:** `Backend CPU Stalls / Total Slots`
        *   **Branch Mispredicts:** `Branch Mispredict Stalls / Total Slots`
        *   **Machine Clears:** `Pipeline Restarts / Total Slots`
        *   **Heavy Ops:** `Microcode Ops Retired / Total Slots`
        *   **Light Ops:** `Total Retiring - Heavy Ops`
        """
    )

    zen4_accordion = mo.accordion({
        "Show AMD Zen 4 TMA reconstruction formulas": amd_zen4_formulas
    })

    fallback_note = mo.md("*Unsupported metrics will automatically use the `perf` fallback.*")

    supported_arch_msg = mo.accordion({
        "View supported TMA architectures": mo.vstack([
            tma_support_content,
            zen4_accordion,
            fallback_note
        ])
    })

    return (supported_arch_msg,)

@app.cell
def _(mo):
    l1_md = mo.md(
        """
        **Array Order:** `[Retiring, Bad Speculation, Frontend Bound, Backend Bound]`

        | Index | Metric | Description |
        |---|---|---|
        | `[0]` | **Retiring** | Fraction of slots utilized by useful work (issued uops that get retired). |
        | `[1]` | **Bad Speculation** | Fraction of slots wasted due to incorrect speculations. |
        | `[2]` | **Frontend Bound** | Fraction of slots where the Frontend undersupplies the Backend. |
        | `[3]` | **Backend Bound** | Fraction of slots where no uops are delivered due to a lack of Backend resources. |
        """
    )

    l2_md = mo.md(
        """
        **Array Order:** `[Light Ops, Heavy Ops, Machine Clears, Branch Mispredicts, Fetch Bandwidth, Fetch Latency, Core Bound, Memory Bound]`

        | Index | Category | Sub-Metric | Description |
        |---|---|---|---|
        | `[0]` | `Retiring` | `light_operations` | Retiring typical, single-uop instructions. |
        | `[1]` | `Retiring` | `heavy_operations` | Retiring complex, multi-uop or microcoded instructions. |
        | `[2]` | `Bad Speculation` | `machine_clears` | Wasted slots due to pipeline flushes (e.g., memory ordering issues, exceptions). |
        | `[3]` | `Bad Speculation` | `branch_mispredicts` | Wasted slots due to incorrect branch predictions. |
        | `[4]` | `Frontend Bound` | `fetch_bandwidth` | Frontend cannot decode or deliver enough instructions per cycle. |
        | `[5]` | `Frontend Bound` | `fetch_latency` | Frontend is completely starved and delivering nothing (e.g., Instruction Cache miss). |
        | `[6]` | `Backend Bound` | `core_bound` | Execution stalled waiting for execution units (ALU/FPU) or due to data dependencies. |
        | `[7]` | `Backend Bound` | `memory_bound` | Execution stalled waiting for data from the memory (L1/L2/L3/RAM). |
        """
    )

    l3_md = mo.md(
        """
        | Index | Category | Metric | Description |
        |---|---|---|---|
        | `[0]` | `Bad Speculation` | `branch_resteers` | Stalls due to branch resteers at execution stage. |
        | `[1]` | `Backend Bound` | `divider` | Cycles where the Divider unit was active. |
        | `[2]` | `Backend Bound` | `dram_bound` | Stalled on external memory (DRAM) accesses. |
        | `[3]` | `Frontend Bound` | `dsb` | Limited by Decoded Stream Buffer (uop cache) pipeline. |
        | `[4]` | `Frontend Bound` | `dsb_switches` | Stalls due to switching from DSB to MITE pipelines. |
        | `[5]` | `Retiring` | `few_uops_instructions` | Instructions decoded into 2 or more uops. |
        | `[6]` | `Retiring` | `fp_arith` | Floating-point (FP) operations executed. |
        | `[7]` | `Retiring` | `fused_instructions` | One uop representing multiple contiguous instructions. |
        | `[8]` | `Frontend Bound` | `icache_misses` | Stalls due to instruction cache misses. |
        | `[9]` | `Frontend Bound` | `itlb_misses` | Stalls due to Instruction TLB (ITLB) misses. |
        | `[10]` | `Backend Bound` | `l1_bound` | Stalled without missing the L1 Data (L1D) cache. |
        | `[11]` | `Backend Bound` | `l2_bound` | Stalled due to L2 cache accesses. |
        | `[12]` | `Backend Bound` | `l3_bound` | Stalled due to L3 cache accesses or sibling Core contention. |
        | `[13]` | `Frontend Bound` | `lcp` | Stalls due to Length Changing Prefixes. |
        | `[14]` | `Retiring` | `memory_operations` | Memory load or store uops retired. |
        | `[15]` | `Retiring` | `microcode_sequencer` | Uops fetched by the Microcode Sequencer (MS) unit. |
        | `[16]` | `Frontend Bound` | `mite` | Limited by MITE pipeline (legacy decode pipeline). |
        | `[17]` | `Frontend Bound` | `ms_switches` | Stalls due to switching uop delivery to the MS unit. |
        | `[18]` | `Retiring` | `non_fused_branches` | Branch instructions that were not fused. |
        | `[19]` | `Retiring` | `other_light_ops` | Remaining light uops not covered by other sibling nodes. |
        | `[20]` | `Bad Speculation` | `other_mispredicts` | Stalls due to other misprediction cases (non-retired branches). |
        | `[21]` | `Bad Speculation` | `other_nukes` | Machine Clears not related to memory ordering. |
        | `[22]` | `Backend Bound` | `pmm_bound` | Stalled on accesses to Persistent Memory Modules (Optane). |
        | `[23]` | `Backend Bound` | `ports_utilization` | Limited due to execution ports saturation (FPU/ALU contention). |
        | `[24]` | `Backend Bound` | `serializing_operation` | Issue-pipeline stalled due to serializing operations. |
        | `[25]` | `Backend Bound` | `store_bound` | Stalled due to store memory accesses and Read For Ownership(RFO) requests. |
        """
    )

    l4_md = mo.md(
        """

        | Metric | Description |
        |---|---|
        | `4k_aliasing` | Load accesses aliased by preceding stores with a 4K address offset. |
        | `assists` | Uops delivered by Microcode Sequencer as a result of Assists. |
        | `cisc` | Uops originated from CISC instructions. |
        | `clears_resteers` | Branch Resteers as a result of Machine Clears. |
        | `code_stlb_hit` / `miss` | ITLB missed, hitting or missing Second-level TLB (STLB). |
        | `contested_accesses` | Memory handling synchronizations due to contested accesses. |
        | `data_sharing` | Memory handling synchronizations due to data-sharing. |
        | `decoder0_alone` | Decoder-0 was the only active decoder. |
        | `dtlb_load` / `store` | DTLB missed by load or store accesses. |
        | `false_sharing` | CPU handling synchronizations due to False Sharing. |
        | `fb_full` | L1D Fill Buffer unavailability limited memory accesses. |
        | `fp_scalar` / `vector` | Arithmetic FP scalar or vector uops retired. |
        | `l1_latency_dependency` | Demand load accesses that hit the L1D cache. |
        | `l2_hit_latency` / `l3` | Demand load accesses that hit L2 or L3 under unloaded scenarios. |
        | `lock_latency` | Cache misses due to lock operations. |
        | `mem_bandwidth` | Approaching bandwidth limits of external memory (DRAM/HBM). |
        | `mem_latency` | Latency from external memory (DRAM/HBM). |
        | `mispredicts_resteers` | Branch Resteers due to Branch Misprediction. |
        | `nop_instructions` | NOP (no op) instructions retired. |
        | `ports_utilized_0/1/2/3m` | CPU executed 0, 1, 2, or 3+ uops per cycle on all execution ports. |
        | `split_loads` / `stores` | Handling memory split accesses crossing 64-byte boundaries. |
        | `sq_full` | Super Queue (SQ) was full. |
        | `store_fwd_blk` | Loads blocked unable to forward data from earlier overlapping stores. |
        | `store_latency` | CPU handling L1D store misses. |
        | `unknown_branches` | Stalls due to new branch address clears. |
        | `x87_use` | Approximation of legacy x87 usage. |
        """
    )

    l5_md = mo.md(
        """

        | Metric | Description |
        |---|---|
        | `alu/load/store_op_utilization` | Cycles CPU dispatched uops on execution ports for ALU, Load, or Store. |
        | `fp_assists` | Uops retired as a result of handing Floating Point (FP) Assists. |
        | `fp_vector_128b/256b/512b` | FP vector uops retired for 128, 256, or 512-bit wide vectors. |
        | `load/store_stlb_hit/miss` | DTLB/TLB missed by load/store accesses, hitting or missing STLB. |
        | `local/remote_mem` | Memory access constrained by local or remote NUMA nodes. |
        | `mixing_vectors` | Penalty for injected blend uops. |
        """
    )

    l6_md = mo.md(
        """

        | Metric | Description |
        |---|---|
        | `port_0` to `port_7` | Fraction of cycles the CPU dispatched uops on specific hardware execution ports (e.g., ALU, Loads, Stores). |
        | `load/store_stlb_miss_X` | Cycles to walk memory paging structures for 4K, 2M or 1G pages. |
        """
    )

    reminder_output_msg = mo.vstack([
        mo.md("**TMA Metrics Dictionary:**"),
        mo.accordion({
            "Level 1 (4 metrics)": l1_md,
            "Level 2 (8 metrics)": l2_md,
            "Level 3 (26 metrics)": l3_md,
            "Level 4 (32 metrics)": l4_md,
            "Level 5 (15 metrics)": l5_md,
            "Level 6 (8 metrics)": l6_md
        })
    ])

    return reminder_output_msg,

@app.cell
def _(reminder_output_msg):
    reminder_output_msg
    return


@app.cell
def _(api_sandbox_editor, btn_api_sandbox):
    mo.stop(not btn_api_sandbox.value, mo.md("*Click 'Run Sandbox' to see the output.*"))

    import io
    from contextlib import redirect_stdout, redirect_stderr
    import traceback

    out = io.StringIO()
    err = io.StringIO()
    local_vars = {}

    with redirect_stdout(out), redirect_stderr(err):
        try:
            exec(api_sandbox_editor.value, globals().copy(), local_vars)
        except Exception as e:
            traceback.print_exc()

    stdout_val = out.getvalue()
    stderr_val = err.getvalue()

    output_blocks = []
    if stdout_val:
        output_blocks.append(mo.md(f"**Standard Output:**\n```text\n{stdout_val}\n```"))
    if stderr_val:
        output_blocks.append(mo.md(f"**Standard Error / Exceptions:**\n```text\n{stderr_val}\n```"))

    if not stdout_val and not stderr_val:
        output_blocks.append(mo.md("*Script executed successfully with no output.*"))

    hw_counters = local_vars.get("hw_counters", [])
    results = local_vars.get("results", [])
    code = local_vars.get("code", -1)

    if code == 0 and hw_counters and results:
        sizes = {"TopdownL1": 4, "TopdownL2": 8, "TopdownL3": 26, "TopdownL3_Mem": 5}
        l1_res, l2_res, l3_res = None, None, None
        curr_idx = 0

        for c in hw_counters:
            s = sizes.get(c, 1)
            chunk = results[curr_idx : curr_idx + s]

            if c == "TopdownL1": l1_res = chunk
            if c == "TopdownL2": l2_res = chunk
            if c in ["TopdownL3", "TopdownL3_Mem"]: l3_res = chunk

            curr_idx += s

        if l1_res and len(l1_res) >= 4 and l1_res[0] >= 0:
            output_blocks.append(mo.md("**Generated Top-down Diagram:**"))
            output_blocks.append(plot_upipe(l1_res, l2_res, l3_res))

    mo.vstack(output_blocks),
    return mo.vstack(output_blocks),


if __name__ == "__main__":
    app.run()
