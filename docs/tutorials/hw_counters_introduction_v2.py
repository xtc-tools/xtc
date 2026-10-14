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
            import json
            import pandas as pd
            from d3blocks import D3Blocks
        except ImportError:
            return mo.callout(mo.md(" **Please install `d3blocks` (`pip install d3blocks`) to visualize the Sankey diagram**"), kind="warn")

        def _render_sankey_html(graph_links, node_info, node_colors, chart_height, synthetic_root):
            """Render D3Blocks output and adapt its browser interactions for Marimo."""
            chart_width = 1024
            title_height = 40
            synthetic_root_id = "xtcroot"

            d3 = D3Blocks()
            html = d3.sankey(
                pd.DataFrame(graph_links, columns=["source", "target", "weight"]),
                color=node_colors,
                node={"align": "justify", "width": 20, "padding": 15, "color": "black"},
                link={"color": "source-target", "stroke_opacity": 0.4},
                fontsize=12,
                title="Top-down Microarchitecture Pipeline µpipe",
                filepath=None,
                figsize=[chart_width, chart_height - title_height],
                showfig=False,
                notebook=False,
                save_button=False,
                show_side_panel=False,
                show_top_panel=False,
                dark_mode=False,
                return_html=True,
            )

            # D3Blocks does not expose all tooltip and label options through
            # sankey(). Append an adapter after its own script rather than
            # patching implementation-specific JavaScript fragments.
            node_info_json = json.dumps(node_info, ensure_ascii=True)
            node_info_json = (
                node_info_json.replace("&", r"\u0026")
                .replace("<", r"\u003c")
                .replace(">", r"\u003e")
            )
            adapter_script = r"""
(() => {
    const nodeInfo = __NODE_INFO__;
    const hasSyntheticRoot = __HAS_SYNTHETIC_ROOT__;
    const syntheticRootId = __SYNTHETIC_ROOT_ID__;
    const svg = document.querySelector("#sankey-main svg");
    if (!svg || !svg.computed) return;

    // D3Blocks adds its own tooltip and native SVG title tooltips. Remove
    // both so only the bounded, description-aware tooltip below is visible.
    if (window.Tooltip) window.Tooltip.remove();
    svg.querySelectorAll("title").forEach((title) => title.remove());

    const tooltip = document.createElement("div");
    tooltip.setAttribute("role", "tooltip");
    tooltip.style.cssText = [
        "position:fixed", "z-index:200", "pointer-events:none", "visibility:hidden",
        "opacity:0", "max-width:min(360px, calc(100vw - 24px))",
        "max-height:calc(100vh - 16px)", "overflow-y:auto", "overflow-wrap:anywhere",
        "box-sizing:border-box", "white-space:normal", "padding:6px 9px",
        "border:1px solid rgba(0,0,0,.18)", "border-radius:5px",
        "background:#fff", "color:#111", "font:12px/1.4 Helvetica,Arial,sans-serif",
        "box-shadow:0 2px 8px rgba(0,0,0,.18)", "transition:opacity .1s"
    ].join(";");
    document.body.appendChild(tooltip);

    svg.querySelectorAll(".sankey-node-label").forEach((label) => {
        const index = Number(label.getAttribute("data-node-index"));
        const node = svg.computed.nodes[index];
        if (nodeInfo[node.id]) label.textContent = nodeInfo[node.id].label;
    });
    svg.querySelectorAll(".sankey-link").forEach((link) => {
        link.style.mixBlendMode = "normal";
    });
    svg.querySelectorAll(".sankey-node").forEach((node) => {
        node.style.strokeWidth = "0.5px";
    });

    if (hasSyntheticRoot) {
        const root = svg.computed.nodes.find((node) => node.id === syntheticRootId);
        svg.querySelectorAll(".sankey-link").forEach((link) => {
            link.style.display = "none";
        });
        if (root) {
            [".sankey-node", ".sankey-node-label"].forEach((selector) => {
                const element = svg.querySelector(
                    `${selector}[data-node-index="${root.index}"]`
                );
                if (element) element.style.display = "none";
            });
        }
    }

    function hideTooltip() {
        tooltip.style.opacity = "0";
        tooltip.style.visibility = "hidden";
    }

    function showTooltip(event, lines) {
        tooltip.replaceChildren(...lines);
        tooltip.style.left = "0px";
        tooltip.style.top = "0px";
        tooltip.style.visibility = "hidden";

        const bounds = tooltip.getBoundingClientRect();
        const margin = 8;
        const gap = 14;
        let left = event.clientX + gap;
        let top = event.clientY + gap;
        if (left + bounds.width > window.innerWidth - margin) {
            left = event.clientX - gap - bounds.width;
        }
        if (top + bounds.height > window.innerHeight - margin) {
            top = event.clientY - gap - bounds.height;
        }
        left = Math.max(margin, Math.min(left, window.innerWidth - bounds.width - margin));
        top = Math.max(margin, Math.min(top, window.innerHeight - bounds.height - margin));

        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
        tooltip.style.visibility = "visible";
        tooltip.style.opacity = "1";
    }

    window.__sankeyNodeHover__ = (node, event) => {
        const info = nodeInfo[node.id];
        if (!info) return;
        const label = document.createElement("strong");
        label.textContent = info.label;
        const description = document.createElement("div");
        description.textContent = info.description;
        showTooltip(event, [label, description]);
    };
    window.__sankeyNodeMove__ = window.__sankeyNodeHover__;
    window.__sankeyNodeLeave__ = hideTooltip;
    window.__sankeyLinkHover__ = (link, event) => {
        const source = nodeInfo[link.source.id];
        const target = nodeInfo[link.target.id];
        if (!source || !target) return;
        const labels = document.createElement("div");
        labels.textContent = `${source.label} → ${target.label}`;
        const value = document.createElement("div");
        value.textContent = `Value: ${Number(link.value).toFixed(1)}%`;
        showTooltip(event, [labels, value]);
    };
    window.__sankeyLinkMove__ = window.__sankeyLinkHover__;
    window.__sankeyLinkLeave__ = hideTooltip;
    window.__sankeyNodeClick__ = (_node, event) => event.stopPropagation();
})();
"""
            adapter_script = (
                adapter_script.replace("__NODE_INFO__", node_info_json)
                .replace("__HAS_SYNTHETIC_ROOT__", json.dumps(synthetic_root))
                .replace("__SYNTHETIC_ROOT_ID__", json.dumps(synthetic_root_id))
            )

            def replace_once(source, old, new):
                if source.count(old) != 1:
                    raise RuntimeError("Unexpected D3Blocks Sankey HTML structure")
                return source.replace(old, new, 1)

            html = replace_once(
                html,
                '<div id="sankey-main"></div>',
                '<div class="xtc-sankey-title">Top-down Microarchitecture Pipeline µpipe</div>\n'
                '<div id="sankey-main"></div>',
            )
            html = replace_once(
                html,
                "</head>",
                "<style>\n"
                "    html, body { height: auto; overflow: hidden; background: #fff; color: #111; }\n"
                "    .xtc-sankey-title { height: 40px; box-sizing: border-box; padding: 10px 20px 0; text-align: center; color: #111; font: 600 12px/20px Helvetica, Arial, sans-serif; }\n"
                "    #sankey-main { margin-top: 0; padding-bottom: 0; }\n"
                "    .sankey-node-label { fill: #111; font-size: 12px; }\n"
                "</style>\n</head>",
            )
            html = replace_once(
                html,
                "</body>",
                f"<script>\n{adapter_script}\n</script>\n</body>",
            )
            return mo.iframe(html, height=f"{chart_height}px")

        try:
            # Color palettes (L1, L2, L3)
            c_ret = "#4ade80" # Green
            c_bad = "#f87171" # Red
            c_fe  = "#60a5fa" # Blue
            c_be  = "#c084fc" # Purple

            desc_l1 = {
                "Retiring": "Slots attributed to completed operations, including arithmetic and bookkeeping; not a measure of algorithmic efficiency.",
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
                "Fetch Lat": "Instruction delivery pauses, for example during code fetch or branch recovery.",
                "Core Bound": "Execution limited by core resources or instruction dependencies.",
                "Mem Bound": "Execution stalled waiting for data from the memory (L1/L2/L3/RAM)."
            }
            desc_l3 = [
                "Frontend delay while fetching a redirected instruction path; can be associated with branch misprediction or pipeline clears.",
                "Intended to indicate divide/square-root unit activity. Native event encoding needs verification; do not diagnose from this value alone.",
                "Stalled on external memory (DRAM).",
                "DSB instruction-delivery activity in the native implementation, not a direct measure of lost capacity.",
                "Stalls due to switching from DSB to MITE.",
                "Intended to describe multi-uop instructions. Native formula does not isolate them; do not diagnose from this value alone.",
                "Floating-point (FP) operations executed.",
                "One uop representing multiple contiguous instructions.",
                "Instruction-cache activity; native events may count misses or stall cycles depending on the CPU. Not a uniform stall percentage.",
                "Intended to indicate code-address translation issues. Native event mapping needs verification, especially on modern Intel.",
                "Estimated load-related stalls without an L1D miss; translation or store-to-load blocking can also contribute.",
                "Estimated stalls associated with L2 accesses; not proof of insufficient data reuse.",
                "Estimated stalls associated with L3 accesses or inter-core contention; not proof of insufficient data reuse.",
                "Stalls due to Length Changing Prefixes.",
                "Memory load or store uops retired.",
                "Uops fetched by the Microcode Sequencer.",
                "Legacy-decoder instruction-delivery activity in the native implementation, not a direct measure of lost capacity.",
                "Intended to indicate transitions to microcode delivery. Native event encoding needs verification; do not diagnose from this value alone.",
                "Branch instructions that were not fused.",
                "Remaining light uops not covered by sibling nodes.",
                "Intended residual speculation category. Native formula uses an incorrect input; do not diagnose from this value alone.",
                "Intended to describe non-memory-ordering machine clears. Native formula uses an incorrect input; do not diagnose from this value alone.",
                "Stalled on accesses to Persistent Memory Modules.",
                "Intended to indicate core-resource or dependency limits, not just port saturation. Native reconstruction needs verification, especially on modern Intel.",
                "Intended to indicate issue stalls from serializing operations. Native event mapping/reconstruction needs verification.",
                "Stalled due to store memory accesses and RFO requests."
            ]

            nodes_label = []
            nodes_color = []
            nodes_desc = []
            links_source = []
            links_target = []
            links_value = []

            def add_node(name, pct, color, desc=""):
                nodes_label.append(f"{name} ({pct:.1f}%)")
                nodes_color.append(color)
                nodes_desc.append(desc)
                return len(nodes_label) - 1

            def add_link(source, target, value):
                links_source.append(source)
                links_target.append(target)
                links_value.append(max(value, 0.1)) # 0.1 ensures 0% metrics remain visible as thin lines

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
                    add_link(parent_node, n_l2, l2_val)

                    if l3_res and l3_mappings and len(l3_res) > 0 and l3_res[0] >= 0:
                        for l3_name, l3_idx in l3_mappings:
                            if l3_idx < len(l3_res):
                                l3_val = l3_res[l3_idx]
                                desc_text = desc_l3[l3_idx] if len(desc_l3) > l3_idx else ""
                                n_l3 = add_node(l3_name, l3_val, color, desc_text)
                                add_link(n_l2, n_l3, l3_val)

                map_light = map_heavy = map_clears = map_misp = map_fbw = map_flat = map_core = map_mem = None

                if l3_res and len(l3_res) >= 26:
                    map_light = [("FP Arith", 6), ("Mem Ops", 14), ("Fused", 7), ("Non-Fused", 18), ("Other Light", 19)]
                    map_heavy = [("Few Uops", 5), ("Microcode", 15)]
                    map_clears = [("Other Nukes", 21)]
                    map_misp = [("Other Misp.", 20)]
                    map_fbw = [("MITE", 16), ("DSB", 3)]
                    map_flat = [("ICache Miss", 8), ("ITLB Miss", 9), ("Branch Rest.", 0), ("MS Switches", 17), ("DSB Swit.", 4), ("LCP", 13)]
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

            # D3Blocks cleans source/target labels, so use stable internal IDs
            # and keep the original display labels and descriptions separately.
            node_ids = [f"n{i}" for i in range(len(nodes_label))]
            node_info = {
                node_id: {
                    "label": label,
                    "description": description,
                }
                for node_id, label, description in zip(node_ids, nodes_label, nodes_desc)
            }
            node_colors = dict(zip(node_ids, nodes_color))
            graph_links = [
                {
                    "source": node_ids[source],
                    "target": node_ids[target],
                    "weight": value,
                }
                for source, target, value in zip(links_source, links_target, links_value)
            ]

            # D3's Sankey layout needs at least one link. Keep a Level-1-only
            # result visible by adding hidden links to an internal helper node.
            synthetic_root = not graph_links
            synthetic_root_id = "xtcroot"
            if synthetic_root:
                node_info[synthetic_root_id] = {"label": "", "description": ""}
                node_colors[synthetic_root_id] = "#ffffff"
                graph_links = [
                    {
                        "source": node_ids[index],
                        "target": synthetic_root_id,
                        "weight": max(l1_res[index], 0.1),
                    }
                    for index in range(4)
                ]

            chart_height = 650 if (l3_res and len(l3_res) > 5) else 450
            return _render_sankey_html(
                graph_links, node_info, node_colors, chart_height, synthetic_root
            )

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
        reference_macs_per_sec = rt.evaluate_flops(dtype)

        # Evaluate pure execution time
        eval_time = module.get_evaluator(validate=False)
        time_results, _, _ = eval_time.evaluate()
        exec_time = min(time_results)

        # Compare MAC throughput with a measured arithmetic microbenchmark, not a theoretical maximum.
        reference_perf_pct = (I * J * K) / exec_time / reference_macs_per_sec * 100

        # Illustrative progress bar relative to the measured reference.
        bar_color = "#ef4444" if reference_perf_pct < 10 else ("#f59e0b" if reference_perf_pct < 50 else "#22c55e")
        fill_width = max(min(reference_perf_pct, 100), 2) # Ensure at least 2% so the text is visible

        perf_ui = mo.Html(f'''
        <div style="margin-bottom: 15px; width: 100%; max-width: 600px;">
            <strong>Throughput (% of Measured Arithmetic Reference):</strong>
            <div style="width: 100%; background-color: #f3f4f6; border-radius: 6px; margin-top: 6px; overflow: hidden; border: 1px solid #d1d5db; display: flex;">
                <div style="width: {fill_width}%; background-color: {bar_color}; color: #000; padding: 4px 0; text-align: center; font-size: 12px; font-family: monospace; font-weight: bold; white-space: nowrap; transition: width 0.5s ease-in-out;">
                    {reference_perf_pct:.1f}%
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
    import re

    def _counter_warning(limitation):
        return mo.callout(mo.md(
            "**Verify counter support and implementation before interpreting.**\n\n"
            f"{limitation}\n\n"
            "Do not infer a cause or select a schedule from this native value alone. "
            "Confirm with independently verified counters, generated assembly, and runtime. "
            "An incorrect event or formula is not merely measurement noise."
        ), kind="warn")

    _step_anchors = {
        2: "step-2-vectorization",
        3: "step-3-register-tiling-2x16",
        4: "step-4-loadstore-hoisting",
        5: "step-5-cache-tiling",
        6: "step-6-register-tile-exploration-3x16",
        7: "step-7-data-packing-memory-layout-optimization",
    }

    def _hint(meaning, causes, check, experiment, caution="", deeper="", native_warning=""):
        def _link_steps(match):
            _steps = [match.group(1)]
            if match.group(2):
                _steps.append(match.group(2))
            return " and ".join(
                f"[Step {step}](#{_step_anchors[int(step)]})" for step in _steps
            )

        _experiment = re.sub(
            r"\bSteps? ([2-7])(?: and ([2-7]))?\b",
            _link_steps,
            experiment,
        )
        _sections = [
            f"#### What it means\n\n{meaning}",
            f"#### Possible causes\n\n{causes}",
            f"#### Check the generated assembly\n\n{check}",
            f"#### Scheduling experiment\n\n{_experiment}",
        ]
        if caution:
            _sections.append(f"#### Interpretation limits\n\n{caution}")
        if deeper:
            _sections.append(f"#### Additional counters\n\n{deeper}")
        _content = mo.md("\n\n".join(_sections))
        if native_warning:
            return mo.vstack([_counter_warning(native_warning), _content])
        return _content

    def _metric_records(group, overview, metrics):
        """Prefix a Level 2 group onto its overview and metric records."""
        records = {f"{group} · Overview": overview}
        records.update({f"{group} · {label}": content for label, content in metrics.items()})
        return records

    _memory = _metric_records(
        "Memory Bound",
        overview=mo.md(r"""
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
        metrics={
            "L1 Bound · l1_bound": _hint(
                "This estimates load-related stalls without an L1 data-cache miss. It is not an L1 miss rate, and does not prove that all the waiting is for L1 hits.",
                "Repeated accumulator loads; dependent load/use chains; address-translation delays (data-TLB misses); loads blocked by earlier stores; limited load throughput.",
                "In the assembly, is C loaded inside every k iteration? Are temporary values stored on the stack (register spills)? Are loads immediately used in a dependent chain?",
                "Keep C accumulators in registers (Step 4), reuse loaded operands, or add independent accumulators (Step 3). Reduce the register tile if it causes spills.",
                "A larger cache tile will not necessarily fix dependencies, translation delays, or load-throughput limits. Packing is an experiment to consider if independent evidence suggests data-TLB pressure.",
                "`dtlb_load`, `l1_latency_dependency`, `store_fwd_blk`, `split_loads`, and load-port metrics, if supported.",
            ),
            "L2 Bound · l2_bound": _hint(
                "This estimates stalls associated with loads accessing L2, typically after an L1 miss.",
                "Unavoidable first accesses; active tiles exceed useful L1 capacity; cache conflicts; loop order delays reuse; strided accesses waste cache-line data.",
                "Estimate the active A/B/C data and inspect which operand is reused by consecutive inner-loop iterations. Include any packed buffers.",
                "Reduce the cache tile or change loop order to reuse data sooner (Step 5). Try packing strided operands (Step 7), measuring the copying cost too.",
                "This does not prove that reuse is poor or that the tile is too large. L2 generally serves data faster than farther caches or DRAM; minimize total runtime, not this percentage in isolation.",
                "`l2_hit_latency`, `dtlb_load`, and `split_loads` can help distinguish latency, address translation, and access-layout problems.",
            ),
            "L3 Bound · l3_bound": _hint(
                "This estimates stalls associated with loads accessing L3 or inter-core contention, typically after misses in closer caches.",
                "Unavoidable first accesses; tiles exceed useful L1/L2 capacity; cache conflicts; loop order delays reuse; large strides waste cache lines; contention with other activity.",
                "Consider A, B, and C together. How much data is accessed between two uses of the same tile? Is the machine running other memory-intensive work?",
                "Reduce cache-tile sizes or reorder loops to reuse a tile sooner (Step 5). Test packing for strided accesses (Step 7). Repeat on a less busy machine.",
                "This proves neither poor reuse nor that L3 is too small. L3 generally serves data faster than DRAM; investigate the access pattern before attributing a cause.",
                "`l3_hit_latency`, `contested_accesses`, and `data_sharing`, where available.",
            ),
            "DRAM Bound · dram_bound": _hint(
                "Loads reaching main memory contribute to stalls.",
                "Little reuse before eviction; a large working set; strided accesses; insufficient independent loads to hide latency; competition for memory bandwidth.",
                "Does the loop repeatedly scan a matrix before reusing it? Is a cache line mostly unused? Compare smaller problem sizes and repeat measurements to assess sensitivity to cache residency.",
                "Use cache tiling and better loop order (Step 5), and test packing (Step 7). Reuse each loaded A/B value for more arithmetic rather than increasing data traffic.",
                "A high value alone cannot distinguish memory latency from bandwidth saturation. Warm-up and cache residency can change the result.",
                "`mem_latency` versus `mem_bandwidth`; `local_mem` / `remote_mem` on multi-socket NUMA machines.",
            ),
            "Store Bound · store_bound": _hint(
                "Store-related resources or obtaining ownership of cache lines contribute to stalls.",
                "C is written during every reduction iteration; register spills; scattered stores; packed buffers are copied too often.",
                "Count stores in the inner loop and inspect stack accesses. Check whether each output vector can be accumulated before writing it once.",
                "Hoist accumulator loads/stores out of the reduction (Step 4). Reduce an oversized register tile and avoid unnecessary packing copies.",
                "Stores may require fetching a cache line for ownership (RFO), so writing can also generate reads. Packing adds stores and is not always beneficial.",
                "`store_latency`, `split_stores`, `store_fwd_blk`, and `dtlb_store`. False sharing matters if multiple threads write the same cache line, not normally in a single-thread kernel.",
            ),
            "PMM Bound · pmm_bound": _hint(
                "This metric concerns accesses to persistent-memory modules, such as Intel Optane, rather than ordinary DRAM.",
                "Data may reside in persistent memory on a supported system, or the metric may be unavailable or reconstructed differently on your CPU.",
                "Check whether the system actually has persistent memory and how this metric is collected. A zero or unavailable result is not evidence of a scheduling improvement.",
                "If applicable, increase data reuse and compare with allocation in ordinary DRAM. Otherwise, focus on the supported cache/DRAM metrics.",
                "The current native Intel implementation reports this metric as zero; it is not a meaningful diagnosis on that path.",
            ),
        },
    )

    _core = _metric_records(
        "Core Bound",
        overview=mo.md(r"""
        **Core Bound:** execution is limited by computational resources or
        dependencies between instructions rather than primarily by fetching data.
        An **execution port** routes operations to units such as arithmetic or
        load/store units. Independent instructions can overlap; a dependent
        accumulator update must wait for its input.

        A fast matrix multiplication may legitimately become compute-bound.
        Check whether the bottleneck is useful SIMD arithmetic or avoidable overhead.
        """),
        metrics={
            "Ports Util. · ports_utilization": _hint(
                "When collected correctly, this suggests core-resource or dependency limits, not a direct measurement of every port's occupancy. The native calculation is a residual estimate.",
                "Instructions compete for execution resources; too many loads or address calculations; too few independent accumulator chains.",
                "Verify the counter first. Inspect assembly for SIMD/FMA instructions, repeated loads, address arithmetic, and spills. Does each FMA wait for the previous update of the same accumulator?",
                "Reuse loaded operands or modestly increase independent register accumulators (Steps 3 and 6). Sweep register-tile sizes rather than assuming that larger is better.",
                "High arithmetic-unit utilization can be desirable. More unrolling can instead cause spills or a larger instruction footprint; accept a change only if runtime improves.",
                "`ports_utilized_0/1/2/3m`, `alu/load/store_op_utilization`, and individual `port_*` metrics. Port assignments depend on the CPU.",
                native_warning="The modern Intel event array does not match the shared calculation's core-stall/scoreboard input order. The resulting Ports Util. value is not a reliable diagnosis there; the older Intel path is also only a rough core-stall proxy.",
            ),
            "Divider · divider": _hint(
                "A correctly collected divider metric indicates divide/square-root unit activity; activity alone does not prove these operations are the bottleneck.",
                "Division, remainder, or square-root instructions; setup or helper code included in the measurement. The operations covered depend on the CPU and event.",
                "Verify the event encoding first, then look for division or square-root instructions in the measured kernel. Plain matrix multiplication should mostly use multiply/add, loads, stores, and loop bookkeeping.",
                "Test tile sizes or indexing that the compiler can simplify, and move invariant calculations outside inner loops where possible.",
                "Do not replace division with an approximate reciprocal without considering numerical correctness.",
                native_warning="The native divider event encoding differs from documented Intel divider events. Until it is verified for the CPU, this value cannot establish that division or square roots are present.",
            ),
            "Serializing · serializing_operation": _hint(
                "When collected correctly, this indicates issue stalls associated with operations that restrict overlapping execution.",
                "Fences, special serializing instructions, synchronization, or helper/instrumentation code in the measured region.",
                "Verify the event and calculation first, then inspect assembly for fences and special instructions. Determine whether any confirmed activity comes from the kernel or measurement/setup overhead.",
                "Repeat with enough kernel work to assess overhead, and keep unnecessary setup outside the timed region. Only remove synchronization if correctness is preserved.",
                "Ordinary dependence between successive FMAs is not the same as an explicitly serializing instruction.",
                native_warning="The native Skylake scoreboard event encoding needs verification, and the modern Intel event array does not match the shared calculation's input order. Do not use this native value as evidence that the kernel contains serializing instructions.",
            ),
        },
    )

    _fetch_latency = _metric_records(
        "Fetch Latency",
        overview=mo.md("**Fetch Latency:** instruction delivery pauses. Possible causes include code fetch, branch recovery, and transitions between instruction-delivery mechanisms. These clues can overlap; check the actual instruction stream before assigning a cause."),
        metrics={
            "ICache Miss · icache_misses": _hint(
                "Instruction-cache activity can be associated with frontend delays. This cache holds code, not A/B/C data; the native metric does not uniformly measure stall cycles.",
                "Aggressive unrolling produces a large kernel; many code paths or helper calls increase the active code footprint.",
                "Compare assembly size across schedules. Does extra unrolling duplicate large blocks of instructions?",
                "Reduce unrolling or the register tile and compare runtime. Favor a compact, repeatedly executed inner loop.",
                "Cache tiling of matrix data does not directly solve instruction-cache misses. The native older-Intel path counts tag misses per cycle, whereas the modern path uses a stall event; neither is a cache miss rate, and they are not directly comparable.",
            ),
            "ITLB Miss · itlb_misses": _hint(
                "A correctly collected instruction-TLB metric concerns code-address translation. The ITLB caches these translations; a miss count alone does not establish a large time cost.",
                "Code spans many pages; large generated kernels or scattered helper calls expand the instruction working set.",
                "Verify which event the label represents, then inspect code size and call structure, especially after increasing unrolling.",
                "Reduce code expansion and compare a smaller register tile or unroll factor.",
                "ITLB concerns instructions. Data packing may help data-TLB misses, but does not directly fix ITLB misses.",
                "`code_stlb_hit` / `code_stlb_miss` distinguish hits in a second-level translation cache from page-table walks.",
                native_warning="The native modern Intel ITLB label currently uses an instruction-cache tag-stall event, not an ITLB event. The older Intel path counts completed instruction page walks per cycle, not translation-stall cycles. Do not infer costly translation from the label or percentage alone.",
            ),
            "MS Switches · ms_switches": _hint(
                "A correctly collected metric estimates the cost of switching to the Microcode Sequencer (MS), which supplies operations for certain complex instructions.",
                "Complex instructions or exceptional cases interrupt the usual instruction-delivery path.",
                "Verify the event encoding first, then inspect assembly and compare Heavy Ops and Microcode. If supported, check assist metrics and whether unusual floating-point inputs trigger slow paths.",
                "Compare schedules that generate simpler hot-loop instructions. If assists are suspected, test representative normal inputs without changing the algorithm's correctness requirements.",
                "The switching cost and the amount of microcode-delivered work are different metrics.",
                "`assists` and `fp_assists`, where supported.",
                native_warning="The native MS Switches event encoding differs from Intel's documented event, and counting transitions alone is not the same as measuring their stall cost. Treat this value as unverified.",
            ),
            "Branch Rest. · branch_resteers": _hint(
                "This estimates frontend delay while fetching from a redirected instruction path. It is a Fetch Latency clue, not a direct branch-misprediction count.",
                "Branch mispredictions, other pipeline clears, or branch-target prediction issues; short loops and remainder paths are possibilities to inspect, not proven causes.",
                "Inspect branches and cleanup paths in the assembly. Compare with Branch Mispredicts and Machine Clears rather than assuming every redirect has the same cause.",
                "If branch overhead is confirmed, compare tile sizes with and without remainder handling or modestly unroll short hot loops. Check runtime and code size.",
                "Recovery costs can appear in both frontend and speculation metrics. These are overlapping clues, not independent percentages to add. Native collection does not cover every component of Intel's full metric.",
                "`mispredicts_resteers`, `clears_resteers`, and `unknown_branches`, if available.",
            ),
            "DSB Swit. · dsb_switches": _hint(
                "This estimates exposed delay when switching from the decoded-uop cache to the legacy decoder; it is a Fetch Latency clue.",
                "Code size, layout, or control flow causes delivery-source transitions; a larger unrolled loop is one possible contributor.",
                "Compare code footprint and branch structure before and after unrolling. Look for a runtime regression as the register tile grows.",
                "Reduce unrolling or simplify the hot loop, then measure again. Exact uop-cache capacity and layout rules are CPU-specific.",
            ),
            "LCP · lcp": _hint(
                "Length-changing instruction prefixes can cause legacy-decoder delay on CPUs that report this metric; Intel groups this under Fetch Latency.",
                "Particular instruction encodings in compiler-generated code; this is usually not a matrix-data locality issue.",
                "Inspect instruction encodings in the assembly and verify that the metric is supported on your CPU.",
                "Compare another schedule or compiler configuration that changes the hot-loop instruction mix. Prioritize larger, actionable bottlenecks first.",
                "This is a specialized diagnosis; students need not hand-edit assembly to proceed with the tutorial.",
            ),
        },
    )

    _fetch_bandwidth = _metric_records(
        "Fetch Bandwidth",
        overview=mo.md("**Fetch Bandwidth:** the frontend delivers instructions, but not fast enough. Too much loop/indexing overhead or expanded code can leave execution units underfed."),
        metrics={
            "MITE / DSB · mite, dsb": _hint(
                "MITE is the legacy instruction decoder; DSB is the Decoded Stream Buffer (a cache of already-decoded micro-operations, or uops).",
                "A hot loop may not fit the decoded-instruction cache well; its layout or instruction mix may constrain delivery; excessive unrolling may enlarge it.",
                "First confirm Fetch Bandwidth is high. Compare instruction count and code size across schedules, and inspect repeated loop/indexing work.",
                "Test moderate unrolling and SIMD to do more arithmetic per instruction, keeping the hot loop compact.",
                "The current native Intel implementation reports MITE/DSB delivery activity rather than Intel's estimated lost capacity. A large DSB or MITE value alone is not proof of a bottleneck.",
                "`decoder0_alone` can help identify legacy-decoder restrictions, where available.",
            ),
        },
    )

    _branches = _metric_records(
        "Branch Mispredicts",
        overview=mo.md("**Branch Mispredicts:** the CPU predicts which instructions come next incorrectly, then discards work. Regular matrix-multiplication loops are often predictable, so inspect branches, short loops, and remainder paths if a verified metric indicates substantial misprediction cost. Test modest unrolling or different tile sizes only after checking the assembly. Related fetching delays appear under **Frontend Bound → Fetch Latency → Branch Rest.**; do not add these overlapping clues as separate costs."),
        metrics={
            "Other Misp. · other_mispredicts": _counter_warning(
                "This is intended as a residual speculation category. The current native Intel calculation uses an instruction-cache input as a machine-clear input. Its value cannot reliably identify misprediction cases; repeating measurements does not fix an incorrect formula.",
            ),
        },
    )

    _clears = _metric_records(
        "Machine Clears",
        overview=mo.md("**Machine Clears:** the CPU restarts its pipeline for reasons other than an ordinary branch misprediction, such as memory-ordering recovery. This is not the same as a cache miss."),
        metrics={
            "Other Nukes · other_nukes": _counter_warning(
                "The standard metric concerns machine clears unrelated to memory ordering; 'nuke' means a pipeline restart. The current native Intel calculation uses an instruction-cache input as a machine-clear input, so this value cannot establish that such restarts occur. If independent evidence indicates clears, investigate CPU-specific events; never remove required synchronization merely to reduce a metric.",
            ),
        },
    )

    _light = _metric_records(
        "Light Ops",
        overview=mo.md("**Light Ops:** ordinary completed operations dominate. They may implement useful arithmetic, but also loads, stores, branches, and address calculations. Retiring does not mean that every instruction advances the matrix multiplication efficiently."),
        metrics={
            "Mem Ops · memory_operations": _hint(
                "Many retired operations are loads or stores. This is an instruction-mix metric, not itself a memory-stall metric.",
                "Repeated C loads/stores; limited reuse of A/B; register spills; packing copies.",
                "Inspect the inner-loop assembly: is C read and written every k iteration? Are values spilled to the stack? Interpret this together with Memory Bound.",
                "Keep accumulators in registers (Step 4), reuse operands, or shrink an oversized register tile. Include packing overhead when judging Step 7.",
                "Necessary, efficient loads can produce a high value. The objective is more useful arithmetic per byte and lower runtime, not zero memory operations.",
            ),
            "FP Arith · fp_arith": _hint(
                "Floating-point arithmetic is being completed; this is generally expected for matrix multiplication.",
                "Useful SIMD arithmetic, or scalar/narrow-vector operations that retire successfully but achieve little of the CPU's arithmetic peak.",
                "Inspect vector width and FMA use in assembly. Are there multiple independent accumulator chains? Check the actual SIMD capabilities of your CPU.",
                "Compare vectorization (Step 2) and register-tile sizes (Steps 3 and 6), using widths supported by the target.",
                "A high FP activity value is not a FLOP count or proof of peak throughput; event weighting can include FMA double-counting. Wider vectors and FMAs do more arithmetic per instruction; frequency and dependencies also matter.",
                "`fp_scalar` / `fp_vector`, `fp_vector_128b/256b/512b`, and `x87_use`, if supported.",
            ),
            "Fused / Non-Fused · fused_instructions, non_fused_branches": _hint(
                "These metrics concern instruction fusion, such as a comparison and branch handled together, and branches not fused that way.",
                "Loop-control instructions occupy a significant share of the work; the instruction sequence may or may not permit fusion on this CPU.",
                "Inspect comparisons and branches in the hot loop. Are very short loops doing little arithmetic per branch?",
                "Use SIMD or modest unrolling to amortize loop control over more useful work; compare runtime and code size.",
                "This fusion is not fused multiply-add (FMA). High fused-instruction activity is not inherently bad, and non-fused branches are not necessarily mispredicted.",
            ),
            "Other Light · other_light_ops": _hint(
                "Completed light operations are not explained by the listed sibling metrics.",
                "Address arithmetic, loop counters, moves, conversions, or residual accounting.",
                "Inspect how many non-arithmetic instructions occur per FMA in the hot loop. Check whether indexing expressions are recomputed unnecessarily.",
                "Compare SIMD, operand reuse, simpler indexing, and modest unrolling to amortize bookkeeping.",
                "This residual category alone does not identify an instruction. Confirm the hypothesis in assembly and runtime.",
                "`nop_instructions` and ALU-port metrics, where supported.",
            ),
        },
    )

    _heavy = _metric_records(
        "Heavy Ops",
        overview=mo.md("**Heavy Ops:** instructions expand into multiple uops or use microcode. 'Heavy' does not automatically mean inefficient; investigate their cost and purpose in this kernel."),
        metrics={
            "Few Uops · few_uops_instructions": _counter_warning(
                "The standard metric concerns retired instructions decoded into multiple uops. The current native Intel formula instead subtracts fused operations from total retired slots and does not isolate multi-uop instructions. Do not infer complex instructions, excess internal work, or an inefficient schedule from this value.",
            ),
            "Microcode · microcode_sequencer": _hint(
                "The Microcode Sequencer supplies a substantial amount of internal operations.",
                "Complex instructions, helper code, or floating-point assists for exceptional input cases, depending on the CPU.",
                "Inspect the measured assembly, compare MS Switches, and check supported assist metrics. Plain matrix-multiplication FMAs should not normally require persistent assisted execution.",
                "Compare a simpler generated kernel and test whether representative input values change the result. Investigate correctness and numerical requirements before changing floating-point behavior.",
                "Microcode activity is not synonymous with a stall, and this percentage alone cannot identify the responsible instruction.",
                "`assists`, `fp_assists`, and `cisc`, where available.",
            ),
        },
    )

    guide_categories = {
        "Backend Bound": {
            "summary": "**Backend Bound:** instructions cannot make progress because data or execution resources are unavailable. Distinguish Memory Bound from Core Bound before choosing between cache tiling and register/SIMD changes.",
            "metrics": {**_memory, **_core},
        },
        "Frontend Bound": {
            "summary": "**Frontend Bound:** the execution machinery is underfed. A larger unrolled kernel can hurt instruction delivery even when it improves data reuse. Distinguish interruptions in delivery (latency) from insufficient delivery rate (bandwidth).",
            "metrics": {**_fetch_latency, **_fetch_bandwidth},
        },
        "Bad Speculation": {
            "summary": "**Bad Speculation:** the CPU starts work before knowing it is needed, then discards it or restarts execution. Separate incorrect branch predictions from other pipeline clears; changing cache tiles is not a general cure.",
            "metrics": {**_branches, **_clears},
        },
        "Retiring": {
            "summary": "**Retiring:** many execution slots complete operations. This is usually better than stalling, but scalar arithmetic, loads/stores, and bookkeeping also retire. A scalar instruction can retire efficiently while computing far fewer elements than SIMD. Check work per instruction and avoidable overhead, rather than trying to minimize Retiring.",
            "metrics": {**_light, **_heavy},
        },
    }

    backend_metric_selector = mo.ui.dropdown(
        options=list(guide_categories["Backend Bound"]["metrics"]),
        value=next(iter(guide_categories["Backend Bound"]["metrics"])),
        searchable=True,
        label="Choose a Level 2 group or metric",
        full_width=True,
    )
    frontend_metric_selector = mo.ui.dropdown(
        options=list(guide_categories["Frontend Bound"]["metrics"]),
        value=next(iter(guide_categories["Frontend Bound"]["metrics"])),
        searchable=True,
        label="Choose a Level 2 group or metric",
        full_width=True,
    )
    speculation_metric_selector = mo.ui.dropdown(
        options=list(guide_categories["Bad Speculation"]["metrics"]),
        value=next(iter(guide_categories["Bad Speculation"]["metrics"])),
        searchable=True,
        label="Choose a Level 2 group or metric",
        full_width=True,
    )
    retiring_metric_selector = mo.ui.dropdown(
        options=list(guide_categories["Retiring"]["metrics"]),
        value=next(iter(guide_categories["Retiring"]["metrics"])),
        searchable=True,
        label="Choose a Level 2 group or metric",
        full_width=True,
    )

    return (
        guide_categories,
        backend_metric_selector,
        frontend_metric_selector,
        speculation_metric_selector,
        retiring_metric_selector,
    )


@app.cell(hide_code=True)
def _(
    mo,
    guide_categories,
    backend_metric_selector,
    frontend_metric_selector,
    speculation_metric_selector,
    retiring_metric_selector,
):
    def _category_panel(category_name, selector):
        category = guide_categories[category_name]
        selected = selector.value
        group_name, selected_item = selected.split(" · ", 1)
        detail = category["metrics"][selected]

        if selected_item == "Overview":
            heading = mo.md(f"### {group_name} overview")
        else:
            metric_name, counter_names = selected_item.rsplit(" · ", 1)
            heading = mo.md(
                f"### {metric_name}\n\n"
                f"**Level 2:** {group_name} · **Counter:** `{counter_names}`"
            )

        return mo.vstack([
            mo.md(category["summary"]),
            mo.hstack(
                [selector, mo.vstack([heading, detail], justify="start")],
                widths=[1, 3],
                justify="start",
                gap=1,
                wrap=True,
            ),
        ])

    category_tabs = mo.ui.tabs({
        "Backend Bound": _category_panel("Backend Bound", backend_metric_selector),
        "Frontend Bound": _category_panel("Frontend Bound", frontend_metric_selector),
        "Bad Speculation": _category_panel("Bad Speculation", speculation_metric_selector),
        "Retiring": _category_panel("Retiring", retiring_metric_selector),
    })

    _how_to_use = mo.md(r"""
    ### How to use this guide

    1. Start with a substantial **Level 1** category in the Sankey, then select a
       Level 2 overview or metric. Treat the largest stall category as a starting
       point, not a verdict that the schedule is bad.
    2. Check **View Generated Assembly** to see whether the proposed cause appears
       in the compiled inner loop.
    3. Change **one scheduling decision at a time**, repeat measurements under similar
       conditions, and keep a change only if **execution time improves**. A smaller
       percentage can hide more absolute time in a slower program.
    4. Validate numerical correctness when experimenting. A faster schedule is only
       useful if it still computes the right result.

    Low throughput relative to the reference is a reason to investigate, not proof
    that a schedule can approach that reference.
    The throughput bar compares matrix multiplication with a **measured arithmetic
    microbenchmark**, not a theoretical hardware maximum. Its colors are illustrative,
    not universal performance thresholds. Hardware counters are clues, not a diagnosis;
    results vary with the CPU, compiler, matrix size, cache residency, and system activity.
    """)

    _vocabulary = mo.md(r"""
    ### Vocabulary

    The **frontend** fetches and decodes instructions; the **backend** executes them
    and accesses data. A **uop** is an internal micro-operation. **Retiring** means
    execution has completed and its result is committed—not that the instruction was
    necessary for an efficient algorithm. **SIMD** performs arithmetic on multiple
    elements per instruction; independent accumulator chains let the CPU overlap work.

    Top-down **Levels 1, 2, and 3 are analysis depths**, not the L1/L2/L3 data caches.
    Retiring and instruction-delivery activity are not themselves wasted work.
    """)

    _counter_limits = mo.md(r"""
    ### Counter interpretation and limitations

    Formulas, denominators, and availability vary by architecture. Reconstructed
    metrics and separate measurement passes need not form exact parent/child percentage
    partitions. Do not add leaf values as if they were mutually exclusive or use a
    universal "bad" threshold. Unsupported metrics are not evidence that a bottleneck
    is absent. The hierarchy uses Intel's conventional grouping as a navigation aid;
    it does not make approximate or unverified values a valid percentage partition.

    Some Level 3 event mappings and formulas in the current runtime are not reliable
    implementations of Intel's standard metrics. Verify **Few Uops, Other Misp., Other
    Nukes, Divider, MS Switches, Ports Util., Serializing, and ITLB Miss** before
    interpreting them. Repeated measurements cannot validate an incorrect event or
    formula. MITE/DSB and ICache values also require care because activity or event
    counts are not necessarily stall percentages. These warnings concern the native
    path; independently verify fallback metrics against your CPU's definitions too.

    Optional deeper clues refer to counters that may be available separately, not
    counters automatically collected here. Step examples are schematic pseudocode, not
    guaranteed generated instructions; boundary and remainder handling is omitted.
    For authoritative definitions, see [Intel's performance-monitoring metrics](https://github.com/intel/perfmon).
    """)

    mo.vstack([
        mo.md(r"""
        ---
        ## Diagnose a schedule bottleneck

        Choose a Level 1 category from the diagram, then search by metric label or raw
        counter name. Open a Level 2 overview for context or select a metric for
        targeted advice about assembly checks and scheduling experiments.
        """),
        category_tabs,
        mo.md("## Reference"),
        mo.ui.tabs({
            "How to use this guide": _how_to_use,
            "Vocabulary": _vocabulary,
            "Counter limitations": _counter_limits,
        }),
    ])
    return category_tabs,


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
    Unrolling exposes work on multiple output vectors, giving the CPU additional independent accumulator chains to overlap. Each accumulator still depends on its own previous update. Register residency and the amount of overlap depend on the generated code; inspect the assembly rather than assuming either is guaranteed.

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
    This schedule changes the placement of the reduction relative to the unrolled/vectorized dimensions, with the aim of exposing register accumulators. The pseudocode shows the intended load/store hoisting: keep `C` in registers through the reduction, then write it back. Check the generated assembly to confirm this happens and that register pressure does not introduce spills.

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
    ## Step 6: Register Tile Exploration (3x16)
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
        | `[0]` | **Retiring** | Slots attributed to completed operations, including arithmetic, data movement, and bookkeeping; not a measure of algorithmic efficiency. |
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
        | `[5]` | `Frontend Bound` | `fetch_latency` | Instruction delivery pauses, for example during code fetch or branch recovery. |
        | `[6]` | `Backend Bound` | `core_bound` | Execution stalled waiting for execution units (ALU/FPU) or due to data dependencies. |
        | `[7]` | `Backend Bound` | `memory_bound` | Execution stalled waiting for data from the memory (L1/L2/L3/RAM). |
        """
    )

    l3_md = mo.md(
        """
        **Interpretation warning:** these names follow Intel's conventional metrics,
        but the current native implementation does not consistently reproduce their
        definitions. Several entries below are unverified or unreliable; see the
        troubleshooting guide before using their values to diagnose a schedule.
        Activity counts, stalls, and slot fractions are not interchangeable.

        | Index | Category | Metric | Description |
        |---|---|---|---|
        | `[0]` | `Frontend Bound` | `branch_resteers` | Estimated fetching delay after control-flow redirects; a Fetch Latency clue that can overlap speculation/recovery costs. |
        | `[1]` | `Backend Bound` | `divider` | Intended divide/square-root unit activity. Native encoding needs verification; do not diagnose from this value alone. |
        | `[2]` | `Backend Bound` | `dram_bound` | Stalled on external memory (DRAM) accesses. |
        | `[3]` | `Frontend Bound` | `dsb` | Native DSB (decoded-uop cache) delivery activity, not a direct bottleneck percentage. |
        | `[4]` | `Frontend Bound` | `dsb_switches` | Estimated delay from switching DSB to MITE; a Fetch Latency clue. |
        | `[5]` | `Retiring` | `few_uops_instructions` | Intended multi-uop instruction metric. Native formula does not isolate these instructions; unreliable for diagnosis. |
        | `[6]` | `Retiring` | `fp_arith` | Retired floating-point activity; not a FLOP count or proof of peak throughput. Event weighting can include FMA double-counting. |
        | `[7]` | `Retiring` | `fused_instructions` | Instruction fusion, such as comparison plus branch; not fused multiply-add (FMA). |
        | `[8]` | `Frontend Bound` | `icache_misses` | Instruction-cache activity; native paths may count misses or stall cycles, so values are not directly comparable. |
        | `[9]` | `Frontend Bound` | `itlb_misses` | Intended code-translation metric. Native modern Intel uses an instruction-cache tag-stall event instead; older Intel counts page walks per cycle. Verify before interpreting. |
        | `[10]` | `Backend Bound` | `l1_bound` | Estimated load-related stalls without an L1D miss; translation and store-to-load blocking can contribute. |
        | `[11]` | `Backend Bound` | `l2_bound` | Estimated stalls associated with L2 accesses; not proof that tiles are too large or reuse is poor. |
        | `[12]` | `Backend Bound` | `l3_bound` | Estimated stalls associated with L3 accesses or inter-core contention; not proof of poor reuse. |
        | `[13]` | `Frontend Bound` | `lcp` | Decode delay associated with length-changing prefixes; a Fetch Latency clue. |
        | `[14]` | `Retiring` | `memory_operations` | Memory load or store uops retired. |
        | `[15]` | `Retiring` | `microcode_sequencer` | Microcode-delivered operations; native delivery activity is not itself a stall metric or necessarily all retired work. |
        | `[16]` | `Frontend Bound` | `mite` | Native legacy-decoder delivery activity, not a direct bottleneck percentage. |
        | `[17]` | `Frontend Bound` | `ms_switches` | Intended delay from microcode-delivery transitions. Native encoding needs verification; transition counts and stall costs differ. |
        | `[18]` | `Retiring` | `non_fused_branches` | Branch instructions that were not fused. |
        | `[19]` | `Retiring` | `other_light_ops` | Remaining light uops not covered by other sibling nodes. |
        | `[20]` | `Bad Speculation` | `other_mispredicts` | Intended residual speculation category. Native formula uses an incorrect input; unreliable for diagnosis. |
        | `[21]` | `Bad Speculation` | `other_nukes` | Intended non-memory-ordering machine clears. Native formula uses an incorrect input; unreliable for diagnosis. |
        | `[22]` | `Backend Bound` | `pmm_bound` | Intended persistent-memory stalls (e.g., Optane); currently zero on the native Intel path. |
        | `[23]` | `Backend Bound` | `ports_utilization` | Intended core-resource/dependency limits, not just port saturation. Native reconstruction is a proxy and has input-order issues on modern Intel. |
        | `[24]` | `Backend Bound` | `serializing_operation` | Intended issue stalls from serializing operations. Native event mapping/calculation needs verification. |
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
