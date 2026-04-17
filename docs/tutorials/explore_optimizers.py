import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import subprocess
    from pathlib import Path
    import numpy as np
    import random
    import xtc.cli.explore as explore
    OPERATORS = explore.OPERATORS
    STRATEGIES_ALIASES = explore.STRATEGIES_ALIASES
    return OPERATORS, STRATEGIES_ALIASES, explore, mo, np, random


@app.cell
def _(OPERATORS, explore, np, random):
    import multiprocessing

    class Args:
        def __init__(self, **overrides):
            self.operator = "matmul"
            self.op_name = None
            self.ops_list = False
            self.func_name = None
            self.strategy = None
            self.search = "random"
            self.backends = ["mlir"]
            self.optimizer = "random-forest-default"
            self.data = None
            self.dims = None
            self.huge_pages = True
            self.test = []
            self.opt_level = 4
            self.dtype = None
            self.trials = 100
            self.threads = 1
            self.max_unroll = 512
            self.seed = 0
            self.output = "results.csv"
            self.resume = False
            self.append = False
            self.eval = "eval"
            self.repeat = 1
            self.number = 1
            self.min_repeat_ms = 100
            self.validate = None
            self.save_temps = None
            self.save_temps_dir = "./save_temps_dir"
            self.explore_dir = "."
            self.optimizer_config = None
            self.child = True
            self.bare_ptr = False
            self.jobs = max(1, multiprocessing.cpu_count() // 2)
            self.execute = True
            self.peak_flops = None
            self.mlir_prefix = None
            self.batch = 1
            self.debug = None
            self.debug_compile = None
            self.debug_xtc = None
            self.debug_optimizer = None
            self.quiet = None
            self.dump = None

            self.__dict__.update(overrides)

            if not self.func_name:
                self.func_name = self.operator

            if not self.strategy:
                self.strategy = OPERATORS[self.operator]["default_strategy"]

            if not self.dims:
                self.dims = OPERATORS[self.operator]["default_dims"]

            if not self.dtype:
                self.dtype = OPERATORS[self.operator]["default_type"]

            if self.op_name:
                self.dims = explore.get_operation_dims(self.operator, self.op_name)

            # backend validation
            for backend in self.backends:
                assert backend in OPERATORS[self.operator]["backends"], (
                    f"backend {backend} not available for operator {self.operator}"
                )

            if self.seed >= 0:
                np.random.seed(self.seed)
                random.seed(self.seed)

            explore.setup_args(self)
    return (Args,)


@app.cell
def _(STRATEGIES_ALIASES, mo):
    from xtc.artifacts import list_operations 
    from xtc.search.strategies import Strategies
    from xtc.search.optimizers import Optimizers

    operators = ["matmul","conv2d"]
    backends = ["mlir","tvm","jir"]
    strategies = list(STRATEGIES_ALIASES.keys()) + list(Strategies.names()) 
    backend_ui = mo.ui.multiselect(options=backends,value=["mlir"],label="backends:")
    operator_ui = mo.ui.radio(options=operators,value="matmul",label="operator:")
    def op_names(operator):
        ops_list = [op[1] for op in list_operations(operator)]
        return mo.ui.dropdown(
            options = ops_list,
            value=ops_list[0],
            label="operator name:",
        )
    strategy_ui = mo.ui.dropdown(
        options=strategies,
        value=strategies[-1],
        label="strategy:"
    )
    seed_ui = mo.ui.number(start=0,stop=200,label="seed:")
    trials_ui = mo.ui.number(start=0,stop=5000,value=100,label="trials:")

    search_types = ["random","iterative","exhaustive"]
    search_select = mo.ui.radio(search_types,value=search_types[0],label="search:")
    batch_ui = mo.ui.number(start=1,stop=50,value=5,label="batch size:")



    def presets_to_marimo(presets, batch_size):
        widgets = {}
        for key, value in presets.items():
            # cheesy fallback
            if value is None:
                value = batch_size

            if isinstance(value, int):
                widgets[key] = mo.ui.number(value=value, step=1)
            elif isinstance(value, float):
                widgets[key] = mo.ui.number(value=value, step=0.1)
            else:
                raise TypeError(f"Unsupported type: {type(value)}")
        return mo.ui.dictionary(widgets)

    opt_names = Optimizers.names()
    opt_presets = {name: dict(Optimizers.from_name(name).PRESET) if hasattr(Optimizers.from_name(name),"PRESET") else None for name in opt_names}
    def get_optimizer_config_ui(opt_name, batch_size):
        print(opt_name)
        if opt_presets[opt_name] == None:
            return mo.md("")
        # have a temp config file with the dict contents 
        ui = presets_to_marimo(opt_presets[opt_name], batch_size)
        print("hello")
        return ui

    optimizer_ui = mo.ui.radio(options=opt_names,value=opt_names[0],label="model type:")
    return (
        backend_ui,
        batch_ui,
        get_optimizer_config_ui,
        op_names,
        operator_ui,
        optimizer_ui,
        search_select,
        seed_ui,
        strategy_ui,
        trials_ui,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loop Explore

    This is an exploration script that attempts to find good parameterizations of select tiling strategies using autotuning. For each operator you can choose the problem size from operators in specific models.
    """)
    return


@app.cell
def _(
    backend_ui,
    batch_ui,
    get_optimizer_config_ui,
    mo,
    op_names,
    operator_ui,
    optimizer_ui,
    search_select,
    seed_ui,
    strategy_ui,
    trials_ui,
):
    search_ui = [search_select]
    optimizer_config_ui = mo.md("")
    if search_select.value == "iterative":
        #search_ui += [mo.vstack([batch_ui,optimizer_ui]), optimizer_config_ui(optimizer_ui.value, batch_ui.value)]
        optimizer_config_ui = get_optimizer_config_ui(optimizer_ui.value, batch_ui.value)
        search_ui += [mo.vstack([batch_ui,optimizer_ui]), optimizer_config_ui]

    vstack_elements = [backend_ui,operator_ui, op_names(operator_ui.value),strategy_ui,trials_ui,seed_ui]
    mo.vstack(vstack_elements, justify="start")
    return optimizer_config_ui, search_ui


@app.cell
def _(mo):
    mo.md(r"""
    Select a search strategy. For iterative searches you have the option to pick a preset and tweak the hyperparameters.
    """)
    return


@app.cell
def _(mo, search_ui):
    mo.hstack(search_ui,widths="equal",justify="start")
    return


@app.cell
def _(
    Args,
    backend_ui,
    batch_ui,
    explore,
    operator_ui,
    optimizer_config_ui,
    optimizer_ui,
    search_select,
    seed_ui,
    strategy_ui,
    trials_ui,
):
    import tempfile
    import os
    import csv
    import yaml

    def extract_config_yaml_from_ui(tmpdir, ui, filename="config.yaml"):
        if not (hasattr(ui, "value") and isinstance(ui.value, dict)):
            return None
        config = ui.value
        path = os.path.join(tmpdir, filename)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        return path

    def config_used(ui):
        pass

    def set_explore_args(config_path):
        args = Args(
            operator = operator_ui.value,
            backends = backend_ui.value,
            strategy = strategy_ui.value,
            search = search_select.value,
            seed = seed_ui.value,
            batch = batch_ui.value,
            optimizer = optimizer_ui.value,
            optimizer_config = config_path,
            trials = trials_ui.value
        )
        return args

    def loop_explore():
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "out.csv")

            config_path = extract_config_yaml_from_ui(
                tmpdir,
                optimizer_config_ui,
            )
            args = set_explore_args(config_path)
            args.output = output_path

            #mo.output.append("optimizing...")
            explore.optimize(args)

            best = 0
            best_info = ""
            with open(output_path, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    try:
                        val = float(row[-2])
                    except (ValueError, IndexError):
                        continue
                    if val > best:
                        best_info = str(row[-4]) + " " + row[-1]
                        best = val
            return best, best_info

    def run_loop_explore():
        best, best_info = loop_explore()
        #mo.output.append(f"best peak perf was {best}")
        return f"best peak perf was {best}\n\n{best_info}"
        #return mo.md(f"best peak perf was {best}")
    return (run_loop_explore,)


@app.cell
def _(mo):
    button = mo.ui.run_button(
        label="run loop explore",
    )
    button
    return (button,)


@app.cell
def _(button, mo, run_loop_explore):
    mo.stop(not button.value)
    with mo.status.spinner():
        explore_out = run_loop_explore()
    mo.md(explore_out)
    return


if __name__ == "__main__":
    app.run()
