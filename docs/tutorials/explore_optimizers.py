import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import argparse
    import xtc.search.explore as explore

    args = argparse.Namespace(
        trials=64,
        search_type="random",
        batch=8,
        opt_name="random-forest-default",
    )
    if not mo.running_in_notebook():
        parser = argparse.ArgumentParser("Test notebook from CLI")
        parser.add_argument("--trials", type=int, default=args.trials)
        parser.add_argument("--search-type", type=str, default=args.search_type)
        parser.add_argument("--opt-name", type=str, default=args.opt_name)
        parser.add_argument("--batch", type=int, default=args.batch)
        args = parser.parse_args()

    return explore, mo, args

@app.cell
def _(mo, args):
    from xtc.artifacts import list_operations
    from xtc.search.strategies import Strategies
    from xtc.search.optimizers import Optimizers

    operators = ["matmul","conv2d"]
    backends = ["mlir","tvm","jir"]
    strategies = list(Strategies.names())
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
        value="tile_oo",
        label="strategy:"
    )
    strategy_param_ui = mo.ui.text(
        value="",
        label="prt strategy scheme:",
        placeholder="e.g. PPWRPRP"
    )
    seed_ui = mo.ui.number(start=0,stop=200,label="seed:")
    trials_ui = mo.ui.number(start=0,stop=2048,value=args.trials,label="trials:")

    search_types = ["random","iterative","exhaustive"]
    search_select = mo.ui.radio(search_types,value=args.search_type,label="search:")
    batch_ui = mo.ui.number(start=1,stop=64,value=args.batch,label="batch size:")



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
        if not opt_presets[opt_name]:
            return mo.md("")
        # have a temp config file with the dict contents 
        ui = presets_to_marimo(opt_presets[opt_name], batch_size)
        return ui

    optimizer_ui = mo.ui.radio(options=opt_names,value=args.opt_name,label="model type:")
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
        strategy_param_ui,
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
    strategy_param_ui,
    trials_ui,
):
    search_ui = [search_select]
    optimizer_config_ui = mo.md("")
    if search_select.value == "iterative":
        #search_ui += [mo.vstack([batch_ui,optimizer_ui]), optimizer_config_ui(optimizer_ui.value, batch_ui.value)]
        optimizer_config_ui = get_optimizer_config_ui(optimizer_ui.value, batch_ui.value)
        search_ui += [mo.vstack([batch_ui,optimizer_ui]), optimizer_config_ui]

    vstack_elements = [backend_ui, operator_ui, op_names(operator_ui.value), strategy_ui]
    if strategy_ui.value == "prt":
        vstack_elements.append(strategy_param_ui)
    vstack_elements += [trials_ui, seed_ui]
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
    backend_ui,
    batch_ui,
    explore,
    operator_ui,
    optimizer_config_ui,
    optimizer_ui,
    search_select,
    seed_ui,
    strategy_ui,
    strategy_param_ui,
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
        strategy_param = strategy_param_ui.value if strategy_ui.value == "prt" else ""
        if strategy_param:
            strategy = f"{strategy_ui.value}:{strategy_param}"
        else:
            strategy = strategy_ui.value
        args = explore.default_exploration_config(
            operator = operator_ui.value,
            backends = backend_ui.value,
            strategy = strategy,
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
            exploration = explore.Exploration(args)
            exploration()

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
        return f"Best peak found is {best*100:.2f}% for {best_info}"

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
    if mo.running_in_notebook():
        mo.stop(not button.value)
        with mo.status.spinner():
            explore_out = run_loop_explore()
    else:
        explore_out = run_loop_explore()
    mo.md(explore_out)
    return

if __name__ == "__main__":
    app.run()
