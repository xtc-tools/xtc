import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from os import path
    from doc_utils import DocUtils

    doc:DocUtils = DocUtils()
    mlir_loop_cpu:str     = "--cpu skylake"
    mlir_loop_arch:str    = "--arch x86-64"
    mlir_loop_args:str    = f"mlir-loop --no-alias {mlir_loop_arch} {mlir_loop_cpu}"

    mlir_path:str         = path.dirname(path.realpath(__file__)) + "/mlir/"
    all_example:mo.ui.text_area = mo.ui.text_area(value=doc.extract_file_content(mlir_path + "all.mlir"))

    output_hidder  = lambda x: mo.accordion({"Output" : x})
    format_llvm_md = lambda lang, x: mo.md(doc.format_code_markdown(lang, x))
    code_output    = lambda lang, x: output_hidder(format_llvm_md(lang, x))


@app.cell(hide_code=True)
def _():
    mo.md(r"""# XTC example""")
    return


@app.cell
def _():
    mo.md(
        r"""
    This example
    has an interchange between the inner "k" and "j" axis
    The "k" axis has a tile size of 8 and "j" a tile size of 32.<br>
    We also explicitly vectorized "j" and unrolled both "j" fully and "k" 8 times.<br>
    <br>
    <sub>Note that we can also parallelize the "i" axis.</sub>
    """
    )
    return


@app.cell
def _():
    all_example
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Source IR""")
    return


@app.cell
def _():
    args_source_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-source-ir", language="shell")
    args_source_ir
    return (args_source_ir,)


@app.cell
def _(args_source_ir: mo.ui.code_editor):
    code_output("llvm", doc.exec_mlir_loop_from_string(args_source_ir.value, all_example.value))
    return


@app.cell
def _():
    mo.md(r"""#Transformed IR""")
    return


@app.cell
def _():
    mo.md(
        r"""
    As you can see, the vectorization worked as intended with the appearance of `fma`, `broadcast`, `transfer_read` and `transfer_write`.<br>
    The tiled axis also got unrolled.
    """
    )
    return


@app.cell
def _():
    args_transformed_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-transformed-ir", language="shell")
    args_transformed_ir
    return (args_transformed_ir,)


@app.cell
def _(args_transformed_ir: mo.ui.code_editor):
    code_output("llvm", doc.exec_mlir_loop_from_string(args_transformed_ir.value, all_example.value))
    return


@app.cell
def _():
    mo.md(r"""# Lowered IR""")
    return


@app.cell
def _():
    mo.md(
        r"""
    On the lowered IR, we can see the use of intrinsic such as `llvm.intr.fmuladd`.<br>
    This means that we are almost guaranteed to see an `vfmadd` instruction appear.
    """
    )
    return


@app.cell
def _():
    args_lowered_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-lowered-ir", language="shell")
    args_lowered_ir
    return (args_lowered_ir,)


@app.cell
def _(args_lowered_ir: mo.ui.code_editor):
    code_output("llvm", doc.exec_mlir_loop_from_string(args_lowered_ir.value, all_example.value))
    return


@app.cell
def _():
    mo.md(r"""# Generated code""")
    return


@app.cell
def _():
    mo.md(
        r"""
    This generated code is great because it has hoisted loads at the top and hoisted stores at the bottom, while the innermost loop is completely filled with `vfmadd` and `vbroadcast`.<br>
    Meaning, that the computation can be pipelined and is interrupted by memory access only when needed.
    """
    )
    return


@app.cell
def _():
    args_assembly:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-assembly", language="shell")
    args_assembly
    return (args_assembly,)


@app.cell
def _(args_assembly: mo.ui.code_editor):
    code_output("asm", doc.exec_mlir_loop_from_string(args_assembly.value, all_example.value))
    return


if __name__ == "__main__":
    app.run()
