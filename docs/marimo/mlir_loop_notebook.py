import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from os import path
    from doc_utils import DocUtils
    doc:DocUtils        = DocUtils()
    mlir_loop_cpu:str   = "--cpu skylake"
    mlir_loop_arch:str  = "--arch x86-64"
    mlir_loop_args:str  = f"mlir-loop --no-alias {mlir_loop_arch} {mlir_loop_cpu}"

    mlir_path:str       = path.dirname(path.realpath(__file__)) + "/mlir/"

    sources:dict[str, mo.ui.text_area] = {
        "vanilla"    : mo.ui.text_area(value=doc.extract_file_content(mlir_path + "vanilla.mlir")    ),
        "tiling"     : mo.ui.text_area(value=doc.extract_file_content(mlir_path + "tiling.mlir")     ),
        "splitting"  : mo.ui.text_area(value=doc.extract_file_content(mlir_path + "splitting.mlir")  ),
        "vectorize"  : mo.ui.text_area(value=doc.extract_file_content(mlir_path + "vectorize.mlir")  ),
        "unroll"     : mo.ui.text_area(value=doc.extract_file_content(mlir_path + "unroll.mlir")     ),
        "parallelize": mo.ui.text_area(value=doc.extract_file_content(mlir_path + "parallelize.mlir")),
        "interchange": mo.ui.text_area(value=doc.extract_file_content(mlir_path + "interchange.mlir")),
        "all"        : mo.ui.text_area(value=doc.extract_file_content(mlir_path + "all.mlir")        )
    }


    # Make the cell code easier to read
    output_hidder  = lambda output_name, x: mo.accordion({output_name : x})                   # Allow us to open and close the output of a cell
    format_llvm_md = lambda lang, x: mo.md(doc.format_code_markdown(lang, x)) # Encapsulate the markdown formating
    code_output    = lambda output_name, lang, x: output_hidder(output_name, format_llvm_md(lang, x))


@app.cell(hide_code=True)
def _():
    mo.md(r"""#Introduction""")
    return


@app.cell
def _():
    mo.md(
        r"""
    Descript aims to be a declarative way to express loop optimization. It allows users to easily test different kind of optimizations without having to reimplement them from scratch. It also enables them to choose the backend.<br>
    This notebook aims to try to explain the syntax of the currently supported optimizations, and enables the reader to modify the MLIR code and the compilation parameters.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    This MLIR code represents a matrix multiplication of a matrix A of size 4x8 matrix and B of size 8x16.<br>
    The parts we are interested in are:<br>
    - `loop.dims` which define axis names.<br>
    - `loop.schedule` which define the axis order and which transformation to apply.<br>

    In this instance, we have 3 dimensions "i", "j" and "k" scheduled in this order.<br>
    Note that the "i" is of size 4, the "j" axis of size 16 and the "k" axis of size 8.
    """
    )
    return


@app.cell
def _():
    vanilla_code:mo.ui.text_area = sources['vanilla']
    vanilla_code
    return (vanilla_code,)


@app.cell
def _():
    mo.md(
        r"""
    The named_sequence called @__transform_main embeds a Transform dialect script, generated from the high-level Descript specification above. This dialect enables one to apply code transformation on linear algebra operators without modifying code semantic, for purposes of optimization.<br>

    For instance, in this source IR, we can see the `transform.structured.match` which locate the operator we want to modify.

    Then for each axis in the order defined above, we have the `transform.structured.tile_using_for` to materialize the loop into a `for` loop of step 1, and a `transform.annotate` to name this newly created loop.
    """
    )
    return


@app.cell
def _():
    vanilla_source_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-source-ir", language="shell")
    vanilla_source_ir
    return (vanilla_source_ir,)


@app.cell
def _(vanilla_code: mo.ui.text_area, vanilla_source_ir: mo.ui.code_editor):
    code_output("Click me to see the outputed source IR", "llvm", doc.exec_mlir_loop_from_string( vanilla_source_ir.value, vanilla_code.value))
    return


@app.cell
def _():
    mo.md(
        r"""
    From the source IR above, we apply those transformations and get the transformed IR below, with our for loops from the `scf` dialect and our annotations on each of them.<br>
    Note that the order defined in the `loop.schedule` is conserved, as the loops respect the dimension sizes that we considered earlier for each axis.
    """
    )
    return


@app.cell
def _():
    vanilla_transformed_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-transformed-ir", language="shell")
    vanilla_transformed_ir
    return (vanilla_transformed_ir,)


@app.cell
def _(
    vanilla_code: mo.ui.text_area,
    vanilla_transformed_ir: mo.ui.code_editor,
):
    code_output("Click me to see the transformed IR", "llvm", doc.exec_mlir_loop_from_string(vanilla_transformed_ir.value, vanilla_code.value))
    return


@app.cell
def _():
    mo.md(r"""Then, everything is lowered in the LLVM dialect ready to be converted to the LLVM IR, as those representations are homomorphic.<br>""")
    return


@app.cell
def _():
    vanilla_lowered_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-lowered-ir", language="shell")
    vanilla_lowered_ir
    return (vanilla_lowered_ir,)


@app.cell
def _(vanilla_code: mo.ui.text_area, vanilla_lowered_ir: mo.ui.code_editor):
    code_output("Click me to see the lowered IR", "llvm", doc.exec_mlir_loop_from_string(vanilla_lowered_ir.value, vanilla_code.value))
    return


@app.cell
def _():
    vanilla_assembly:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-assembly", language="shell")
    vanilla_assembly
    return (vanilla_assembly,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Finally, the code can be compiled.
    Here we decided to compile it in an x86-64 architecture with a skylake processor, which include the AVX2 instruction set.
    """
    )
    return


@app.cell
def _(vanilla_assembly: mo.ui.code_editor, vanilla_code: mo.ui.text_area):
    code_output("Click me to see the generated code", "asm",doc.exec_mlir_loop_from_string(vanilla_assembly.value, vanilla_code.value))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""#Interchange""")
    return


@app.cell
def _():
    mo.md(
        r"""
    As we saw above, we can define the order of each axis. So, in this example, let's decide to swap "j" and "k" order (The dimensions sizes remain unchanged).<br><br>
    This schedule isn't much different from the schedule in the introduction, meaning that both source IR will be similar.<br>
    However, the order of each `transform.structured.tile_using_for` is important, as it goes from the outermost loop to the innermost.
    """
    )
    return


@app.cell
def _():
    interchange_code:mo.ui.text_area = sources['interchange']
    interchange_code
    return (interchange_code,)


@app.cell
def _():
    interchange_source_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-source-ir", language="shell")
    interchange_source_ir
    return (interchange_source_ir,)


@app.cell
def _(
    interchange_code: mo.ui.text_area,
    interchange_source_ir: mo.ui.code_editor,
):
    code_output("Click to see the outputed source IR", "llvm", doc.exec_mlir_loop_from_string(interchange_source_ir.value, interchange_code.value))
    return


@app.cell
def _():
    interchange_transformed_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-transformed-ir", language="shell")
    interchange_transformed_ir
    return (interchange_transformed_ir,)


@app.cell
def _():
    mo.md(r"""In this transformed IR, we can notice, thanks to the annotations on each loop, that the order of the schedule is conserved.<br>""")
    return


@app.cell
def _(
    interchange_code: mo.ui.text_area,
    interchange_transformed_ir: mo.ui.code_editor,
):
    code_output("Click to see the output", "llvm", doc.exec_mlir_loop_from_string(interchange_transformed_ir.value, interchange_code.value))
    return


@app.cell
def _():
    mo.md(r"""# Tiling""")
    return


@app.cell
def _():
    mo.md(
        r"""
    To tile our axis, we need to first define the outer loops interchange, and then we can define our tiles on each axis with : <br>
    `"axis_name#tile_size"`<br>
    <br>
    You can define multiple tile level, but each inner level have to divide previous ones.<br>
    Here is an example :
    """
    )
    return


@app.cell
def _():
    tiling_code:mo.ui.text_area = sources['tiling']
    tiling_code
    return (tiling_code,)


@app.cell
def _():
    mo.md(
        r"""
    As you can see, the source IR has three new loops, with the same instructions as before.<br>
    However we can see that the tile size isn't always 1 anymore, for instance, the j axis has a tile size of 8.
    """
    )
    return


@app.cell
def _():
    tiling_source_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-source-ir", language="shell")
    tiling_source_ir
    return (tiling_source_ir,)


@app.cell
def _(tiling_code: mo.ui.text_area, tiling_source_ir: mo.ui.code_editor):
    code_output("Click to see the output", "llvm", doc.exec_mlir_loop_from_string(tiling_source_ir.value, tiling_code.value))
    return


@app.cell
def _():
    mo.md(
        r"""
    The previous tile size are converted into:<br>
    - The step size for an outer tile.<br>
    - The iteration domain size for an inner tile.

    Again with j:<br>
    - The outer tile has a step size of 8.
    - The inner tile has a iteration domain from 0 to 8 (excluded).
    """
    )
    return


@app.cell
def _():
    tiling_transformed_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-transformed-ir", language="shell")
    tiling_transformed_ir
    return (tiling_transformed_ir,)


@app.cell
def _(tiling_code: mo.ui.text_area, tiling_transformed_ir: mo.ui.code_editor):
    code_output("Click to see the output", "llvm", doc.exec_mlir_loop_from_string(tiling_transformed_ir.value, tiling_code.value))
    return


@app.cell
def _():
    mo.md(r"""# Splitting""")
    return


@app.cell
def _():
    mo.md(
        r"""
    The splitting operator allows you to cut an axis in multiple loops, with their own sizes, and their own constraints.<br>
    This can be useful to simplify the divisibility constraint of tiling.
    For instance, let $A$ be a loop of size $10$. You can split it in two with a size of $B = 8$ and $C = 2$, then you can tile $B$.
    """
    )
    return


@app.cell
def _():
    splitting_code:mo.ui.text_area = sources["splitting"]
    splitting_code
    return (splitting_code,)


@app.cell
def _():
    mo.md(r"""Here we split the "j" axis of size $16$ in two part of size $8$.""")
    return


@app.cell
def _():
    mo.md(
        r"""
    Note that the above split can also be written as :
    ```llvm
    "j[:8]" = { "j", "k" },
    "j[:]" = { "j", "k" }
    ```
    As the [:] notation will restart from 8 and stop directly at the end of the axis.
    """
    )
    return


@app.cell
def _():
    mo.md(r"""After the definition of the "i" axis, we can see the `transform.structured.split` which create two loops instead of one, and then these loops will have their own content defined below, again, in the same order as the interchange.""")
    return


@app.cell
def _():
    splitting_source_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-source-ir", language="shell")
    splitting_source_ir
    return (splitting_source_ir,)


@app.cell
def _(splitting_code: mo.ui.text_area, splitting_source_ir: mo.ui.code_editor):
    code_output("Click to see the output", "mlir", doc.exec_mlir_loop_from_string(splitting_source_ir.value, splitting_code.value))
    return


@app.cell
def _():
    mo.md(
        r"""
    These transformations are materialized by two `scf.for` loops on the "j" axis, with their own "k" axis inside."
    """
    )
    return


@app.cell
def _():
    splitting_transformed_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-transformed-ir", language="shell")
    splitting_transformed_ir
    return (splitting_transformed_ir,)


@app.cell
def _(
    splitting_code: mo.ui.text_area,
    splitting_transformed_ir: mo.ui.code_editor,
):
    code_output("Click to see the output", "llvm", doc.exec_mlir_loop_from_string(splitting_transformed_ir.value, splitting_code.value))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Unroll""")
    return


@app.cell
def _():
    mo.md(
        r"""
    The "unroll" annotation will unroll the associated axis.<br>
    You can control the unrolling factor with a positive integer.<br>
    Giving no parameters means a full unroll.
    Full unroll are only supported on tiled or splitted axis.
    """
    )
    return


@app.cell
def _():
    unroll_code:mo.ui.text_area = sources["unroll"]
    unroll_code
    return (unroll_code,)


@app.cell
def _():
    mo.md(r"""We chose to unroll the "j" axis, and in the corresponding IR below we can see a `transform.loop.unroll` on the corresponding axis, with an unroll factor of 8 which corresponds to the tile size of this axis.""")
    return


@app.cell
def _():
    unroll_source_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-source-ir", language="shell")
    unroll_source_ir
    return (unroll_source_ir,)


@app.cell
def _(unroll_code: mo.ui.text_area, unroll_source_ir: mo.ui.code_editor):
    code_output("Click to see the output", "llvm", doc.exec_mlir_loop_from_string(unroll_source_ir.value, unroll_code.value))
    return


@app.cell
def _():
    mo.md(r"""And in the transformed IR, the loop got completely unrolled at this level, meaning that the unrolled transformation is done at this level.""")
    return


@app.cell
def _():
    unroll_transformed_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-transformed-ir", language="shell")
    unroll_transformed_ir
    return (unroll_transformed_ir,)


@app.cell
def _(unroll_code: mo.ui.text_area, unroll_transformed_ir: mo.ui.code_editor):
    code_output("Click to see the output", "llvm", doc.exec_mlir_loop_from_string(unroll_transformed_ir.value, unroll_code.value))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Vectorization""")
    return


@app.cell
def _():
    mo.md(r"""To explicitly tell the compiler to vectorize an axis, we can use the "vectorize" annotation.<br>""")
    return


@app.cell
def _():
    vectorize_code:mo.ui.text_area = sources["vectorize"]
    vectorize_code
    return (vectorize_code,)


@app.cell
def _():
    vectorize_transformed_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-transformed-ir", language="shell")
    vectorize_transformed_ir
    return (vectorize_transformed_ir,)


@app.cell
def _():
    mo.md(
        r"""
    If successful, this transformation will use `vector` dialect, which abstracts low-level, assembly vector instructions like AVX.<br>
    Indeed, this dialect contains well-known operations such as :<br>
    - `broadcast` corresponding to `vbroadcast` family in AVX<br>
    - `fma` corresponding to `vfmadd` family <br>
    - `transfer_read` and `transfer_write` that can be translated into `vmov` <br>
    """
    )
    return


@app.cell
def _(
    vectorize_code: mo.ui.text_area,
    vectorize_transformed_ir: mo.ui.code_editor,
):
    code_output("Click to see the output", "llvm", doc.exec_mlir_loop_from_string(vectorize_transformed_ir.value, vectorize_code.value))
    return


@app.cell
def _():
    mo.md(r"""Something striking here, is the use of `llvm.intr.fmuladd` operation on 3 `vector<16xf64>` which corresponds to intrinsic, meaning that in lower levels the conversion into vectorized instruction is almost guaranteed.""")
    return


@app.cell
def _():
    vectorize_lowered_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-lowered-ir", language="shell")
    vectorize_lowered_ir
    return (vectorize_lowered_ir,)


@app.cell
def _(
    vectorize_code: mo.ui.text_area,
    vectorize_lowered_ir: mo.ui.code_editor,
):
    code_output("Click to see the output", "mlir", doc.exec_mlir_loop_from_string(vectorize_lowered_ir.value, vectorize_code.value))
    return


@app.cell
def _():
    mo.md(r"""# Parallelize""")
    return


@app.cell
def _():
    mo.md(r"""This annotation will enable multithreading on the annotated axis.<br/>""")
    return


@app.cell
def _():
    parallelize_code:mo.ui.text_area = sources["parallelize"]
    parallelize_code
    return (parallelize_code,)


@app.cell
def _():
    mo.md(r"""In the lowered IR, we notice the use of the `omp` dialect (OpenMP) on the parallelized (here "i") axis.""")
    return


@app.cell
def _():
    parallelize_lowered_ir:mo.ui.code_editor = mo.ui.code_editor(value=mlir_loop_args + " --print-lowered-ir", language="shell")
    parallelize_lowered_ir
    return (parallelize_lowered_ir,)


@app.cell
def _(
    parallelize_code: mo.ui.text_area,
    parallelize_lowered_ir: mo.ui.code_editor,
):
    code_output("Click to see the output", "llvm", doc.exec_mlir_loop_from_string(parallelize_lowered_ir.value, parallelize_code.value))
    return


if __name__ == "__main__":
    app.run()
