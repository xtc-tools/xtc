import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __():
    import sys
    from io import StringIO
    from contextlib import redirect_stderr
    import marimo as mo

@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # XTC Tutorial - CGO 2026

    Welcome to the XTC tutorial! This interactive notebook will guide you through the fundamentals of performance engineering using XTC, a research platform for optimizing AI operators.

    By the end of this tutorial, you will understand how to:
    - Define computational graphs with XTC
    - Compile and evaluate operator performance
    - Apply high-level scheduling transformations using Descript
    - Explore the scheduling space to find optimal configurations
    """)
    return

if __name__ == "__main__":
    app.run()
