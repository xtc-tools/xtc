from setup_xdsl_mm import mm1

impl = mm1()

impl.compile(
    print_source_ir=False,
    print_transformed_ir=False,
    print_ir_after=[],
    print_ir_before=[],
    print_assembly=True,
    color=True,
    debug=False,
)
