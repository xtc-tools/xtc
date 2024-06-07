from setup_mlir_mm import generic_mm1

impl = generic_mm1()

e = impl.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_ir_after=[],
    print_ir_before=[],
    print_assembly=True,
    color = True,
    debug = False,
)

print(e)
