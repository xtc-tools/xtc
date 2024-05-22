from setup_xdsl_mm import mm1

impl = mm1()

mlircode = impl.generate_without_compilation()
print(mlircode)
