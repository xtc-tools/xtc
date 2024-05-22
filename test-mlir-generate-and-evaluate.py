from setup_mlir_mm import mm1

source_path = "/tmp/test.mlir"

impl = mm1()

e = impl.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_ir_after=[],
    print_ir_before=[],
    print_assembly=True,
    color=True,
    debug=False,
)

mlircode = impl.generate_without_compilation()
f = open(source_path, "w")
f.write(mlircode)
f.close()

from PartialImplementer import PartialImplementer
import os

home = os.environ.get("HOME", "")

impl = PartialImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_path=source_path,
    payload_name=impl.payload_name,
)

from PartialImplementer import PartialImplementer

impl = PartialImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_path=source_path,
    payload_name=impl.payload_name,
)

e = impl.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_ir_after=[],
    print_ir_before=[],
    print_assembly=True,
    color=True,
    debug=False,
)
print(e)
