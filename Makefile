check:
	pyright
	mypy
	lit tests/filecheck
	pytests tests/unit
	pytests tests/mlir
	pytests tests/tvm
	pytests tests/jir
