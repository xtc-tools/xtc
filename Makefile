check:
	pyright
	mypy
	lit tests/filecheck
	pytest tests/unit
	pytest tests/mlir
	pytest tests/tvm
	pytest tests/jir
