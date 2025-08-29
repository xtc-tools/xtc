check: check-type check-lit check-pytest

check-type: check-pyright check-mypy

check-pyright:
	pyright

check-mypy:
	mypy

check-lit:
	lit -v tests/filecheck
	env XTC_MLIR_TARGET=c lit -v tests/filecheck/backends
	env XTC_MLIR_TARGET=c lit -v tests/filecheck/mlir_loop

check-pytest:
	scripts/pytest/run_pytest.sh -v tests/pytest

.PHONY: check check-lit check-pytest check-type check-pyright check-mypy
.SUFFIXES:
