help:
	@echo "Available make targets:"
	@echo
	@echo "  make test            # run minimal tests"
	@echo "  make check           # run all acceptance tests (all targets below)"
	@echo "    make check-type    # run all type checks tests"
	@echo "    make check-lit     # run all lit checks for binary target"
	@echo "    make check-lit-c   # run all lit checks for C target"
	@echo "    make check-pytest  # run all pytest tests"
	@echo

test:
	pytest tests/pytest/unit tests/pytest/mlir tests/pytest/tvm tests/pytest/jir

check: check-type check-lit-all check-pytest

check-type: check-pyright check-mypy

check-pyright:
	pyright

check-mypy:
	mypy

check-lit-all:
	$(MAKE) check-lit
	$(MAKE) check-lit-c

check-lit:
	lit -v tests/filecheck

check-lit-c:
	env XTC_MLIR_TARGET=c lit -v tests/filecheck/backends tests/filecheck/mlir_loop

check-pytest:
	scripts/pytest/run_pytest.sh -v

.PHONY: help test check check-lit-all check-lit check-lit-c check-pytest check-type check-pyright check-mypy
.SUFFIXES:
