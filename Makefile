help:
	@echo "Available make targets:"
	@echo
	@echo "  make test            # run minimal tests"
	@echo "  make check           # run all acceptance tests (all targets below)"
	@echo "    make check-format  # run all format checks tests"
	@echo "    make check-type    # run all type checks tests"
	@echo "    make check-lit     # run all lit checks for binary target"
	@echo "    make check-lit-c   # run all lit checks for C target"
	@echo "    make check-lit-nvgpu # run all lit checks for NVGPU target"
	@echo "    make check-pytest  # run all pytest tests"
	@echo "    make check-banwords # run banned words checks"
	@echo "  make format          # apply formatting (warning: change files in place)"
	@echo "    make format-license # add licenses"
	@echo "    make format-ruff   # format python files with ruff"
	@echo "  make claude          # create CLAUDE.md"
	@echo


test:
	pytest tests/pytest/unit tests/pytest/mlir tests/pytest/tvm tests/pytest/jir

check: check-format check-banwords check-type check-lit-all check-pytest

format: format-license format-ruff

check-format: check-format-ruff check-license

check-format-ruff:
	scripts/ruff/format.sh --check

check-license:
	scripts/licensing/licensing.py --check

check-banwords:
	scripts/banwords/banwords.py --check

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

check-lit-nvgpu:
	env XTC_MLIR_TARGET=nvgpu lit -v tests/filecheck/backends tests/filecheck/mlir_loop tests/filecheck/evaluation

check-pytest:
	scripts/pytest/run_pytest.sh -v

format-ruff:
	scripts/ruff/format.sh

format-license:
	scripts/licensing/licensing.py --apply

claude:
	scripts/llms/init_claude.py README.md "Links" "AI assistants" > CLAUDE.md

run-tutorial:
	marimo run docs/tutorials/xtc_101.py

.PHONY: help test check check-lit-all check-lit check-lit-c check-lit-nvpgu check-pytest check-type check-pyright check-mypy check-format check-format-ruff check-license check-banwords format format-ruff format-license run-tutorial
.SUFFIXES:
