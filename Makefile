help:
	@echo "Available make targets:"
	@echo
	@echo "  make test            # run minimal tests"
	@echo "  make check           # run all acceptance tests (all targets below)"
	@echo "    make check-format  # run all format checks tests"
	@echo "    make check-type    # run all type checks tests"
	@echo "    make check-dependencies # check that pyproject.toml is up to date
	@echo "    make check-lit     # run all lit checks for binary target"
	@echo "    make check-lit-c   # run all lit checks for C target"
	@echo "    make check-lit-nvgpu # run all lit checks for NVGPU target"
	@echo "    make check-pytest  # run all pytest tests"
	@echo "    make check-banwords # run banned words checks"
	@echo "    make check-tutorials # run tutorials checks
	@echo "  make format          # apply formatting (warning: change files in place)"
	@echo "    make format-license # add licenses"
	@echo "    make format-ruff   # format python files with ruff"
	@echo "  make dependencies    # update pyproject.toml dependencies from dependencies.toml
	@echo "  make wheel           # install build tools, build and check distributions"
	@echo "  make pages           # build the documentation site"
	@echo "  make agents          # create AGENTS.md"
	@echo "  make claude          # create CLAUDE.md"
	@echo


test:
	pytest tests/pytest/unit tests/pytest/mlir tests/pytest/tvm tests/pytest/jir

check: check-format check-banwords check-type check-dependencies check-lit-all check-pytest check-tutorials

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

check-dependencies:
	scripts/pyproject/update_dependencies.py --check

check-lit-all:
	$(MAKE) check-lit
	$(MAKE) check-lit-c

check-lit:
	lit -v tests/filecheck

check-lit-c:
	env XTC_MLIR_TARGET=c lit -v tests/filecheck/backends tests/filecheck/mlir_loop

check-lit-nvgpu:
	[ `uname -s` = Darwin ] || env XTC_MLIR_TARGET=nvgpu lit -v tests/filecheck/backends tests/filecheck/mlir_loop tests/filecheck/evaluation

check-lit-mppa:
	[ `uname -s` = Darwin ] || env XTC_MLIR_TARGET=mppa lit -v -j 1 tests/filecheck/backends/target_mppa tests/filecheck/evaluation/test_matmul_pmu_counters_mppa.py

check-pytest:
	scripts/pytest/run_pytest.sh -v

check-tutorials:
	scripts/tutorials/test_marimos.sh
	scripts/tutorials/test_tutorial_explore_optimizers.sh
	scripts/tutorials/test_tutorial_xtc_101.sh

format-ruff:
	scripts/ruff/format.sh

format-license:
	scripts/licensing/licensing.py --apply

dependencies:
	scripts/pyproject/update_dependencies.py

wheel:
	python -m ensurepip --upgrade
	python -m pip install --upgrade build twine
	rm -rf build dist src/*.egg-info
	python -m build
	python -m twine check dist/*
	python scripts/pyproject/check_wheel.py

pages:
	$(MAKE) -C docs/site site

agents:
	scripts/llms/init_agents.py agents README.md "Links" "local LLVM" "AI assistants" > AGENTS.md

claude:
	scripts/llms/init_agents.py claude README.md "Links" "local LLVM" "AI assistants" > CLAUDE.md

run-tutorial:
	marimo run docs/tutorials/xtc_101.py

.PHONY: help test check check-lit-all check-lit check-lit-c check-lit-nvpgu check-pytest check-type check-pyright check-mypy check-format check-format-ruff check-license check-banwords format format-ruff format-license agents claude check-tutorials run-tutorial check-dependencies dependencies wheel pages
.SUFFIXES:
