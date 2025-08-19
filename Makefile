check: check-type check-lit check-pytest

check-type: check-pyright check-mypy

check-pyright:
	pyright

check-mypy:
	mypy

check-lit:
	lit -v tests/filecheck

check-pytest:
	scripts/pytest/run_pytest.sh -v tests/pytest

.PHONY: check check-lit check-pytest check-type check-pyright check-mypy
.SUFFIXES:
