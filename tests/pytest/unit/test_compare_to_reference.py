import numpy as np

from xtc.utils.evaluation import compare_to_reference

# Shared validation helper. Pure numpy: no backend needed.


def _add_reference(a, b, out):
    out[:] = a + b


def test_compare_to_reference_match():
    a = np.ones((4,), dtype="float32")
    b = np.full((4,), 2.0, dtype="float32")
    assert compare_to_reference([a + b], [a, b], _add_reference) == (0, "")


def test_compare_to_reference_mismatch():
    a = np.ones((4,), dtype="float32")
    b = np.full((4,), 2.0, dtype="float32")
    wrong = np.zeros((4,), dtype="float32")
    code, msg = compare_to_reference([wrong], [a, b], _add_reference)
    assert code == 1
    assert "differ" in msg
