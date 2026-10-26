#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, Sequence
import numpy as np
import numpy.typing
import ml_dtypes

from .math import mulall


def np_init(shape: Sequence[int], dtype: str) -> numpy.typing.NDArray[Any]:
    """
    Initialize and return a NP array filled with the 17 evenly
    spaced values in [-1, 1] with step 1/8: -1, -0.875, ..., 0, ..., 1.
    Values are exact in binary floating point.
    """
    vals = np.arange(mulall(list(shape)))
    return ((vals % 17 - 8) / 8).reshape(shape).astype(dtype)


def np_allclose(ref: Any, out: Any) -> bool:
    """
    Compare arrays with a tolerance scaled to the dtype: low precision
    dtypes such as float16 accumulate rounding differences between
    distinct summation orders, which the default allclose tolerance
    (rtol=1e-5) cannot absorb.
    """
    dtype = np.promote_types(np.asarray(ref).dtype, np.asarray(out).dtype)
    if dtype == np.float16:
        # 2e-2 is about 20x the float16 machine epsilon: it absorbs
        # reordering noise of float16 accumulations while still catching
        # wrong kernels, which err by far more than that.
        return np.allclose(ref, out, rtol=2e-2, atol=2e-2)
    elif dtype == ml_dtypes.bfloat16:
        # bfloat16 has ~8x larger machine epsilon than float16.
        # Use a looser tolerance to account for reduced precision
        # and accumulation/reordering differences.
        return np.allclose(ref, out, rtol=5e-2, atol=2e-2)
    return np.allclose(ref, out)
