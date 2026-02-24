#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
import numpy as np
import numpy.typing

from .math import mulall


def np_init(shape: tuple, dtype: str, **attrs: Any) -> numpy.typing.NDArray[Any]:
    """
    Initialize and return a NP array filled
    with numbers in [1, 9].
    """
    vals = np.arange(mulall(list(shape)))
    vals = vals % 9 + 1
    return vals.reshape(shape).astype(dtype)
