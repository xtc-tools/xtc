#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod

import xtc.itf


class Evaluator(ABC):
    """An abstract implementation of a Module performance evaluator.

    An Evaluator measures and validates the performance of compiled Modules.
    It works alongside Executors to provide performance metrics and correctness
    validation of the compiled code. This is crucial for assessing the
    effectiveness of different compilation strategies and optimizations.

    Evaluators can measure metrics like execution time, throughput, and
    validate output correctness against reference implementations.
    """

    @abstractmethod
    def evaluate(self) -> tuple[list[float], int, str]:
        """Evaluates the performance of the associated Module.

        Executes the Module multiple times to gather performance metrics,
        potentially validating correctness against reference implementations.

        Returns:
            List of performance measurements (typically execution times in seconds), error coe and error message
        """
        ...

    @property
    @abstractmethod
    def module(self) -> "xtc.itf.comp.Module":
        """Returns the Module being evaluated.

        Returns:
            The compiled Module this evaluator is measuring
        """
        ...
