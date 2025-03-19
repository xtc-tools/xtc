#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
import xtc.itf


class Schedule(ABC):
    """An abstract representation of the result of transformations from a scheduler.

    A Schedule captures all the transformations and optimizations that have been
    applied to an implementation by its associated Scheduler. It serves as an
    intermediate representation between scheduling operations and code generation,
    allowing Compilers to generate optimized executable code based on the
    specified transformations.

    Schedules are backend-specific and contain the necessary information for
    their associated Compiler to generate code optimized for the target
    platform and runtime. Common transformations captured in a Schedule include:
    - Tiling
    - Loop interchange
    - Vectorization
    - Parallelization
    - Loop unrolling
    """

    @property
    @abstractmethod
    def scheduler(self) -> "xtc.itf.schd.Scheduler":
        """Returns the scheduler that generated this schedule.

        Returns:
        """
        ...
