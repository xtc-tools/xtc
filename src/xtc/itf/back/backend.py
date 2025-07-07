#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing import Any

from ..graph.graph import Graph
from ..schd.scheduler import Scheduler
from ..comp.compiler import Compiler


class Backend(ABC):
    """An abstract implementation of specific Graph implementation.

    A Backend is constructed from an input Graph and provides backend-specific
    implementations of the graph operations. It serves as a bridge between the abstract
    graph representation and concrete backend implementations (e.g., MLIR, TVM, JIR).

    The Implementer provides access to associated Scheduler and Compiler instances
    for applying transformations and generating executable code.
    """

    @abstractmethod
    def get_scheduler(self, **kwargs: Any) -> Scheduler:
        """Returns the scheduler associated with this implementation.

        Args:
            kwargs: scheduler configuration

        Returns:
            The scheduler for applying transformations
        """
        ...

    @abstractmethod
    def get_compiler(self, **kwargs: Any) -> Compiler:
        """Returns the compiler associated with this implementation.

        Args:
            kwargs: compiler configuration

        Returns:
            The compiler for generating executable code
        """
        ...

    @property
    @abstractmethod
    def graph(self) -> Graph:
        """Returns the graph being implemented.

        Returns:
            The source graph for this implementation
        """
        ...
