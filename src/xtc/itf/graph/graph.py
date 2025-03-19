#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from .node import Node


class Graph(ABC):
    """An abstract representation of a dataflow graph over Tensor types.

    A Graph is a directed acyclic graph (DAG) over Node objects with input Tensors
    and output Tensors. From given input Tensor dimensions and type, all node inputs,
    outputs and graph outputs Tensor dimensions and types can be inferred.

    A Graph can be evaluated by interpretation or compiled through various backends
    (mlir, tvm, jir) using corresponding Backend, Scheduler, and Compiler classes.

    Nodes in the graph are keyed by their name which is unique within the graph.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of this graph.

        Returns:
            The graph's globally unique identifier
        """
        ...

    @property
    @abstractmethod
    def nodes(self) -> dict[str, Node]:
        """Returns a dictionary of all nodes in the graph, keyed by node name.

        Returns:
            Dictionary mapping node names to Node objects
        """
        ...

    @property
    @abstractmethod
    def inputs(self) -> list[str]:
        """Returns the list of input tensor names for this graph.

        Returns:
            List of input tensor names
        """
        ...

    @property
    @abstractmethod
    def outputs(self) -> list[str]:
        """Returns the list of output tensor names for the graph.

        Returns:
            List of output tensor names
        """
        ...
