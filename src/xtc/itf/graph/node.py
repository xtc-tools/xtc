#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from ..operator.operator import Operator
from ..data import TensorType, Tensor


class Node(ABC):
    """An abstract representation of a node in a dataflow graph.

    A Node represents a pure operation on input Tensor objects, resulting in output
    Tensor objects. Each node has a unique name within its graph, a set of input
    and output tensors, and an associated Operator that defines its semantic behavior.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of this node.

        Returns:
            The node's unique identifier within its graph
        """
        ...

    @property
    @abstractmethod
    def inputs(self) -> list[str]:
        """Returns the list of input tensor names for this node.

        Returns:
            List of input tensor names
        """
        ...

    @property
    @abstractmethod
    def outputs(self) -> list[str]:
        """Returns the list of output tensor names for this node.

        Returns:
            List of output tensor names
        """
        ...

    @property
    @abstractmethod
    def operator(self) -> Operator:
        """Returns the operator that defines this node's behavior.

        Returns:
            The algebraic operation associated with this node
        """
        ...

    @abstractmethod
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        """Infers output tensor types from input tensor types.

        Args:
            inputs: List of input tensor types

        Returns:
            List of inferred output tensor types
        """
        ...

    @abstractmethod
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        """Evaluate the node with input tensors to produce output tensors.

        Args:
            inputs: List of input tensors

        Returns:
            List of output tensors
        """
        ...
