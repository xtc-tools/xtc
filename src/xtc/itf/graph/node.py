#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from collections.abc import Sequence
from ..operator.operator import Operator
from ..data import TensorType, Tensor
from .operation import Operation


class Node(ABC):
    """An abstract representation of a node in a dataflow graph.

    A Node represents a pure operation on input Tensor objects, resulting in output
    Tensor objects. Each node has a unique global uid, a set of input
    and output tensors, and an associated Operator that defines its semantic behavior.
    """

    @property
    @abstractmethod
    def uid(self) -> str:
        """Returns the globally unique id over all created nodes.

        Returns:
            The node's globally unique id
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of this node. Can be non-unique or empty.

        Returns:
            The node's name
        """
        ...

    @property
    @abstractmethod
    def inputs(self) -> list[str]:
        """Returns the list of input tensors uids for this node.

        As of now nodes can only have one output, hence a
        tensor uid is the same as the producing node uid.

        Returns:
            List of input tensor names
        """
        ...

    @property
    @abstractmethod
    def outputs(self) -> list[str]:
        """Returns the list of output tensors uids for this node.

        As of now nodes can only have one output, hence the
        node outputs is the list containing the node uid itself.

        Returns:
            List of output tensor names
        """
        ...

    @property
    @abstractmethod
    def preds(self) -> Sequence[str]:
        """Returns the list of predecessor nodes uids.

        The list is such that:
        - nodes appear in the same order of inputs
        - nodes appear only once, hence the number of preds
          may not be equal to the number of inputs

        Returns:
            List of predecessors nodes uids
        """
        ...

    @property
    @abstractmethod
    def preds_nodes(self) -> Sequence["Node"]:
        """Returns the list of predecessor nodes.

        The list is the same as for preds, but contains the nodes
        instead of the nodes' uids.

        Returns:
            List of predecessors nodes
        """
        ...

    @property
    @abstractmethod
    def operator(self) -> Operator:
        """Returns the operator that defines this node's behavior.

        Returns:
            The algebraic operator associated with this node
        """
        ...

    @property
    @abstractmethod
    def operation(self) -> Operation:
        """Returns the operation that defines this node's behavior.

        The operation specifies the operator behavior and the instanciated
        operator dimensions.

        Returns:
            The algebraic operation associated with this node
        """
        ...

    @abstractmethod
    def forward_types(self, inputs_types: Sequence[TensorType]) -> Sequence[TensorType]:
        """Infers output tensor types from input tensor types.

        Args:
            inputs_types: List of input tensor types

        Returns:
            List of inferred output tensor types
        """
        ...

    @abstractmethod
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[Tensor]:
        """Evaluate the node with input tensors to produce output tensors.

        Args:
            inputs: List of input tensors

        Returns:
            List of output tensors
        """
        ...
