#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from collections.abc import Sequence
from ..data.tensor import Tensor, TensorType


class Operator(ABC):
    """An abstract representation of the algebraic operation for a node.

    An Operator defines the semantic behavior of operations in the graph, including
    how it transforms input tensor types and how it processes tensor data. It provides
    both type inference capabilities and concrete implementations of the operation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique identifier for this operator type.

        Returns:
            The operator's name
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
        """Evaluate the operator with input tensors to produce output tensors.

        Args:
            inputs: List of input tensors

        Returns:
            List of output tensors
        """
        ...
