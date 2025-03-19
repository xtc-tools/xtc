#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing import Any
import numpy.typing


class TensorType(ABC):
    """An abstract representation of a tensor's type information.

    TensorType defines the shape and data type characteristics of a tensor,
    providing the necessary information for type inference and validation
    during graph operations. This includes the tensor's dimensionality,
    size along each dimension, and the underlying data type.
    """

    @abstractmethod
    def shape(self) -> tuple[int]:
        """Returns the tensor's shape as a tuple of dimension sizes.

        Returns:
            The size of each dimension in the tensor
        """
        ...

    @abstractmethod
    def dtype(self) -> str:
        """Returns the tensor's data type.

        Returns:
            The underlying data type of the tensor elements
        """
        ...

    @abstractmethod
    def ndims(self) -> int:
        """Returns the number of dimensions in the tensor.

        Returns:
            The tensor's dimensionality
        """
        ...


class Tensor(ABC):
    """An abstract representation of a multidimensional object.

    A Tensor is a fundamental input/output type in the dataflow graph,
    representing multidimensional data with associated type information.
    Tensors are used as inputs and outputs for Node operations in the Graph,
    and their dimensions and types can be used for inference throughout
    the compilation process.
    """

    @abstractmethod
    def type(self) -> TensorType:
        """Returns the tensor's type information.

        Returns:
            The type descriptor containing shape and dtype information
        """
        ...

    @abstractmethod
    def data(self) -> Any:
        """Returns the tensor's linearized data.

        Returns:
            any: The tensor's data
        """
        ...

    @abstractmethod
    def numpy(self) -> numpy.typing.NDArray:
        """Convert the tensor to a numpy array.

        Returns:
            The tensor's data as a numpy array
        """
        ...
