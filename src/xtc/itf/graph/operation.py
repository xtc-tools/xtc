#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing import TypeAlias, Any
from collections.abc import Sequence, Mapping
from ..data import TensorType


DimSize: TypeAlias = int | str
DimsSizes: TypeAlias = tuple[DimSize, ...]
DimSpec: TypeAlias = str
DimsSpecs: TypeAlias = tuple[DimSpec, ...]
DimAccess: TypeAlias = str
InputAccess: TypeAlias = tuple[DimAccess, ...]
InputsAccesses: TypeAlias = tuple[InputAccess, ...]
OutputAccess: TypeAlias = tuple[DimAccess, ...]
OutputsAccesses: TypeAlias = tuple[OutputAccess, ...]
AccessesMaps: TypeAlias = tuple[DimsSpecs, InputsAccesses, OutputsAccesses]
OperationAttr: TypeAlias = Any
OperationAttrs: TypeAlias = Mapping[str, OperationAttr]


class Operation(ABC):
    """An abstract representation of an Operation, itself a specialized Operator.

    An Operation represents the computation performed by a Node, i.e. an Operator
    specification and instanciated dimensions and types for the inputs and
    outputs.

    The Operation computation is currently internal,
    though the inputs and outputs accesses functions are available through the
    accesses_maps.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of this operations's operator.

        Returns:
            The node's operator unique name
        """
        ...

    @property
    @abstractmethod
    def attrs(self) -> OperationAttrs:
        """Returns the dict of attributes for this operation.

        Returns:
            Dict of attributes per name
        """
        ...

    @property
    @abstractmethod
    def inputs_types(self) -> Sequence[TensorType]:
        """Returns the list of input tensors types for this operation.

        Returns:
            List of input tensors types
        """
        ...

    @property
    @abstractmethod
    def outputs_types(self) -> Sequence[TensorType]:
        """Returns the list of output tensors types for this operation.

        Returns:
            List of output tensors types
        """
        ...

    @property
    @abstractmethod
    def dims(self) -> Mapping[DimSpec, DimSize]:
        """Returns the dict of dimensions size for this operation.

        A dimension size may be resolved (int) or symbolic (str).

        Returns:
            Dict of dim name, dim size
        """
        ...

    @abstractmethod
    def dims_kind(self, kind: str) -> Sequence[DimSpec]:
        """Returns the list of dimensions of the given kind.

        The kind argument is currently one of:
        - "P" for parallel dims
        - "R" for reduction axes

        Returns:
            List of dims names
        """
        ...

    @property
    @abstractmethod
    def accesses_maps(self) -> AccessesMaps:
        """Returns the accesses map for this operation.

        Accesses maps are a 3-tuple with:
        - operation dimensions names tuple,
        - tuple of inputs accesses tuples for each input,
        - tuple of outputs accesses tuples for each output,

        Returns:
            Accesses map for this operation
        """
        ...
