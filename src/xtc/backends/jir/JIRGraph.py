#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override

import xtc.itf as itf

from .JIROps import JIROperation


class JIROperator(itf.operator.Operator):
    def __init__(
        self,
        source_op: JIROperation,
    ):
        self._operator = source_op.operator

    @property
    @override
    def name(self) -> str:
        return self._operator.name

    @override
    def apply(self, inputs: list[itf.data.Tensor]) -> list[itf.data.Tensor]:
        # TODO
        return []

    @override
    def applyType(self, inputs: list[itf.data.TensorType]) -> list[itf.data.TensorType]:
        # TODO
        return []


class JIRNode(itf.graph.Node):
    def __init__(
        self,
        name: str,
        operator: JIROperator,
        dims: dict[str, int],
    ) -> None:
        self._name = name
        self._operator = operator
        self._dims = dims

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def operator(self) -> itf.operator.Operator:
        return self._operator

    @property
    @override
    def inputs(self) -> list[str]:
        # TODO: Node inputs undefined for now
        return []

    @property
    @override
    def outputs(self) -> list[str]:
        # Tensor output 0 is node name
        return [self.name]


class JIRGraph(itf.graph.Graph):
    def __init__(
        self,
        name: str,
        nodes: list[JIRNode],
    ) -> None:
        assert len(nodes) > 0
        self._name = name
        self._nodes = nodes
        self._inputs = [nodes[0].name]
        self._outputs = [nodes[-1].name]

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def nodes(self) -> dict[str, itf.graph.Node]:
        return {node.name: node for node in self._nodes}

    @property
    @override
    def inputs(self) -> list[str]:
        return self._inputs

    @property
    @override
    def outputs(self) -> list[str]:
        return self._outputs
