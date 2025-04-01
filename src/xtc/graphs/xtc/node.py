#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import cast
import threading

from xtc.itf.graph import Node
from xtc.itf.operator import Operator
from xtc.itf.data import TensorType, Tensor

from .data import XTCTensorType
from .expr import XTCExpr, XTCTensorExpr, XTCOpExpr

__all__ = [
    "XTCNode",
]


class XTCNode(Node):
    _node_map: dict[str, "XTCNode"] = {}
    _node_id_map: dict[int, "XTCNode"] = {}

    @classmethod
    def get_node(cls, name: str) -> "XTCNode":
        node = cls._node_map.get(name)
        assert node is not None, (
            f"node name not found in nodes map: {name}: {cls._node_map}"
        )
        return node

    def __init__(self, expr: "XTCExpr", name: str | None = None) -> None:
        self._inputs_types: tuple[XTCTensorType, ...] | None = None
        self._outputs_types: tuple[XTCTensorType, ...] | None = None
        self._expr = expr
        if name is None:
            name = f"%{expr._idx}"
        self._name = name
        with threading.Lock():
            if name in self._node_map:
                raise RuntimeError(f"non unique name for node: {name}")
            self._node_map[name] = self
            self._node_id_map[expr._idx] = self

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def operator(self) -> Operator:
        assert isinstance(self._expr, XTCOpExpr) or isinstance(
            self._expr, XTCTensorExpr
        )
        assert self._expr._op is not None
        return self._expr._op

    @property
    @override
    def inputs(self) -> list[str]:
        return [node.name for node in self.inputs_nodes()]

    @property
    @override
    def outputs(self) -> list[str]:
        return [self.name]

    @property
    def node_idx(self) -> int:
        return self._expr._idx

    @override
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        if isinstance(self._expr, XTCTensorExpr):
            assert self._expr.value and self._expr.value.type.is_constant(), (
                f"Tensor type not constant in tensor initializer"
            )
            outputs_types = [self._expr.value.type]
        else:
            assert isinstance(self._expr, XTCOpExpr)
            outputs_types = self._expr.forward_types(inputs_types)
        return cast(list[TensorType], outputs_types)

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        if isinstance(self._expr, XTCTensorExpr):
            assert self._expr.value and self._expr.value.type.is_constant(), (
                f"Tensor type not constant in tensor initializer"
            )
            outputs = [self._expr.value]
        else:
            assert isinstance(self._expr, XTCOpExpr)
            outputs = self._expr.forward(inputs)
        return outputs

    def inputs_nodes(self) -> list["XTCNode"]:
        if not isinstance(self._expr, XTCOpExpr):
            return []
        inputs_nodes = [
            self._node_id_map.get(idx) for idx in [arg._idx for arg in self._expr.args]
        ]
        assert all([node is not None for node in inputs_nodes]), (
            f"unexpected non connected input for node: {self}, inputs: {inputs_nodes}"
        )
        return cast(list[XTCNode], inputs_nodes)

    @override
    def __str__(self) -> str:
        return str(self._expr).split("=", 1)[1].strip()
