#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from collections.abc import Sequence
from typing import cast

from xtc.itf.graph import Node
from xtc.itf.operator import Operator
from xtc.itf.data import TensorType, Tensor

from .data import XTCTensor, XTCTensorType
from .expr import XTCExpr, XTCTensorExpr, XTCOpExpr
from .operation import XTCOperation

__all__ = [
    "XTCNode",
]


class XTCNode(Node):
    _node_uid_map: dict[str, "XTCNode"] = {}

    @classmethod
    def _get_node(cls, uid: str) -> "XTCNode":
        node = cls._node_uid_map.get(uid)
        assert node is not None, (
            f"node name not found in nodes idx map: {uid}: {cls._node_uid_map}"
        )
        return node

    def __init__(self, expr: "XTCExpr", name: str | None = None) -> None:
        # TODO: assume a single node output for now,
        # i.e. a node is equivalent to it's first output and to the expression
        self._inputs_types: list[XTCTensorType] | None = None
        self._outputs_types: list[XTCTensorType] | None = None
        self._expr = expr
        self._node_uid_map[self.uid] = self
        if name is None:
            name = self.uid
        self._name = name

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def uid(self) -> str:
        return self._expr.uid

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
        # TODO: assume single output for all nodes,
        # i.e. input argument is the expression itself
        return [arg[0] for arg in self._inputs_args()]

    @property
    @override
    def outputs(self) -> list[str]:
        # TODO: assume single output for all nodes,
        # i.e. output is the expression itself
        return [self.uid]

    @property
    @override
    def preds(self) -> Sequence[str]:
        return list(dict.fromkeys([arg[0] for arg in self._inputs_args()]))

    @property
    @override
    def preds_nodes(self) -> Sequence["XTCNode"]:
        return [self._get_node(node_uid) for node_uid in self.preds]

    @property
    @override
    def inputs_types(self) -> Sequence[XTCTensorType] | None:
        return self._inputs_types

    @property
    @override
    def outputs_types(self) -> Sequence[XTCTensorType] | None:
        return self._outputs_types

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        self._inputs_types = [cast(XTCTensorType, typ) for typ in inputs_types]
        if isinstance(self._expr, XTCTensorExpr):
            assert self._expr.type.is_constant(), (
                f"Tensor type not constant in tensor initializer"
            )
            outputs_types: Sequence[XTCTensorType] = [self._expr.type]
        else:
            assert isinstance(self._expr, XTCOpExpr)
            outputs_types = self._expr.forward_types(inputs_types)
        self._outputs_types = list(outputs_types)
        return outputs_types

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        if isinstance(self._expr, XTCTensorExpr):
            assert self._expr.type.is_constant(), (
                f"Tensor type not constant in tensor initializer"
            )
            outputs: Sequence[XTCTensor] = [self._expr.value]
        else:
            assert isinstance(self._expr, XTCOpExpr)
            outputs = self._expr.forward(inputs)
        return outputs

    def _inputs_args(self) -> Sequence[tuple[str, int]]:
        inps_args = [(inp.uid, 0) for inp in self._expr.args]
        return inps_args

    @property
    @override
    def operation(self) -> XTCOperation:
        return self._operation()

    def _operation(self) -> XTCOperation:
        assert self._inputs_types is not None
        assert self._outputs_types is not None
        inputs_types = self._inputs_types
        outputs_types = self._outputs_types
        assert isinstance(self._expr, XTCTensorExpr) or isinstance(
            self._expr, XTCOpExpr
        )
        return self._expr._op.get_operation(
            inps_types=inputs_types,
            outs_types=outputs_types,
        )

    @override
    def __str__(self) -> str:
        attrs = {}
        if self.uid != self.name:
            attrs["name"] = self.name
        attrs_str = ", ".join([f"{k} = {repr(v)}" for k, v in attrs.items()])
        if attrs_str:
            attrs_str = f" {{{attrs_str}}}"
        type_str = ""
        if self.inputs_types is not None and self.outputs_types is not None:
            type_str = f" : {self.inputs_types} -> {self.outputs_types}"
        return str(self._expr).split("=", 1)[1].strip() + attrs_str + type_str
