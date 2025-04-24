#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from collections.abc import Sequence, Mapping
from typing import TypeAlias, cast

from xtc.itf.graph import Graph, Node
from xtc.itf.data import TensorType, Tensor

from .node import XTCNode
from .utils import XTCGraphUtils
from .data import XTCTensor, XTCTensorType

__all__ = [
    "XTCGraph",
]


DefLoc: TypeAlias = "XTCNode"
UseLoc: TypeAlias = "XTCNode"
InputsType: TypeAlias = list[UseLoc]
OutputsType: TypeAlias = list[DefLoc]
NodesType: TypeAlias = list["XTCNode"]


class XTCGraph(Graph):
    def __init__(self, name: str | None = None) -> None:
        self._inputs: InputsType = []
        self._outputs: OutputsType = []
        self._nodes: NodesType = []
        self._name = name

    @property
    @override
    def name(self) -> str:
        return "" if self._name is None else self._name

    @property
    @override
    def nodes(self) -> Mapping[str, Node]:
        return {node.uid: node for node in self._nodes}

    @property
    @override
    def inputs(self) -> Sequence[str]:
        return [node.uid for node in self._inputs]

    @property
    @override
    def outputs(self) -> Sequence[str]:
        return [node.uid for node in self._outputs]

    @property
    @override
    def inputs_nodes(self) -> Sequence["XTCNode"]:
        return self._inputs

    @property
    @override
    def outputs_nodes(self) -> Sequence["XTCNode"]:
        return self._outputs

    def add_nodes(self, nodes: NodesType) -> None:
        self._nodes.extend(nodes)

    def set_inputs(self, inputs: InputsType) -> None:
        self._inputs = inputs

    def set_outputs(self, outputs: OutputsType) -> None:
        self._outputs = outputs

    @override
    def forward_types(
        self, inputs_types: Sequence[TensorType]
    ) -> Sequence[XTCTensorType]:
        assert len(inputs_types) == len(self._inputs), (
            f"forward types inputs size mismatch: {len(inputs_types)} != {len(self._inputs)}"
        )
        nodes = XTCGraphUtils.get_nodes_topological(self._nodes)
        outputs_map = {
            node_uid: cast(XTCTensorType, inp_type)
            for node_uid, inp_type in zip(self.inputs, inputs_types)
        }
        for node in nodes:
            inp_types = [outputs_map[node_uid] for node_uid in node.inputs]
            out_types = node.forward_types(inp_types)
            outputs_map[node.uid] = out_types[0]
        outputs_types = [outputs_map[node_uid] for node_uid in self.outputs]
        return outputs_types

    @override
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[XTCTensor]:
        assert len(inputs) == len(self._inputs), (
            f"forward types inputs size mismatch: {len(inputs)} != {len(self._inputs)}"
        )
        nodes = XTCGraphUtils.get_nodes_topological(self._nodes)
        outputs_map = {
            node_uid: cast(XTCTensor, inp) for node_uid, inp in zip(self.inputs, inputs)
        }
        for node in nodes:
            inps = [outputs_map[node_uid] for node_uid in node.inputs]
            outs = node.forward(inps)
            outputs_map[node.uid] = outs[0]
        outputs = [outputs_map[node_uid] for node_uid in self.outputs]
        return outputs

    @override
    def __str__(self) -> str:
        nodes = XTCGraphUtils.get_nodes_topological_from_seed(
            self._nodes, self._outputs
        )
        graph_str = "graph:\n"
        if self.name != "":
            graph_str += f"  name: {self._name}\n"
        if len(self._inputs) > 0:
            graph_str += "  inputs:\n"
            for node_uid in self.inputs:
                graph_str += f"  - {node_uid}\n"
        else:
            graph_str += "  inputs: []\n"
        if len(self._outputs) > 0:
            graph_str += "  outputs:\n"
            for node_uid in self.outputs:
                graph_str += f"  - {node_uid}\n"
        else:
            graph_str += "  outputs: []\n"
        if len(self._nodes) > 0:
            graph_str += "  nodes:\n"
            for node in nodes:
                graph_str += f"  - {node.uid}: {node}\n"
        else:
            graph_str += "  nodes: {}\n"
        return graph_str
