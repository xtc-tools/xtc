#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import TypeAlias, cast

from xtc.itf.graph import Graph, Node
from xtc.itf.data import TensorType, Tensor

from .node import XTCNode
from .utils import XTCGraphUtils
from .data import XTCTensorType

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
    def nodes(self) -> dict[str, Node]:
        return {node.name: node for node in self._nodes}

    @property
    @override
    def inputs(self) -> list[str]:
        return [node.name for node in self._inputs]

    @property
    @override
    def outputs(self) -> list[str]:
        return [node.name for node in self._outputs]

    def add_nodes(self, nodes: NodesType) -> None:
        self._nodes.extend(nodes)

    @property
    def inputs_nodes(self) -> list["XTCNode"]:
        return self._inputs

    @property
    def outputs_nodes(self) -> list["XTCNode"]:
        return self._outputs

    def set_inputs(self, inputs: InputsType) -> None:
        self._inputs = inputs

    def set_outputs(self, outputs: OutputsType) -> None:
        self._outputs = outputs

    @override
    def forward_types(self, inputs_types: list[TensorType]) -> list[TensorType]:
        assert len(inputs_types) == len(self._inputs), (
            f"forward types inputs size mismatch: {len(inputs_types)} != {len(self._inputs)}"
        )
        nodes = XTCGraphUtils.get_nodes_topological(self._nodes)
        outputs_map = {
            node: inp_type for node, inp_type in zip(self.inputs, inputs_types)
        }
        for node in nodes:
            inp_types = [outputs_map[inp_node] for inp_node in node.inputs]
            out_types = node.forward_types(inp_types)
            outputs_map[node.name] = out_types[0]
        outputs_types = [outputs_map[out_node] for out_node in self.outputs]
        return outputs_types

    @override
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        assert len(inputs) == len(self._inputs), (
            f"forward types inputs size mismatch: {len(inputs)} != {len(self._inputs)}"
        )
        nodes = XTCGraphUtils.get_nodes_topological(self._nodes)
        outputs_map = {node: inp for node, inp in zip(self.inputs, inputs)}
        for node in nodes:
            inps = [outputs_map[inp_node] for inp_node in node.inputs]
            outs = node.forward(inps)
            outputs_map[node.name] = outs[0]
        outputs = [outputs_map[out_node] for out_node in self.outputs]
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
            for name in self.inputs:
                graph_str += f"  - {name}\n"
        else:
            graph_str += "  inputs: []\n"
        if len(self._outputs) > 0:
            graph_str += "  outputs:\n"
            for name in self.outputs:
                graph_str += f"  - {name}\n"
        else:
            graph_str += "  outputs: []\n"
        if len(self._nodes) > 0:
            graph_str += "  nodes:\n"
            for node in nodes:
                graph_str += f"    {node.name}: {node}\n"
        else:
            graph_str += "  nodes: {}\n"
        return graph_str
