#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from .node import XTCNode


__all__ = [
    "XTCGraphUtils",
]


class XTCGraphUtils:
    @staticmethod
    def get_nodes_outputs(nodes: list[XTCNode]) -> list[XTCNode]:
        unique_nodes = list({node.node_idx: node for node in nodes}.values())
        assert len(unique_nodes) == len(nodes)
        used = set()
        for node in nodes:
            inputs_nodes = node.inputs_nodes()
            for src_node in inputs_nodes:
                used.add(src_node)
        outputs = [node for node in nodes if node not in used]
        return outputs

    @staticmethod
    def get_nodes_topological_from_seed(
        nodes: list[XTCNode], seed: list[XTCNode]
    ) -> list[XTCNode]:
        unique_nodes = list({node.node_idx: node for node in nodes}.values())
        assert len(unique_nodes) == len(nodes)
        nodes_set = set(nodes)
        assert all([node in nodes_set for node in seed])
        seen = set()
        rwalk = []

        def reverse_walk(out: XTCNode) -> None:
            if out in seen or out not in nodes_set:
                return
            seen.add(out)
            inputs_nodes = out.inputs_nodes()
            for src_node in inputs_nodes:
                reverse_walk(src_node)
            rwalk.append(out)

        for node in seed:
            reverse_walk(node)
        return rwalk

    @staticmethod
    def get_nodes_topological(nodes: list[XTCNode]) -> list[XTCNode]:
        outputs = XTCGraphUtils.get_nodes_outputs(nodes)
        sorted = XTCGraphUtils.get_nodes_topological_from_seed(nodes, outputs)
        return sorted
