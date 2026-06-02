#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import torch
import torch.nn.functional as F

from torch._inductor.custom_graph_pass import CustomGraphPass

_LINEAR_TARGETS = frozenset(
    {
        F.linear,
        torch.ops.aten.linear.default,
    }
)


def replace_linear_pass(graph: torch.fx.Graph) -> None:
    for node in graph.nodes:
        if node.op == "call_function" and node.target in _LINEAR_TARGETS:
            node.target = torch.ops.xtc.matmul.default
    graph.eliminate_dead_code()


class ReplaceLinearPass(CustomGraphPass):
    def __call__(self, graph: torch.fx.Graph) -> None:
        replace_linear_pass(graph)

    def uuid(self) -> str:
        return "replace_linear_with_xtc_matmul"


def register_pre_grad_pass() -> None:
    import torch._inductor.config as inductor_config

    inductor_config.pre_grad_custom_pass = ReplaceLinearPass()
