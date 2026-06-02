#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""PyTorch integration: XTC-backed ``xtc::matmul`` and ``torch.compile`` hooks."""

from __future__ import annotations

from enum import Enum

from xtc.integration.pytorch.fx_rewrite import register_pre_grad_pass
from xtc.integration.pytorch.ops import register_ops


class XtcIntegration(Enum):
    """How XTC-backed ops are invoked under ``torch.compile``."""

    EAGER = "eager"
    INDUCTOR_CPP = "inductor-cpp"


def register_torch_xtc_extensions(
    *,
    xtc_integration: XtcIntegration | str = XtcIntegration.EAGER,
) -> None:
    """Register XTC custom ops, Inductor pass, and integration mode."""
    import torch._inductor.config as inductor_config

    # IMPORT CUSTOM OPS AND REWRITE PASS

    register_ops()

    register_pre_grad_pass()

    # REGISTER INDUCTOR CPP LOWERING IF NEEDED

    integration = (
        xtc_integration
        if isinstance(xtc_integration, XtcIntegration)
        else XtcIntegration(xtc_integration)
    )

    if integration == XtcIntegration.INDUCTOR_CPP:
        inductor_config.cpp_wrapper = True
        from xtc.integration.pytorch.inductor_cpp import register_inductor_cpp_hooks
        from xtc.integration.pytorch.inductor_lowering import (
            register_inductor_lowerings,
        )

        register_inductor_cpp_hooks()
        register_inductor_lowerings()
