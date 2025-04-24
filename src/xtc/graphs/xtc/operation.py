#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from collections.abc import Mapping, Sequence
from typing import Any

from xtc.itf.graph import Operation
from xtc.itf.graph.operation import AccessesMaps
from xtc.itf.data import TensorType


class XTCOperation(Operation):
    def __init__(
        self,
        name: str,
        attrs: Mapping[str, Any],
        inputs_types: Sequence[TensorType],
        outputs_types: Sequence[TensorType],
        dims: Mapping[str, int | str],
        kinds: Sequence[str],
        inps_maps: Sequence[Sequence[str]],
        outs_maps: Sequence[Sequence[str]],
    ) -> None:
        self._name = name
        self._inputs_types = tuple(inputs_types)
        self._outputs_types = tuple(outputs_types)
        self._attrs = dict(attrs)
        self._dims = dict(dims)
        self._kinds = tuple(kinds)
        assert len(self._dims) == len(self._kinds)
        self._maps = (
            tuple(dims.keys()),
            tuple([tuple(inp_map) for inp_map in inps_maps]),
            tuple([tuple(out_map) for out_map in outs_maps]),
        )

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def attrs(self) -> Mapping[str, Any]:
        return self._attrs

    @property
    @override
    def inputs_types(self) -> Sequence[TensorType]:
        return self._inputs_types

    @property
    @override
    def outputs_types(self) -> Sequence[TensorType]:
        return self._outputs_types

    @property
    @override
    def dims(self) -> Mapping[str, int | str]:
        return self._dims

    @override
    def dims_kind(self, kind: str) -> Sequence[str]:
        return [d for d, k in zip(self._dims, self._kinds) if k == kind]

    @property
    @override
    def accesses_maps(self) -> AccessesMaps:
        return self._maps
