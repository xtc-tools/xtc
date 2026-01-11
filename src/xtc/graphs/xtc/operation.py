#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from collections.abc import Mapping, Sequence
from typing import Any
import json

from xtc.graphs.xtc.data import XTCTensorType
from xtc.itf.graph import Operation
from xtc.itf.graph.operation import AccessesMaps
from xtc.utils.math import mulall


class XTCOperation(Operation):
    def __init__(
        self,
        name: str,
        attrs: Mapping[str, Any],
        inputs_types: Sequence[XTCTensorType],
        outputs_types: Sequence[XTCTensorType],
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
    def inputs_types(self) -> Sequence[XTCTensorType]:
        return self._inputs_types

    @property
    @override
    def outputs_types(self) -> Sequence[XTCTensorType]:
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

    @property
    @override
    def ops_count(self) -> int:
        # Assume single output, hence estimate
        # ops as the product of all dimensions
        # in the iteration space
        shape = self._outputs_types[0].constant_shape
        ops_count = mulall(list(shape))
        return ops_count

    @property
    @override
    def ops_dtype(self) -> str:
        # Assume single output, hence estimate
        # dtype as the first output dtype
        return self._outputs_types[0].constant_dtype

    @property
    @override
    def signature(self) -> list[Any]:
        # Normalize json
        return json.loads(
            json.dumps(
                [self.name, list(self.dims.values()), self.ops_dtype, dict(self.attrs)]
            )
        )
