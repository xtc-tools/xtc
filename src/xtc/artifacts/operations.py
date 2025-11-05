#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import itertools
import logging

_OPERATION_REGISTRY: dict[str, dict[str, dict]] = {}

__all__ = [
    "register_operation",
    "get_operation",
    "has_operation",
    "list_operations",
]

logger = logging.getLogger(__name__)


def register_operation(
    operator: str, name: str, dims: dict[str, int], params: dict[str, int] | None = None
):
    if operator not in _OPERATION_REGISTRY:
        _OPERATION_REGISTRY[operator] = {}
    if params is None:
        params = {}
    operations = _OPERATION_REGISTRY[operator]
    canonical = name.lower()
    if canonical in operations:
        logger.warning(f"operation {operator}/{canonical} is already registered")
    _OPERATION_REGISTRY[operator][canonical] = dict(
        name=name,
        dims=dims,
        params=params,
    )


def get_operation(operator: str, name: str) -> dict:
    if operator not in _OPERATION_REGISTRY:
        raise ValueError(f"operator {operator} not registered in operation registry")
    operations = _OPERATION_REGISTRY[operator]
    canonical = name.lower()
    if canonical not in operations:
        raise ValueError(
            f"operation name {name} for operator {operator} not registered in operation registry"
        )
    return operations[canonical]


def has_operation(operator: str, name: str) -> bool:
    try:
        get_operation(operator, name)
    except ValueError:
        return False
    return True


def list_operations(operator: str = "") -> list[tuple[str, str]]:
    if operator == "":
        operations = list(
            itertools.chain(
                *[list_operations(operator) for operator in _OPERATION_REGISTRY.keys()]
            )
        )
        return operations
    if operator not in _OPERATION_REGISTRY:
        raise ValueError(f"operator {operator} not registered in operation registry")
    operations = [
        (operator, op["name"]) for op in _OPERATION_REGISTRY[operator].values()
    ]
    return operations
