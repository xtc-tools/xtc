#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from xdsl.dialects import func as func
from xdsl.utils.hints import isa
from xdsl.ir import (
    Block,
    Region,
    Operation,
)
from xdsl.dialects.builtin import (
    AnyMemRefType,
    AnyIntegerAttr,
    ArrayAttr,
    DictionaryAttr,
    UnitAttr,
)


def xdsl_operator_to_function(source_op: Operation, name: str) -> func.FuncOp:
    # Fetch data
    operands = source_op.operands
    shaped_types, scalar_types = [], []
    for o in operands:
        if isa(o.type, AnyMemRefType):
            shaped_types.append(o.type)
        else:
            scalar_types.append(o.type)

    #
    payload = Block(arg_types=shaped_types)
    concrete_operands = []
    shaped_count, scalar_count = 0, 0
    for o in operands:
        if isa(o.type, AnyMemRefType):
            concrete_operands.append(payload.args[shaped_count])
            shaped_count += 1
        else:
            if isa(o.type, xdslIntegerType):
                attr = AnyIntegerAttr(0, scalar_types[scalar_count])
            else:
                attr = xdslFloatAttr(0.0, scalar_types[scalar_count])
            constant = xdslConstant(attr)
            payload.add_ops([constant])
            concrete_operands.append(constant.results[0])
            scalar_count += 1

    value_mapper = {o: p for o, p in zip(operands, concrete_operands)}

    new_op = source_op.clone(value_mapper=value_mapper)
    payload.add_ops([new_op, func.Return()])
    payload_func = func.FuncOp(
        name=name,
        function_type=(shaped_types, ()),
        region=Region(payload),
        arg_attrs=ArrayAttr(
            param=[
                DictionaryAttr(data={"llvm.noalias": UnitAttr()}) for t in shaped_types
            ]
        ),
    )

    return payload_func
