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
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    AnyMemRefType,
    AnyIntegerAttr,
    FloatAttr,
    DictionaryAttr,
    UnitAttr,
    IntegerType,
)

from xdsl.context import MLContext
from xdsl.parser import Parser
from xdsl.dialects import func, linalg, arith, memref
from xdsl.dialects.builtin import ModuleOp


def parse_xdsl_module(source: str) -> ModuleOp:
    context = MLContext()
    context.load_dialect(func.Func)
    context.load_dialect(linalg.Linalg)
    context.load_dialect(arith.Arith)
    context.load_dialect(memref.MemRef)
    parser = Parser(context, source)
    module = parser.parse_module()
    return module


def brand_inputs_with_noalias(myfunc: func.FuncOp):
    alias_attrs = [
        DictionaryAttr(value={"llvm.noalias": UnitAttr()})
        for ity in myfunc.function_type.inputs
    ]
    myfunc.arg_attrs = alias_attrs


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
            if isa(o.type, IntegerType):
                attr = AnyIntegerAttr(0, scalar_types[scalar_count])
            else:
                attr = FloatAttr(0.0, scalar_types[scalar_count])
            constant = ConstantOp(attr)
            payload.add_ops([constant])
            concrete_operands.append(constant.results[0])
            scalar_count += 1

    value_mapper = {o: p for o, p in zip(operands, concrete_operands)}

    new_op = source_op.clone(value_mapper=value_mapper)
    payload.add_ops([new_op, func.ReturnOp()])
    payload_func = func.FuncOp(
        name=name,
        function_type=(shaped_types, ()),
        region=Region(payload),
    )

    return payload_func
