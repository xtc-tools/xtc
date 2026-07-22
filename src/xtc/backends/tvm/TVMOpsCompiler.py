#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing_extensions import override
from typing import Any, TypeAlias

import tvm
import tvm.te as te

from .TVMOps import (
    TVMBaseExpr,
    TVMOperation,
    TVMGraph,
)

__all__ = [
    "TVMExprCompiler",
    "TVMSchedulableExpr",
    "TVMSchedulableExpr",
    "TVMSchedulableExprTE",
    "TVMSchedulableExprTIR",
    "TVMScheduledExpr",
    "TVMScheduledExprTE",
    "TVMScheduledExprTIR",
]


TETensor: TypeAlias = te.Tensor
TIRSchedule: TypeAlias = tvm.tir.Schedule
TIRFunc: TypeAlias = tvm.tir.PrimFunc
TESchedule: TypeAlias = Any  # te.Schedule not available on tvm > 0.19


class TVMExprCompiler:
    def __init__(self, expr: TVMBaseExpr, tir_schedule: bool = True):
        self._expr = expr
        self._tir_schedule = tir_schedule

    def generate(self) -> "TVMSchedulableExpr":
        if isinstance(self._expr, TVMGraph):
            vars, params = [
                list(vars.values()) for vars in self._expr._te_expr_from_graph()
            ]
        else:
            assert isinstance(self._expr, TVMOperation)
            params = list(self._expr.operator.generate_op())
            vars = params
        if self._tir_schedule:
            prim_func = te.create_prim_func(params)
            return TVMSchedulableExprTIR(self._expr, prim_func)
        return TVMSchedulableExprTE(self._expr, params, vars)


class TVMSchedulableExpr(ABC):
    @abstractmethod
    def schedule(self, schedule: Any = None) -> "TVMScheduledExpr": ...

    @property
    @abstractmethod
    def expr(self) -> TVMBaseExpr: ...


class TVMSchedulableExprTE(TVMSchedulableExpr):
    def __init__(
        self,
        expr: TVMBaseExpr,
        params: Sequence[TETensor],
        tensors: Sequence[TETensor] | None = None,
    ):
        self._expr = expr
        self._params = list(params)
        self._tensors = list(params) if tensors is None else list(tensors)

    @property
    @override
    def expr(self) -> TVMBaseExpr:
        return self._expr

    @override
    def schedule(self, schedule: Any = None) -> "TVMScheduledExprTE":
        sch = te.create_schedule(self._params[-1].op)  # type: ignore
        if schedule is not None:
            schedule_map = schedule.schedule_impl
            tensors_map = {t.name: t for t in self._tensors}
            for sched in schedule_map.values():
                if sched:
                    exec(sched, {"sch": sch, "obj": tensors_map}, {})
        return TVMScheduledExprTE(self, sch)


class TVMSchedulableExprTIR(TVMSchedulableExpr):
    def __init__(self, expr: TVMBaseExpr, func: TIRFunc):
        self._expr = expr
        self._func = func

    @property
    @override
    def expr(self) -> TVMBaseExpr:
        return self._expr

    @override
    def schedule(self, schedule: Any = None) -> "TVMScheduledExprTIR":
        func_name = self._expr.name
        func = self._func.with_attr("global_symbol", self._expr.name)
        mod = tvm.IRModule({func_name: func})
        sch = tvm.tir.Schedule(mod)
        if schedule is None:
            return TVMScheduledExprTIR(self, sch)
        # TODO: schedule TIR
        schedule_map = schedule.schedule_impl
        sch.work_on(func_name)
        for sched in schedule_map.values():
            if sched:
                exec(sched, {"sch": sch}, {})
        return TVMScheduledExprTIR(self, sch)


class TVMScheduledExpr(ABC):
    @property
    @abstractmethod
    def schedulable(self) -> TVMSchedulableExpr: ...

    @abstractmethod
    def dumps(self) -> str: ...


class TVMScheduledExprTE(TVMScheduledExpr):
    def __init__(self, schedulable: TVMSchedulableExprTE, schedule: TESchedule):
        self._schedulable = schedulable
        self._schedule = schedule

    @property
    @override
    def schedulable(self) -> TVMSchedulableExprTE:
        return self._schedulable

    @override
    def dumps(self) -> str:
        return str(
            tvm.lower(self._schedule, self._schedulable._params, simple_mode=True)  # type: ignore
        )


class TVMScheduledExprTIR(TVMScheduledExpr):
    def __init__(self, schedulable: TVMSchedulableExprTIR, schedule: TIRSchedule):
        self._schedulable = schedulable
        self._schedule = schedule

    @property
    @override
    def schedulable(self) -> TVMSchedulableExprTIR:
        return self._schedulable

    @override
    def dumps(self) -> str:
        return str(self._schedule.mod)
