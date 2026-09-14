#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Any, TypeAlias, cast
import tempfile
from pathlib import Path
import shutil

import tvm
import tvm.te as te
import tvm.s_tir

from .TVMOps import (
    TVMBaseExpr,
    TVMOperation,
    TVMGraph,
)
from .TVMScheduleTransforms import externalize_tile_below, loop_partition_rebased

__all__ = [
    "TVMExprCompiler",
    "TVMSchedulableExpr",
    "TVMSchedulableExpr",
    "TVMSchedulableExprTIR",
    "TVMScheduledExpr",
    "TVMScheduledExprTIR",
]


TETensor: TypeAlias = te.Tensor
TIRSchedule: TypeAlias = tvm.s_tir.Schedule
TIRFunc: TypeAlias = tvm.tirx.PrimFunc
TESchedule: TypeAlias = Any  # te.Schedule not available on tvm > 0.19
TEParam: TypeAlias = te.Tensor | tvm.tirx.Var


class TVMExprCompiler:
    def __init__(self, expr: TVMBaseExpr):
        self._expr = expr

    def generate(self) -> "TVMSchedulableExpr":
        if isinstance(self._expr, TVMGraph):
            _, params = [
                list(vars.values()) for vars in self._expr._te_expr_from_graph()
            ]
        else:
            assert isinstance(self._expr, TVMOperation)
            params = list(self._expr.operator.generate_op())
        args = cast(list[TEParam], params)
        prim_func = te.create_prim_func(args)
        return TVMSchedulableExprTIR(self._expr, prim_func)


class TVMSchedulableExpr(ABC):
    @abstractmethod
    def schedule(self, schedule: Any = None) -> "TVMScheduledExpr": ...

    @property
    @abstractmethod
    def expr(self) -> TVMBaseExpr: ...


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
        sch = tvm.s_tir.Schedule(mod)
        if schedule is None:
            return TVMScheduledExprTIR(self, sch)
        schedule_map = schedule.schedule_impl
        sch.work_on(func_name)
        namespace = {
            "sch": sch,
            "loop_partition_rebased": loop_partition_rebased,
            "externalize_tile_below": externalize_tile_below,
        }
        for sched in schedule_map.values():
            if sched:
                self._exec_schedule(sched, namespace)
        sch = cast(TIRSchedule, namespace["sch"])
        return TVMScheduledExprTIR(self, sch)

    def _exec_schedule(self, sched: str, namespace: dict[str, Any]):
        tdir = Path(tempfile.mkdtemp(dir="."))
        try:
            outf = tdir / "schedule.py"
            outf.write_text(sched)
            code = compile(sched, outf, "exec")
            exec(code, namespace, namespace)
        except Exception:
            raise
        else:
            shutil.rmtree(tdir)


class TVMScheduledExpr(ABC):
    @property
    @abstractmethod
    def schedulable(self) -> TVMSchedulableExpr: ...

    @abstractmethod
    def dumps(self) -> str: ...


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
