#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os
import sys
import tempfile
from typing import Any, TYPE_CHECKING, cast
from typing_extensions import override

from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects import bufferization
from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.context import Context
from xdsl.ir import Block
from xdsl.parser import Parser

import xtc.itf as itf
from xtc.targets.iree import IREEModule

from .IREEConfig import IREEConfig
from .IREEScheduler import IREESchedule

if TYPE_CHECKING:
    from .IREEBackend import IREEBackend

__all__ = [
    "IREECompiler",
]


def _rewrite_to_tensor_return(fn: FuncOp, n_out: int) -> FuncOp:
    """Turn XTC's ``(tensors..., memref outputs...) -> ()`` function into an
    IREE-friendly ``(tensors...) -> (tensors...)`` function."""
    block = cast(Block, fn.body.first_block)
    mats = [
        op
        for op in block.ops
        if isinstance(op, bufferization.MaterializeInDestinationOp)
    ]
    assert len(mats) == n_out, (
        f"expected {n_out} materialize_in_destination ops, found {len(mats)}"
    )
    out_tensors = [m.operands[0] for m in mats]
    out_args = list(block.args)[-n_out:]
    for op in list(mats):
        op.detach()
        op.erase()
    terminator = block.last_op
    assert terminator is not None
    terminator.detach()
    terminator.erase()
    block.add_op(ReturnOp(*out_tensors))
    for arg in reversed(out_args):
        block.erase_arg(arg)
    in_types = [arg.type for arg in block.args]
    out_types = [t.type for t in out_tensors]
    fn.properties["function_type"] = FunctionType.from_lists(in_types, out_types)
    return fn


def _attach_compilation_info(fn: FuncOp, configs: dict[str, str]) -> None:
    """Attach each ``compilation_info`` attribute onto the op carrying its id."""
    ctx = Context(allow_unregistered=True)
    remaining = dict(configs)
    for op in fn.walk():
        for op_id in list(remaining):
            if op_id in op.attributes:
                info = remaining.pop(op_id)
                op.attributes["compilation_info"] = Parser(ctx, info).parse_attribute()
    assert not remaining, f"markers not found in the module: {list(remaining)}"


class IREECompiler(itf.comp.Compiler):
    """Compiler carrying `IREESchedule` to IREE."""

    def __init__(self, backend: "IREEBackend", **kwargs: Any) -> None:
        self._backend = backend
        # shared_lib is accepted for API parity with the other backends but
        # ignored: IREE always produces a vmfb.
        kwargs.pop("shared_lib", None)
        self._config = IREEConfig(**kwargs)

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    @property
    def config(self) -> IREEConfig:
        return self._config

    def generate_mlir(self, schedule: IREESchedule) -> str:
        """Emit the annotated MLIR module IREE will compile."""
        fn = cast(FuncOp, self._backend.mlir_backend.xdsl_func.clone())
        n_out = len(self._backend.graph.outputs)
        _rewrite_to_tensor_return(fn, n_out)
        _attach_compilation_info(fn, schedule.lowering_configs())
        return str(ModuleOp([fn]))

    @override
    def compile(self, schedule: itf.schd.Schedule) -> itf.comp.Module:
        """Compile the annotated MLIR to an IREE ``.vmfb`` and wrap it."""
        from iree.compiler import tools as ireec

        assert isinstance(schedule, IREESchedule)
        mlir_text = self.generate_mlir(schedule)
        if self._config.print_source_ir:
            print("// -----// IREE input MLIR //----- //", file=sys.stderr)
            print(mlir_text, file=sys.stderr)

        name = self._backend.graph.name
        dump_file = self._config.dump_file
        if dump_file is None:
            fd, dump_file = tempfile.mkstemp(prefix=f"{name}_")
            os.close(fd)
        vmfb_path = f"{dump_file}.vmfb"

        vmfb = ireec.compile_str(
            mlir_text,
            target_backends=[self._config.target_backend],
            extra_args=self._config.iree_compile_args(),
        )
        assert vmfb is not None, "iree-compile produced no output"
        with open(vmfb_path, "wb") as f:
            f.write(vmfb)

        return IREEModule(
            name=name,
            payload_name=name,
            file_name=vmfb_path,
            graph=self._backend.graph,
            parallelized=schedule.parallelized,
        )
