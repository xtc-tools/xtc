#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from typing import Any
from dataclasses import dataclass
import re
import strictyaml
from typing_extensions import override

from .exceptions import ScheduleParseError

literal = int | str


def isint(s: str):
    return s.lstrip("-").isdigit()


@dataclass(frozen=True)
class Annotations:
    """AST Type : annotations that can be applied to a loop.

    Attributes:
        unroll_factor: The unroll factor. None means "unroll fully" (use loop size).
            Only meaningful when unroll_specified is True.
        unroll_specified: True if unroll was explicitly requested.
        vectorize: True if vectorization was requested.
        parallelize: True if parallelization was requested.
        buffer: The memory type for the buffer. None means default memory type.
            Only meaningful when buffer_specified is True.
        buffer_specified: True if buffer was explicitly requested.
        pack: Pack configuration as (input_idx, mtype, pad). mtype is None for default.
            Only meaningful when pack_specified is True.
        pack_specified: True if pack was explicitly requested.
    """

    unroll_factor: literal | None = None
    unroll_specified: bool = False
    vectorize: bool | str = False
    parallelize: bool | str = False
    buffer: str | None = None
    buffer_specified: bool = False
    pack: tuple[literal, str | None, bool | str] | None = None
    pack_specified: bool = False
    partial: bool = False
    full: bool = False


@dataclass(frozen=True)
class SplitDecl:
    """AST Type: a split declaration like 'axis[start:end]'."""

    axis: str
    start: literal | None
    end: literal | None
    size: literal | None
    body: ScheduleSpec

    @override
    def __str__(self) -> str:
        if self.size is not None:
            return f"{self.axis}[:{self.size}:]"
        start_str = "" if self.start is None else str(self.start)
        end_str = "" if self.end is None else str(self.end)
        decl = f"{self.axis}[{start_str}:{end_str}]"
        return decl


@dataclass(frozen=True)
class TileDecl:
    """AST Type: a tile declaration like 'axis#size'."""

    axis: str
    size: literal
    annotations: Annotations

    @override
    def __str__(self) -> str:
        return f"{self.axis}#{self.size}"


@dataclass(frozen=True)
class AxisDecl:
    """AST Type: a direct axis reference."""

    axis: str
    annotations: Annotations


ScheduleItem = SplitDecl | TileDecl | AxisDecl


@dataclass(frozen=True)
class ScheduleSpec:
    """AST Type: the complete parsed schedule specification."""

    items: tuple[ScheduleItem, ...]


@dataclass()
class ScheduleParser:
    """Parses a dict-based schedule specification into an AST."""

    _SPLIT_PATTERN = re.compile(r"^(.*)\[(-\d+|\d*)?:(-\d+|\d*)?\]$")
    _SPLIT_MIDDLE_PATTERN = re.compile(r"^(.*)\[:(\w*):\]$")

    def parse(self, spec: dict[str, Any]) -> ScheduleSpec:
        """Parse a schedule specification dict into an AST."""
        items: list[ScheduleItem] = []

        for declaration, value in spec.items():
            item = self._parse_declaration(declaration, value)
            items.append(item)

        return ScheduleSpec(items=tuple(items))

    def _parse_declaration(self, declaration: str, value: Any) -> ScheduleItem:
        """Parse a single declaration into a ScheduleItem."""
        assert isinstance(value, dict)
        # Try split declaration first (e.g., "axis[0:10]")
        if ":" in declaration:
            return self._parse_split(declaration, value)

        # Try tile declaration (e.g., "axis#32")
        if "#" in declaration:
            return self._parse_tile(declaration, value)

        # Must be a direct axis reference
        return self._parse_axis_ref(declaration, value)

    def _parse_split(self, declaration: str, value: dict) -> SplitDecl:
        """Parse a split declaration like 'axis[start:end]'."""
        axis_name, start, end, size = self._parse_split_syntax(declaration)

        body = self.parse(value)
        return SplitDecl(axis=axis_name, start=start, end=end, body=body, size=size)

    def _parse_tile(self, declaration: str, value: dict) -> TileDecl:
        """Parse a tile declaration like 'axis#size'."""
        parts = declaration.split("#")
        if len(parts) != 2:
            raise ScheduleParseError(
                f"`{declaration}`: invalid tile syntax, expected 'axis#size'"
            )

        axis_name, size_str = parts

        size = int(size_str) if isint(size_str) else size_str

        annotations = self._parse_annotations(value, declaration)
        return TileDecl(axis=axis_name, size=size, annotations=annotations)

    def _parse_axis_ref(self, declaration: str, value: dict) -> AxisDecl:
        """Parse a direct axis reference."""

        annotations = self._parse_annotations(value, declaration)
        return AxisDecl(axis=declaration, annotations=annotations)

    def _parse_annotations(self, value: dict[str, Any], context: str) -> Annotations:
        """Parse annotation dict into Annotations object."""

        unroll_factor: literal | None = None
        unroll_specified = False
        vectorize = False
        parallelize = False
        buffer: str | None = None
        buffer_specified = False
        pack: tuple[literal, str | None, bool | str] | None = None
        pack_specified = False
        partial = False
        full = False

        for key, param in value.items():
            match key:
                case "unroll":
                    if param is True or param is None:
                        unroll_factor = None
                        unroll_specified = True
                    elif param is False:
                        pass
                    elif isinstance(param, int | str):
                        unroll_factor = param
                        unroll_specified = True
                    else:
                        raise ScheduleParseError(
                            f'`{{"unroll" = {param}}}`: unroll parameter should be True, False, or a string or integer.'
                        )
                case "vectorize":
                    if not isinstance(param, bool | str):
                        raise ScheduleParseError(
                            f'`{{"vectorize" = {param}}}`: vectorization parameter should be True, False or a string.'
                        )
                    vectorize = param
                case "parallelize":
                    if not isinstance(param, bool | str):
                        raise ScheduleParseError(
                            f'`{{"parallelize" = {param}}}`: parallelization parameter should be True, False or a string.'
                        )
                    parallelize = param
                case "buffer":
                    if not isinstance(param, str):
                        raise ScheduleParseError(
                            f'`{{"buffer" = {param}}}`: buffer parameter should be a string (mtype).'
                        )
                    buffer = None if param == "default" else param
                    buffer_specified = True
                case "pack":
                    pack = self._parse_pack_param(param, context)
                    pack_specified = True
                case "partial":
                    partial = True
                case "full":
                    full = True
                case _:
                    raise ScheduleParseError(f"Unknown annotation on {context}: {key}")

        if partial and full:
            raise ScheduleParseError(f"{context} has both annotations full and partial")

        return Annotations(
            unroll_factor=unroll_factor,
            unroll_specified=unroll_specified,
            vectorize=vectorize,
            parallelize=parallelize,
            buffer=buffer,
            buffer_specified=buffer_specified,
            pack=pack,
            pack_specified=pack_specified,
            partial=partial,
            full=full,
        )

    def _parse_pack_param(
        self, param: Any, context: str
    ) -> tuple[literal, str | None, bool | str]:
        """Parse pack parameter into (input_idx, mtype, pad) tuple."""
        if not isinstance(param, (list, tuple)) or len(param) != 3:
            raise ScheduleParseError(
                f'`{{"pack" = {param}}}` on {context}: pack parameter should be a tuple (input_idx, mtype, pad).'
            )

        input_idx, mtype, pad = param

        if not isinstance(input_idx, literal):
            raise ScheduleParseError(
                f'`{{"pack" = {param}}}` on {context}: input_idx should be an integer.'
            )

        if not isinstance(mtype, str):
            raise ScheduleParseError(
                f'`{{"pack" = {param}}}` on {context}: mtype should be a string or None.'
            )

        if not isinstance(pad, bool | str):
            raise ScheduleParseError(
                f'`{{"pack" = {param}}}` on {context}: pad should be a boolean.'
            )

        # Convert "default" to None for mtype
        if mtype == "default":
            mtype = None

        return (input_idx, mtype, pad)

    def _parse_split_syntax(
        self, declaration: str
    ) -> tuple[str, literal | None, literal | None, literal | None]:
        """Parse the syntax of a split declaration."""
        match = self._SPLIT_PATTERN.match(declaration)
        if not match:
            match = self._SPLIT_MIDDLE_PATTERN.match(declaration)
            if not match:
                raise ScheduleParseError(f"Wrong format {declaration}")
            prefix, z = match.groups()
            z = int(z) if isint(z) else z
            return prefix, None, None, z

        prefix, x_str, y_str = match.groups()
        x = int(x_str) if isint(x_str) else x_str
        y = int(y_str) if isint(y_str) else y_str
        x = x if x else None
        y = y if y else None
        return prefix, x, y, None


class YAMLParser:
    """Parses a YAML specification into a dict-based schedule specification."""

    def parse(self, spec: str):
        descript_spec = strictyaml.load(spec).data
        if not isinstance(descript_spec, dict):
            raise ScheduleParseError(
                f"Wrong format: YAML input parses to {type(descript_spec)}."
            )
        return self._parse(descript_spec)

    def _parse(self, spec: dict[str, Any]) -> dict[str, dict]:
        descript_spec = dict()
        for a, v in spec.items():
            if isinstance(v, str):
                d = self._split(v)
            elif isinstance(v, dict):
                d = v
            else:
                raise ScheduleParseError(
                    f"Value {v} of key {a} is neither a string nor a dict."
                )
            size = d.get("size", None)
            if size:
                d.pop("size")
                a = f"{a}#{size}"
            if ":" in a:
                descript_spec[a] = self._parse(d)
            else:
                descript_spec[a] = d
        return descript_spec

    def _split(self, s: str) -> dict[str, Any]:
        d = dict()
        for s in s.split():
            if "=" not in s:
                d[s] = None
            else:
                x, y = s.split("=")
                try:
                    tmp = eval(y)
                except (NameError, SyntaxError):
                    tmp = y
                d[x] = tmp
        return d
