#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from typing import Any
from dataclasses import dataclass
import re
import yaml
from typing_extensions import override

from .exceptions import ScheduleParseError

literal = int | str
ansor_tile = "PRTUOWF"
tup_list = list[tuple[str, Any]]


def toliteral(s: str) -> literal:
    """Returns int(s) if possible, otherwise returns s."""
    if s.lstrip("-").isdigit():
        return int(s)
    return s


def pre_parse(s: dict[str, Any] | tup_list | list[dict[str, Any]]) -> tup_list:
    if isinstance(s, dict):
        return [(k, _pre_parse(v)) for k, v in s.items()]
    out: tup_list = []
    for s_ in s:
        if isinstance(s_, dict):
            out += [(k, _pre_parse(v)) for k, v in s_.items()]
        else:
            k, v = s_
            out.append((k, _pre_parse(v)))
    return out


def _pre_parse(v: Any) -> Any:
    if isinstance(v, dict):
        return pre_parse(v)
    elif isinstance(v, list) and isinstance(v[0], dict | list):
        return pre_parse(v)
    else:
        return v


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
    buffer_specified: bool | str = False
    pack: tuple[literal, str | None, bool | str] | None = None
    pack_specified: bool | str = False
    interchange: str = ""
    partial: bool = False
    full: bool = False


@dataclass(frozen=True)
class SplitDecl:
    """AST Type: a split declaration like 'axis[start:end]' or 'axis[:size:]'."""

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


@dataclass(frozen=True)
class PRTDecl:
    """AST Type: Ansor-style tile declarations"""

    shape: str
    annotations: Annotations

    def __post__init__(self):
        for s in self.shape:
            if s not in ansor_tile:
                # Should be unreachable
                raise ScheduleParseError(f"Invalid tile declaration {s}")


ScheduleItem = SplitDecl | TileDecl | AxisDecl | PRTDecl


@dataclass(frozen=True)
class ScheduleSpec:
    """AST Type: the complete parsed schedule specification."""

    items: tuple[ScheduleItem, ...]


@dataclass()
class ScheduleParser:
    """Parses a dict-based schedule specification into an AST."""

    _SPLIT_PATTERN = re.compile(r"^(.*)\[(-\w+|\w*)?:(-\w+|\w*)?\]$")
    _SPLIT_MIDDLE_PATTERN = re.compile(r"^(.*)\[:(\w*):\]$")

    def parse(self, spec: list[tuple[str, list]] | dict[str, Any]) -> ScheduleSpec:
        """Parse a schedule specification dict into an AST."""
        if isinstance(spec, dict):
            spec = pre_parse(spec)

        items: list[ScheduleItem] = []

        for declaration, value in spec:
            item = self._parse_declaration(declaration, value)
            items.append(item)

        return ScheduleSpec(items=tuple(items))

    def _parse_declaration(self, declaration: str, value: list) -> ScheduleItem:
        """Parse a single declaration into a ScheduleItem."""
        # Try split declaration first (e.g., "axis[0:10]")
        if ":" in declaration:
            return self._parse_split(declaration, value)

        # Try tile declaration (e.g., "axis#32")
        if "#" in declaration:
            return self._parse_tile(declaration, value)

        if declaration[0] in ansor_tile:
            return self._parse_ansor_tile(declaration, value)

        # Must be a direct axis reference
        return self._parse_axis_ref(declaration, value)

    def _parse_split(self, declaration: str, value: list) -> SplitDecl:
        """Parse a split declaration like 'axis[start:end]' or 'axis[:size:]'."""
        axis_name, start, end, size = self._parse_split_syntax(declaration)

        body = self.parse(value)
        return SplitDecl(axis=axis_name, start=start, end=end, body=body, size=size)

    def _parse_tile(self, declaration: str, value: list) -> TileDecl:
        """Parse a tile declaration like 'axis#size'."""
        parts = declaration.split("#")
        if len(parts) != 2:
            raise ScheduleParseError(
                f"`{declaration}`: invalid tile syntax, expected 'axis#size'"
            )

        axis_name, size_str = parts

        size = toliteral(size_str)

        annotations = self._parse_annotations(value, declaration)
        return TileDecl(axis=axis_name, size=size, annotations=annotations)

    def _parse_ansor_tile(self, declaration: str, value: list) -> PRTDecl:
        """Parse an ansor-style tile reference."""

        annotations = self._parse_annotations(value, declaration)
        return PRTDecl(shape=declaration, annotations=annotations)

    def _parse_axis_ref(self, declaration: str, value: list) -> AxisDecl:
        """Parse a direct axis reference."""

        annotations = self._parse_annotations(value, declaration)
        return AxisDecl(axis=declaration, annotations=annotations)

    def _parse_annotations(
        self, value: list[tuple[str, Any]], declaration: str
    ) -> Annotations:
        """Parse annotation dict into Annotations object."""

        unroll_factor: literal | None = None
        unroll_specified = False
        vectorize: bool | str = False
        parallelize: bool | str = False
        buffer: str | None = None
        buffer_specified: bool | str = False
        pack: tuple[literal, str | None, bool | str] | None = None
        pack_specified: bool | str = False
        interchange: str = ""
        partial = False
        full = False

        for key, param in value:
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
                    if param is None:
                        param = True
                    if not isinstance(param, bool | str):
                        raise ScheduleParseError(
                            f'`{{"vectorize" = {param}}}`: vectorization parameter should be True, False or a string.'
                        )
                    vectorize = param
                case "parallelize":
                    if param is None:
                        param = True
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
                    if isinstance(param, str):
                        pack = (declaration, None, False)
                        pack_specified = param
                    else:
                        pack = self._parse_pack_param(param, declaration)
                        pack_specified = True
                case "pad":
                    if pack is None:
                        raise ScheduleParseError(
                            f"pad annotation before/without pack on {declaration}: {key}"
                        )
                    declaration_, mtype_, pad_ = pack
                    pack = (declaration_, mtype_, param)
                case "interchange":
                    if param is None:
                        interchange = "interchange"
                    else:
                        interchange = param
                case "partial":
                    partial = True
                case "full":
                    full = True
                case _:
                    raise ScheduleParseError(
                        f"Unknown annotation on {declaration}: {key}"
                    )

        if partial and full:
            raise ScheduleParseError(
                f"{declaration} has both annotations full and partial"
            )

        return Annotations(
            unroll_factor=unroll_factor,
            unroll_specified=unroll_specified,
            vectorize=vectorize,
            parallelize=parallelize,
            buffer=buffer,
            buffer_specified=buffer_specified,
            pack=pack,
            pack_specified=pack_specified,
            interchange=interchange,
            partial=partial,
            full=full,
        )

    def _parse_pack_param(
        self, param: Any, declaration: str
    ) -> tuple[literal, str | None, bool | str] | None:
        """Parse pack parameter into (input_idx, mtype, pad) tuple."""

        if param is None:
            return (declaration, None, False)

        if not isinstance(param, (list, tuple)) or len(param) != 3:
            raise ScheduleParseError(
                f'`{{"pack" = {param}}}` on {declaration}: pack parameter should be a tuple (input_idx, mtype, pad).'
            )

        input_idx, mtype, pad = param

        if not isinstance(input_idx, literal):
            raise ScheduleParseError(
                f'`{{"pack" = {param}}}` on {declaration}: input_idx should be an integer.'
            )

        if not isinstance(mtype, str | None):
            raise ScheduleParseError(
                f'`{{"pack" = {param}}}` on {declaration}: mtype should be a string or None.'
            )

        if not isinstance(pad, bool | str):
            raise ScheduleParseError(
                f'`{{"pack" = {param}}}` on {declaration}: pad should be a boolean.'
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
            z = toliteral(z)
            return prefix, None, None, z

        prefix, x_str, y_str = match.groups()
        x: literal | None = toliteral(x_str)
        y: literal | None = toliteral(y_str)
        if not x:
            x = None
        if not y:
            y = None
        return prefix, x, y, None


class YAMLParser:
    """Parses a YAML specification into a dict-based schedule specification."""

    def parse(self, spec: str) -> tup_list:
        """Parses a YAML specification into a dict-based schedule specification."""
        descript_spec = yaml.safe_load(spec)
        if not isinstance(descript_spec, dict | list):
            raise ScheduleParseError(
                f"Wrong format: YAML input parses to {type(descript_spec)}."
            )
        descript_spec = pre_parse(descript_spec)
        return self._parse(descript_spec)

    def _parse(self, spec: tup_list) -> tup_list:
        """Parses a dict YAML specification into a schedule specification."""
        # constraints = spec.pop("constraints", [])
        descript_spec: tup_list = []
        for a, v in spec:
            if a == "constraints":
                descript_spec.append((a, v))
                continue
            if isinstance(v, str):
                d = self._split(v)
            elif isinstance(v, list):
                d = v
            elif v is None:
                d = []
            else:
                raise ScheduleParseError(
                    f"Value {v} of key {a} is neither a string nor a dict."
                )
            size = [v for k, v in d if k == "size"]
            if size:
                d = [(k, v) for k, v in d if k != "size"]
                a = f"{a}#{size[0]}"
            if ":" in a:
                descript_spec.append((a, self._parse(d)))
            else:
                descript_spec.append((a, d))
        return descript_spec

    def _split(self, s: str) -> tup_list:
        """Splits a string of 'keyword's and 'keyword=value's separated by spaces into a dict."""
        d: tup_list = []
        for s in s.split():
            if "=" not in s:
                d.append((s, None))
            else:
                x, y = s.split("=")
                try:
                    tmp = eval(y)
                except (NameError, SyntaxError):
                    tmp = y
                d.append((x, tmp))
        return d
