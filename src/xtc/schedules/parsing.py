#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from typing import Any
from dataclasses import dataclass
import re
from typing_extensions import override

from .exceptions import ScheduleParseError


@dataclass(frozen=True)
class Annotations:
    """AST Type : annotations that can be applied to a loop.

    Attributes:
        unroll_factor: The unroll factor. None means "unroll fully" (use loop size).
            Only meaningful when unroll_specified is True.
        unroll_specified: True if unroll was explicitly requested.
        vectorize: True if vectorization was requested.
        parallelize: True if parallelization was requested.
    """

    unroll_factor: int | None = None
    unroll_specified: bool = False
    vectorize: bool = False
    parallelize: bool = False


@dataclass(frozen=True)
class SplitDecl:
    """AST Type: a split declaration like 'axis[start:end]'."""

    axis: str
    start: int | None
    end: int | None
    body: ScheduleSpec

    @override
    def __str__(self) -> str:
        start_str = "" if self.start is None else str(self.start)
        end_str = "" if self.end is None else str(self.end)
        decl = f"{self.axis}[{start_str}:{end_str}]"
        return decl


@dataclass(frozen=True)
class TileDecl:
    """AST Type: a tile declaration like 'axis#size'."""

    axis: str
    size: int
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


class ScheduleParser:
    """Parses a dict-based schedule specification into an AST."""

    _SPLIT_PATTERN = re.compile(r"^(.*)\[(-\d+|\d*)?:(-\d+|\d*)?\]$")

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
        axis_name, start, end = self._parse_split_syntax(declaration)

        body = self.parse(value)
        return SplitDecl(axis=axis_name, start=start, end=end, body=body)

    def _parse_tile(self, declaration: str, value: dict) -> TileDecl:
        """Parse a tile declaration like 'axis#size'."""
        parts = declaration.split("#")
        if len(parts) != 2:
            raise ScheduleParseError(
                f"`{declaration}`: invalid tile syntax, expected 'axis#size'"
            )

        axis_name, size_str = parts

        try:
            size = int(size_str)
        except ValueError:
            raise ScheduleParseError(f"`{declaration}`: {size_str} is not an integer.")

        annotations = self._parse_annotations(value, declaration)
        return TileDecl(axis=axis_name, size=size, annotations=annotations)

    def _parse_axis_ref(self, declaration: str, value: dict) -> AxisDecl:
        """Parse a direct axis reference."""

        annotations = self._parse_annotations(value, declaration)
        return AxisDecl(axis=declaration, annotations=annotations)

    def _parse_annotations(self, value: dict[str, Any], context: str) -> Annotations:
        """Parse annotation dict into Annotations object."""

        unroll_factor: int | None = None
        unroll_specified = False
        vectorize = False
        parallelize = False

        for key, param in value.items():
            if key == "unroll":
                if param is True:
                    unroll_factor = None
                    unroll_specified = True
                elif param is False:
                    pass
                elif isinstance(param, int):
                    unroll_factor = param
                    unroll_specified = True
                else:
                    raise ScheduleParseError(
                        f'`{{"unroll" = {param}}}`: unroll parameter should be True, False, or an integer.'
                    )
            elif key == "vectorize":
                if not isinstance(param, bool):
                    raise ScheduleParseError(
                        f'`{{"vectorize" = {param}}}`: parameterized vectorization not implemented.'
                    )
                vectorize = param
            elif key == "parallelize":
                if not isinstance(param, bool):
                    raise ScheduleParseError(
                        f'`{{"parallelize" = {param}}}`: parameterized parallelization not implemented.'
                    )
                parallelize = param
            else:
                raise ScheduleParseError(f"Unknown annotation on {context}: {key}")

        return Annotations(
            unroll_factor=unroll_factor,
            unroll_specified=unroll_specified,
            vectorize=vectorize,
            parallelize=parallelize,
        )

    def _parse_split_syntax(
        self, declaration: str
    ) -> tuple[str, int | None, int | None]:
        """Parse the syntax of a split declaration."""
        match = self._SPLIT_PATTERN.match(declaration)
        if not match:
            raise ScheduleParseError(f"Wrong format {declaration}")

        prefix, x_str, y_str = match.groups()
        x = int(x_str) if x_str else None
        y = int(y_str) if y_str else None
        return prefix, x, y
