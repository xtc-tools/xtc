#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Human-readable rendering of loop-nest nodes."""

from collections.abc import Iterator

from .loop_nest import LoopNestNode


def pretty_print_loop_node(
    node: LoopNestNode, indent: int = 0, *, detailed: bool = False
) -> str:
    """Render a loop-nest node and its children, starting at `indent` spaces.

    The compact notation uses `loop X` for a dimension, `tile(X, N)` for a
    tile, `split(X, start, end)` for a partition branch, and `...` for the
    innermost body. Entries follow interchange order; branch bounds follow
    cut declaration order. This renderer is used in validation errors: it
    must not require a checked nest. Mark missing children and revisited
    nodes so malformed trees can still be rendered for diagnostics. In
    detailed mode, the root is labeled on its first line and each child on
    its branch header.
    """
    lines: list[str] = []
    seen = {id(node)}
    pending = [(node.root, _node_entries(node, indent))]
    while pending:
        path, entries = pending[-1]
        entry = next(entries, None)
        if entry is None:
            pending.pop()
            continue
        suffix = f"  <- {node.root}" if detailed and not lines else ""
        if isinstance(entry, str):
            lines.append(entry + suffix)
            continue

        line, child, child_indent = entry
        child_path = f"{path}/{child.root}"
        if child.root.startswith(f"{path}/"):
            child_path = child.root
        if detailed and id(child) not in seen:
            suffix += f"  <- {child_path}"
        lines.append(line + suffix)
        if id(child) in seen:
            lines.append(" " * child_indent + f"<cycle or reused node {child.root!r}>")
        else:
            seen.add(id(child))
            pending.append((child_path, _node_entries(child, child_indent)))
    return "\n".join(lines)


def _node_entries(
    node: LoopNestNode, indent: int
) -> Iterator[str | tuple[str, LoopNestNode, int]]:
    tiles_info = {
        name: (axis, size)
        for axis, tiles in node.tiles.items()
        for name, size in tiles.items()
    }
    splits_info: dict[str, tuple[str, int, int | None, LoopNestNode | None]] = {}
    for axis, branches in node.splits.items():
        cuts = list(branches.items())
        for index, (name, start) in enumerate(cuts):
            end = cuts[index + 1][1] if index + 1 < len(cuts) else None
            child = node.children[index] if index < len(node.children) else None
            splits_info[name] = (axis, start, end, child)

    printed_split = False
    current_indent = indent
    for loop_name in node.interchange:
        if loop_name in splits_info:
            printed_split = True
            split_axis, start, end, child = splits_info[loop_name]
            end_str = str(end) if end is not None else "..."
            line = f"split({split_axis}, {start}, {end_str})"
            line = " " * current_indent + _add_annotations(node, line, loop_name)
            if child is None:
                yield line
                yield " " * (current_indent + 2) + "<missing child>"
            else:
                yield line, child, current_indent + 2
        else:
            if loop_name in tiles_info:
                axis, size = tiles_info[loop_name]
                line = f"tile({axis}, {size})"
            else:
                basename = loop_name.split("/")[-1]
                line = f"loop {basename}"

            line = _add_annotations(node, line, loop_name)
            yield " " * current_indent + line
            current_indent += 2

    if not printed_split:
        yield " " * current_indent + "..."


def _add_annotations(node: LoopNestNode, line: str, loop_name: str) -> str:
    """Add loop annotations to a loop line."""
    annotations: list[str] = []
    if loop_name in node.parallelize:
        annotations.append("parallelized")
    if loop_name in node.vectorize:
        annotations.append("vectorized")
    if loop_name in node.unroll:
        annotations.append(f"unroll({node.unroll[loop_name]})")
    if loop_name in node.buffer_at:
        mtype = node.buffer_at[loop_name]
        if mtype is not None:
            annotations.append(f"buffer({mtype})")
        else:
            annotations.append("buffer")
    if loop_name in node.pack_at:
        input_idx, mtype, pad = node.pack_at[loop_name]
        parts = [str(input_idx)]
        if mtype is not None:
            parts.append(mtype)
        if pad:
            parts.append("pad")
        annotations.append(f"pack({', '.join(parts)})")
    if loop_name in node.fuse_producer_at:
        prod_idx = node.fuse_producer_at[loop_name]
        annotations.append(f"fuse_producer({prod_idx})")
    if loop_name in node.fuse_consumer_at:
        annotations.append("fuse_consumer")
    if loop_name in node.external_at:
        annotations.append(f"external({node.external_at[loop_name]})")
    if annotations:
        line += "  // " + ", ".join(annotations)
    return line
