#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from pathlib import Path
import subprocess
import shlex


def target_arch(arch: str = "") -> str:
    """
    Normalize given arch to target arch as
    defined in target triple.
    """
    match arch:
        case "x86_64" | "x86-64":
            return "x86_64"
        case "aarch64" | "native" | "":
            return arch
        case _:
            raise ValueError(f"target triple: unknown arch: {arch}")


def target_triple(arch: str = "") -> str:
    """
    Returns the target triple prefix given the arch.
    When arch is unspecified or native, returns "".
    """
    arch = target_arch(arch)
    match arch:
        case "aarch64" | "x86_64":
            return f"{arch}-linux-gnu"
        case "native" | "":
            return ""
        case _:
            raise ValueError(f"target triple: unknown arch: {arch}")


def binutils_prefix(arch: str = "") -> str:
    """
    Returns the target triple prefix given the arch.
    When arch is unspecified, assume no prefix.
    """
    triple = target_triple(arch)
    if not triple:
        return ""
    return f"{triple}-"


def binutils_command(command: str, arch: str = "") -> str:
    """
    Returns the binutils command name from the given base name.
    Add the target triple prefix when arch specified.
    """
    prefix = binutils_prefix(arch)
    return f"{prefix}{command}"


def disassemble(
    obj_path: str | Path,
    function: str = "",
    section: str = "",
    arch: str = "",
    color: bool = False,
    visualize_jumps: bool = True,
) -> str:
    """
    Returns the disassembled multi-line string for the given obj_path.
    Optionally disassembling only the given function.
    """
    base_opts = [
        "-dr",
        "--no-addresses",
        "--no-show-raw-insn",
    ]
    jumps_opts = [
        "--visualize-jumps",
    ]
    color_opts = [
        "--disassembler-color=on",
    ]

    obj_path = Path(obj_path)
    args = [
        *base_opts,
        *([f"--disassemble={function}"] if function else ["--disassemble"]),
        *(jumps_opts if visualize_jumps else []),
        *(color_opts if color else []),
        str(obj_path),
    ]
    objdump = binutils_command("objdump", arch=arch)
    cmd = [objdump] + args
    p = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Executing command: {shlex.join(cmd)}:\n"
            f"  stdout:\n{p.stdout}\n"
            f"  stderr:\n{p.stderr}\n"
            f"  error code: {p.returncode}"
        )
    output = ""
    emit = False
    # Filter out file header and optionally section
    for line in p.stdout.splitlines():
        if "section" in line:
            emit = True
            if section and not f"section {section}" in line:
                emit = False
        if emit:
            output += f"{line}\n"
    return output
