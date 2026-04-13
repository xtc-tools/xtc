#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from pathlib import Path
import subprocess
import shlex
import platform


def target_arch(arch: str = "") -> str:
    """
    Normalize given arch to target arch as
    defined in target triple.
    """
    match arch:
        case "x86_64" | "x86-64":
            return "x86_64"
        case "arm64" | "aarch64" | "native" | "":
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
        case "arm64" | "native" | "":
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
    if platform.system() == "Darwin" and triple == "aarch64-linux-gnu":
        # On darwin cross aarch64 binutils from aarch64-elf-binutils
        triple = "aarch64-elf"
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
    target = target_arch(arch)
    if target in ["x86_64", "aarch64"] or platform.system() == "Linux":
        jumps_opts = []
        disass_symbol_opt = [f"--disassemble={function}"]
    elif platform.system() == "Darwin":
        jumps_opts = []
        disass_symbol_opt = ["--disassemble-symbols=ltmp0"]
    color_opts = [
        "--disassembler-color=on",
    ]

    obj_path = Path(obj_path)
    args = [
        *base_opts,
        *(disass_symbol_opt if function else ["--disassemble"]),
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
