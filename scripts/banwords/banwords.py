#!/usr/bin/env python3

import re
import sys
from pathlib import Path
import base64
import logging
import argparse
import subprocess
import shlex
import os
import glob
import itertools

logger = logging.getLogger(__file__)

BAN_FILE = str(Path(__file__).parent / "banwords.b64")


def cmd_output(cmd: str, cwd: str) -> str:
    cmd_lst = shlex.split(cmd)
    p = subprocess.run(cmd_lst, capture_output=True, check=True, text=True, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(
            f"executing command failed: {cmd}:\n"
            " stdout: {p.stdout}\n"
            " stderr: {p.stderr}"
        )
    return p.stdout


def get_git_paths(top_dir: str, dirs: list[str]) -> list[str]:
    out = []
    for path in dirs:
        cmd = f"git ls-files {path}"
        out += cmd_output(cmd, cwd=top_dir).splitlines()
    paths = [p for p in list(dict.fromkeys(out)) if Path(p).is_file()]
    return paths


def get_all_paths(top_dir: str, dirs: list[str]) -> list[str]:
    out = []
    for path in dirs:
        dir = Path(top_dir) / path
        out += [str(p) for p in dir.rglob("*") if p.is_file()]
    paths = list(dict.fromkeys(out))
    return paths


def filter_paths(
    top_dir: str,
    files: list[str],
    includes: list[str],
    excludes: list[str] = [],
) -> list[str]:
    include_set = set(
        itertools.chain(
            *(glob.glob(pat, recursive=True, root_dir=top_dir) for pat in includes)
        )
    )
    exclude_set = set(
        itertools.chain(
            *(glob.glob(pat, recursive=True, root_dir=top_dir) for pat in excludes)
        )
    )
    filter_set = include_set - exclude_set
    top_path = Path(top_dir).resolve()
    files = [str(Path(p).resolve().relative_to(top_path)) for p in files]
    filtered = [p for p in files if p in filter_set]
    return filtered


def decode_banfile(fname: str) -> str:
    return base64.b64decode(Path(fname).read_text()).decode()


def load_banlist(fname: str) -> re.Pattern:
    content = decode_banfile(fname)
    words = [
        re.escape(l)
        for l in [l.strip() for l in content.splitlines()]
        if l and not l.startswith("#")
    ]
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.IGNORECASE)


def check_file(fname: str, pattern: re.Pattern, top: str = ".", show: bool = False):
    count = 0
    path = Path(top) / fname
    try:
        text = path.read_text()
    except UnicodeDecodeError as e:
        text = ""
    for idx, line in enumerate(text.splitlines()):
        matches = list(dict.fromkeys(pattern.findall(line)))
        if matches != []:
            logger.error(
                "found %d banned word\n%s:%d: %s",
                len(matches),
                path,
                idx + 1,
                ", ".join(matches) if show else f"<hidden>",
            )
        count += len(matches)
    return count


def main():
    TOP_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[2], Path.cwd()))
    DIRS = [Path(".")]

    LICENSE = TOP_DIR / "LICENSE"
    INCLUDES = ["**"]
    EXCLUDES = []
    BAN_FILE = Path(__file__).parent / "banwords.b64"

    parser = argparse.ArgumentParser(
        description="Check/apply LICENSE file to sources",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--apply",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply removal of banned words",
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="check banned words",
    )
    parser.add_argument(
        "--list",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="only list matched files",
    )
    parser.add_argument(
        "--ban",
        type=str,
        default=BAN_FILE,
        help="banned words base64 file",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show banned words in output",
    )
    parser.add_argument(
        "--top",
        type=str,
        default=str(TOP_DIR),
        help="top level directory",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        type=str,
        default=[str(d) for d in DIRS],
        help="dirs to apply",
    )
    parser.add_argument(
        "--git-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="search git files only, otherwise all files",
    )
    parser.add_argument(
        "--includes",
        nargs="+",
        type=str,
        default=INCLUDES,
        help="includes globs patterns",
    )
    parser.add_argument(
        "--excludes",
        nargs="+",
        type=str,
        default=EXCLUDES,
        help="excludes globs patterns",
    )
    args = parser.parse_args()

    logging.basicConfig()
    logger.setLevel(logging.INFO)

    if args.git_files:
        paths = get_git_paths(args.top, args.dirs)
    else:
        paths = get_all_paths(args.top, args.dirs)
    paths = filter_paths(args.top, paths, args.includes, args.excludes)
    if args.list:
        for path in paths:
            print(path)
        raise SystemExit()
    if len(paths) == 0:
        logger.warning("No file found")
        raise SystemExit()

    pattern = load_banlist(args.ban)
    total_count = 0
    for file in paths:
        count = check_file(file, pattern, top=args.top, show=args.show)
        total_count += count

    if total_count:
        suffix = "" if args.show else ", run with --show to see actual banned words"
        logger.error(
            "found %d banned word in %d files%s", total_count, len(paths), suffix
        )
        raise SystemExit(1)
    logger.info("Checked %d files", len(paths))


if __name__ == "__main__":
    main()
