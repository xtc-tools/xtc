#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
from pathlib import Path
import re
import jinja2
from typing import Sequence


class Replace:
    """
    Replace a serie of {{key}} value in a text.
    """

    def __init__(self, keys: Sequence[str]):
        self.pattern = re.compile("|".join([re.escape("{{" + k + "}}") for k in keys]))

    def replace(self, text: str, **replaces: str):
        rep = dict((re.escape("{{" + k + "}}"), v) for k, v in replaces.items())
        return self.pattern.sub(lambda m: rep[re.escape(m.group(0))], text)


_ALPHANUM_RE = None


def to_cname(name: str) -> str:
    """
    Convert a string to a C identifier compatible name.
    This may be of use to generate either function names,
    or tensor names from input graphs.
    Rules are:
    - empty -> "_"
    - replace any non-alpha and non-"_" to "_"
    - add leading "_" when starting with number
    - add trailing "_" if len without leading "_" is >= 2
    Note that this last rule is to avoid C keywords.
    """
    global _ALPHANUM_RE
    if _ALPHANUM_RE is None:
        _ALPHANUM_RE = re.compile(r"[^a-zA-Z0-9_]")
    if name == "":
        name = "_"
    name = _ALPHANUM_RE.sub("_", name)
    if name[0] in "0123456789":
        name = "_" + name
    if name[-1] != "_":
        if name[0] == "_" and len(name) >= 3 or name[0] != "_" and len(name) >= 2:
            name = name + "_"
    return name


def jinja_generate_file(fname: str, template_fname: str, **kwargs: Any) -> None:
    """
    Generates into fname from the jinja2 template_fname given kwargs.
    """
    file_path = Path(fname)
    template_path = Path(template_fname)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    template = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_path.parent)
    ).get_template(template_path.name)
    with open(file_path, mode="w", encoding="utf-8") as file:
        file.write(template.render(kwargs))
