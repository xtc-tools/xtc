#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import re


class Replace:
    """
    Replace a serie of {{key}} value in a text.
    """

    def __init__(self, keys):
        self.pattern = re.compile("|".join([re.escape("{{" + k + "}}") for k in keys]))

    def replace(self, text, **replaces):
        rep = dict((re.escape("{{" + k + "}}"), v) for k, v in replaces.items())
        return self.pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
