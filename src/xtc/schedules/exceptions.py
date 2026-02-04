#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Schedule-related exceptions."""


class ScheduleParseError(RuntimeError):
    """Raised when schedule parsing fails."""

    pass


class ScheduleInterpretError(RuntimeError):
    """Raised when schedule interpretation fails."""

    pass


class ScheduleValidationError(RuntimeError):
    """Raised when schedule validation fails."""

    pass
