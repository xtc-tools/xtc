# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod

from xtc.itf.runtime.common import CommonRuntimeInterface


class EmbeddedDevice(CommonRuntimeInterface, ABC):
    """Abstract interface for an embedded device."""

    @abstractmethod
    def flash(self, image_path: str) -> None:
        """Flash a binary image to the device.

        Args:
            image_path (str): Path to the binary image to flash.
        """
        ...

    # TODO
