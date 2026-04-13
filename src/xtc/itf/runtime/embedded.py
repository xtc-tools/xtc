# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod

from xtc.itf.runtime.common import CommonRuntimeInterface


class EmbeddedDevice(CommonRuntimeInterface, ABC):
    """Abstract interface for an embedded device.
    Unlike Host or Accelerator, tensors must not be transfered to the
    device, so the runtime has to handle tensors creation and validation.
    This class is suited for microcontrollers and remote execution.
    """

    @abstractmethod
    def flash(self, image_path: str) -> None:
        """Flash a binary image to the device.

        Args:
            image_path (str): Path to the binary image to flash.
        """
        ...

    # TODO Add embedded device specific methods
