#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import os
from typing import Any
from typing_extensions import override

from xtc.backends.mlir.MlirConfig import MlirConfig

VALID_PLATFORMS = ["hw", "iss", "qemu"]
VALID_ARCHS = ["kv3-1", "kv3-2"]
VALID_FIRMWARES = ["ocl_fw_l1.elf"]  # TODO add other firmwares

DEFAULT_WORK_DIR = "/tmp/" + os.getlogin() + "/mlir_mppa"
DEFAULT_PLATFORM = "hw"
DEFAULT_ARCH = "kv3-2"
DEFAULT_FIRMWARE = "ocl_fw_l1.elf"
DEFAULT_VERBOSE = True
DEFAULT_BUILD_VERBOSE = 0
DEFAULT_BENCHMARK = False
DEFAULT_MPPA_TRACE_ENABLE = False
DEFAULT_MPPA_TRACE_USE_SYSCALL = True


class MppaConfig:
    """A class to gather all configs"""

    def __init__(self, mlir_config: MlirConfig | None = None):
        if mlir_config is None:
            mlir_config = MlirConfig()
        # Default Configuration
        self.work_dir: str = DEFAULT_WORK_DIR
        self.platform: str = get_platform()
        self.arch: str = DEFAULT_ARCH
        self.firmware: str = DEFAULT_FIRMWARE
        self.verbose: bool = mlir_config.debug
        self.build_verbose: int = mlir_config.debug
        self.benchmark: bool = DEFAULT_BENCHMARK
        self.mppa_trace_enable: bool = DEFAULT_MPPA_TRACE_ENABLE
        self.mppa_trace_use_syscall: bool = DEFAULT_MPPA_TRACE_USE_SYSCALL
        self.mlir_config: MlirConfig = mlir_config
        # Read from env
        self.set_platform(get_platform())
        self.set_benchmark(is_benchmark())
        self.set_mppa_trace_enable(mppa_trace_enable())
        self.set_mppa_trace_use_syscall(mppa_trace_use_syscall())
        if os.getenv("CLEAN_WORK_DIR", "0") in ["1", "true", "True"]:
            self.clean_work_dir()

    def set_work_dir(self, work_dir: str) -> None:
        if self.work_dir != "" and self.work_dir != "/":
            self.work_dir = work_dir

    def set_platform(self, platform: str) -> None:
        if platform in VALID_PLATFORMS:
            self.platform = platform

    def set_arch(self, arch: str) -> None:
        if arch in VALID_ARCHS:
            self.arch = arch

    def set_firmware(self, firmware: str) -> None:
        if firmware in VALID_FIRMWARES:
            self.firmware = firmware

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose

    def set_build_verbose(self, build_verbose: int) -> None:
        self.build_verbose = build_verbose

    def set_benchmark(self, benchmark: bool) -> None:
        self.benchmark = benchmark

    def set_mppa_trace_enable(self, mppa_trace_enable: bool) -> None:
        self.mppa_trace_enable = mppa_trace_enable

    def set_mppa_trace_use_syscall(self, mppa_trace_use_syscall: bool) -> None:
        self.mppa_trace_use_syscall = mppa_trace_use_syscall

    def clean_work_dir(self):
        if os.path.exists(self.work_dir):
            os.system("rm -r " + self.work_dir)

    @override
    def __str__(self) -> str:
        s = "Mppa configuration:\n"
        s += " - work_dir: " + self.work_dir + "\n"
        s += " - platform: " + self.platform + "\n"
        s += " - arch: " + self.arch + "\n"
        s += " - firmware: " + self.firmware + "\n"
        s += " - verbose: " + str(self.verbose) + "\n"
        s += " - build_verbose: " + str(self.build_verbose) + "\n"
        s += " - benchmark: " + str(self.benchmark)
        s += " - mppa_trace_enable: " + str(self.mppa_trace_enable) + "\n"
        s += " - mppa_trace_use_syscall: " + str(self.mppa_trace_use_syscall) + "\n"
        return s

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MppaConfig):
            return False
        return (
            self.work_dir == other.work_dir
            and self.platform == other.platform
            and self.arch == other.arch
            and self.firmware == other.firmware
            and self.verbose == other.verbose
            and self.build_verbose == other.build_verbose
            and self.benchmark == other.benchmark
            and self.mppa_trace_enable == other.mppa_trace_enable
            and self.mppa_trace_use_syscall == other.mppa_trace_use_syscall
        )


# Creation of a MppaConfig from env


def get_platform() -> str:
    platform = os.getenv("PLATFORM")
    if platform is not None:
        if platform in VALID_PLATFORMS:
            return platform
        else:
            print(f"\033[91mUnknown platform: {platform}\033[0m")
            exit(1)
    return DEFAULT_PLATFORM


def is_benchmark():
    if "BENCHMARK" in os.environ:
        if os.getenv("BENCHMARK") in ["1", "true", "True"]:
            return True
        else:
            return False
    return DEFAULT_BENCHMARK


def mppa_trace_enable():
    if "MPPA_TRACE_ENABLE" in os.environ:
        if os.getenv("MPPA_TRACE_ENABLE") in ["1", "true", "True"]:
            return True
        else:
            return False
    if is_benchmark():
        return True
    return DEFAULT_MPPA_TRACE_ENABLE


def mppa_trace_use_syscall():
    if "MPPA_TRACE_USE_SYSCALL" in os.environ:
        if os.getenv("MPPA_TRACE_USE_SYSCALL") in ["1", "true", "True"]:
            return True
        else:
            return False
    return DEFAULT_MPPA_TRACE_USE_SYSCALL


def mppa_trace_use_oculink():
    return not mppa_trace_use_syscall()
