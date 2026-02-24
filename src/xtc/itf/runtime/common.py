# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing import Any

from xtc.runtimes.types.dlpack import DLDataType, DLDevice
from xtc.utils.cfunc import CFunc


class CommonRuntimeInterface(ABC):
    """Abstract interface for a common runtime interface."""

    @abstractmethod
    def target_name(self) -> str:
        """Get the name of the target.

        Returns:
            A string representing the name of the target.
        """
        ...

    @abstractmethod
    def device_name(self) -> str:
        """Get the name of the device.

        Returns:
            A string representing the name of the device.
        """
        ...

    @abstractmethod
    def device_arch(self) -> str:
        """Get the architecture of the device.

        Returns:
            A string representing the architecture of the device.
        """
        ...

    @abstractmethod
    def device_id(self) -> int:
        """Get the ID of the device.

        Returns:
            An integer representing the ID of the device.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        results: Any,
        repeat: int,
        number: int,
        nargs: int,
        cfunc: CFunc,
        args: Any,
    ) -> None:
        """Evaluate a function with timing measurements.

        Args:
            results: Pointer to array of doubles to store timing results.
            repeat: Number of times to repeat the measurement.
            number: Number of function calls per repeat.
            nargs: Number of arguments passed to the function.
            cfunc: Function pointer to evaluate.
            args: Pointer to array of void pointers containing function arguments.
        """
        ...

    @abstractmethod
    def evaluate_perf(
        self,
        pmu_events: list[str],
        repeat: int,
        number: int,
        min_repeat_ms: int,
        cfunc: CFunc,
        args_tuples: list[Any],
    ) -> list[float]:
        """Evaluate a function with performance counter measurements.

        Args:
            pmu_events: List of performance events to measure.
            repeat: Number of times to repeat the measurement.
            number: Number of function calls per repeat.
            min_repeat_ms: Minimum time in milliseconds for each repeat.
            cfunc: Function pointer to evaluate.
            args_tuples: List of argument tuples.
        """
        ...

    @abstractmethod
    def evaluate_packed(
        self,
        results: Any,
        repeat: int,
        number: int,
        min_repeat_ms: int,
        cfunc: CFunc,
        args: Any,
        codes: Any,
        nargs: int,
    ) -> None:
        """Evaluate a packed function with timing measurements.

        Args:
            results: Pointer to array of doubles to store timing results.
            repeat: Number of times to repeat the measurement.
            number: Number of function calls per repeat.
            min_repeat_ms: Minimum time in milliseconds for each repeat.
            cfunc: Packed function pointer to evaluate.
            args: Pointer to array of packed arguments.
            codes: Pointer to array of integers containing argument type codes.
            nargs: Number of arguments.
        """
        ...

    @abstractmethod
    def evaluate_packed_perf(
        self,
        results: Any,
        pmu_events: list[str],
        repeat: int,
        number: int,
        min_repeat_ms: int,
        cfunc: CFunc,
        args: Any,
        codes: Any,
        nargs: int,
    ) -> None:
        """Evaluate a packed function with performance counter measurements.

        Args:
            results: Pointer to array of doubles to store performance results.
            pmu_events: List of performance events to measure.
            repeat: Number of times to repeat the measurement.
            number: Number of function calls per repeat.
            min_repeat_ms: Minimum time in milliseconds for each repeat.
            cfunc: Packed function pointer to evaluate.
            args: Pointer to array of packed arguments.
            codes: Pointer to array of integers containing argument type codes.
            nargs: Number of arguments.
        """
        ...

    @abstractmethod
    def cndarray_new(
        self,
        ndim: int,
        shape: Any,
        dtype: DLDataType,
        device: DLDevice,
    ) -> Any:
        """Create a new CNDArray.

        Args:
            ndim: Number of dimensions.
            shape: Pointer to array of int64 containing shape dimensions.
            dtype: Data type descriptor.
            device: Device descriptor.

        Returns:
            Pointer to the created CNDArray, or None on failure.
        """
        ...

    @abstractmethod
    def cndarray_del(self, handle: Any) -> None:
        """Delete a CNDArray.

        Args:
            handle: Pointer to the CNDArray to delete.
        """
        ...

    @abstractmethod
    def cndarray_copy_from_data(self, handle: Any, data_handle: Any) -> None:
        """Copy data from a data handle into a CNDArray.

        Args:
            handle: Pointer to the destination CNDArray.
            data_handle: Pointer to the source data.
        """
        ...

    @abstractmethod
    def cndarray_copy_to_data(self, handle: Any, data_handle: Any) -> None:
        """Copy data from a CNDArray to a data handle.

        Args:
            handle: Pointer to the source CNDArray.
            data_handle: Pointer to the destination data.
        """
        ...

    @abstractmethod
    def evaluate_flops(self, dtype_name: str | bytes) -> float:
        """Evaluate the peak floating-point operations per second for a given data type.

        Args:
            dtype_name: Data type name as string or bytes (e.g., "float32").

        Returns:
            Peak FLOPS as a double, or 0.0 if the data type is not supported.
        """
        ...
