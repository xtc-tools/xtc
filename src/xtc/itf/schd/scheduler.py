#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from .schedule import Schedule
import xtc.itf


class Scheduler(ABC):
    """An abstract implementation of the backend scheduler.

    A Scheduler is constructed from a given Backend and is responsible for
    applying primitive scheduling operations and transformations to the implementation.
    It generates a Schedule that captures these transformations, which can then be
    used by a Compiler to generate optimized executable code.

    Schedulers are backend-specific and work with their associated Backend
    to provide optimization capabilities appropriate for the target platform
    and runtime.
    """

    @abstractmethod
    def schedule(self) -> Schedule:
        """Creates a Schedule from the applied transformations.

        Returns a Schedule object that captures all the transformations and
        optimizations that have been applied to the implementation. This
        Schedule can then be used by a Compiler to generate executable code.

        Returns:
            Schedule: The resulting schedule containing all applied transformations
        """
        ...

    @property
    @abstractmethod
    def backend(self) -> "xtc.itf.back.Backend":
        """Returns the backend associated with this scheduler.

        Returns:
            Backend: The backend-specific implementation this scheduler
                     applies transformations to
        """
        ...

    @abstractmethod
    def split(self, dim: str, segments: dict[str, int]) -> None:
        """Split a dimension into `len(segments)` segments.

        Each segment is characterized by a starting/cutting point,
        which is also the endpoint of the previous segment, and by
        the name of the new axis created by the cut. The segments
        items must be provided in ascending order of the cut points
        on the axis.

        Args:
            dim: name of the dimension to split
            segments: ordered dict of new root name and segment
                      starting point
        """
        ...

    @abstractmethod
    def tile(self, dim: str, tiles: dict[str, int]) -> None:
        """Apply a multi level tiling operation on a dimension.

        The given tile sizes is interpreter outer to inner and
        each size must divide the dimension size. After this
        transformation, the number of axis for the given initial
        dimension is `1 + len(tiles)` where the first axis inherits
        the name of the dimension, and the remaining axis names are
        given by the given tiles keys.

        Args:
            dim: name of the dimension to tile
            tiles: dict outer to inner of axis name and tile size
        """
        ...

    @abstractmethod
    def interchange(self, permutation: list[str]) -> None:
        """Apply interchange over all axes.

        The given permutation of axes names is interpreted
        outer to inner and must have the same size as the
        number of axes after tiling.

        Args:
            permutation: outer to inner axes names permutation
        """
        ...

    @abstractmethod
    def vectorize(self, axes: list[str]) -> None:
        """Apply vectorizations on the given axes names.

        The axes names must given must all be inner axis, full
        unrolling and vectorization of all given axes is implied.

        Args:
            axes: axes names to vectorize
        """
        ...

    @abstractmethod
    def parallelize(self, axes: list[str]) -> None:
        """Apply parallelization on the given axes names.

        The axes names must given must all be outer axis.

        Args:
            axes: axes names to parallelize
        """
        ...

    @abstractmethod
    def unroll(self, unrolls: dict[str, int]) -> None:
        """Apply unrolling on the given axes names.

        Each given axes name is unrolled with the specified unroll
        factor.

        Args:
            unrolls: dict of axes names and unroll factor
        """

    @abstractmethod
    def buffer_at(self, axis: str, mtype: str | None = None) -> None:
        """Apply a write bufferoization at a given level,

        A write buffer is created for the output under the given
        axis, The buffer memory type can be specified or defaults
        to the local memory at this level.

        Args:
            axis: localisation of the write buffer
            mtype: buffer memory type for the allocation
        """
        ...
