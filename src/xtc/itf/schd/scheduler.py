#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from .schedule import Schedule
import xtc.itf

DEFAULT_ROOT = "."


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
    def set_dims(self, dims: list[str]) -> None:
        """Redefines dimensions names.

        Use provided abstract dimensions names for the scheduler
        transformantions instead of the default operation dimensions names.

        This should be set before applying the transformations

        Args:
            dims: list of dimensions names
        """
        ...

    @abstractmethod
    def split(
        self, dim: str, segments: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None:
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

    def strip_mine(
        self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None:
        """Apply a multi level strip mining transformation on the given dimension.

        The strip mining can be seen as a multi level 1D tiling where the
        given tile sizes are interpreter outer to inner.
        After this transformation, the number of axis for the given initial
        dimension is `1 + len(tiles)` where the first axis inherits
        the name of the dimension, and the remaining axis names are
        given by the given tiles keys.
        Each 1D tile size must be greater or equal to the inner tile sizes.
        Some backend may not support non-divisible tile sizes, in which
        case an assertion is raised.

        Args:
            dim: name of the dimension to strip mine
            tiles: dict outer to inner of axis name and tile size
            root: the parent split (or the operator's absolute root)
        """
        self.tile(dim=dim, tiles=tiles, root=root)

    @abstractmethod
    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        """Apply a multi level tiling operation.

        As of now the interface is limited to a single dimension tiling,
        hence it is equivalent to strip mining the given dimension.

        In order to create multi dimensional tiles, strip mine each dimension
        with tile or stip_mine and use interchange to reorder generated axes
        accordingly.

        Args:
            dim: name of the dimension to tile
            tiles: dict outer to inner of axis name and tile size
            root: the parent split (or the operator's absolute root)
        """
        ...

    @abstractmethod
    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT) -> None:
        """Apply interchange over all axes.

        The given permutation of axes names is interpreted
        outer to inner and must have the same size as the
        number of axes after tiling.

        Args:
            permutation: outer to inner axes names permutation
            root: the parent split (or the operator's absolute root)
        """
        ...

    @abstractmethod
    def vectorize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        """Apply vectorizations on the given axes names.

        The axes names given must all be inner axes and parallel axes, full
        unrolling and vectorization of all given axes is implied.

        Args:
            axes: axes names to vectorize
            root: the parent split (or the operator's absolute root)
        """
        ...

    @abstractmethod
    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        """Apply parallelization on the given axes names.

        The axes names must given must all be outer axes and parallel axes.

        Args:
            axes: axes names to parallelize
            root: the parent split (or the operator's absolute root)
        """
        ...

    @abstractmethod
    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        """Apply unrolling on the given axes names.

        Each given axes name is unrolled with the specified unroll
        factor. The unroll factors must be greater or equal to 1.

        Args:
            unrolls: dict of axes names and unroll factor
            root: the parent split (or the operator's absolute root)
        """

    @abstractmethod
    def buffer_at(
        self, axis: str, mtype: str | None = None, root: str = DEFAULT_ROOT
    ) -> None:
        """Create a write buffer at a given level.

        A write buffer is created for the output under the given
        axis, The buffer memory type can be specified or defaults
        to the local memory at this level.

        Args:
            axis: localisation of the write buffer
            mtype: buffer memory type for the allocation
            root: the parent split (or the operator's absolute root)
        """
        ...

    @abstractmethod
    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        root: str = DEFAULT_ROOT,
    ) -> None:
        """Create a packed read buffer at a given level.

        A packed read buffer is created for the given input buffer index.
        The buffer memory type can be specified or defaults
        to the local memory at this level.
        When pad is true, a padding strategy is applied in order to reduce
        sets/banks conflicts.

        Args:
            axis: localisation of the write buffer
            input_idx: input buffer index for the scheduled computation
            mtype: buffer memory type for the allocation
            pad: whether to add padding or not
            root: the parent split (or the operator's absolute root)
        """
        ...

    @abstractmethod
    def define_memory_mesh(self, axes: dict[str, int]) -> None:
        """Define a memory mesh.

        Args:
            axes: dictionary where keys are axes names and values are the number of memories along each axis
        """
        ...

    @abstractmethod
    def define_processor_mesh(self, axes: dict[str, int]) -> None:
        """Define a processor mesh. It must be a superset of the memory mesh.

        Args:
            axes: dictionary where keys are axes names and values are the number of processors along each axis
        """
        ...

    @abstractmethod
    def distribute(
        self, axis: str, processor_axis: str, root: str = DEFAULT_ROOT
    ) -> None:
        """Distribute computation across processors along a given axis.

        This method distributes the computation of the specified axis across
        multiple processors or cores. The processor_axis parameter defines
        the axis that represents the processor dimension for this distribution.

        Args:
            axis: the axis to distribute across processors
            processor_axis: the axis representing the processor dimension
            root: the parent split (or the operator's absolute root)
        """
        ...

    @abstractmethod
    def distributed_buffer_at(
        self,
        axis: str,
        input_idx: int,
        memory_axes: list[str],
        root: str = DEFAULT_ROOT,
    ) -> None:
        """Create a distributed buffer at a given level across multiple memory axes.

        This method creates a distributed buffer for the given input buffer index
        at the specified axis level. The buffer is distributed across the provided
        memory axes, enabling distributed memory management and access patterns
        for improved performance in distributed computing environments.

        Args:
            axis: the axis level where the distributed buffer should be created
            input_idx: input buffer index for the scheduled computation
            memory_axes: list of memory axes across which to distribute the buffer
            root: the parent split (or the operator's absolute root)
        """
        ...
