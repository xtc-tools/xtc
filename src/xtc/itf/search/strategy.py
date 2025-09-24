#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing import TypeAlias, Any
from collections.abc import Iterator

from ..graph import Graph
from ..schd import Scheduler


__all__ = [
    "Sample",
    "Strategy",
]


Sample: TypeAlias = Any


class Strategy(ABC):
    """Base abstract class for implementing a strategy.

    A strategy provides a predefined template for scheduling some operations
    embedded in a Graph object.

    From a strategy, one can:
    - generate an exhaustive list of samples in the search space
    - randomly sample the search space
    - get a default sample depending on some optimization level
    - actually schedule a Scheduler object for the given graph.
    """

    @property
    @abstractmethod
    def graph(self) -> Graph:
        """The graph associated with this strategy.

        This graph nust be the same object as the scheduler's graph
        when calling the generate method.

        Returns:
            The graph object
        """
        ...

    @abstractmethod
    def generate(self, scheduler: Scheduler, sample: Sample) -> None:
        """Generate and execute scheduling operations for the sample.

        This method applies the passed sample from the strategy sample
        space to the given scheduler.

        Note that the scheduler state is changed.
        In order to get the final schedule, after calling this method,
        call sch.schedule().

        Args:
            scheduler: The Scheduler object
            sample: The sample to apply
        """
        ...

    @abstractmethod
    def exhaustive(self) -> Iterator[Sample]:
        """Generates the exhaustive space of samples for this strategy.

        The actual space size may be huge, hence it is not recommended
        to convert this output to a list for instance without knowing
        the space size upperbound.

        Note that the returned samples are not randomized, hence the
        order is deterministic, though probably not suitable for
        random exploration unless all samples are retrieved.

        Returns:
            An iterator to the generated samples
        """
        ...

    @abstractmethod
    def sample(self, num: int, seed: int | None = 0) -> Iterator[Sample]:
        """Generates unique random samples from this strategy.

        The implementation should ensure that the search space
        is sampled uniformly, i.e. each distinct point in the
        search space should be equally probable.

        The number of requested samples must be greater than 0.

        If the seed provided is None, the generated sample list
        is not deterministic.

        Note that the returned number of samples may be less than
        the requested number of sample either because:
        - the search space is smaller than the requested number
        - the stop condition for sampling distinct samples is reached

        Args:
            num: number of samples requested
            seed: optional fixed seed, defaults to 0

        Returns:
            An iterator to the generated samples
        """
        ...

    @abstractmethod
    def default_schedule(self, opt_level: int = 2) -> Sample:
        """Generates a default sample for some optimization level.

        The returned sample should be a reasonable schedule given the
        strategy and passed opt_level. There is no rule there, though,
        typically vectorization and tilings are done at opt_level >= 3.

        Args:
            opt_level: The optimization level in [0, 3]

        Returns:
            The selected sample
        """
        ...

    @property
    @abstractmethod
    def sample_names(self) -> list[str]:
        """The names of the sample variables associated with the strategy.

        The order of the names must correspond to the order of the values
        in a sample.

        Returns:
            The list of the names of the sample variables.
        """
        ...

    @abstractmethod
    def dict_to_sample(self, sample: dict[str, Any]) -> Sample:
        """Generates a VecSample from a given Sample.

        The variables in the VecSample are in the order given by self.sample_names.

        Args:
            sample: The Sample to convert

        Returns:
            The equivalent VecSample
        """
        ...

    @abstractmethod
    def sample_to_dict(self, sample: Sample) -> dict[str, int]:
        """Generates a Sample from a given VecSample.

        The variables in the VecSample must be in the order given by self.sample_names.

        Args:
            sample: The VecSample to convert

        Returns:
            The equivalent Sample
        """
        ...
