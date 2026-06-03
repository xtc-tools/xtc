#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing import TypeAlias

VecSample: TypeAlias = list[int]


class Optimizer(ABC):
    """Base abstract class for implementing an optimizer

    An Optimizer is used in iterative evaluation during loop-explore to
    suggest samples for each batch using observations from previous batch.
    """

    @abstractmethod
    def suggest(self) -> list[VecSample]:
        """Suggests a new batch of samples to be evaluated.

        It gets a large sample of size batch_candidates, and then from that
        either returns random choices or uses a model to pick the predicted best samples.

        Returns:
            A list of samples representing a new batch of samples.
        """
        ...

    @abstractmethod
    def observe(self, x: list[VecSample], y: list[float]):
        """Observes the result of the batch evaluation and updates the model.

        The model is first fit after update_first samples are observed
        and is subsequently refit every update_period additional samples.

        Args:
            x: the batch of samples that were evaluated
            y: the evaluation result for each sample in the batch
        """
        ...

    @abstractmethod
    def finished(self):
        """Gets called when the evaluation for all iterations has been completed

        Used for cleaner logging
        """
        ...
