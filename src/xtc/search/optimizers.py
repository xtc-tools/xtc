#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import random
import numpy as np
import logging
import yaml
from collections.abc import Sequence, Iterator
from sklearn.ensemble import RandomForestRegressor
from typing import Callable, TypeAlias, Any, ClassVar
from typing_extensions import override
from xtc.itf.search.optimizer import Optimizer

VecSample: TypeAlias = list[int]

logger = logging.getLogger(__name__)


class BaseOptimizer(Optimizer):
    # Maximum number of draws when looking for unseen candidates
    max_draws: ClassVar[int] = 5

    def __init__(
        self,
        sample_fn: Callable[[int, int], Iterator[VecSample]],
        batch: int,
        seed: int,
        **kwargs: Any,
    ):
        self.sample_fn = sample_fn
        self.batch = batch
        self.seed = seed
        self.rng = random.Random(seed)
        # Samples already evaluated, never suggested again
        self.seen: set[tuple[int, ...]] = set()
        # logging
        self.last_filtered = 0
        self.n_filtered = 0
        self.n_redraws = 0

    @override
    def suggest(self) -> list[VecSample]:
        raise NotImplementedError

    @override
    def observe(self, x: list[VecSample], y: list[float]):
        self.seen.update(tuple(s) for s in x)

    @override
    def restore(self, x: list[VecSample], y: list[float]):
        if not x:
            return
        # Replay the random draws of each batch of the previous run, so that
        # the resumed run continues the same sequence of candidates
        n_batches = -(-len(x) // self.batch)
        for i in range(n_batches):
            self._replay_batch(i * self.batch)
        self.observe(x, y)
        logger.info(
            f"restored {len(x)} samples, replayed {n_batches} batches, seed: {self.seed}"
        )

    @override
    def finished(self):
        logger.info(
            f"filtered {self.n_filtered} already evaluated candidates, "
            f"{self.n_redraws} redraws"
        )

    def _replay_batch(self, n_observed: int):
        """Replays the random draws of a suggest() made after n_observed samples."""
        self._next_seed()

    def _next_seed(self) -> int:
        return self.rng.randint(0, 2**32 - 1)

    def _sample(self, num: int) -> list[VecSample]:
        return list(self.sample_fn(num, self._next_seed()))

    def _sample_unseen(self, num: int, need: int) -> list[VecSample]:
        """Samples num candidates that were not already evaluated.

        Draws again with a doubled size while less than need unseen candidates
        are found, returns an empty list when no unseen candidate can be found.
        """
        candidates: list[VecSample] = []
        drawn: set[tuple[int, ...]] = set()
        filtered = 0
        for draw in range(self.max_draws):
            if draw > 0:
                self.n_redraws += 1
            for x in self._sample(num * 2**draw):
                key = tuple(x)
                if key in drawn:
                    continue
                drawn.add(key)
                if key in self.seen:
                    filtered += 1
                else:
                    candidates.append(x)
            if len(candidates) >= need:
                break
        self.last_filtered = filtered
        self.n_filtered += filtered
        return candidates[:num]


class RandomOptimizer(BaseOptimizer):
    def __init__(
        self,
        sample_fn: Callable[[int, int], Iterator[VecSample]],
        batch: int,
        seed: int,
    ):
        super().__init__(
            sample_fn=sample_fn,
            batch=batch,
            seed=seed,
        )

    @override
    def suggest(self):
        return self._sample_unseen(self.batch, self.batch)[: self.batch]


class RandomForestOptimizer(BaseOptimizer):
    def __init__(
        self,
        sample_fn: Callable[[int, int], Iterator[VecSample]],
        batch: int,
        seed: int,
        batch_candidates: int,
        beta: float,
        update_first: int,
        update_period: int,
        n_estimators: int,
        max_depth: int,
        min_samples_leaf: int,
        max_features: float,
        beta_decay: bool = True,
        alpha: float = 0.0,
    ):
        super().__init__(sample_fn, batch, seed)
        self.batch_candidates = batch_candidates
        self.update_first = update_first if update_first else batch
        self.update_period = update_period if update_period else batch
        self.beta = beta
        self.alpha = alpha
        self.beta_min = 0.05
        self.update_last = 0
        self.beta_decay = beta_decay

        self.X: list[VecSample] = []
        self.y: list[float] = []
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=1,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_leaf * 2,
            max_features=max_features,
            random_state=seed,
        )
        self.best_y: float = 0.0
        self.best_x: VecSample = []
        # logging
        self.last_beta: float | None = None
        self.log_str = ""

    @override
    def suggest(self):
        candidates = self._sample_unseen(self.batch_candidates, self.batch)
        if len(candidates) <= self.batch:
            return candidates
        if len(self.X) < self.update_first:
            return self.rng.choices(candidates, k=self.batch)

        preds = np.stack([t.predict(candidates) for t in self.rf.estimators_])
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)

        # UCB acquisition modified with inverse time beta decay
        beta = (
            max(self.beta_min, self.beta / (len(self.y) ** self.alpha))
            if self.beta_decay
            else self.beta
        )
        scores = mean + beta * std
        top_idx = np.argpartition(scores, -self.batch)[-self.batch :]
        top_candidates = [candidates[i] for i in top_idx]

        self.last_beta = beta
        return top_candidates

    @override
    def observe(self, x: list[VecSample], y: list[float]):
        super().observe(x, y)
        self.X += x
        self.y += y
        for i in range(len(y)):
            if y[i] > self.best_y:
                self.best_x = x[i]
                self.best_y = y[i]

        self.log_str += f"trial: {len(self.X)} best_y: {self.best_y} y: {y} last_beta: {self.last_beta} filtered: {self.last_filtered}\n"

        if len(self.X) < self.update_first:
            return

        steps_since_first = (len(self.X) - self.update_first) // self.update_period
        update_n = self.update_first + steps_since_first * self.update_period

        if update_n > self.update_last:
            self.rf.fit(self.X[:update_n], self.y[:update_n])
            self.update_last = update_n

    @override
    def _replay_batch(self, n_observed: int):
        super()._replay_batch(n_observed)
        if n_observed < self.update_first:
            # Warm-up batches also draw their random choices
            self.rng.choices([None], k=self.batch)

    @override
    def restore(self, x: list[VecSample], y: list[float]):
        super().restore(x, y)
        if x:
            logger.info(f"restored best_y: {self.best_y} best_x: {self.best_x}")

    @override
    def finished(self):
        logger.info("finished!")
        logger.info(self.log_str)
        super().finished()


class BaseOptimizerPreset(BaseOptimizer):
    PRESET: dict[str, Any] = {}

    def __init__(
        self,
        sample_fn: Callable[[int, int], Iterator[VecSample]],
        batch: int,
        seed: int,
        config_file: str = "",
    ):
        config = self.resolve_config(config_file)
        super().__init__(sample_fn, batch, seed, **config)

    @classmethod
    def resolve_config(cls, config_file: str = "") -> dict[str, Any]:
        """Returns the preset parameters updated with the config file ones."""
        config = dict(cls.PRESET)
        if config_file:
            with open(config_file, "r") as f:
                config.update(yaml.safe_load(f))
        return config


class RandomOptimizerPreset(BaseOptimizerPreset, RandomOptimizer):
    pass


class RandomForestPreset_Default(BaseOptimizerPreset, RandomForestOptimizer):
    PRESET: dict[str, Any] = {
        "batch_candidates": 1000,
        "beta": 5,
        "alpha": 0.6,
        "update_first": None,
        "update_period": None,
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 3,
        "max_features": 0.9,
        "beta_decay": True,
    }


class RandomForestPreset_UCB(BaseOptimizerPreset, RandomForestOptimizer):
    PRESET: dict[str, Any] = {
        "batch_candidates": 1000,
        "beta": 1,
        "update_first": None,
        "update_period": None,
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": 0.8,
        "beta_decay": False,
    }


class RandomForestPreset_UCB_Explore(BaseOptimizerPreset, RandomForestOptimizer):
    PRESET: dict[str, Any] = {
        "batch_candidates": 1000,
        "beta": 5,
        "update_first": None,
        "update_period": None,
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": 0.8,
        "beta_decay": False,
    }


class Optimizers:
    @classmethod
    def names(cls) -> Sequence[str]:
        return list(cls._map.keys())

    @classmethod
    def from_name(cls, name: str) -> type[BaseOptimizerPreset]:
        if name not in cls._map:
            raise ValueError(f"unknown optimizer name: {name}")
        return cls._map[name]

    _map: ClassVar[dict[str, type[BaseOptimizerPreset]]] = {
        "random": RandomOptimizerPreset,
        "random-forest-default": RandomForestPreset_Default,
        "random-forest-ucb": RandomForestPreset_UCB,
        "random-forest-ucb-explore": RandomForestPreset_UCB_Explore,
    }
