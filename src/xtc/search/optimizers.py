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
    def __init__(
        self,
        sample_fn: Callable[[int, int], Iterator[VecSample]],
        batch: int,
        seed: int,
        **kwargs: Any,
    ):
        self.sample_fn = sample_fn
        self.batch = batch
        self.rng = random.Random(seed)

    @override
    def suggest(self) -> list[VecSample]:
        raise NotImplementedError

    @override
    def observe(self, x: list[VecSample], y: list[float]):
        pass

    @override
    def finished(self):
        pass

    def _sample(self, num: int) -> list[VecSample]:
        seed = self.rng.randint(0, 2**32 - 1)
        return list(self.sample_fn(num, seed))


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
        return self._sample(self.batch)


class RandomForestOptimizer(BaseOptimizer):
    def __init__(
        self,
        sample_fn: Callable[[int, int], Iterator[VecSample]],
        batch: int,
        seed: int,
        batch_candidates: int,
        beta: float,
        alpha: float,
        update_first: int,
        update_period: int,
        n_estimators: int,
        max_depth: int,
        min_samples_leaf: int,
        max_features: float,
    ):
        super().__init__(sample_fn, batch, seed)
        self.batch_candidates = batch_candidates
        self.update_first = update_first if update_first else batch
        self.update_period = update_period if update_period else batch
        self.beta = beta
        self.seed = seed
        self.alpha = alpha
        self.beta_min = 0.05
        self.update_last = 0

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
        self.last_beta = None
        self.log_str = ""

    @override
    def suggest(self):
        candidates = self._sample(self.batch_candidates)
        if len(self.X) < self.update_first:
            return self.rng.choices(candidates, k=self.batch)

        preds = np.stack([t.predict(candidates) for t in self.rf.estimators_])
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)

        # UCB acquisition modified with improvement and inverse time beta decay
        beta = max(self.beta_min, self.beta / (len(self.y) ** self.alpha))
        improvement = mean - self.best_y
        scores = improvement + beta * std
        top_idx = np.argpartition(scores, -self.batch)[-self.batch :]
        top_candidates = [candidates[i] for i in top_idx]

        self.last_beta = beta
        return top_candidates

    @override
    def observe(self, x: list[VecSample], y: list[float]):
        self.X += x
        self.y += y
        for i in range(len(y)):
            if y[i] > self.best_y:
                self.best_x = x[i]
                self.best_y = y[i]

        self.log_str += f"trial: {len(self.X)} best_y: {self.best_y} y: {y} last_beta: {self.last_beta}\n"

        if len(self.X) < self.update_first:
            return

        steps_since_first = (len(self.X) - self.update_first) // self.update_period
        update_n = self.update_first + steps_since_first * self.update_period

        if update_n > self.update_last:
            self.rf.fit(self.X[:update_n], self.y[:update_n])
            self.update_last = update_n

    @override
    def finished(self):
        logger.info("finished!")
        logger.info(self.log_str)


class BaseOptimizerPreset(BaseOptimizer):
    PRESET: dict[str, Any] = {}

    def __init__(
        self,
        sample_fn: Callable[[int, int], Iterator[VecSample]],
        batch: int,
        seed: int,
        config_file: str = "",
    ):
        config = dict(self.PRESET)
        if config_file:
            with open(config_file, "r") as f:
                config.update(yaml.safe_load(f))
        super().__init__(sample_fn, batch, seed, **config)


class RandomOptimizerPreset(BaseOptimizerPreset, RandomOptimizer):
    pass


class RandomForestPreset_Default(BaseOptimizerPreset, RandomForestOptimizer):
    PRESET: dict[str, Any] = {
        "batch_candidates": 1000,
        "beta": 2.5,
        "alpha": 0.7,
        "update_first": None,
        "update_period": None,
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": 0.8,
    }


class RandomForestPreset_Explore(BaseOptimizerPreset, RandomForestOptimizer):
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
    }


class RandomForestPreset_Aggressive(BaseOptimizerPreset, RandomForestOptimizer):
    PRESET: dict[str, Any] = {
        "batch_candidates": 1000,
        "beta": 2,
        "alpha": 0.8,
        "update_first": None,
        "update_period": None,
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "max_features": 0.7,
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
        "random-forest-explore": RandomForestPreset_Explore,
        "random-forest-default": RandomForestPreset_Default,
        "random-forest-aggressive": RandomForestPreset_Aggressive,
    }
