#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import random
import numpy as np
import logging
import yaml
from collections.abc import Sequence
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


class Optimizer:
    @abstractmethod
    def suggest():
        pass

    @abstractmethod
    def observe(self, x, y):
        pass

    def finished(self):
        pass


class RandomForestOptimizer(Optimizer):
    def __init__(
        self,
        sample_fn,
        batch,
        seed,
        batch_candidates,
        beta,
        alpha,
        update_first,
        update_period,
        n_estimators,
        max_depth,
        min_samples_leaf,
        max_features,
    ):
        self.sample_fn = sample_fn
        self.batch = batch
        self.batch_candidates = batch_candidates
        self.update_first = update_first if update_first else batch
        self.update_period = update_period if update_period else batch
        self.beta = beta
        self.rng = random.Random(seed)
        self.seed = seed
        self.alpha = alpha
        self.beta_min = 0.05

        self.X = []
        self.y = []
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=1,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_leaf * 2,
            max_features=max_features,
            random_state=seed,
        )
        self.best_y = 0
        self.best_x = None
        # logging
        self.last_beta = None
        self.log_str = ""

    def suggest(self):
        seed = self.rng.randint(0, 2**32 - 1)
        candidates = list(self.sample_fn(self.batch_candidates, seed))

        if len(self.X) < self.update_first:
            return self.rng.choices(candidates, k=self.batch)

        preds = np.stack([t.predict(candidates) for t in self.rf.estimators_])
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)

        beta = max(self.beta_min, self.beta / (len(self.y) ** self.alpha))
        improvement = mean - self.best_y
        scores = improvement + beta * std
        top_idx = np.argpartition(scores, -self.batch)[-self.batch :]
        top_candidates = [candidates[i] for i in top_idx]

        self.last_beta = beta
        return top_candidates

    def observe(self, x, y):
        self.X += x
        self.y += y
        for i in range(0, self.batch):
            if y[i] > self.best_y:
                self.best_x = x[i]
                self.best_y = y[i]

        self.log_str += f"trial: {len(self.X)} best_y: {self.best_y} y: {y} last_beta: {self.last_beta}\n"
        if len(self.X) >= self.update_first and len(self.X) % self.update_period == 0:
            self.rf.fit(self.X, self.y)

    def finished(self):
        logger.info("finished!")
        logger.info(self.log_str)


class RandomForestOptimizer_Explore(RandomForestOptimizer):
    def __init__(self, sample_fn, batch, seed, config_file):
        preset = {
            "batch_candidates": 1000,
            "beta": 5,
            "alpha": 0.6,
            "update_first": None,
            "update_period": None,
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 3,
            "max_features": 0.9,
        }
        super().__init__(sample_fn, batch, seed, **preset)


class RandomForestOptimizer_Default(RandomForestOptimizer):
    def __init__(self, sample_fn, batch, seed, config_file):
        preset = {
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
        super().__init__(sample_fn, batch, seed, **preset)


class RandomForestOptimizer_Aggressive(RandomForestOptimizer):
    def __init__(self, sample_fn, batch, seed, config_file):
        preset = {
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
        super().__init__(sample_fn, batch, seed, **preset)


class RandomForestOptimizer_Custom(RandomForestOptimizer):
    def __init__(self, sample_fn, batch, seed, config_file):
        with open(config_file, "r") as f:
            custom = yaml.safe_load(f)
            super().__init__(sample_fn, batch, seed, **custom)


class Optimizers:
    @classmethod
    def names(cls) -> Sequence[str]:
        return list(cls._map.keys())

    @classmethod
    def from_name(cls, name: str) -> type[Optimizer]:
        if name not in cls._map:
            raise ValueError(f"unknown optimizer name: {name}")
        return cls._map[name]

    _map = {
        "random-forest-explore": RandomForestOptimizer_Explore,
        "random-forest-default": RandomForestOptimizer_Default,
        "random-forest-aggressive": RandomForestOptimizer_Aggressive,
        "random-forest-custom": RandomForestOptimizer_Custom,
    }
