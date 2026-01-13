import random
import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor

class Optimizer():
    @abstractmethod
    def suggest():
        pass
    @abstractmethod
    def observe(self, x, y):
        pass

class RandomForestOptimizer:
    def __init__(self, sample_fn, batch_size=128, min_samples=5, beta=1.0, seed=0):
        self.sample_fn = sample_fn
        self.batch_size = batch_size
        self.min_samples = min_samples
        self.beta = beta
        self.rng = random.Random(seed)
        self.seed = seed

        self.X = []
        self.y = []

        self.rf = RandomForestRegressor(
            n_estimators=200,
            n_jobs=-1,
            random_state=seed,
        )

    def suggest(self):
        candidates = list(self.sample_fn(self.batch_size, self.seed))

        if len(self.X) < self.min_samples:
            return self.rng.choice(candidates)

        preds = np.stack([t.predict(candidates) for t in self.rf.estimators_])
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        # want low loss with high certainty (low std)
        score = mean - self.beta * std
        return candidates[np.argmin(score)]

    def observe(self, x, y):
        self.X.append(list(x))
        self.y.append(y)

        if len(self.X) >= self.min_samples:
            self.rf.fit(self.X, self.y)
