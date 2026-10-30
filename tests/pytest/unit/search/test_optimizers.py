import itertools
import random
from collections.abc import Iterator

from xtc.search.optimizers import RandomForestOptimizer, RandomOptimizer

GRID = [list(x) for x in itertools.product(range(1, 9), range(1, 9))]


def _sample_fn(num: int, seed: int) -> Iterator[list[int]]:
    return iter(random.Random(seed).sample(GRID, min(num, len(GRID))))


def _score(x: list[int]) -> float:
    return 1.0 / (1 + abs(x[0] - 5) + abs(x[1] - 3))


def _rf_optimizer(batch: int = 4, update_first: int = 8) -> RandomForestOptimizer:
    return RandomForestOptimizer(
        _sample_fn,
        batch,
        seed=0,
        batch_candidates=32,
        beta=2.5,
        alpha=0.7,
        update_first=update_first,
        update_period=batch,
        n_estimators=10,
        max_depth=4,
        min_samples_leaf=1,
        max_features=1.0,
    )


def test_random_forest_restore_matches_observe():
    x = GRID[:10]
    y = [_score(s) for s in x]

    restored = _rf_optimizer()
    restored.restore(x, y)

    observed = _rf_optimizer()
    for i in range(0, len(x), 4):
        observed.observe(x[i : i + 4], y[i : i + 4])

    assert restored.X == observed.X == x
    assert restored.y == observed.y == y
    assert restored.best_y == observed.best_y == max(y)
    assert restored.best_x == observed.best_x == x[y.index(max(y))]
    # Fit on the first 8 samples, the next refit happens at 12.
    assert restored.update_last == observed.update_last == 8
    assert len(restored.rf.estimators_) == 10


def test_random_forest_restore_below_update_first_does_not_fit():
    opt = _rf_optimizer(update_first=8)
    opt.restore(GRID[:3], [_score(s) for s in GRID[:3]])

    assert opt.update_last == 0
    assert not hasattr(opt.rf, "estimators_")


def test_random_forest_suggest_after_restore_uses_model():
    opt = _rf_optimizer()
    opt.restore(GRID[:16], [_score(s) for s in GRID[:16]])

    suggested = opt.suggest()

    assert len(suggested) == 4
    # last_beta is only set on the model driven path.
    assert opt.last_beta is not None


def test_restore_empty_is_noop():
    opt = _rf_optimizer()
    opt.restore([], [])

    assert opt.X == []
    assert opt.y == []
    assert opt.update_last == 0

    rnd = RandomOptimizer(_sample_fn, 4, seed=0)
    rnd.restore([], [])
    assert len(rnd.suggest()) == 4
