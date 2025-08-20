import math

import numpy as np
import optuna
import pytest

from batched_sampler import BatchedSampler

N_TRIALS = 200
SEED = 1234
DIMENSION = 10

# generate random rotate matrix using QR decomposition
rng = np.random.default_rng(SEED)
R, _ = np.linalg.qr(rng.normal(size=(DIMENSION, DIMENSION)))
w = rng.random(DIMENSION) * 1000
w /= w.sum()
assert np.allclose(R @ R.T, np.eye(DIMENSION))


def objective(trial):
    x = np.array([trial.suggest_float(f"x{i}", -10, 10) for i in range(DIMENSION)])
    z = R @ x
    return w.dot((z - 2) ** 2)


@pytest.mark.parametrize("processes", [1, 2, 4])
def test_multiprocessing_sampler_matches_original(processes):
    # Reference (original GPSampler)
    original_sampler = optuna.samplers.GPSampler(seed=SEED)
    original_study = optuna.create_study(sampler=original_sampler)
    original_study.optimize(objective, n_trials=N_TRIALS)

    # Batched (parallelized local search inside acquisition optimization)
    batched_sampler = BatchedSampler(mode="multiprocessing", seed=SEED)
    batched_sampler.create_worker_pool(processes=processes)
    batched_study = optuna.create_study(sampler=batched_sampler)
    with batched_sampler:
        batched_study.optimize(objective, n_trials=N_TRIALS)

    # Ensure same number of trials
    assert len(original_study.trials) == len(batched_study.trials) == N_TRIALS

    # Compare trial-by-trial (order should be identical if deterministic)
    for t_orig, t_batch in zip(
        sorted(original_study.trials, key=lambda t: t.number),
        sorted(batched_study.trials, key=lambda t: t.number),
    ):
        # Parameters match (allow tiny numerical jitter)
        for k in t_orig.params.keys():
            assert k in t_orig.params and k in t_batch.params
            assert math.isclose(
                t_orig.params[k], t_batch.params[k], rel_tol=1e-9, abs_tol=1e-12
            ), (
                f"Param {k} differs at trial {t_orig.number}: {t_orig.params[k]} vs {t_batch.params[k]}"
            )
        # Objective values match
        assert math.isclose(t_orig.value, t_batch.value, rel_tol=1e-9, abs_tol=1e-12), (  # type: ignore
            f"Value differs at trial {t_orig.number}: {t_orig.value} vs {t_batch.value}"
        )
