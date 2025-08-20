import math

import optuna

from batched_sampler import BatchedSampler

# In this test, easy objective function and small number of trials are used because batched acquisition functions can make different decisions than original ones as the optimization goes on.

N_TRIALS = 20
SEED = 1234


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return (x - 2) ** 2 + (y - 3) ** 2


def test_batched_acqf_eval_sampler_matches_original():
    # Reference (original GPSampler)
    original_sampler = optuna.samplers.GPSampler(seed=SEED)
    original_study = optuna.create_study(sampler=original_sampler)
    original_study.optimize(objective, n_trials=N_TRIALS)

    # BatchedSampler with mode="batched_acqf_eval"
    batched_sampler = BatchedSampler(mode="batched_acqf_eval", seed=SEED)
    batched_study = optuna.create_study(sampler=batched_sampler)
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
                t_orig.params[k], t_batch.params[k], rel_tol=1e-2, abs_tol=1e-2
            ), (
                f"Param {k} differs at trial {t_orig.number}: {t_orig.params[k]} vs {t_batch.params[k]}"
            )
        # Objective values match
        assert math.isclose(t_orig.value, t_batch.value, rel_tol=1e-2, abs_tol=1e-2), (  # type: ignore
            f"Value differs at trial {t_orig.number}: {t_orig.value} vs {t_batch.value}"
        )
