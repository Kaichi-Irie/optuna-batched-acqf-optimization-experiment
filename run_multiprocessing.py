import numpy as np
import optuna

from batched_sampler import BatchedSampler

N_TRIALS = 300
SEED = 42
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


def run_multiprocessing(processes):
    sampler = BatchedSampler(mode="multiprocessing", seed=SEED)
    sampler.create_worker_pool(processes=processes)
    study = optuna.create_study(sampler=sampler)
    with sampler:
        study.optimize(objective, n_trials=N_TRIALS)

    elapsed = (
        study.trials[-1].datetime_complete - study.trials[0].datetime_start
    ).total_seconds()

    print(
        f"{processes} processes took {elapsed:.2f} seconds. Best trial value: {study.best_trial.value:.2e}"
    )


if __name__ == "__main__":
    # if processes is None, then multiprocessing will use all available cores
    run_multiprocessing(processes=8)
