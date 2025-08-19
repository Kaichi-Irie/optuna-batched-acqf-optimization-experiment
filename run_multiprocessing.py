import time

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
    study.optimize(objective, n_trials=N_TRIALS)


if __name__ == "__main__":
    processes = 4
    start = time.time()
    run_multiprocessing(processes=processes)
    end = time.time()
    print(f"{processes} processes took {end - start:.2e} seconds")
