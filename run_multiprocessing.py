import time

import optuna

from batched_sampler import BatchedSampler

N_TRIALS = 50
SEED = 42


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2


def run_multiprocessing(processes):
    sampler = BatchedSampler(batch_size=10, mode="multiprocessing", seed=SEED)
    sampler.create_worker_pool(processes=processes)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)


if __name__ == "__main__":
    processes = 4
    start = time.time()
    run_multiprocessing(processes=processes)
    end = time.time()
    print(f"{processes} processes took {end - start:.2f} seconds")
