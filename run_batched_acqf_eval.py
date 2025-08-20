# %%
import importlib

import numpy as np
import optuna

import batched_acqf_eval_optim_mixed
import batched_sampler
from batched_sampler import BatchedSampler

importlib.reload(batched_sampler)
importlib.reload(batched_acqf_eval_optim_mixed)

N_TRIALS = 500
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


# %%
sampler = BatchedSampler(mode="batched_acqf_eval", seed=SEED)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)

elapsed = (
    study.trials[-1].datetime_complete - study.trials[0].datetime_start
).total_seconds()

print(
    f"batched took {elapsed:.2f} seconds. Best trial value: {study.best_trial.value:.2e}"
)

# %%
