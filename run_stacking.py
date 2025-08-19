import numpy as np
import optuna

from batched_sampler import BatchedSampler

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


sampler = BatchedSampler(mode="stacking", seed=SEED)
study_stacking = optuna.create_study(sampler=sampler)
study_stacking.optimize(objective, n_trials=N_TRIALS)

t_stacking = (
    study_stacking.trials[-1].datetime_complete
    - study_stacking.trials[0].datetime_start
).total_seconds()

print(
    f"Stacking took {t_stacking:.2f} seconds. Best trial value: {study_stacking.best_trial.value:.2e}"
)
