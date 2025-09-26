#%%
import numpy as np
import optuna
import sys
import importlib
# import pytest

NUM_PARAMS = 20
ROTATION_MATRIX, _ = np.linalg.qr(np.random.rand(NUM_PARAMS, NUM_PARAMS))

CONDITIONING = 10
WEIGHTS = np.array([CONDITIONING**(i / (NUM_PARAMS - 1)) for i in range(NUM_PARAMS)])
N_TRIALS = 80

def objective(trial: optuna.Trial) -> float:
    xs = np.array([trial.suggest_float(f"x_{i}", -1, 1) for i in range(NUM_PARAMS)])
    rotated_xs = ROTATION_MATRIX @ xs

    zs = np.array([trial.suggest_int(f"z{i}", 0, 5) for i in range(NUM_PARAMS )])
    rotated_zs = ROTATION_MATRIX @ zs
    return  np.sum(WEIGHTS * (rotated_xs**2)) + np.sum((rotated_zs - 2) ** 2)
#%%
sampler = optuna.samplers.GPSampler(seed=42)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)

elapsed = (study.trials[-1].datetime_complete - study.trials[0].datetime_start).total_seconds() # type: ignore
print(f"GP took {elapsed:f} seconds. ")

# %%
import numpy as np
import optuna
import sys
import importlib
# import pytest

# def test_behavior_without_greenlet(monkeypatch: pytest.MonkeyPatch) -> None:
#     monkeypatch.setitem(sys.modules, "greenlet", None)
#     import optuna._gp.batched_lbfgsb as my_module

#     importlib.reload(my_module)
#     assert my_module._greenlet_imports.is_successful() is False

#     # See if optimization still works without greenlet
#     import optuna

#     sampler = optuna.samplers.GPSampler(seed=42)
#     study = optuna.create_study(sampler=sampler)
#     study.optimize(objective, n_trials=N_TRIALS)
#     elapsed = (study.trials[-1].datetime_complete - study.trials[0].datetime_start).total_seconds() # type: ignore
#     print(f"GP without greenlet took {elapsed:f} seconds. ")

# test_behavior_without_greenlet(pytest.MonkeyPatch())

# # %%
