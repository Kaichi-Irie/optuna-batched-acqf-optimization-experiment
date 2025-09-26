#%%
import json
import os
from itertools import product

import numpy as np
import optunahub

import optuna

np.random.seed(42)
#%%

NUM_CONTINUOUS_PARAMS = 15
ROTATION_MATRIX, _ = np.linalg.qr(np.random.rand(NUM_CONTINUOUS_PARAMS, NUM_CONTINUOUS_PARAMS))
CONDITIONING = 10
WEIGHTS = np.array([CONDITIONING**(i / (NUM_CONTINUOUS_PARAMS - 1)) for i in range(NUM_CONTINUOUS_PARAMS)])


NUM_DISCRETE_PARAMS = 5
ROTATION_MATRIX_D, _ = np.linalg.qr(np.random.rand(NUM_DISCRETE_PARAMS, NUM_DISCRETE_PARAMS))
# WEIGHTS_D = np.array([CONDITIONING**(i / (NUM_DISCRETE_PARAMS - 1)) for i in range(NUM_DISCRETE_PARAMS)])

def rotated_ellipsoid(trial):
    xs = np.array([trial.suggest_float(f"x_{i}", -1, 1) for i in range(NUM_CONTINUOUS_PARAMS)])
    rotated_xs = ROTATION_MATRIX @ xs
    return np.sum(WEIGHTS * (rotated_xs**2))


def rotated_ellipsoid_with_discrete(trial):
    xs = np.array([trial.suggest_float(f"x_{i}", -1, 1) for i in range(NUM_CONTINUOUS_PARAMS)])
    rotated_xs = ROTATION_MATRIX @ xs
    zs = np.array([trial.suggest_int(f"z{i}", 0, 5) for i in range(NUM_DISCRETE_PARAMS)])
    rotated_zs = ROTATION_MATRIX_D @ zs
    return  np.sum(WEIGHTS * (rotated_xs**2)) + np.sum((rotated_zs - 2) ** 2)
#%%
bbob = optunahub.load_module("benchmarks/bbob")
f6 = bbob.Problem(function_id=6, dimension=20, instance_id=1)
#%%

wfg = optunahub.load_module("benchmarks/wfg")
wfg4 = wfg.Problem(function_id=4, n_objectives=2, dimension=3, k=1)
#%%

objectives = {
    "rotated_ellipsoid": rotated_ellipsoid,
    "rotated_ellipsoid_with_discrete": rotated_ellipsoid_with_discrete,
    "f6": f6,
}

directions = {
    "rotated_ellipsoid": ["minimize"],
    "f6": f6.directions,
}

def execute_benchmark(
    mode: str,
    objective_type: str,
    n_trials: int,
    seed: int,
    results_file: str,
    output_dir="output",
):
    sampler = optuna.samplers.GPSampler(seed=seed)
    name = f"{objective_type}_{seed}_{mode}_{n_trials}"
    log_file = f"{name}.jsonl"
    study = optuna.create_study(study_name=name, sampler=sampler)

    study.optimize(func=objectives[objective_type], n_trials=n_trials)
    # print(study.best_trial.params, study.best_trial.value)
    elapsed = (study.trials[-1].datetime_complete - study.trials[0].datetime_start).total_seconds() # type: ignore
    print(f"{mode} took {elapsed:f} seconds. ")

    result = {
        "objective_type": objective_type,
        "seed": seed,
        "mode": mode,
        "elapsed": round(elapsed, 2),
        "n_trials": n_trials,
        "best_value": study.best_trial.value,
    }

    with open(os.path.join(output_dir, results_file), "a") as f:
        f.write(json.dumps(result) + "\n")

seeds =[42,]# [42,43,44]
n_trials = 300

# Only "GL_Batched_Eval" is enabled in this PR
# Edit the source code directly to run other modes.
modes: list[str] = [
    # "GL_Batched_Eval",
# "GL_Batched_Eval_refactored",
"GL_Batched_Eval_refactoredv2",
# "GL_Fallback",
# "GL_Fallback_refactored",
# "GL_Fallback_refactoredv2",
# "original",
]

objective_types = [
    # "rotated_ellipsoid",
    "rotated_ellipsoid_with_discrete",
    # "f6",
    # "wfg",
]
print("Starting benchmark...")
for seed, mode, objective_type in product(seeds, modes, objective_types):
    execute_benchmark(
        mode=mode,
        objective_type=objective_type,
        results_file="results0924.jsonl",
        n_trials=n_trials,
        seed=seed,
    )

# %%
