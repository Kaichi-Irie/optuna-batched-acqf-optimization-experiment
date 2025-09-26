#%%
import json
import os
from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import optunahub

import optuna
from optuna.storages.journal import JournalFileBackend, JournalStorage

DIMENSION = 20
wfg = optunahub.load_module("benchmarks/wfg")
wfg4 = wfg.Problem(function_id=4, n_objectives=3, dimension=DIMENSION, k=4)


def compute_hypervolume_history(study: optuna.Study, ref_point: np.ndarray) -> np.ndarray:
    loss_values = np.array([t.values for t in study.trials])
    hypervolume_history = np.zeros(len(loss_values), dtype=float)
    for i in range(len(loss_values)):
        hypervolume_history[i] = optuna._hypervolume.wfg.compute_hypervolume(
            loss_values[:i + 1], np.array(wfg4.reference_point)
        )
    return hypervolume_history


#%%
study_pareto = optuna.create_study(
    study_name="ParetoFront", directions=wfg4.directions
)
for x in wfg4.get_optimal_solutions(1000):  # Generate 1000 Pareto optimal solutions
    study_pareto.enqueue_trial(params={
        f"x{i}": x.phenome[i] for i in range(3)
    })
study_pareto.optimize(wfg4, n_trials=1000)


#%%

def execute_benchmark(
    mode: str,
    n_trials: int,
    seed: int,
    results_file="results.jsonl",
    output_dir="output",
):
    sampler = optuna.samplers.GPSampler(seed=seed)
    objective_type = "wfg"
    name = f"{objective_type}_{DIMENSION}D_{seed}_{mode}_{n_trials}_trials"
    log_file = f"{name}.jsonl"
    storage = JournalStorage(JournalFileBackend(os.path.join(output_dir, log_file)))
    study = optuna.create_study(study_name=name, sampler=sampler,directions=wfg4.directions, storage=storage)

    study.optimize(func=wfg4, n_trials=n_trials)
    elapsed = (study.trials[-1].datetime_complete - study.trials[0].datetime_start).total_seconds() # type: ignore
    print(f"{mode} took {elapsed:f} seconds. ")

    result = {
        "objective_type": objective_type,
        "dimension": DIMENSION,
        "seed": seed,
        "mode": mode,
        "elapsed": round(elapsed, 2),
        "n_trials": n_trials,
    }

    with open(os.path.join(output_dir, results_file), "a") as f:
        f.write(json.dumps(result) + "\n")
    return study

seeds = [42, 43, 44]  # [42,43,44]
n_trials = 100

# Only "GL_Batched_Eval" is enabled in this PR
# Edit the source code directly to run other modes.
modes: list[str] = [
    # "GL_Batched_Eval",
"GL_Fallback",
# "original",
]


studies = []
for seed, mode in product(seeds, modes):
    study = execute_benchmark(
        mode=mode,
        n_trials=n_trials,
        seed=seed,
    )
    studies.append(study)



# %%
# objective_type = "wfg"
# n_trials = 100
# seeds = [43]# [42,43,44]
# modes = [
#     "GL_Batched_Eval",
# # "GL_Fallback",
# # "original",
# ]


# study_names = [f"{objective_type}_{DIMENSION}D_{seed}_{mode}_{n_trials}_trials" for seed, mode in product(seeds, modes)]
# study_names



# # %%
# studies = []
# for name in study_names:
#     storage = JournalStorage(JournalFileBackend(os.path.join("output", f"{name}.jsonl")))
#     study = optuna.load_study(study_name=name, storage=storage)
#     studies.append(study)

# # %%
# # plot hypervolume history for all studies
# labels = {}
# for name in study_names:
#     if "Batched" in name:
#         labels[name] = "This PR (Batched Eval)"
#     elif "Fallback" in name:
#         labels[name] = "This PR (Fallback)"
#     elif "original" in name:
#         labels[name] = "Original"
#     else:
#         raise ValueError(f"Unknown study name: {name}")

# #%%
# for study in studies:
#     hh = compute_hypervolume_history(study, np.array(wfg4.reference_point))
#     plt.plot(hh, label=labels[study.study_name])

# plt.xlabel("Number of Trials")
# plt.ylabel("Hypervolume")
# plt.title("Hypervolume History of WFG (2 objectives, D=3), seed=43")
# plt.legend()
# plt.grid()
# plt.savefig("hypervolume_history.png", dpi=600, bbox_inches="tight")
# plt.show()


# # %%
