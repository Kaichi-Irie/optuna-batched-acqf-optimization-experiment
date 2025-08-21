# %%
import json
import os
import sys
from itertools import product
from pathlib import Path

import optuna
import optunahub
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

# プロジェクトのルートディレクトリをPythonのパスに追加
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import benchmark.iterinfo_global_variables as cfg
from batched_sampler import SAMPLERMODE, BatchedSampler

# %%
BBOB = optunahub.load_module("benchmarks/bbob")


# %%
def execute_benchmark(
    mode: SAMPLERMODE,
    dimension,
    function_id,
    n_trials=50,
    seed=42,
    summary_file="summary.jsonl",
    output_dir="benchmark/output",
):
    objective = BBOB.Problem(
        function_id=function_id, dimension=dimension, instance_id=1
    )
    sampler = BatchedSampler(mode=mode, seed=seed)
    log_file = f"f{function_id}_{dimension}D_{seed}_{mode}_{n_trials}_trials.jsonl"
    storage = JournalStorage(JournalFileBackend(os.path.join(output_dir, log_file)))

    study = optuna.create_study(
        directions=objective.directions, sampler=sampler, storage=storage
    )

    with sampler:
        cfg.MAX_NITS = []
        cfg.TOTAL_NITS = []
        study.optimize(objective, n_trials=n_trials)

    print(study.best_trial.params, study.best_trial.value)
    elapsed = (
        study.trials[-1].datetime_complete - study.trials[0].datetime_start
    ).total_seconds()
    print(f"{mode} took {elapsed:.2f} seconds. ")

    summary = {
        "function_id": function_id,
        "dimension": dimension,
        "seed": seed,
        "mode": mode,
        "elapsed": round(elapsed, 2),
        "n_trials": n_trials,
        "best_value": study.best_trial.value,
        # "best_params": study.best_trial.params,
    }

    with open(os.path.join(output_dir, summary_file), "a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"{len(cfg.MAX_NITS)=}, {len(cfg.TOTAL_NITS)=}")

    if cfg.MAX_NITS and cfg.TOTAL_NITS:
        # first 10 trials are warm-up
        assert len(cfg.MAX_NITS) == len(cfg.TOTAL_NITS) == len(study.trials) - 10
        iteration_info = {
            "max_nits": cfg.MAX_NITS,
            "total_nits": cfg.TOTAL_NITS,
        }
        # save as JSONL
        with open(os.path.join(output_dir, "iterinfo_" + log_file), "w") as f:
            f.write(json.dumps(iteration_info) + "\n")


# %%
if __name__ == "__main__":
    seeds = [42, 43, 44]  # [42, 43, 44]
    n_trials = 300  # 回せるだけ回す~500
    # https://numbbo.github.io/coco/testsuites/bbob
    function_ids = [1, 6, 10, 15, 20]  # [1,6,10,15,20]
    dimensions = [5, 10, 20]  # [5,10,20]
    modes: list[SAMPLERMODE] = [
        "stacking",
        "batched_acqf_eval",
        "multiprocessing",
        "original",
    ]
    for function_id, dimension, seed, mode in product(
        function_ids, dimensions, seeds, modes
    ):
        execute_benchmark(
            function_id=function_id,
            dimension=dimension,
            mode=mode,
            n_trials=n_trials,
            seed=seed,
        )
