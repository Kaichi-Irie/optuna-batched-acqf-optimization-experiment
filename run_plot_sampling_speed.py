# %%
import importlib
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import optuna
from optuna.storages.journal import JournalFileBackend, JournalStorage

import plot_sampling_speed as plot_sampling_speed

importlib.reload(plot_sampling_speed)

INPUT_DIR = "benchmark/output"
OUTPUT_DIR = "benchmark/plot"


def load(mode, dimension=20, function_id=1, seed=42, n_trials=300):
    log_file = f"f{function_id}_{dimension}D_{seed}_{mode}_{n_trials}_trials.jsonl"
    storage = JournalStorage(JournalFileBackend(os.path.join(INPUT_DIR, log_file)))
    names = [study.study_name for study in storage.get_all_studies()]
    study = optuna.load_study(study_name=names[-1], storage=storage)
    return study


color_dict = {
    "batched_acqf_eval": "darkred",
    "stacking": "blue",
    "multiprocessing": "gray",
    "original": "black",
}

marker_dict = {
    "batched_acqf_eval": None,
    "stacking": None,
    "multiprocessing": None,
    "original": "x",
}

# %%
function_id = 1
dimension = 20
studies = defaultdict(lambda: [])
seed = 42
mode = "stacking"
n_trials = 300
log_file = f"f{function_id}_{dimension}D_{seed}_{mode}_{n_trials}_trials.jsonl"
storage = JournalStorage(JournalFileBackend(os.path.join(INPUT_DIR, log_file)))
names = [study.study_name for study in storage.get_all_studies()]
study = optuna.load_study(study_name=names[-1], storage=storage)
# %%
for mode in ["batched_acqf_eval", "stacking", "multiprocessing", "original"]:
    study = load(mode=mode, function_id=function_id)
    studies[mode].append(study)

plt.rcParams["font.size"] = 15
plt.figure(figsize=(12, 9))


ax = plot_sampling_speed.plot_sampling_speed(
    dict(studies), color_dict=color_dict, marker_dict=marker_dict
)

ax.set_xscale("linear")
ax.set_yscale("linear")
ax.set_title(f"Elapsed Time at Each Trial (FID={function_id}, D={dimension})")

plot_image = f"plot_f{function_id}_{dimension}D.png"
plt.savefig(os.path.join(OUTPUT_DIR, plot_image), bbox_inches="tight", dpi=900)
