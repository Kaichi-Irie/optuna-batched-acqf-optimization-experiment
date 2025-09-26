# %%
import json

import pandas as pd
import plotly.express as px

# --- データ準備 ---
# summary.jsonlを読み込む
df_list = []
with open("benchmark/output/summary.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        df = pd.DataFrame(data, index=[0])
        df_list.append(df)

# すべてのデータフレームを結合
df = pd.concat(df_list, ignore_index=True)
df
# %%
# 各設定（function_id, dimension, mode）ごとに平均値を計算
# seedの違いを平均して、結果を安定させる
summary_df = (
    df.groupby(["function_id", "dimension", "mode"])
    .agg(
        elapsed_mean=("elapsed", "mean"),
        best_value_mean=("best_value", "mean"),
    )
    .reset_index()
)
summary_df
# %%
# --- グラフ作成 ---
fig = px.bar(
    summary_df,
    x="mode",
    y="elapsed_mean",
    color="mode",
    barmode="group",  # 棒をグループ化して表示
    facet_row="function_id",  # 関数ごとにグラフを分ける
    facet_col="dimension",
    labels={
        "dimension": "Dim",
        "elapsed_mean": "Runtime (sec) ",
        "function_id": "テスト関数 ID",
    },
    title="<b>手法別・次元別の計算時間比較</b>",
    template="plotly_white",
)

fig.update_layout(title_x=0.5, font=dict(size=14))
fig.show()

# %%

# %%
# 基準となるoriginalモードの実行時間を抽出
original_times = summary_df[summary_df["mode"] == "original"][
    ["function_id", "dimension", "elapsed_mean"]
].rename(columns={"elapsed_mean": "original_elapsed"})

# 元のDataFrameにoriginalモードの実行時間を結合
summary_with_ratio = pd.merge(
    summary_df, original_times, on=["function_id", "dimension"]
)

# originalモードに対する実行時間の割合（%）を計算
summary_with_ratio["percentage_vs_original"] = (
    summary_with_ratio["elapsed_mean"] / summary_with_ratio["original_elapsed"]
) * 100

# 表示用にフォーマットを整える
summary_with_ratio["percentage_vs_original"] = summary_with_ratio[
    "percentage_vs_original"
].map("{:.2f}%".format)


summary_with_ratio

# %%


def create_df(file):
    df = []
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)
            df.append(data)

    df = pd.DataFrame(df)
    df.head()
    # drop unnecessary columns
    # unnecessary_columns = ["directions", "op_code", "worker_id", "study_name", "study_id"]

    necessary_columns = ["trial_id", "values"]
    df = df[necessary_columns]
    df.dropna(inplace=True)
    print(f"{df.shape=}")
    df["value"] = df["values"].apply(lambda x: x[0] if isinstance(x, list) else x)
    df.drop(columns=["values"], inplace=True)
    # make column trial_id as int
    df["trial_id"] = df["trial_id"].astype(int)

    # remove index
    df.reset_index(drop=True, inplace=True)

    df["cummin_value"] = df["value"].cummin()
    return df


# %%
import plotly.express as px

file = "benchmark/output/f1_5D_42_batched_acqf_eval_30_trials.jsonl"
modes = [
    "stacking",
    "batched_acqf_eval",
    "multiprocessing",
    "original",
]

dim = 10
seed = 42
mode = modes[0]
file = f"benchmark/output/f1_{dim}D_{seed}_{mode}_300_trials.jsonl"
# show cummin of all modes in one graph
for mode in modes:
    file = f"benchmark/output/f1_{dim}D_{seed}_{mode}_300_trials.jsonl"
    df = create_df(file)
    fig = px.line(
        df, x="trial_id", y="cummin_value", title=f"Cumulative Minimum Value - {mode}"
    )
    fig.show()
# %%

dim = 10
seed = 42

# 各モードのDataFrameを格納するリスト
all_dfs = []

# 全てのモードのデータを読み込み、リストに追加
for mode in modes:
    file = f"benchmark/output/f1_{dim}D_{seed}_{mode}_300_trials.jsonl"
    df = create_df(file)
    df["mode"] = mode  # どのモードのデータか分かるように列を追加
    all_dfs.append(df)

# リスト内の全てのDataFrameを1つに結合
combined_df = pd.concat(all_dfs, ignore_index=True)

# 結合したDataFrameを使い、`color="mode"`で色分けしてプロット
fig = px.line(
    combined_df,
    x="trial_id",
    y="cummin_value",
    color="mode",  # modeごとに線の色を変える
    title=f"Cumulative Minimum Value Comparison (Dim={dim}, Seed={seed})",
    labels={
        "trial_id": "Trial Number",
        "cummin_value": "Best Value Found",
        "mode": "Sampler Mode",
    },
    template="plotly_white",
)
fig.show()

# %%
import os

import optuna
from optuna.storages.journal import JournalFileBackend, JournalStorage

function_id = 1
dimension = 5
seed = 42
mode = "batched_acqf_eval"
n_trials = 300
output_dir = "benchmark/output"
log_file = f"f{function_id}_{dimension}D_{seed}_{mode}_{n_trials}_trials.jsonl"
storage = JournalStorage(JournalFileBackend(os.path.join(output_dir, log_file)))

study = storage.get_all_studies()[0]
name = study.study_name
names = [study.study_name for study in storage.get_all_studies()]
study = optuna.load_study(study_name=name, storage=storage)


def load(mode, dimension=20, function_id=1, seed=42, n_trials=300):
    log_file = f"f{function_id}_{dimension}D_{seed}_{mode}_{n_trials}_trials.jsonl"
    storage = JournalStorage(JournalFileBackend(os.path.join(output_dir, log_file)))
    names = [study.study_name for study in storage.get_all_studies()]
    study = optuna.load_study(study_name=names[-1], storage=storage)
    return study


# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import optuna
import optunahub

studies = defaultdict(lambda: [])
for mode in ["batched_acqf_eval", "original", "stacking", "multiprocessing"]:
    study = load(
        mode=mode,
        dimension=dimension,
        function_id=function_id,
        seed=seed,
        n_trials=n_trials,
    )
    studies[mode].append(study)


plot_sampling_speed = optunahub.load_module(
    package="visualization/plot_sampling_speed",
)
ax = plot_sampling_speed.plot_sampling_speed(dict(studies))
ax.set_xscale("linear")
ax.set_yscale("linear")
plt.show()

# %%
# 【変更前】逐次実行：N個の最適化を一つずつ実行


# N個の初期値
starting_points = ...
results = []

# 1. 初期値ごとに外側のループが回る
for x0 in starting_points:
    x = x0
    # 2. 反復法による更新ループ
    for _ in range(max_iter):
        # 評価と勾配計算を一つずつ行う（ボトルネック）
        f, grad = func_and_grad(x)
        x = update(x, f, grad)
    results.append(x)


# N個の初期値を一つのTensorにまとめる
xs = starting_points
# 1. 最適化ループは1回だけ
for _ in range(max_iter):
    # N個の点に対する評価と勾配計算を一度に実行！
    fs, grads = func_and_grad_batched(xs)

    # N個の点の座標を一度に更新
    xs = update_batched(xs, fs, grads)

results = xs
# %%
import json

dimension = 20
function_id = 1
n_trials = 300
seed = 42
with open(
    f"benchmark/output/iterinfo_f{function_id}_{dimension}D_{seed}_stacking_{n_trials}_trials.jsonl",
    "r",
) as f:
    json_data = [json.loads(line) for line in f]
    data = json_data[0]

total_nits_stacking = data["total_nits"]
with open(
    f"benchmark/output/iterinfo_f{function_id}_{dimension}D_{seed}_batched_acqf_eval_{n_trials}_trials.jsonl",
    "r",
) as f:
    json_data = [json.loads(line) for line in f]
    data = json_data[0]

max_nits_original = data["max_nits"]
total_nits_original = data["total_nits"]
assert len(max_nits_original) == len(total_nits_original)


# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- 2. データをグラフ描画に適した形式に整形 ---
# 各リストをPandas DataFrameに変換し、一つのDataFrameに結合します
batch_size = 10
total_nits_stacking_modified = [x * batch_size for x in total_nits_stacking]
data_dict = {
    "Stacking Total Nits": total_nits_stacking_modified,
    "Original Max Nits": max_nits_original,
    "Original Total Nits": total_nits_original,
}

# グラフ描画用のリストを作成
plot_data = []
for method, iter_list in data_dict.items():
    for trial_num, iterations in enumerate(iter_list):
        plot_data.append(
            {"Trial": trial_num, "Method": method, "Iterations": iterations}
        )

df = pd.DataFrame(plot_data)


# --- 3. グラフの描画 ---
# スタイルやフォントサイズをプレゼン用に設定
sns.set_theme(style="whitegrid")
sns.set_context("talk")

# グラフのサイズを指定 (2つのグラフを縦に並べる)
fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
fig.suptitle(
    f"Trial and Iterations (f{function_id}, {dimension}D, {seed}, {n_trials} trials)",
    fontsize=24,
    y=1.02,
)


# --- グラフ1: 生のデータをプロット ---
sns.lineplot(
    data=df,
    x="Trial",
    y="Iterations",
    hue="Method",
    ax=axes[0],
    alpha=0.6,  # 少し透明にして傾向線を見やすくする
)
axes[0].set_title("Raw Iteration Counts")
axes[0].set_ylabel("Number of Iterations")


# --- グラフ2: 移動平均で滑らかにしたデータをプロット ---
# windowサイズで平滑化の度合いを調整 (例: 10 trial分の平均)
window_size = 10
df["Smoothed Iterations"] = df.groupby("Method")["Iterations"].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean()
)

sns.lineplot(
    data=df,
    x="Trial",
    y="Smoothed Iterations",
    hue="Method",
    ax=axes[1],
    linewidth=2.5,  # 線を太くして見やすくする
)
axes[1].set_title(f"Smoothed with Moving Average (window={window_size})")
axes[1].set_xlabel("Trial Number")
axes[1].set_ylabel("Smoothed Iterations")


# 凡例を一つにまとめる
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    handles, labels, title="Method", bbox_to_anchor=(1.05, 0.85), loc="upper left"
)
axes[0].get_legend().remove()
axes[1].get_legend().remove()


# レイアウトを自動調整
# plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("benchmark/output/iteration_plot.png")
plt.show()


# %%


def update():
    pass


def acqf():
    pass


def acqf_grad():
    pass


def scipy_optim():
    pass


# %%


# min_fval=0.0, min_x=0.0
min_fval, min_x = scipy_optim(
    f=lambda x: x**2,
    g=lambda x: 2 * x,
    x0=1.0,
)


best_x, best_fval = ...
for x0 in x0_list:
    min_f, min_x = scipy_optim(acqf, acqf_grad, x0)
    best_x, best_fval = ...


# %%


def acqf_grad_stacked(x_list):
    pass


# %%


def acqf_stacked(x_list):
    return acqf(x_list).sum()


sum_fvals, min_xs = scipy_optim(
    lambda x_list: acqf(x_list).sum(),
    acqf_grad_stacked,
    x0_list.flatten(),
)
best_fval, best_x = ...


#%%
...
from scipy.optimize import _lbfgsb as scipy_lbfgsb
...
scipy_lbfgsb.main(...)
#%%

# %%
