# %%
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_iter_data(dimension, function_id, seed, n_trials, mode):
    with open(
        f"benchmark/output/iterinfo_f{function_id}_{dimension}D_{seed}_{mode}_{n_trials}_trials.jsonl",
        "r",
    ) as f:
        json_data = [json.loads(line) for line in f]
        data = json_data[0]
    return data


dimension = 20
function_id = 1
n_trials = 300
seed = 42
data_stacking = read_iter_data(dimension, function_id, seed, n_trials, "stacking")
data_batched_acqf = read_iter_data(
    dimension, function_id, seed, n_trials, "batched_acqf_eval"
)
total_nits_stacking = data_stacking["total_nits"]
batch_size = 10
total_nits_stacking_modified = [x * batch_size for x in total_nits_stacking]
max_nits_original = data_batched_acqf["max_nits"]
total_nits_original = data_batched_acqf["total_nits"]
# %%
# --- 2. データをグラフ描画に適した形式に整形 ---
# 各リストをPandas DataFrameに変換し、一つのDataFrameに結合します
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

plt.savefig(
    f"benchmark/plot/iteration_plot_f{function_id}_{dimension}D_{seed}_{n_trials}_trials.png",
    bbox_inches="tight",
    dpi=600,
)
plt.show()

# %%
