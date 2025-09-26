# %%
import json

import pandas as pd

# --- データ準備 ---
# summary.jsonlを読み込む
df_list = []
with open("summary_pfncluster.jsonl", "r") as f:
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
    df.groupby(["dimension", "mode"])
    .agg(
        elapsed_mean=("elapsed", "mean"),
        best_value_mean=("best_value", "mean"),
    )
    .reset_index()
)
summary_df

# %%
# 基準となるoriginalモードの実行時間を抽出
original_times = summary_df[summary_df["mode"] == "original"][
    ["dimension", "elapsed_mean"]
].rename(columns={"elapsed_mean": "original_elapsed"})

# 元のDataFrameにoriginalモードの実行時間を結合
summary_with_ratio = pd.merge(summary_df, original_times, on=["dimension"])

# originalモードに対する実行時間の割合を計算
summary_with_ratio["speedup"] = (
    summary_with_ratio["original_elapsed"] / summary_with_ratio["elapsed_mean"]
)

summary_with_ratio.to_csv("summary_with_speedup.csv", index=False)
# %%
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- 1. データの準備 ---
csv_data = """dimension,mode,elapsed_mean,best_value_mean,original_elapsed,speedup
5,batched_acqf_eval,62.85066666666667,167.2695963733847,162.5186666666667,2.585790657219229
5,multiprocessing,88.79866666666666,194.2571098956067,162.5186666666667,1.8301926455352187
5,original,162.5186666666667,194.2571098956067,162.5186666666667,1.0
5,stacking,59.374,168.12468400583256,162.5186666666667,2.7372025914822427
10,batched_acqf_eval,67.07333333333334,2296.7963930113833,155.64399999999998,2.320504919988072
10,multiprocessing,94.08066666666667,4138.540890698442,155.64399999999998,1.6543675285747688
10,original,155.64399999999998,4138.540890698442,155.64399999999998,1.0
10,stacking,76.41000000000001,2224.4870453387093,155.64399999999998,2.0369585132836012
20,batched_acqf_eval,86.032,40873.261356120245,171.5173333333333,1.9936457752154235
20,multiprocessing,111.816,48971.57990088368,171.5173333333333,1.5339247811881422
20,original,171.5173333333333,48971.57990088368,171.5173333333333,1.0
20,stacking,106.636,54602.87414648433,171.5173333333333,1.608437425759906
"""
df = pd.read_csv(io.StringIO(csv_data), skipinitialspace=True)

# --- 2. 高速化係数の計算 ---
original_times = df[df["mode"] == "original"].set_index("dimension")["elapsed_mean"]
df["Original Time"] = df["dimension"].map(original_times)

# replace batched_acqf_eval with This PR (Batched)
df["mode"] = df["mode"].replace({"batched_acqf_eval": "Batched Eval."})
df["mode"] = df["mode"].replace({"multiprocessing": "MP"})
df["mode"] = df["mode"].replace({"stacking": "Stacking"})

df["speedup"] = df["speedup"] * 100  # パーセンテージに変換

# 'original' と 'Batched Eval.' のみをプロット対象としてフィルタリング
# plot_df = df[df["mode"].isin(["original", "Batched Eval."])].copy()
plot_df = df[df["mode"] != "original"].copy()

# `mode` 列のカテゴリカル順序を設定
plot_df["mode"] = pd.Categorical(
    plot_df["mode"],
    categories=[
        "Batched Eval.",
        "Stacking",
        "MP",
    ],
    ordered=True,
)
# --- 3. スタイリッシュな比較プロット作成 ---

# モダンなプロットスタイルを設定
sns.set_style("whitegrid", {"axes.grid": False})
plt.rcParams["font.family"] = "sans-serif"

# グラフのフィギュアとアックスを作成
fig, ax = plt.subplots(figsize=(11, 6))

# カスタムカラーパレットを定義
# palette = {"original": "#cccccc", "This PR (Batched)": "#2ecc71"}
palette = {
    "Batched Eval.": "#2ecc71",
    "MP": "#b0b0b0",
    "Stacking": "#5598eb",
}

# グループ化された棒グラフを作成
bar_plot = sns.barplot(
    data=plot_df, x="dimension", y="speedup", hue="mode", palette=palette, ax=ax
)

# 棒グラフの上に数値を表示
for p in bar_plot.patches:
    # 0やNaNのような無効な高さをチェック
    height = p.get_height()
    if pd.notna(height) and height > 0:
        ax.annotate(
            format(height, ".0f"),
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            size=12,
            weight="bold",
            xytext=(0, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

# タイトルと軸ラベルを設定
ax.set_title("Speedup Comparison", fontsize=24, weight="bold", pad=20)
ax.set_xlabel("Dimension of the Objective Function", fontsize=20, weight="bold")
ax.set_ylabel("Mean Speedup (%)", fontsize=20, weight="bold")
# 軸の目盛りを大きくする
ax.tick_params(axis="both", which="major", labelsize=18)

# 上と右の枠線を消してスッキリさせる
sns.despine()

# グリッド線を追加
ax.yaxis.grid(True, linestyle="--", which="major", color="lightgrey", alpha=1)
ax.set_axisbelow(True)


# 凡例を調整
ax.legend(fontsize=16)

# レイアウトを自動調整
plt.tight_layout()

# 画像をファイルとして保存
plt.savefig("three_batching_methods_speedup_comparison.png", dpi=900)

plt.show()


# %%
19 / 34

# %%
