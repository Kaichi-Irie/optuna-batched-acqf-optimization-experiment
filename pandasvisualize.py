# %%
import json

import pandas as pd

# --- データ準備 ---
# summary.jsonlを読み込む
df_list = []
with open("benchmark/output/summary_pfncluster.jsonl", "r") as f:
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
    df.groupby(["objective_type","mode"])
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
    ["function_id", "dimension", "elapsed_mean"]
].rename(columns={"elapsed_mean": "original_elapsed"})

# 元のDataFrameにoriginalモードの実行時間を結合
summary_with_ratio = pd.merge(
    summary_df, original_times, on=["function_id", "dimension"]
)

# originalモードに対する実行時間の割合（%）を計算
summary_with_ratio["speedup"] = (
    summary_with_ratio["original_elapsed"] / summary_with_ratio["elapsed_mean"]
) * 100

# 整数に丸める
summary_with_ratio["speedup"] = summary_with_ratio["speedup"].map(round)

summary_with_ratio

# %%
# dimension毎にspeedupの平均を計算
mean_speedup = (
    summary_with_ratio.groupby(["dimension", "mode"])["speedup"].mean().reset_index()
)
mean_speedup


# %%
# 棒グラフにプロット
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(data=mean_speedup, x="dimension", y="speedup", hue="mode")
# set y-axis title
plt.xlabel("Dimension of the Problem")
plt.ylabel("Mean Speedup (%)")
plt.title("Mean Speedup Compared to Original by Dimension and Mode")
# %%
import matplotlib.pyplot as plt
import pandas as pd  # サンプルデータ作成のため
import seaborn as sns

# ----------------------------------------------------

# グラフのスタイルとフォントサイズを設定
sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=1.7)

# グラフのサイズを指定
plt.figure(figsize=(12, 9))
# plt.figure(figsize=(7.5, 10))
# fig, ax = plt.subplots(figsize=(7, 12))

hue_order = [
    # "original",
    "multiprocessing",
    "stacking",
    "batched_acqf_eval",
]

palette = {
    # "original": "gray",
    "multiprocessing": "darkgray",  # 灰色
    "stacking": "#0A00C9",
    "batched_acqf_eval": "#9E0404",  # 赤
}

# グラフの描画
mean_speedup_5D = mean_speedup[mean_speedup["dimension"] == 5]
mean_speedup_wo_original = mean_speedup[mean_speedup["mode"] != "original"]
ax = sns.barplot(
    data=mean_speedup_wo_original,
    x="dimension",
    y="speedup",
    hue="mode",
    hue_order=hue_order,
    palette=palette,
    legend=False,
)


ax.axhline(
    100, color="black", linestyle="--", linewidth=1.5, label="Baseline (Original)"
)

# ラベルとタイトルの設定
plt.xlabel("Dimension")
plt.ylabel("Mean Speedup (%)")
plt.title("Speedup Comparison")

# 凡例の調整
# plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="lower center", ncols=2)
# ax.legend()

# レイアウトを自動調整
plt.tight_layout()

# グラフを表示
plt.savefig("speedup_comparison.png", dpi=900, bbox_inches="tight")

# %%
