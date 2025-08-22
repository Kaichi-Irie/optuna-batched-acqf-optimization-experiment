# %%
import json

import pandas as pd

# --- データ準備 ---
# summary.jsonlを読み込む
df_list = []
with open("benchmark/output/summary_tmp.jsonl", "r") as f:
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

# 表示用にフォーマットを整える
summary_with_ratio["speedup"] = summary_with_ratio["speedup"].map("{:.2f}%".format)


summary_with_ratio

# %%
