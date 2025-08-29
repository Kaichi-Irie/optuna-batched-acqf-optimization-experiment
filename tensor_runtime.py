# %%
from timeit import timeit

import matplotlib.pyplot as plt
import torch

# 実験用のデータサイズを定義
batch_size = 10
dimensions = [10**i for i in range(2, 5)] + [5 * 10**i for i in range(2, 4)]
dimensions.sort()


# 各手法の実行時間を格納するリスト
runtimes_numpy_vectorized = []
runtimes_numpy_loop = []
runtimes_torch_vectorized = []
runtimes_torch_loop = []
num_iters = 10
# %%

# PyTorchでの計算時間測定（CPUを使用）
for dimension in dimensions:
    a = torch.rand(batch_size, dimension)
    b = torch.rand(dimension, dimension)
    runtime_torch_vectorized = timeit(lambda: a @ b, number=num_iters)
    runtimes_torch_vectorized.append(runtime_torch_vectorized)

for dimension in dimensions:
    a = torch.rand(batch_size, dimension)
    b = torch.rand(dimension, dimension)
    runtime_torch_loop = timeit(
        "for i in range(batch_size): a[i] @ b",
        globals={"a": a, "b": b, "batch_size": batch_size},
        number=num_iters,
    )
    runtimes_torch_loop.append(runtime_torch_loop)
print("PyTorchの計測完了")
print(f"PyTorch (Vectorized): {runtimes_torch_vectorized}")
print(f"PyTorch (Loop): {runtimes_torch_loop}")
# %%
# グラフの作成
plt.rcParams["font.size"] = 20
fig, ax = plt.subplots(figsize=(12, 9))
ax.plot(
    dimensions,
    runtimes_torch_vectorized,
    marker="s",
    lw=4,
    ms=15,
    label="PyTorch (Batched)",
    color="darkred",
)
ax.plot(
    dimensions,
    runtimes_torch_loop,
    marker="s",
    lw=4,
    ms=15,
    label="PyTorch (Python For Loop)",
    color="gray",
)


# グラフの装飾
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Tensor/Array Size", fontsize=20)
ax.set_ylabel("Runtime (sec)", fontsize=20)
ax.set_title("Batched vs Python For Loop", fontsize=20)
ax.grid(True, which="minor", ls=":", color="gray")
ax.grid(True, which="major", color="black")
ax.legend()


# グラフをファイルとして保存
plt.savefig("vectorization_performance.png", bbox_inches="tight")

print("グラフが 'vectorization_performance.png' として保存されました。")
print("\n--- 実行時間データ ---")
print(f"Sizes: {dimensions}")
print(f"NumPy (Vectorized): {runtimes_numpy_vectorized}")
print(f"NumPy (Loop): {runtimes_numpy_loop}")
print(f"PyTorch (Vectorized): {runtimes_torch_vectorized}")
print(f"PyTorch (Loop): {runtimes_torch_loop}")
