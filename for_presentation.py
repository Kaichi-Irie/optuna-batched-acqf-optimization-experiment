# %%
# 最適化の初期値のリスト
starting_points = [...]
min_fs = []
for x0 in starting_points:
    # 初期値x0から最適化した時の極小値 min_f
    min_f = minimize(acqf, x0=x0)
    min_fs.append(min_f)
# 各最適化で得られた極小値min_fの最小値をとる
best_f = min(min_fs)
#%%
# ...existing code...


def optimize_acqf(
    acqf: BaseAcquisitionFunc,
    ...
) -> tuple[np.ndarray, float]:
    # ...省略...
    starting_points=np.array([...])
    best_f = math.inf
    for x0 in starting_points:
        x, f = local_search(acqf, x0,...)
        if f > best_f:
            best_x = x
            best_f = f

    return best_x, best_f


#%%
import numpy as np
# ...existing code...


def optimize_acqf(
    acqf: BaseAcquisitionFunc,
    ...
) -> tuple[np.ndarray, float]:
    # バッチ初期点をまとめて作成
    starting_points=np.array([...])
    # バッチでローカルサーチを実行
    xs, fs = local_search_batched(acqf, starting_points, ...)
    assert xs.shape == (batch_size, dimension)
    assert fs.shape == (batch_size,)

    # バッチ結果からベストを選択
    best_x = xs[np.argmax(fs)]
    best_f = float(np.max(fs))

    return best_x, best_f
