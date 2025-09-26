# %%
import time
from typing import Any

import numpy as np
import scipy.optimize as so
from greenlet import greenlet

from benchmark_tensor_operations import TensorOperationsBenchmark

np.random.seed(42)


def fmin_l_bfgs_b_batched(func_and_grad_batch, xs0: np.ndarray, bounds: list[Any]) -> list[Any]:
    n, d = xs0.shape
    assert len(bounds) == n

    results: list[Any]
    results = [None] * n

    def run(i: int) -> None:
        def func(x: np.ndarray) -> tuple[float, np.ndarray]:
            assert x.shape == (d,)
            y, grad = greenlet.getcurrent().parent.switch(x)
            assert grad.shape == (d,)
            return y, grad

        results[i] = so.fmin_l_bfgs_b(
            func=func,
            x0=xs0[i],
            bounds=bounds[i],
            iprint=-1,  # 途中経過を非表示
        )

    # 最初のリクエスト収集
    xs = []
    greenlets = []
    for i in range(n):
        gl = greenlet(run)
        x = gl.switch(i)
        if x is None:  # そのgreenletは終了済みで次のリクエストはなし
            continue
        assert x.shape == (d,)
        xs.append(x)
        greenlets.append(gl)

    while len(xs) > 0:
        # バッチ評価
        ys, grads = func_and_grad_batch(np.stack(xs))
        assert ys.shape == (len(xs),)
        assert grads.shape == (len(xs), d)

        # 結果の分配と次のリクエスト収集
        xs_next = []
        greenlets_next = []
        for j, gl in enumerate(greenlets):
            x = gl.switch((ys[j], grads[j]))
            if x is None:  # そのgreenletは終了済みで次のリクエストはなし
                continue
            assert x.shape == (d,)
            xs_next.append(x)
            greenlets_next.append(gl)
        xs = xs_next
        greenlets = greenlets_next

    return results


# %%
D = 50  # 次元
SCALE = np.random.rand(D) * 10 + 1.0  # 各次元のスケール

# --- メイン処理 ---
N_LOCAL_SEARCH = 10
DIMENSION = D
BOUNDS = [(-5.0, 5.0) for _ in range(DIMENSION)]
STARTING_POINTS = np.array(
    [
        np.random.uniform(BOUNDS[0][0], BOUNDS[0][1], size=len(BOUNDS))
        for _ in range(N_LOCAL_SEARCH)
    ]
)
N_TRIALS = 300

t = TensorOperationsBenchmark(n_trials=N_TRIALS, dimension=DIMENSION, batch_size=N_LOCAL_SEARCH)
t_batched = TensorOperationsBenchmark(n_trials=N_TRIALS, dimension=DIMENSION, batch_size=1)


# %%
def func_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    # 差が分かりやすいように人為的にsleepを入れる
    t.execute()
    x = x * SCALE  # スケールをかける
    return np.sum(x**2), 2 * x


# --- 評価対象の関数（バッチ対応版） ---

# %%


def func_and_grad_batched(x_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    (N, D)形状のN個の点xをバッチで受け取り、
    N個の関数値と(N, D)形状の勾配を返す
    """
    # 実際の重い計算をシミュレート
    x_batch = x_batch * SCALE  # スケールをかける
    fvals = np.sum(x_batch**2, axis=1)
    grads = 2 * x_batch

    # 差が分かりやすいように人為的にsleepを入れる
    # バッチで計算すると90%の時間で計算できると仮定
    # time.sleep(0.1 * len(x_batch) * 0.6)
    t_batched.execute()
    return fvals, grads


fvals, grads = func_and_grad_batched(np.random.rand(10, D))  # 動作確認
print(f"fvals: {fvals}")
print(f"grads shape: {grads.shape}")
# %%


if __name__ == "__main__":
    # Optimize Batch
    start = time.time()
    final_results = fmin_l_bfgs_b_batched(
        func_and_grad_batched, STARTING_POINTS, [BOUNDS] * N_LOCAL_SEARCH
    )
    elapsed = time.time() - start

    best_result = min(final_results, key=lambda r: r[1])
    print("\n" + "=" * 50)
    print("🚀 Overall Best Result:")
    print(f"  Minimum function value: {best_result[1]:.6f}")
    print(f"  Elapsed time: {elapsed:3e} seconds")
    print(f"  Found at x: {best_result[0]}")
    print("=" * 50)

    # Optimize Sequentially
    best_fval = float("inf")
    best_x = None
    start = time.time()
    for i in range(N_LOCAL_SEARCH):
        min_x, min_f, info = so.fmin_l_bfgs_b(
            func=func_and_grad,
            x0=STARTING_POINTS[i],
            bounds=BOUNDS,
            iprint=-1,  # 途中経過を非表示
        )
        # cast np.ndarray to float for
        min_f = min_f.item()
        print(f"  🎉 [Seq {i}] Finished. f(x) = {min_f:.4f}")
        if min_f < best_fval:
            best_fval = min_f
            best_x = min_x
    elapsed = time.time() - start

    print("\n" + "=" * 50)
    print("🚀 Overall Best Result:")
    print(f"  Minimum function value: {best_fval:.6f}")
    print(f"  Found at x: {best_x}")
    print(f"  Elapsed time: {elapsed:3e} seconds")
    print("=" * 50)
