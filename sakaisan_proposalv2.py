# %%
import queue
import threading
import time

import numpy as np
import scipy.optimize as so

from benchmark_tensor_operations import TensorOperationsBenchmark

# --- 設定 ---
N_LOCAL_SEARCH = 10  # ワーカースレッドの数（＝並列実行する局所探索の数）
BATCH_SIZE = 5  # バッチ処理の最大サイズ
MASTER_TIMEOUT = 0.01  # マスターがバッチを確定するまでの待機時間 (秒)
N_TRIALS = 10000
DIMENSION = 100

BATCHED_TENSOR_OPS = TensorOperationsBenchmark(
    n_trials=N_TRIALS, dimension=DIMENSION, batch_size=BATCH_SIZE
)

SINGLE_TENSOR_OPS = TensorOperationsBenchmark(
    n_trials=N_TRIALS, dimension=DIMENSION, batch_size=1
)


WEIGHTS = np.random.rand(DIMENSION) * 40
WEIGHTS_batched = np.tile(WEIGHTS, (BATCH_SIZE, 1))  # shape: (BATCH_SIZE, DIMENSION)
# --- 共有オブジェクト ---
request_queue = queue.Queue()


def func_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    SINGLE_TENSOR_OPS.execute()
    x = x
    return np.sum(x**2), 2 * x


# --- 評価対象の関数（バッチ対応版） ---
def func_and_grad_batched(x_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    (N, D)形状のN個の点xをバッチで受け取り、
    N個の関数値と(N, D)形状の勾配を返す
    """
    # 実際の重い計算をシミュレート
    BATCHED_TENSOR_OPS.execute()
    x_batch = x_batch
    fvals = np.sum(x_batch**2, axis=1)
    grads = 2 * x_batch
    return fvals, grads


# --- マスタースレッドのロジック ---
def master_thread_logic():
    """リクエストを収集し、バッチ処理を実行して結果を返す"""
    print("🤖 [Master]  Master thread started.")
    while True:
        x_list = []
        result_containers = []
        try:
            # バッチサイズに達するかタイムアウトするまでリクエストを収集
            while len(x_list) < BATCH_SIZE:
                # 最初の1件はタイムアウトなしで待つ
                timeout = MASTER_TIMEOUT if x_list else None
                req, res_q = request_queue.get(timeout=timeout)

                if req is None:  # 終了シグナル
                    print("👋 [Master]  Master thread finished.")
                    return

                x_list.append(req)
                result_containers.append(res_q)
        except queue.Empty:
            pass  # タイムアウトしたので現在のバッチで処理を続行

        if x_list:
            print(f"⚙️  [Master]  Batch processing {len(x_list)} items...")
            x_batch = np.vstack(x_list)
            fvals, grads = func_and_grad_batched(x_batch)

            # 各ワーカーの専用キューに結果を返す
            for i, res_q in enumerate(result_containers):
                res_q.put((fvals[i], grads[i]))


# --- SciPyオプティマイザと非同期処理を繋ぐラッパークラス ---
class FuncAndGradWrapper:
    """
    fmin_l_bfgs_bに渡すためのcallableオブジェクト。
    内部でマスターとの非同期通信を行う。
    """

    def __init__(self, req_q: queue.Queue):
        self.request_queue = req_q
        self.result_queue = queue.Queue(maxsize=1)

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        # SciPyから呼ばれたら、マスターに評価を依頼
        self.request_queue.put((x, self.result_queue))

        # マスターからの結果を待つ
        fval, grad = self.result_queue.get()
        return fval, grad


# --- ワーカースレッドのロジック ---
def worker_thread_logic(worker_id: int, bounds: list, results_list: list):
    """一つの独立したL-BFGS-B最適化を実行する"""
    print(f"  👷 [Worker {worker_id}] Started.")

    # このワーカー（最適化セッション）専用のラッパーインスタンスを作成
    wrapped_func_and_grad = FuncAndGradWrapper(request_queue)

    min_x, min_f, info = so.fmin_l_bfgs_b(
        func=wrapped_func_and_grad,
        x0=STARTING_POINTS[worker_id],
        bounds=bounds,
        iprint=-1,  # 途中経過を非表示
    )

    print(f"  🎉 [Worker {worker_id}] Finished. f(x) = {min_f:.4f} ")
    results_list.append({"fval": min_f, "x": min_x, "info": info})


# %%
# --- メイン処理 ---
BOUNDS = [(-5.0, 5.0) for _ in range(DIMENSION)]
STARTING_POINTS = [
    np.random.uniform(BOUNDS[0][0], BOUNDS[0][1], size=len(BOUNDS))
    for _ in range(N_LOCAL_SEARCH)
]

final_results = []
# 1. マスタースレッドをデーモンとして起動
# (メインスレッドが終了したら自動的に終了する)
master = threading.Thread(target=master_thread_logic, daemon=True)
master.start()
start = time.time()

# 2. ワーカースレッドをN個起動
workers = []
for i in range(N_LOCAL_SEARCH):
    # results_listを渡して、各スレッドの結果を収集できるようにする
    worker = threading.Thread(
        target=worker_thread_logic, args=(i, BOUNDS, final_results)
    )
    workers.append(worker)
    worker.start()

# 3. すべてのワーカーの終了を待つ
for worker in workers:
    worker.join()

# 4. マスターに終了シグナルを送る
# (ワーカが全て終了したので、もうリクエストは来ない)
# デーモンスレッドなので厳密には不要ですが、明示的に終了させます。
request_queue.put((None, None))
master.join(timeout=1)
elapsed = time.time() - start

# 5. 全探索の中から最良の結果を見つける
if final_results:
    best_result = min(final_results, key=lambda r: r["fval"])
    print("\n" + "=" * 50)
    print("🚀 Overall Best Result of Parallel Optimization:")
    print(f"  Minimum function value: {best_result['fval']:.6f}")
    print(f"  Elapsed time: {elapsed:.2f} seconds")
    print(f"  Found at x: {best_result['x']}")
    print("=" * 50)

# %%
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
    print(f"  🎉 [Seq {i}] Finished. f(x) = {min_f:.4f} at x = {min_x}")
    if min_f < best_fval:
        best_fval = min_f
        best_x = min_x
elapsed = time.time() - start

print("\n" + "=" * 50)
print("🚀 Overall Best Result of Sequential Optimization:")
print(f"  Minimum function value: {best_fval:.6f}")
print(f"  Found at x: {best_x}")
print(f"  Elapsed time: {elapsed:.2f} seconds")
print("=" * 50)

# %%
