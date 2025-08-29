import threading
import time
from typing import Any

import numpy as np
import scipy.optimize as so

from benchmark_tensor_operations import TensorOperationsBenchmark


class Canceled(Exception):
    pass


def fmin_l_bfgs_b_batched(
    func_and_grad_batch, xs0: np.ndarray, bounds: list[Any]
) -> list[Any]:
    n, d = xs0.shape
    assert len(bounds) == n

    cond_req_or_finished = [threading.Condition() for _ in range(n)]
    cond_evaluated_or_terminated = [threading.Condition() for _ in range(n)]

    eval_requests: list[np.ndarray | None]
    eval_requests = [None] * n

    eval_responses: list[tuple[float, np.ndarray] | None]
    eval_responses = [None] * n

    worker_finished: list[bool]
    worker_finished = [False] * n

    worker_results: list[Any]
    worker_results = [None] * n

    terminated = False

    def worker_main(i: int) -> None:
        def func(x: np.ndarray) -> tuple[float, np.ndarray]:
            assert x.shape == (d,)

            # 関数評価をリクエスト
            with cond_req_or_finished[i]:
                assert eval_requests[i] is None
                eval_requests[i] = x
                cond_req_or_finished[i].notify()

            # 関数評価結果または処理の中断を待つ
            with cond_evaluated_or_terminated[i]:
                if not terminated and eval_responses[i] is None:
                    cond_evaluated_or_terminated[i].wait()
                if terminated:
                    raise Canceled()
                assert eval_responses[i] is not None
                ret, eval_responses[i] = eval_responses[i], None

            assert ret is not None
            y, grad = ret
            assert grad.shape == (d,)
            return ret

        try:
            worker_results[i] = so.fmin_l_bfgs_b(
                func=func,
                x0=xs0[i],
                bounds=bounds[i],
                iprint=-1,  # 途中経過を非表示
            )
        except Canceled:
            # 意図した終了なので、スタックトレースを表示させないよう、例外を握り潰す
            pass
        finally:
            # ワーカーの処理終了を通知
            with cond_req_or_finished[i]:
                worker_finished[i] = True
                cond_req_or_finished[i].notify()

    try:
        # ワーカー起動
        workers = []
        for i in range(n):
            t = threading.Thread(
                target=worker_main, args=[i], name=f"worker {i}", daemon=True
            )
            workers.append(t)
            t.start()

        # ワーカーによる関数評価リクエストを集めて、バッチ評価するループ
        while True:
            # リクエスト収集
            xs = []
            indices = []
            for i in range(n):
                with cond_req_or_finished[i]:
                    if not worker_finished[i] and eval_requests[i] is None:
                        cond_req_or_finished[i].wait()
                    if worker_finished[i]:
                        continue
                    assert eval_requests[i] is not None
                    x, eval_requests[i] = eval_requests[i], None
                    assert x is not None
                    assert x.shape == (d,)
                    xs.append(x)
                    indices.append(i)

            # 全てのワーカーが終了済みなので、処理を終了
            if len(xs) == 0:
                break

            # バッチ評価
            ys, grads = func_and_grad_batch(np.stack(xs))
            assert ys.shape == (len(indices),)
            assert grads.shape == (len(indices), d)

            # 結果の分配
            for j, i in enumerate(indices):
                with cond_evaluated_or_terminated[i]:
                    assert eval_responses[i] is None
                    eval_responses[i] = (ys[j], grads[j])
                    cond_evaluated_or_terminated[i].notify()
    finally:
        # 後処理
        terminated = True
        for i, worker in enumerate(workers):
            # 例外による終了の場合には、ワーカーが関数の評価結果待ちの可能性があるので、中断を通知
            with cond_evaluated_or_terminated[i]:
                cond_evaluated_or_terminated[i].notify()
            worker.join()

    return worker_results


N_LOCAL_SEARCH = 20
DIMENSION = 100
BOUNDS = [(-5.0, 5.0) for _ in range(DIMENSION)]
STARTING_POINTS = np.array(
    [
        np.random.uniform(BOUNDS[0][0], BOUNDS[0][1], size=len(BOUNDS))
        for _ in range(N_LOCAL_SEARCH)
    ]
)
N_TRIALS = 500
T_OPS_BATCHED = TensorOperationsBenchmark(
    n_trials=N_TRIALS, dimension=DIMENSION, batch_size=N_LOCAL_SEARCH
)
T_OPS_SINGLE = TensorOperationsBenchmark(
    n_trials=N_TRIALS, dimension=DIMENSION, batch_size=1
)

WEIGHTS = (np.random.rand(DIMENSION) - 0.5) * 9.5


def func_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    T_OPS_SINGLE.execute()
    x = x * WEIGHTS
    return np.sum(x**2), 2 * x


# --- 評価対象の関数（バッチ対応版） ---
def func_and_grad_batched(x_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    (N, D)形状のN個の点xをバッチで受け取り、
    N個の関数値と(N, D)形状の勾配を返す
    """
    # 実際の重い計算をシミュレート
    x_batch = x_batch * WEIGHTS
    fvals = np.sum(x_batch**2, axis=1)
    grads = 2 * x_batch

    T_OPS_BATCHED.execute()
    return fvals, grads


# --- メイン処理 ---


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
    print(f"  Elapsed time: {elapsed:.4e} seconds")
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
    # print(f"  Found at x: {best_x}")
    print(f"  Elapsed time: {elapsed:.4e} seconds")
    print("=" * 50)
