# %%

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar, rosen, rosen_der


# BFGSの1ステップ更新を行う関数 (変更なし)
def bfgs_update(B_k, s_k, y_k):
    y_k_s_k = y_k.T @ s_k
    if np.abs(y_k_s_k) < 1e-8:
        return B_k
    term1 = (y_k @ y_k.T) / y_k_s_k
    B_k_s_k = B_k @ s_k
    term2 = -(B_k_s_k @ B_k_s_k.T) / (s_k.T @ B_k_s_k)
    B_k_plus_1 = B_k + term1 + term2
    return B_k_plus_1


# Stacking用の目的関数
def stacked_rosen(X):
    """4次元ベクトルを受け取り、2つのRosenbrock関数の値の和を返す"""
    f1 = rosen(X[0:2])
    f2 = rosen(X[2:4])
    return f1 + f2


# --- シミュレーション設定 ---
N_ITERATIONS = 30  # シミュレーションする反復回数
# %%

# --- シナリオA: 個別最適化 (Separate Optimization) ---
print("--- シナリオA: 個別最適化 (厳密なラインサーチ) ---")
x1 = np.array([0.0, 0.0])
x2 = np.array([2.0, 2.0])
B1 = np.identity(2)
B2 = np.identity(2)

approx_conds1 = [np.linalg.cond(B1)]
approx_conds2 = [np.linalg.cond(B2)]


for i in range(N_ITERATIONS):
    # --- Problem 1 ---
    grad1_k = rosen_der(x1)
    p1 = -np.linalg.inv(B1) @ grad1_k
    # ラインサーチで最適なalphaを決定
    line_search_res1 = minimize_scalar(
        lambda alpha: rosen(x1 + alpha * p1), bounds=(0, 1), method="bounded"
    )
    alpha1 = line_search_res1.x
    x1_new = x1 + alpha1 * p1
    s1 = x1_new - x1
    grad1_k_plus_1 = rosen_der(x1_new)
    y1 = grad1_k_plus_1 - grad1_k
    B1 = bfgs_update(B1, s1, y1)
    x1 = x1_new
    approx_conds1.append(np.linalg.cond(B1))

    # --- Problem 2 ---
    grad2_k = rosen_der(x2)
    p2 = -np.linalg.inv(B2) @ grad2_k
    # ラインサーチで最適なalphaを決定
    line_search_res2 = minimize_scalar(
        lambda alpha: rosen(x2 + alpha * p2), bounds=(0, 1), method="bounded"
    )
    alpha2 = line_search_res2.x
    x2_new = x2 + alpha2 * p2
    s2 = x2_new - x2
    grad2_k_plus_1 = rosen_der(x2_new)
    y2 = grad2_k_plus_1 - grad2_k
    B2 = bfgs_update(B2, s2, y2)
    x2 = x2_new
    approx_conds2.append(np.linalg.cond(B2))

    print(f"Iter {i + 1}: Cond(B1)={approx_conds1[-1]:.2e}, Cond(B2)={approx_conds2[-1]:.2e}")
# %%

# --- シナリオB: Stackingによる最適化 ---
print("\n--- シナリオB: Stackingによる最適化 (厳密なラインサーチ) ---")
X_stacked = np.array([0.0, 0.0, 2.0, 2.0])
B_stacked = np.identity(4)

approx_conds_stacked = [np.linalg.cond(B_stacked)]

for i in range(N_ITERATIONS):
    grad_stacked_k = np.hstack([rosen_der(X_stacked[0:2]), rosen_der(X_stacked[2:4])])
    P = -np.linalg.inv(B_stacked) @ grad_stacked_k

    # ラインサーチで最適なalphaを決定
    line_search_res_stacked = minimize_scalar(
        lambda alpha: stacked_rosen(X_stacked + alpha * P), bounds=(0, 1), method="bounded"
    )
    alpha = line_search_res_stacked.x

    X_stacked_new = X_stacked + alpha * P
    S = X_stacked_new - X_stacked
    grad_stacked_k_plus_1 = np.hstack(
        [rosen_der(X_stacked_new[0:2]), rosen_der(X_stacked_new[2:4])]
    )
    Y = grad_stacked_k_plus_1 - grad_stacked_k

    B_stacked = bfgs_update(B_stacked, S, Y)
    X_stacked = X_stacked_new
    approx_conds_stacked.append(np.linalg.cond(B_stacked))
    print(f"Iter {i + 1}: Cond(B_stacked)={approx_conds_stacked[-1]:.2e}")

print("\n--- 最終的な近似ヘッセ行列 (Stacking) ---")
print(np.round(B_stacked, 2))
# %%

# --- 結果のプロット ---
plt.figure(figsize=(10, 6))
plt.plot(approx_conds1, marker="o", linestyle="--", label="Separate Opt. 1: Cond(B1)")
plt.plot(approx_conds2, marker="s", linestyle="--", label="Separate Opt. 2: Cond(B2)")
plt.plot(
    approx_conds_stacked,
    marker="x",
    linestyle="-",
    label="Stacked Opt.: Cond(B_stacked)",
    color="red",
)
plt.yscale("log")
plt.title("Condition Number of Approximate Hessian (with Exact Line Search)")
plt.xlabel("Iteration")
plt.ylabel("Condition Number (log scale)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("BFGS_condition_number_exact_line_search.png", dpi=300)
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar, rosen_der, rosen_hess

# # Rosenbrock関数の真のヘッセ行列を計算する関数
# def rosen_hess(x):
#     """Calculates the true Hessian matrix of the 2D Rosenbrock function."""
#     x_val, y_val = x[0], x[1]
#     h11 = 2.0 - 400.0 * (y_val - x_val**2) + 800.0 * x_val**2
#     h12 = -400.0 * x_val
#     h21 = -400.0 * x_val
#     h22 = 200.0
#     return np.array([[h11, h12], [h21, h22]])


# BFGSの1ステップ更新を行う関数 (変更なし)
def bfgs_update(B_k, s_k, y_k):
    y_k_s_k = y_k.T @ s_k
    if np.abs(y_k_s_k) < 1e-8:
        return B_k
    term1 = (y_k @ y_k.T) / y_k_s_k
    B_k_s_k = B_k @ s_k
    term2 = -(B_k_s_k @ B_k_s_k.T) / (s_k.T @ B_k_s_k)
    return B_k + term1 + term2


def stacked_rosen(X):
    return rosen(X[0:2]) + rosen(X[2:4])


# --- シミュレーション設定 ---
N_ITERATIONS = 50

# %%
# --- シナリオA: 個別最適化 ---
print("--- シナリオA: 個別最適化 (真のヘッセ行列と比較) ---")
x1, x2 = np.array([0.0, 0.0]), np.array([2.0, 2.0])
B1, B2 = np.identity(2), np.identity(2)

# 近似値と真値の両方の条件数を記録するリスト
conds1_bfgs, conds2_bfgs = [np.linalg.cond(B1)], [np.linalg.cond(B2)]
conds1_true, conds2_true = (
    [np.linalg.cond(rosen_hess(x1))],
    [np.linalg.cond(rosen_hess(x2))],
)

for i in range(N_ITERATIONS):
    # Problem 1
    p1 = -np.linalg.inv(B1) @ rosen_der(x1)
    alpha1 = minimize_scalar(
        lambda alpha: rosen(x1 + alpha * p1), bounds=(0, 1), method="bounded"
    ).x
    x1_new = x1 + alpha1 * p1
    s1, y1 = x1_new - x1, rosen_der(x1_new) - rosen_der(x1)
    B1 = bfgs_update(B1, s1, y1)
    x1 = x1_new
    conds1_bfgs.append(np.linalg.cond(B1))
    conds1_true.append(np.linalg.cond(rosen_hess(x1)))

    # Problem 2
    p2 = -np.linalg.inv(B2) @ rosen_der(x2)
    alpha2 = minimize_scalar(
        lambda alpha: rosen(x2 + alpha * p2), bounds=(0, 1), method="bounded"
    ).x
    x2_new = x2 + alpha2 * p2
    s2, y2 = x2_new - x2, rosen_der(x2_new) - rosen_der(x2)
    B2 = bfgs_update(B2, s2, y2)
    x2 = x2_new
    conds2_bfgs.append(np.linalg.cond(B2))
    conds2_true.append(np.linalg.cond(rosen_hess(x2)))
# %%

# --- シナリオB: Stackingによる最適化 ---
print("\n--- シナリオB: Stackingによる最適化 (真のヘッセ行列と比較) ---")
X_stacked = np.array([0.0, 0.0, 2.0, 2.0])
B_stacked = np.identity(4)

conds_stacked_bfgs = [np.linalg.cond(B_stacked)]
# 真のブロック対角ヘッセ行列の条件数
h1_true = rosen_hess(X_stacked[0:2])
h2_true = rosen_hess(X_stacked[2:4])
conds_stacked_true = [max(np.linalg.cond(h1_true), np.linalg.cond(h2_true))]


for i in range(N_ITERATIONS):
    grad_stacked = np.hstack([rosen_der(X_stacked[0:2]), rosen_der(X_stacked[2:4])])
    P = -np.linalg.inv(B_stacked) @ grad_stacked
    alpha = minimize_scalar(
        lambda a: stacked_rosen(X_stacked + a * P), bounds=(0, 1), method="bounded"
    ).x
    X_stacked_new = X_stacked + alpha * P
    S, Y = (
        X_stacked_new - X_stacked,
        np.hstack([rosen_der(X_stacked_new[0:2]), rosen_der(X_stacked_new[2:4])]) - grad_stacked,
    )
    B_stacked = bfgs_update(B_stacked, S, Y)
    X_stacked = X_stacked_new

    conds_stacked_bfgs.append(np.linalg.cond(B_stacked))
    h1_true = rosen_hess(X_stacked[0:2])
    h2_true = rosen_hess(X_stacked[2:4])
    conds_stacked_true.append(max(np.linalg.cond(h1_true), np.linalg.cond(h2_true)))

    print(
        f"Iter {i + 1}: Cond(B_stacked)={conds_stacked_bfgs[-1]:.2e}, Cond(H_true)={conds_stacked_true[-1]:.2e}"
    )
# %%

# --- 結果のプロット ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 7))

# シナリオA
ax.plot(conds1_true, color="blue", linestyle="-", label="True Hessian 1: Cond(H1)")
ax.plot(conds1_bfgs, color="blue", linestyle="--", marker="o", label="BFGS Approx. 1: Cond(B1)")
ax.plot(
    conds2_true, color="green", linestyle="-", label="True Hessian 2: Cond(H2)"
)  # Problem2は変化が少ないため省略可
ax.plot(conds2_bfgs, color="green", linestyle="--", marker="s", label="BFGS Approx. 2: Cond(B2)")

# シナリオB
ax.plot(
    conds_stacked_true,
    color="black",
    linestyle="-",
    linewidth=2,
    label="True Stacked Hessian: max(Cond(Hi))",
)
ax.plot(
    conds_stacked_bfgs,
    color="red",
    linestyle="--",
    marker="x",
    label="BFGS Stacked Approx.: Cond(B_stacked)",
)

ax.set_yscale("log")
ax.set_title("Condition Number: BFGS Approximation vs. True Hessian", fontsize=16)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Condition Number (log scale)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# %%
