from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from optuna._gp.scipy_blas_thread_patch import (
    single_blas_thread_if_scipy_v1_15_or_newer,
)
from optuna.logging import get_logger

if TYPE_CHECKING:
    import scipy.optimize as so
    from optuna._gp.acqf import BaseAcquisitionFunc
else:
    from optuna import _LazyImport

    so = _LazyImport("scipy.optimize")

_logger = get_logger(__name__)


def _gradient_ascent(
    acqf: BaseAcquisitionFunc,
    initial_params: np.ndarray,
    initial_fval: float,
    continuous_indices: np.ndarray,
    lengthscales: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, float, bool]:
    """
    This function optimizes the acquisition function using preconditioning.
    Preconditioning equalizes the variances caused by each parameter and
    speeds up the convergence.

    In Optuna, acquisition functions use Matern 5/2 kernel, which is a function of `x / l`
    where `x` is `normalized_params` and `l` is the corresponding lengthscales.
    Then acquisition functions are a function of `x / l`, i.e. `f(x / l)`.
    As `l` has different values for each param, it makes the function ill-conditioned.
    By transforming `x / l` to `zl / l = z`, the function becomes `f(z)` and has
    equal variances w.r.t. `z`.
    So optimization w.r.t. `z` instead of `x` is the preconditioning here and
    speeds up the convergence.
    As the domain of `x` is [0, 1], that of `z` becomes [0, 1/l].
    """
    if len(continuous_indices) == 0:
        return initial_params, initial_fval, False
    normalized_params = initial_params.copy()

    def negative_acqf_with_grad(scaled_x: np.ndarray) -> tuple[float, np.ndarray]:
        # Scale back to the original domain, i.e. [0, 1], from [0, 1/s].
        normalized_params[continuous_indices] = scaled_x * lengthscales
        (fval, grad) = acqf.eval_acqf_with_grad(normalized_params)
        # Flip sign because scipy minimizes functions.
        # Let the scaled acqf be g(x) and the acqf be f(sx), then dg/dx = df/dx * s.
        return -fval, -grad[continuous_indices] * lengthscales

    with single_blas_thread_if_scipy_v1_15_or_newer():
        scaled_cont_x_opt, neg_fval_opt, info = so.fmin_l_bfgs_b(
            func=negative_acqf_with_grad,
            x0=normalized_params[continuous_indices] / lengthscales,
            bounds=[(0, 1 / s) for s in lengthscales],
            pgtol=math.sqrt(tol),
            maxiter=200,
        )

    if -neg_fval_opt > initial_fval and info["nit"] > 0:  # Improved.
        # `nit` is the number of iterations.
        normalized_params[continuous_indices] = scaled_cont_x_opt * lengthscales
        return normalized_params, -neg_fval_opt, True

    return initial_params, initial_fval, False  # No improvement.


def local_search_mixed(
    acqf: BaseAcquisitionFunc,
    initial_normalized_params: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> tuple[np.ndarray, float]:
    continuous_indices = acqf.search_space.continuous_indices
    if len(continuous_indices) != len(initial_normalized_params):
        raise ValueError("Only continuous optimization is supported.")
    # This is a technique for speeding up optimization.
    # We use an isotropic kernel, so scaling the gradient will make
    # the hessian better-conditioned.
    # NOTE: Ideally, separating lengthscales should be used for the constraint functions,
    # but for simplicity, the ones from the objective function are being reused.
    # TODO(kAIto47802): Think of a better way to handle this.
    lengthscales = acqf.length_scales
    best_normalized_params = initial_normalized_params.copy()
    best_fval = float(acqf.eval_acqf_no_grad(best_normalized_params))

    (best_normalized_params, best_fval, _) = _gradient_ascent(
        acqf,
        best_normalized_params,
        best_fval,
        continuous_indices,
        lengthscales,
        tol,
    )
    return best_normalized_params, best_fval


def run_local_search_wrapper_with_kwargs(
    acqf: BaseAcquisitionFunc, initial_normalized_params: np.ndarray, tol: float = 1e-4
):
    return local_search_mixed(acqf, initial_normalized_params, tol=tol)


def optimize_acqf_mixed(
    acqf: BaseAcquisitionFunc,
    *,
    worker_pool,
    warmstart_normalized_params_array: np.ndarray | None = None,
    n_preliminary_samples: int = 2048,
    n_local_search: int = 10,
    tol: float = 1e-4,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, float]:
    if worker_pool is None:
        raise ValueError("Worker pool must be provided for multiprocessing.")

    rng = rng or np.random.RandomState()

    if warmstart_normalized_params_array is None:
        warmstart_normalized_params_array = np.empty((0, acqf.search_space.dim))

    assert len(warmstart_normalized_params_array) <= n_local_search - 1, (
        "We must choose at least 1 best sampled point + given_initial_xs as start points."
    )

    sampled_xs = acqf.search_space.sample_normalized_params(
        n_preliminary_samples, rng=rng
    )

    # Evaluate all values at initial samples
    f_vals = acqf.eval_acqf_no_grad(sampled_xs)
    assert isinstance(f_vals, np.ndarray)

    max_i = np.argmax(f_vals)

    # TODO(nabenabe): Benchmark the BoTorch roulette selection as well.
    # https://github.com/pytorch/botorch/blob/v0.14.0/botorch/optim/initializers.py#L942
    # We use a modified roulette wheel selection to pick the initial param for each local search.
    probs = np.exp(f_vals - f_vals[max_i])
    probs[max_i] = 0.0  # We already picked the best param, so remove it from roulette.
    probs /= probs.sum()
    n_non_zero_probs_improvement = int(np.count_nonzero(probs > 0.0))
    # n_additional_warmstart becomes smaller when study starts to converge.
    n_additional_warmstart = min(
        n_local_search - len(warmstart_normalized_params_array) - 1,
        n_non_zero_probs_improvement,
    )
    if n_additional_warmstart == n_non_zero_probs_improvement:
        _logger.warning(
            "Study already converged, so the number of local search is reduced."
        )
    chosen_idxs = np.array([max_i])
    if n_additional_warmstart > 0:
        additional_idxs = rng.choice(
            len(sampled_xs), size=n_additional_warmstart, replace=False, p=probs
        )
        chosen_idxs = np.append(chosen_idxs, additional_idxs)

    best_x = sampled_xs[max_i, :]
    best_f = float(f_vals[max_i])

    # If the worker pool is available, we run local search in parallel.
    results = worker_pool.starmap(
        run_local_search_wrapper_with_kwargs,
        [
            (acqf, x_warmstart, tol)
            for x_warmstart in np.vstack(
                [sampled_xs[chosen_idxs, :], warmstart_normalized_params_array]
            )
        ],
    )
    for x, f in results:
        if f > best_f:
            best_x = x
            best_f = f
    return best_x, best_f
