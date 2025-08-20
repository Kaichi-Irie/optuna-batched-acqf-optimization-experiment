from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from optuna._gp.scipy_blas_thread_patch import (
    single_blas_thread_if_scipy_v1_15_or_newer,
)
from optuna.logging import get_logger

from batched_lbfgsb import batched_lbfgsb

if TYPE_CHECKING:
    import scipy.optimize as so
    from optuna._gp.acqf import BaseAcquisitionFunc
else:
    from optuna import _LazyImport

    so = _LazyImport("scipy.optimize")

_logger = get_logger(__name__)


def _gradient_ascent(
    acqf: BaseAcquisitionFunc,
    initial_params_list: np.ndarray,  # (B,D)
    initial_fvals: np.ndarray,  # (B,)
    continuous_indices: np.ndarray,
    lengthscales: np.ndarray,  # (D,)
    tol: float,
) -> tuple[np.ndarray, np.ndarray, bool]:  # ((B,D), (B,), bool)
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
        return initial_params_list, initial_fvals, False
    batch_size, dimension = initial_params_list.shape
    if len(continuous_indices) != dimension:
        raise ValueError("Incompatible continuous indices.")

    normalized_params_buffer = initial_params_list.copy()
    assert normalized_params_buffer.shape == (batch_size, dimension)

    def negative_acqf_with_grad(scaled_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        (b,D) -> ((b,), (b,D))
        b corresponds to the number of batches that have not yet converged.
        """
        # Scale back to the original domain, i.e. [0, 1], from [0, 1/s].
        not_converged_batch_size = len(scaled_x)
        assert scaled_x.shape == (not_converged_batch_size, dimension), (
            f"Expected (b,D), got {scaled_x.shape}."
        )
        normalized_params_buffer[:not_converged_batch_size, continuous_indices] = (
            scaled_x * lengthscales
        )
        # (fvals, grads) = acqf.eval_acqf_with_grad(normalized_params)
        x_tensor = torch.from_numpy(
            normalized_params_buffer[:not_converged_batch_size]
        ).requires_grad_(True)
        assert x_tensor.shape == (not_converged_batch_size, dimension)
        fvals = acqf.eval_acqf(x_tensor)
        fvals.sum().backward()
        grads = x_tensor.grad.detach().numpy()  # type: ignore
        fvals = fvals.detach().numpy()
        # Flip sign because scipy minimizes functions.
        # Let the scaled acqf be g(x) and the acqf be f(sx), then dg/dx = df/dx * s.
        assert fvals.shape == (not_converged_batch_size,)
        assert grads.shape == (not_converged_batch_size, dimension)
        return -fvals, -grads[:, continuous_indices] * lengthscales

    with single_blas_thread_if_scipy_v1_15_or_newer():
        # x0: (B,D) - flatten -> (B*D,)
        x0 = normalized_params_buffer[:, continuous_indices] / lengthscales
        assert lengthscales.shape == (dimension,)
        assert len(continuous_indices) == dimension
        assert normalized_params_buffer.shape == (batch_size, dimension)
        assert x0.shape == (batch_size, dimension)
        # individual_bounds = [(0, 1 / s) for s in lengthscales]  # (D,2)
        # make individual bounds numpy array
        bounds = np.array([(0, 1 / s) for s in lengthscales])  # (D, 2)
        # TODO
        scaled_cont_x_opts, neg_fval_opts, infos = batched_lbfgsb(
            func_and_grad=negative_acqf_with_grad,  # type: ignore
            bounds=bounds,
            x0=x0,
            max_iters=200,
            pgtol=math.sqrt(tol),
        )
        assert scaled_cont_x_opts.shape == (batch_size, dimension)
        assert neg_fval_opts.shape == (batch_size,)
        # scaled_cont_x_opts = scaled_cont_x_opts.reshape(batch_size, dimension)

    normalized_params_buffer[:, continuous_indices] = (
        scaled_cont_x_opts * lengthscales
    )  # (B,D)
    return normalized_params_buffer, -neg_fval_opts, True


def local_search_mixed_stacked(
    acqf: BaseAcquisitionFunc,
    initial_normalized_params_list: np.ndarray,  # (D,) -> (B,D)
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    xs: (B,D)
    fs: (B,)
    """
    continuous_indices = acqf.search_space.continuous_indices
    # assert initial_normalized_params_list.ndim == 2
    # if len(continuous_indices) != len(initial_normalized_params_list):
    #     raise ValueError("Only continuous optimization is supported.")

    # This is a technique for speeding up optimization.
    # We use an isotropic kernel, so scaling the gradient will make
    # the hessian better-conditioned.
    # NOTE: Ideally, separating lengthscales should be used for the constraint functions,
    # but for simplicity, the ones from the objective function are being reused.
    # TODO(kAIto47802): Think of a better way to handle this.
    batch_size, dimension = initial_normalized_params_list.shape
    lengthscales = acqf.length_scales  # (D,)
    assert lengthscales.shape == (dimension,)
    best_normalized_params_list = initial_normalized_params_list.copy()  # (D,) -> (B,D)
    assert best_normalized_params_list.shape == (batch_size, dimension)
    best_fvals = np.array(
        [float(acqf.eval_acqf_no_grad(p)) for p in best_normalized_params_list]
    )

    (best_normalized_params_list, best_fvals, _) = _gradient_ascent(
        acqf,
        best_normalized_params_list,
        best_fvals,
        continuous_indices,
        lengthscales,
        tol,
    )
    return best_normalized_params_list, best_fvals


def optimize_acqf_mixed(
    acqf: BaseAcquisitionFunc,
    *,
    warmstart_normalized_params_array: np.ndarray | None = None,
    n_preliminary_samples: int = 2048,
    n_local_search: int = 10,
    tol: float = 1e-4,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, float]:
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
    # sampled_xs: (2048,D)
    best_x = sampled_xs[max_i, :]
    best_f = float(f_vals[max_i])

    dimension = sampled_xs.shape[1]

    batch_size = n_local_search

    # x_warmstarts: (B,D)
    x_warmstarts = np.vstack(
        [sampled_xs[chosen_idxs, :], warmstart_normalized_params_array]
    )
    assert x_warmstarts.shape == (batch_size, dimension)
    # xs: (B,D)
    xs, fs = local_search_mixed_stacked(acqf, x_warmstarts, tol=tol)
    assert xs.shape == (batch_size, dimension)
    assert fs.shape == (batch_size,)
    for x, f in zip(xs, fs):
        if f > best_f:
            best_x = x
            best_f = f
    return best_x, best_f
