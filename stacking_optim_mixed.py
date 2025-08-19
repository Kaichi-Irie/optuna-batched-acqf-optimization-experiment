from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from optuna._gp.scipy_blas_thread_patch import (
    single_blas_thread_if_scipy_v1_15_or_newer,
)
from optuna.logging import get_logger

from acqf_wrapper import AcqfWrapper

if TYPE_CHECKING:
    import scipy.optimize as so
    from optuna._gp.acqf import BaseAcquisitionFunc
else:
    from optuna import _LazyImport

    so = _LazyImport("scipy.optimize")

_logger = get_logger(__name__)


def _gradient_ascent(
    acqf_wrapper: AcqfWrapper,
    initial_params: np.ndarray,  # (B,D)
    initial_fval: float,  #  (B,)
    continuous_indices: np.ndarray,
    lengthscales: np.ndarray,  # (D,)
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
    batch_size, dimension = acqf_wrapper.batch_size, acqf_wrapper.dimension
    if len(continuous_indices) != dimension:
        raise ValueError("Incompatible continuous indices.")

    normalized_params = initial_params.copy()
    assert normalized_params.shape == (batch_size, dimension)

    def negative_acqf_with_grad(scaled_x: np.ndarray) -> tuple[float, np.ndarray]:
        """
        scaled_x: (B,D)
        ->
        f: (1,) # Sum
        grad: (B*D,) # Flattened
        """
        scaled_x = scaled_x.reshape(batch_size, dimension)
        # Scale back to the original domain, i.e. [0, 1], from [0, 1/s].
        # (B,D) = (B,D) * (D,)
        normalized_params[:, continuous_indices] = scaled_x * lengthscales

        # (fval, grad) = acqf.eval_acqf_with_grad(normalized_params)  # (1,), (D,)

        # fvals: (B,), x_tensor: (B,D)
        fvals, x_tensor = acqf_wrapper.eval_acqf_from_numpy(normalized_params)
        # fval: (1,)
        fval = fvals.sum()
        fval.backward()  # type: ignore
        # grad: (B,D)
        grad = x_tensor.grad.detach().numpy()  # type: ignore
        assert grad.shape == (batch_size, dimension)
        # Flip sign because scipy minimizes functions.
        # Let the scaled acqf be g(x) and the acqf be f(sx), then dg/dx = df/dx * s.
        grad = -grad[:, continuous_indices] * lengthscales
        assert grad.shape == (batch_size, dimension)
        grad = grad.ravel()  # (B*D,)
        assert grad.shape == (batch_size * dimension,)
        return -fval.item(), grad

    with single_blas_thread_if_scipy_v1_15_or_newer():
        # x0: (B,D) - flatten -> (B*D,)
        x0 = normalized_params[:, continuous_indices] / lengthscales
        assert lengthscales.shape == (dimension,)
        assert len(continuous_indices) == dimension
        assert normalized_params.shape == (batch_size, dimension)
        assert x0.shape == (batch_size, dimension)
        # individual_bounds = [(0, 1 / s) for s in lengthscales]  # (D,2)
        # make individual bounds numpy array
        individual_bounds = np.array([(0, 1 / s) for s in lengthscales])
        assert individual_bounds.shape == (dimension, 2)
        bounds = np.tile(individual_bounds, (batch_size, 1))  # (B*D,2)
        assert bounds.shape == (batch_size * dimension, 2)
        # bounds = [tuple(b) for b in bounds]
        scaled_cont_x_opt, neg_fval_opt, info = so.fmin_l_bfgs_b(
            func=negative_acqf_with_grad,
            x0=x0.flatten(),
            bounds=bounds.tolist(),
            pgtol=math.sqrt(tol),
            maxiter=200,
        )
        # reshape from (B*D,) to (B,D)
        scaled_cont_x_opt = scaled_cont_x_opt.reshape(batch_size, dimension)

    if -neg_fval_opt > initial_fval and info["nit"] > 0:  # Improved.
        # `nit` is the number of iterations.
        normalized_params[:, continuous_indices] = (
            scaled_cont_x_opt * lengthscales
        )  # (B,D)
        return normalized_params, -neg_fval_opt, True

    return initial_params, initial_fval, False  # No improvement.


def local_search_mixed_stacked(
    acqf_wrapper: AcqfWrapper,
    initial_normalized_params_list: np.ndarray,  # (D,) -> (B,D)
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> tuple[np.ndarray, float]:
    continuous_indices = acqf_wrapper.search_space.continuous_indices
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
    lengthscales = acqf_wrapper.length_scales  # (D,)
    assert lengthscales.shape == (dimension,)
    best_normalized_params_list = initial_normalized_params_list.copy()  # (D,) -> (B,D)
    assert best_normalized_params_list.shape == (batch_size, dimension)
    best_fval = sum(
        [float(acqf_wrapper.eval_acqf_no_grad(p)) for p in best_normalized_params_list]
    )

    (best_normalized_params_list, best_fval, _) = _gradient_ascent(
        acqf_wrapper,
        best_normalized_params_list,
        best_fval,
        continuous_indices,
        lengthscales,
        tol,
    )
    return best_normalized_params_list, best_fval


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
    acqf_wrapper = AcqfWrapper(acqf, batch_size=batch_size, dimension=dimension)

    # x_warmstarts: (B,D)
    x_warmstarts = np.vstack(
        [sampled_xs[chosen_idxs, :], warmstart_normalized_params_array]
    )
    assert x_warmstarts.shape == (batch_size, dimension)
    # xs: (B,D)
    xs, f_sum = local_search_mixed_stacked(acqf_wrapper, x_warmstarts, tol=tol)
    assert xs.shape == (batch_size, dimension)
    for x in xs:
        f = float(acqf_wrapper.eval_acqf_no_grad(x))
        if f > best_f:
            best_x = x
            best_f = f
    return best_x, best_f
