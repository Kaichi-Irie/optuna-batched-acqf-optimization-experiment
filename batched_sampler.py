import multiprocessing

import numpy as np
import optuna._gp.acqf as acqf_module
from optuna.samplers import GPSampler

import optim_mixed


class BatchedSampler(GPSampler):
    def __init__(self, processes: int | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_pool = None
        if processes is None:
            processes = multiprocessing.cpu_count()
        self.processes: int | None = processes

    def __enter__(self):
        """
        Create a worker pool for multiprocessing.
        """
        ctx = multiprocessing.get_context("spawn")
        self.worker_pool = ctx.Pool(processes=self.processes)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Shut down the worker pool.
        """
        if self.worker_pool is not None:
            self.worker_pool.close()
            self.worker_pool.join()
            self.worker_pool = None
            self.processes = None

    def _optimize_acqf(
        self, acqf: acqf_module.BaseAcquisitionFunc, best_params: np.ndarray | None
    ) -> np.ndarray:
        assert best_params is None or len(best_params.shape) == 2
        if self.worker_pool is None:
            raise ValueError("Worker pool must be created before optimization.")
        normalized_params, _acqf_val = optim_mixed.optimize_acqf_mixed(
            acqf,
            worker_pool=self.worker_pool,
            warmstart_normalized_params_array=best_params,
            n_preliminary_samples=self._n_preliminary_samples,
            n_local_search=self._n_local_search,
            tol=self._tol,
            rng=self._rng.rng,
        )
        return normalized_params
