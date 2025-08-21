import multiprocessing
from typing import Literal

import numpy as np
import optuna._gp.acqf as acqf_module
from optuna.samplers import GPSampler

import batched_acqf_eval_optim_mixed
import multiprocessing_optim_mixed
import stacking_optim_mixed

SAMPLERMODE = Literal["stacking", "batched_acqf_eval", "multiprocessing", "original"]


class BatchedSampler(GPSampler):
    def __init__(
        self, mode: SAMPLERMODE, processes: int | None = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.worker_pool = None
        if mode == "multiprocessing" and processes is None:
            processes = multiprocessing.cpu_count()
        if mode != "multiprocessing" and processes is not None:
            raise ValueError(
                "Processes must not be specified for non-multiprocessing mode."
            )
        self.processes: int | None = processes
        self.mode: SAMPLERMODE = mode

    def create_worker_pool(self, processes: int):
        if self.mode != "multiprocessing":
            raise ValueError(f"Invalid mode: {self.mode}.")
        self.processes = processes
        self.worker_pool = multiprocessing.Pool(processes=processes)

    def __enter__(self):
        """
        Create a worker pool for multiprocessing.
        """
        if self.mode == "multiprocessing" and self.processes is not None:
            self.create_worker_pool(self.processes)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Shut down the worker pool.
        """
        if self.mode == "multiprocessing" and self.worker_pool is not None:
            self.worker_pool.close()
            self.worker_pool.join()
            self.worker_pool = None
            self.processes = None

    def _optimize_acqf(
        self, acqf: acqf_module.BaseAcquisitionFunc, best_params: np.ndarray | None
    ) -> np.ndarray:
        assert best_params is None or len(best_params.shape) == 2

        if self.mode == "stacking":
            # batched_size is set as n_local_search (=10) inside stacking_optim_mixed
            normalized_params, _acqf_val = stacking_optim_mixed.optimize_acqf_mixed(
                acqf,
                warmstart_normalized_params_array=best_params,
                n_preliminary_samples=self._n_preliminary_samples,
                n_local_search=self._n_local_search,
                tol=self._tol,
                rng=self._rng.rng,
            )
            return normalized_params
            # raise ValueError("Stacking mode is not implemented.")
        if self.mode == "batched_acqf_eval":
            normalized_params, _acqf_val = (
                batched_acqf_eval_optim_mixed.optimize_acqf_mixed(
                    acqf,
                    warmstart_normalized_params_array=best_params,
                    n_preliminary_samples=self._n_preliminary_samples,
                    n_local_search=self._n_local_search,
                    tol=self._tol,
                    rng=self._rng.rng,
                )
            )
            return normalized_params
        if self.mode == "multiprocessing":
            if self.worker_pool is None:
                raise ValueError("Worker pool must be created before optimization.")
            normalized_params, _acqf_val = (
                multiprocessing_optim_mixed.optimize_acqf_mixed(
                    acqf,
                    worker_pool=self.worker_pool,
                    warmstart_normalized_params_array=best_params,
                    n_preliminary_samples=self._n_preliminary_samples,
                    n_local_search=self._n_local_search,
                    tol=self._tol,
                    rng=self._rng.rng,
                )
            )
            return normalized_params
        if self.mode == "original":
            return super()._optimize_acqf(acqf, best_params)
        else:
            raise ValueError(f"Invalid mode: {self.mode}.")
