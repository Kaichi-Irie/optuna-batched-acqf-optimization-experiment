import numpy as np
import optuna._gp.acqf as acqf_module
from optuna.samplers import GPSampler

import batched_optim_mixed


class BatchedSampler(GPSampler):
    def __init__(self, batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.worker_pool = None

    def _optimize_acqf(
        self, acqf: acqf_module.BaseAcquisitionFunc, best_params: np.ndarray | None
    ) -> np.ndarray:
        assert best_params is None or len(best_params.shape) == 2
        normalized_params, _acqf_val = batched_optim_mixed.optimize_acqf_mixed(
            acqf,
            warmstart_normalized_params_array=best_params,
            n_preliminary_samples=self._n_preliminary_samples,
            n_local_search=self._n_local_search,
            tol=self._tol,
            rng=self._rng.rng,
        )
        return normalized_params
