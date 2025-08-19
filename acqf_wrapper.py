import numpy as np
import torch
from optuna._gp.acqf import BaseAcquisitionFunc


class AcqfWrapper(BaseAcquisitionFunc):
    def __init__(
        self, acqf: BaseAcquisitionFunc, batch_size: int, dimension: int
    ) -> None:
        self._acqf = acqf
        self.batch_size = batch_size
        self.dimension = dimension
        super().__init__(
            length_scales=acqf.length_scales, search_space=acqf.search_space
        )

    def eval_acqf_from_numpy(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        x:(B,D)

        Returns
        (B,)
        """
        # TODO: xをtensorで受け取るようにする
        x_tensor = torch.from_numpy(x).requires_grad_(True)
        vals = self._acqf.eval_acqf(x_tensor)
        assert x_tensor.shape == (self.batch_size, self.dimension)
        assert vals.shape == (self.batch_size,)
        return vals, x_tensor

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        return self._acqf.eval_acqf(x)
