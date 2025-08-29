import torch


class TensorOperationsBenchmark:
    def __init__(
        self,
        n_trials: int,
        dimension: int,
        batch_size: int,
    ):
        self.n_trials = n_trials
        self.dimension = dimension
        self.batch_size = batch_size
        self.batch_matrix = torch.rand((batch_size, dimension), dtype=torch.float64)
        self.X = torch.rand((n_trials, dimension), dtype=torch.float64)
        self.weights = torch.rand((dimension,), dtype=torch.float64)
        self.k = torch.rand((batch_size, n_trials), dtype=torch.float64)
        self.C = torch.rand((n_trials, n_trials), dtype=torch.float64)

    def execute(self) -> None:
        ((self.batch_matrix[..., None, :] - self.X[..., None, :, :]) ** 2).matmul(
            self.weights
        )
        torch.linalg.vecdot(self.k, self.k.matmul(self.C))
