# %%
from timeit import timeit

from benchmark_tensor_operations import TensorOperationsBenchmark

# %%
batch_size = 10
dimension = 50
n_trials = 300
t_batched = TensorOperationsBenchmark(
    n_trials=n_trials, dimension=dimension, batch_size=batch_size
)
t = TensorOperationsBenchmark(n_trials=n_trials, dimension=dimension, batch_size=1)


# %%
elapsed = timeit(t_batched.execute, number=100)
print(f"Batched execution time (batch size={batch_size}): {elapsed:.3e}")
# %%
elapsed = timeit(lambda: [t.execute() for _ in range(batch_size)], number=100)
print(f"Sequential execution time (batch size={batch_size}): {elapsed:.3e}")

# %%
