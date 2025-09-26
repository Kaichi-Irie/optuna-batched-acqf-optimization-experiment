def func_and_grad_batched(xs):
    return [], []


def update(*args):
    pass


max_iter = 0

starting_points = [...]

xs=starting_points

for _ in range(max_iter):
f, grad = func_and_grad_batched(xs)
        x = update(x, f, grad)
    results.append(x)
