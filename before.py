def func_and_grad(xs):
    return [], []


def update(*args):
    pass


max_iter = 0

starting_points = [...]
results = []

for x0 in starting_points:
    x = x0
    for _ in range(max_iter):
        f, grad = func_and_grad(x)
        x = update(x, f, grad)
    results.append(x)


# %%
def minimize(*args):
    pass


def acqf():
    pass


starting_points = [...]
min_fs = []
for x0 in starting_points:
    min_f = minimize(acqf, x0)
    min_fs.append(min_f)
best_f = min(min_fs)
