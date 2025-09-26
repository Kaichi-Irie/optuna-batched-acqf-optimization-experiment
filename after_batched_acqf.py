def func_and_grad_batched(xs):
    return [], []


def f_batched():
    pass


def g_batched():
    pass


def update(*args):
    pass


max_iter = 0


def scipy_optim(f, g, x0, maxiter=...):
    min_fval, min_x = ...
    x = x0
    for _ in range(maxiter):
        fval, grad = f(x), g(x)
        x = update(x, fval, grad)
        min_fval, min_x = ...
    return min_fval, min_x


starting_points = [...]
xs = starting_points

for _ in range(max_iter):
    fvals = f_batched(xs)
    grads = g_batched(xs)
    for x, fval, grad in zip(xs, fvals, grads):
        x = update(x, fval, grad)

results = xs


def minimize_stacking(*args):
    pass


def acqf():
    pass


def acqf_grad():
    pass


starting_points = [...]
min_fs = minimize_stacking(acqf, starting_points)
best_f = min(min_fs)
# %%


# %%
def scipy_optim(f, g, x0, maxiter=...):
    min_fval, min_x = ...
    x = x0
    for _ in range(maxiter):
        fval, grad = f(x), g(x)
        x = update(x, fval, grad)
        min_fval, min_x = ...
    return min_fval, min_x


# %%
for x0 in x0_list:
    x = x0
    for _ in range(maxiter):
        fval, grad = f(x), g(x)
        x = update(x, fval, grad)
    min_fval, min_x = ...
best_fval, best_x = ...
# %%
min_fvals = []
for x0 in x0_list:
    min_fval, min_x = scipy_optim(acqf, acqf_grad, x0)
    min_fvals.append(min_fval)
best_fval = min(min_fvals)


# %%
def func():
    pass


def optimize(func, x0):
    pass
def optimize_batched(func, x0):
    pass
x0_list = [...]


min_fvals = []
for x0 in x0_list:
    min_fval = optimize(func, x0)
    min_fvals.append(min_fval)
best_fval = min(min_fvals)



min_fvals = optimize_batched(func, x0_list)
best_fval = min(min_fvals)


# %%
def scipy_optim_batched(f, g, x0, maxiter=...):
    min_fval, min_x = ...
    x = x0
    for _ in range(maxiter):
        fval, grad = f(x), g(x)
        x = update(x, fval, grad)
        min_fval, min_x = ...
    return min_fval, min_x


min_fvals = []
min_xs = []
xs = x0_list
for _ in range(maxiter):
    fvals = f_batched(xs)
    grads = g_batched(xs)
    for x, fval, grad in zip(xs, fvals, grads):
        x = update(x, fval, grad)
    min_fval, min_x = ...
best_fval, best_x = ...


# %%
