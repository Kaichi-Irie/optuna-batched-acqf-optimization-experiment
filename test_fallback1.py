import sys
import importlib
from unittest.mock import MagicMock



def test_behavior_with_greenlet(monkeypatch):
    monkeypatch.setitem(sys.modules, "greenlet", MagicMock())
    import optuna._gp.batched_lbfgsb as optimization_module
    importlib.reload(optimization_module)
    assert optimization_module._greenlet_imports.is_successful() is True

def test_behavior_without_greenlet(monkeypatch):
    monkeypatch.setitem(sys.modules, "greenlet", None)
    import optuna._gp.batched_lbfgsb as optimization_module
    importlib.reload(optimization_module)
    assert optimization_module._greenlet_imports.is_successful() is False

    # See if optimization still works without greenlet
    import optuna
    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda trial: trial.suggest_float("x", -10, 10) ** 2, n_trials=15)
