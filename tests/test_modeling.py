import numpy as np
from hw2lib.modeling import build_models


def test_models_select_fourteen_features():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 20))
    y = X[:, 0] + rng.normal(size=80)
    models = build_models()
    for name in ('kbest', 'rfe'):
        model = models[name].fit(X, y)
        assert model.named_steps['select'].get_support().sum() == 14
    assert set(models) == {'dummy', 'ols', 'kbest', 'rfe', 'ridge', 'lasso'}
