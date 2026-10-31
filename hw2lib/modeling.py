"""Preprocessing and regression estimators used within grouped CV."""
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models():
    def pipe(model, select=None):
        steps = [('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]
        if select is not None:
            steps.append(('select', select))
        return Pipeline(steps + [('model', model)])
    return {'dummy': pipe(DummyRegressor()), 'ols': pipe(LinearRegression()),
            'kbest': pipe(LinearRegression(), SelectKBest(f_regression, k=14)),
            'rfe': pipe(LinearRegression(), RFE(LinearRegression(), n_features_to_select=14)),
            'ridge': pipe(Ridge()), 'lasso': pipe(Lasso(max_iter=50000))}
