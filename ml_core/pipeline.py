import numpy as np
from .base import BaseEstimator, ClassifierMixin, RegressorMixin, check_X_y, check_array


class StandardScaler(BaseEstimator):
    def __init__(self, with_mean=True, with_std=True):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = np.mean(X, axis=0) if self.with_mean else np.zeros(X.shape[1])
        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0.0] = 1.0
        else:
            self.scale_ = np.ones(X.shape[1])
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = check_array(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class CorrelationFeatureSelector(BaseEstimator):
    def __init__(self, k=20):
        super().__init__()
        self.k = k
        self.selected_indices_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y_flat = y.ravel().astype(np.float64)
        n_features = X.shape[1]
        correlations = np.zeros(n_features)

        y_centered = y_flat - np.mean(y_flat)
        y_norm = np.linalg.norm(y_centered)

        for i in range(n_features):
            x_col = X[:, i]
            x_centered = x_col - np.mean(x_col)
            x_norm = np.linalg.norm(x_centered)
            if x_norm == 0 or y_norm == 0:
                correlations[i] = 0.0
            else:
                correlations[i] = np.abs(np.dot(x_centered, y_centered) / (x_norm * y_norm))

        k_eff = min(self.k, n_features)
        self.selected_indices_ = np.argsort(correlations)[-k_eff:]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = check_array(X)
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class Pipeline(BaseEstimator):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def fit(self, X, y):
        current_X = X
        for _, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                current_X = step.fit_transform(current_X, y)
            else:
                current_X = step.fit(current_X, y).transform(current_X)

        self.steps[-1][1].fit(current_X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        current_X = X
        for _, step in self.steps[:-1]:
            current_X = step.transform(current_X)
        return self.steps[-1][1].predict(current_X)

    def score(self, X, y):
        self._check_is_fitted()
        current_X = X
        for _, step in self.steps[:-1]:
            current_X = step.transform(current_X)
        return self.steps[-1][1].score(current_X, y)
