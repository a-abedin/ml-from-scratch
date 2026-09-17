import numpy as np
from itertools import combinations_with_replacement
from .base import BaseEstimator, RegressorMixin, check_X_y, check_array


class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n_samples), X])
        else:
            X_design = X

        # Solve normal equations using least squares for numerical stability
        weights, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights

        self.is_fitted_ = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = check_array(X)
        return X @ self.coef_ + self.intercept_


class PolynomialFeatures(BaseEstimator):
    def __init__(self, degree=2, include_bias=False):
        super().__init__()
        self.degree = degree
        self.include_bias = include_bias
        self.combinations_ = []

    def fit(self, X, y=None):
        X = check_array(X)
        n_features = X.shape[1]

        combos = []
        if self.include_bias:
            combos.append(())

        for d in range(1, self.degree + 1):
            combos.extend(combinations_with_replacement(range(n_features), d))

        self.combinations_ = combos
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = check_array(X)
        n_samples = X.shape[0]

        output_cols = []
        for combo in self.combinations_:
            if len(combo) == 0:
                output_cols.append(np.ones((n_samples, 1)))
            else:
                output_cols.append(np.prod(X[:, combo], axis=1, keepdims=True))

        return np.hstack(output_cols)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
