from abc import ABC, abstractmethod
import numpy as np


def check_X_y(X, y):
    """Validate input arrays for training."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatched samples: X has {X.shape[0]}, y has {y.shape[0]}"
        )

    return X, y


def check_array(X):
    """Validate input array for inference."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X.shape}")
    return X


class BaseEstimator(ABC):
    """Base class for all estimators and transformers."""

    def __init__(self):
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X, y=None):
        """Fit estimator or transformer on training data."""
        pass

    def _check_is_fitted(self):
        """Ensure estimator is fitted before running inference or transformation."""
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )


class ClassifierMixin:
    """Mixin class providing standard accuracy evaluation for classifiers."""

    def score(self, X, y):
        X, y = check_X_y(X, y)
        predictions = self.predict(X)
        return np.mean(predictions == y)


class RegressorMixin:
    """Mixin class providing R-squared evaluation for regressors."""

    def score(self, X, y):
        X, y = check_X_y(X, y)
        predictions = self.predict(X)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - predictions) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - (ss_res / ss_tot)


class TransformerMixin:
    """Mixin class providing fit_transform implementation for transformers."""

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)