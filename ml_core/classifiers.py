import numpy as np
from .base import BaseEstimator, ClassifierMixin, check_X_y, check_array


class LogisticRegressionSoftmax(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.01, epochs=1000, reg=1e-4, fit_intercept=True):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.fit_intercept = fit_intercept
        self.W_ = None
        self.b_ = None
        self.classes_ = None

    def _softmax(self, z):
        # Numerical stability trick to prevent overflow
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Vectorized one-hot encoding
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y_indices] = 1.0

        # Parameter initialization
        self.W_ = np.random.randn(n_features, n_classes) * 0.01
        self.b_ = np.zeros((1, n_classes)) if self.fit_intercept else None

        for _ in range(self.epochs):
            logits = X @ self.W_
            if self.fit_intercept:
                logits += self.b_

            probs = self._softmax(logits)
            error = probs - y_onehot

            # Analytical gradients with L2 regularization penalty
            grad_W = (X.T @ error) / n_samples + self.reg * self.W_
            self.W_ -= self.lr * grad_W

            if self.fit_intercept:
                grad_b = np.sum(error, axis=0, keepdims=True) / n_samples
                self.b_ -= self.lr * grad_b

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        self._check_is_fitted()
        X = check_array(X)
        logits = X @ self.W_
        if self.fit_intercept:
            logits += self.b_
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        predicted_indices = np.argmax(probs, axis=1)
        return self.classes_[predicted_indices]
