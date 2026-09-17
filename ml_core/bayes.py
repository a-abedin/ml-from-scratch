import numpy as np
from .base import BaseEstimator, ClassifierMixin, check_X_y, check_array


class LinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self, reg=1e-4):
        super().__init__()
        self.reg = reg
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.means_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)
        pooled_cov = np.zeros((n_features, n_features))

        for idx, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.priors_[idx] = X_c.shape[0] / n_samples
            self.means_[idx] = np.mean(X_c, axis=0)
            diff = X_c - self.means_[idx]
            pooled_cov += diff.T @ diff

        self.covariance_ = pooled_cov / (n_samples - n_classes)
        # Regularization for numerical stability and invertibility
        self.covariance_ += np.eye(n_features) * self.reg
        inv_cov = np.linalg.pinv(self.covariance_)

        self.coef_ = inv_cov @ self.means_.T  # Shape: (n_features, n_classes)
        self.intercept_ = -0.5 * np.sum(self.means_ * (self.means_ @ inv_cov), axis=1) + np.log(self.priors_)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = check_array(X)
        scores = X @ self.coef_ + self.intercept_
        return self.classes_[np.argmax(scores, axis=1)]


class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self, reg=1e-4):
        super().__init__()
        self.reg = reg
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariances_ = []

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.means_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)
        self.covariances_ = []

        for idx, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            n_c = X_c.shape[0]
            self.priors_[idx] = n_c / n_samples
            self.means_[idx] = np.mean(X_c, axis=0)
            
            diff = X_c - self.means_[idx]
            cov_c = (diff.T @ diff) / (n_c - 1)
            cov_c += np.eye(n_features) * self.reg
            self.covariances_.append(cov_c)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = check_array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_posteriors = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            mean = self.means_[idx]
            cov = self.covariances_[idx]
            diff = X - mean

            sign, logdet = np.linalg.slogdet(cov)
            inv_cov = np.linalg.pinv(cov)
            
            # Vectorized quadratic form: (x - mu)^T * Sigma^-1 * (x - mu)
            mahalanobis = np.sum((diff @ inv_cov) * diff, axis=1)
            log_posteriors[:, idx] = -0.5 * logdet - 0.5 * mahalanobis + np.log(self.priors_[idx])

        return self.classes_[np.argmax(log_posteriors, axis=1)]
