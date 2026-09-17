import numpy as np
from .base import BaseEstimator, ClassifierMixin, check_X_y, check_array


class MultiLayerPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=(64, 32), lr=0.01, epochs=500, reg=1e-4):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.weights_ = []
        self.biases_ = []
        self.classes_ = None

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_grad(self, z):
        return (z > 0).astype(float)

    def _softmax(self, z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y_indices] = 1.0

        layer_dims = [n_features] + list(self.hidden_layers) + [n_classes]
        self.weights_ = []
        self.biases_ = []

        # He initialization
        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i + 1]) * np.sqrt(2.0 / layer_dims[i])
            b = np.zeros((1, layer_dims[i + 1]))
            self.weights_.append(w)
            self.biases_.append(b)

        for _ in range(self.epochs):
            # Forward propagation
            activations = [X]
            linear_outputs = []

            for i in range(len(self.weights_) - 1):
                z = activations[-1] @ self.weights_[i] + self.biases_[i]
                linear_outputs.append(z)
                activations.append(self._relu(z))

            # Output layer with Softmax
            z_final = activations[-1] @ self.weights_[-1] + self.biases_[-1]
            linear_outputs.append(z_final)
            activations.append(self._softmax(z_final))

            # Backpropagation
            delta = (activations[-1] - y_onehot) / n_samples
            grad_w = activations[-2].T @ delta + self.reg * self.weights_[-1]
            grad_b = np.sum(delta, axis=0, keepdims=True)

            self.weights_[-1] -= self.lr * grad_w
            self.biases_[-1] -= self.lr * grad_b

            for i in range(len(self.weights_) - 2, -1, -1):
                delta = (delta @ self.weights_[i + 1].T) * self._relu_grad(linear_outputs[i])
                grad_w = activations[i].T @ delta + self.reg * self.weights_[i]
                grad_b = np.sum(delta, axis=0, keepdims=True)

                self.weights_[i] -= self.lr * grad_w
                self.biases_[i] -= self.lr * grad_b

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        self._check_is_fitted()
        X = check_array(X)
        curr = X
        for i in range(len(self.weights_) - 1):
            curr = self._relu(curr @ self.weights_[i] + self.biases_[i])
        return self._softmax(curr @ self.weights_[-1] + self.biases_[-1])

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
