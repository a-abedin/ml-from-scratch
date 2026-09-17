# ML-Core: Scikit-Learn Compatible Machine Learning From Scratch

[![Tests](https://github.com/a-abedin/ml-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/a-abedin/ml-from-scratch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, fully vectorized machine learning library built purely with **NumPy** and standard Python libraries. Designed with a clean, Scikit-Learn-compatible API (`fit`, `predict`, `score`), strict mathematical precision, and leakage-free preprocessing pipelines.

---

## Key Highlights

* **Pure NumPy Vectorization:** Zero nested loops across gradient updates, mahalanobis distance calculations, and backpropagation.
* **Numerical Stability Built-In:** LogSumExp and subtract-max softmax implementations to eliminate floating-point overflow.
* **Parity Tested:** Unit tests validate statistical equivalence and coefficient parity (< 1e-4 tolerance) against Scikit-Learn reference baselines.
* **Leakage-Safe Architecture:** Custom pipelines ensure feature selection and scaling are strictly learned on training folds.

---

## Benchmark & Parity Verification

Our custom implementations match standard libraries in accuracy while providing transparent, inspectable mathematical operations:

| Estimator | Dataset | Custom Accuracy / R² | Scikit-Learn Accuracy / R² | Tolerance Match |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (OLS)** | Synthetic Regression | R²: 0.984 | R²: 0.984 | Δ < 1e-4 |
| **Linear Discriminant (LDA)** | Fisher Iris | 98.0% | 98.0% | Exact match |
| **Softmax Logistic Regression** | Wine Dataset | 96.7% | 97.2% | Δ < 0.01 |
| **Modular Neural Network (MLP)**| Moons Non-Linear | 94.5% | 95.0% | Δ < 0.01 |

---

## Project Architecture

ml-from-scratch/
├── ml_core/
│   ├── base.py              # BaseEstimator, ClassifierMixin, RegressorMixin
│   ├── linear_models.py     # Closed-form OLS and polynomial expansion
│   ├── classifiers.py       # Vectorized Softmax Logistic Regression
│   ├── bayes.py             # Analytical LDA (pooled) and QDA (per-class)
│   ├── neural_networks.py   # Multi-layer Perceptron with He initialization
│   └── pipeline.py          # Leakage-free Standard Scaler & Feature Selector
├── tests/
│   └── test_parity.py       # Pytest suite against Scikit-Learn baselines
├── requirements.txt
└── README.md

---

## Quickstart

### 1. Installation
```bash
git clone [https://github.com/a-abedin/ml-from-scratch.git](https://github.com/a-abedin/ml-from-scratch.git)
cd ml-from-scratch
pip install -r requirements.txt
```

### 2. Training an End-to-End Pipeline
```python
from ml_core.pipeline import Pipeline, StandardScaler, CorrelationFeatureSelector
from ml_core.classifiers import LogisticRegressionSoftmax
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a leakage-free pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", CorrelationFeatureSelector(k=8)),
    ("classifier", LogisticRegressionSoftmax(lr=0.05, epochs=1500))
])

# Train and evaluate using the standardized API
pipe.fit(X_train, y_train)
test_acc = pipe.score(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2%}")
```

### 3. Running Unit Tests
```bash
pytest tests/test_parity.py -v
```

---

## Engineering Details

* **He Normal Initialization:** Neural network weights are scaled by sqrt(2 / n_in) to prevent vanishing/exploding gradients in deep configurations.
* **Covariance Inversion Stability:** Inverted matrices in LDA and QDA utilize Moore-Penrose pseudo-inverses alongside diagonal regularization (ε · I) to guarantee numerical solvability under collinearity.