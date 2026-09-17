import numpy as np
import pytest
from sklearn.datasets import make_regression, load_iris
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA

from ml_core.linear_models import LinearRegression
from ml_core.bayes import LinearDiscriminantAnalysis
from ml_core.pipeline import Pipeline, StandardScaler, CorrelationFeatureSelector


def test_linear_regression_parity():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)

    custom_model = LinearRegression(fit_intercept=True).fit(X, y)
    sk_model = SklearnLinearRegression(fit_intercept=True).fit(X, y)

    np.testing.assert_allclose(custom_model.coef_, sk_model.coef_, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(custom_model.intercept_, sk_model.intercept_, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(custom_model.predict(X), sk_model.predict(X), rtol=1e-4, atol=1e-4)


def test_lda_accuracy_parity():
    iris = load_iris()
    X, y = iris.data, iris.target

    custom_lda = LinearDiscriminantAnalysis(reg=1e-5).fit(X, y)
    sk_lda = SklearnLDA(solver="svd").fit(X, y)

    custom_score = custom_lda.score(X, y)
    sk_score = sk_lda.score(X, y)

    assert abs(custom_score - sk_score) < 0.05


def test_pipeline_execution():
    X, y = make_regression(n_samples=150, n_features=30, noise=0.2, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", CorrelationFeatureSelector(k=10)),
        ("regressor", LinearRegression())
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)
    score = pipe.score(X, y)

    assert preds.shape == y.shape
    assert score > 0.8
