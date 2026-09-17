import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

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
