import pandas as pd
import pytest
from sklearn import linear_model
from sklearn import datasets
from sklearn import neighbors
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing

import ethik


def load_boston():
    boston = datasets.load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.Series(boston.target, name="House price")
    return X, y


def load_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target).map(lambda x: iris.target_names[x])
    return X, y


def test_check_alpha():
    for alpha in (-1, -0.1, 0.5, 1):
        with pytest.raises(ValueError):
            ethik.ClassificationExplainer(alpha=alpha)


def test_check_n_taus():
    for n_taus in (-1, 0):
        with pytest.raises(ValueError):
            ethik.ClassificationExplainer(n_taus=n_taus)


def test_check_n_samples():
    for n_samples in (-1, 0):
        with pytest.raises(ValueError):
            ethik.ClassificationExplainer(n_samples=n_samples)


def test_check_sample_frac():
    for sample_frac in (-1, 0, 1):
        with pytest.raises(ValueError):
            ethik.ClassificationExplainer(sample_frac=sample_frac)


def test_check_conf_level():
    for conf_level in (-1, 0, 0.5):
        with pytest.raises(ValueError):
            ethik.ClassificationExplainer(conf_level=conf_level)


def test_check_max_iterations():
    for max_iterations in (-1, 0):
        with pytest.raises(ValueError):
            ethik.ClassificationExplainer(max_iterations=max_iterations)


def test_check_tol():
    for tol in (-1, 0):
        with pytest.raises(ValueError):
            ethik.ClassificationExplainer(tol=tol)


def test_memoization():
    X, y = load_boston()

    # Train a model and make predictions
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, shuffle=True
    )
    model = pipeline.make_pipeline(
        preprocessing.StandardScaler(), linear_model.LinearRegression()
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Without memoization
    explainer = ethik.RegressionExplainer(memoize=False, n_jobs=1)
    explainer.explain_influence(X_test[["INDUS", "NOX"]], y_pred)
    assert set(explainer.info["feature"].unique()) == set(["INDUS", "NOX"])
    explainer.explain_influence(X_test[["INDUS"]], y_pred)
    assert set(explainer.info["feature"].unique()) == set(["INDUS"])

    # With memoization
    explainer = ethik.RegressionExplainer(memoize=True, n_jobs=1)
    influence = explainer.explain_influence(X_test[["INDUS", "NOX"]], y_pred)
    assert set(influence["feature"].unique()) == set(["INDUS", "NOX"])
    assert set(explainer.info["feature"].unique()) == set(["INDUS", "NOX"])
    influence = explainer.explain_influence(X_test[["INDUS"]], y_pred)
    assert set(influence["feature"].unique()) == set(["INDUS"])
    assert set(explainer.info["feature"].unique()) == set(["INDUS", "NOX"])


def test_target_identity():
    X, y = load_iris()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, shuffle=True, random_state=42
    )

    model = pipeline.make_pipeline(
        preprocessing.StandardScaler(), neighbors.KNeighborsClassifier()
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)
    y_pred = pd.DataFrame(y_pred, columns=model.classes_)

    explainer = ethik.ClassificationExplainer(memoize=True)
    label = y_pred.columns[0]

    explainer.explain_influence(X_test=X_test, y_pred=y_pred[label])

    explainer.explain_influence(X_test=X_test["petal width (cm)"], y_pred=y_pred[label])

    rows = explainer.info.query(
        f"label == '{label}' and feature == 'petal width (cm)' and tau == -0.85"
    )
    assert len(rows) == 1
