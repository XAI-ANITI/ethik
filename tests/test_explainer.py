import pandas as pd
import pytest
from sklearn import linear_model
from sklearn import datasets
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
    explainer.explain_bias(X_test[["INDUS", "NOX"]], y_pred)
    assert explainer.info["feature"].unique().tolist() == ["INDUS", "NOX"]
    explainer.explain_bias(X_test[["INDUS"]], y_pred)
    assert explainer.info["feature"].unique().tolist() == ["INDUS"]

    # With memoization
    explainer = ethik.RegressionExplainer(memoize=True, n_jobs=1)
    bias = explainer.explain_bias(X_test[["INDUS", "NOX"]], y_pred)
    assert bias["feature"].unique().tolist() == ["INDUS", "NOX"]
    assert explainer.info["feature"].unique().tolist() == ["INDUS", "NOX"]
    bias = explainer.explain_bias(X_test[["INDUS"]], y_pred)
    assert bias["feature"].unique().tolist() == ["INDUS"]
    assert explainer.info["feature"].unique().tolist() == ["INDUS", "NOX"]


def test_determine_pairs_to_do():
    X, y = load_iris()

    # Train a model and make predictions
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, shuffle=True
    )
    model = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(solver="lbfgs", multi_class="auto"),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    y_pred = pd.DataFrame(y_pred, columns=model.classes_)

    # With memoization

    explainer = ethik.ClassificationExplainer(memoize=True, n_jobs=1)

    to_do = explainer._determine_pairs_to_do(["sepal length (cm)"], ["setosa"])
    assert to_do == {"sepal length (cm)": ["setosa"]}

    explainer.explain_bias(X_test["sepal length (cm)"], y_pred["setosa"])

    to_do = explainer._determine_pairs_to_do(["sepal length (cm)"], ["setosa"])
    assert to_do == {}

    to_do = explainer._determine_pairs_to_do(
        ["sepal length (cm)", "petal width (cm)"], ["setosa", "virginica"]
    )
    assert to_do == {
        "petal width (cm)": ["setosa", "virginica"],
        "sepal length (cm)": ["virginica"],
    }
