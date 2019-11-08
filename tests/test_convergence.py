import warnings

import pandas as pd
from sklearn import datasets
from sklearn import ensemble
from sklearn import model_selection

import ethik


def test_boston():

    boston = datasets.load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.Series(boston.target, name="House price")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, shuffle=True, random_state=42
    )

    model = ensemble.RandomForestRegressor(n_estimators=10, max_depth=5)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, name=y_test.name)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("ignore", category=ethik.warnings.ConstantWarning)
        explainer = ethik.RegressionExplainer(alpha=0, verbose=False)
        explainer.explain_influence(X_test=X_test, y_pred=y_pred)

    assert len(caught_warnings) == 0
