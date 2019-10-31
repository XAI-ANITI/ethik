import os

import ethik
import json
import lightgbm as lgb
import pandas as pd
from plotly.utils import PlotlyJSONEncoder
from sklearn import model_selection


def save_fig(fig, name):
    here = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(here, "assets", "plots")
    path = os.path.join(folder, f"{name}.json")

    if not os.path.exists(folder):
        os.makedirs(folder)

    fig.update_layout(
        plot_bgcolor="rgba(255, 255, 255, 0)", paper_bgcolor="rgba(255, 255, 255, 0)"
    )

    data = json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder))
    layout = json.loads(json.dumps(fig.layout, cls=PlotlyJSONEncoder))

    fig_json = json.dumps({"data": data, "layout": layout})
    with open(path, "w") as f:
        f.write(fig_json)


if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    ]
    dtypes = {
        "workclass": "category",
        "education": "category",
        "marital-status": "category",
        "occupation": "category",
        "relationship": "category",
        "race": "category",
        "gender": "category",
        "native-country": "category",
    }

    X = pd.read_csv(url, names=names, header=None, dtype=dtypes)
    X["gender"] = (
        X["gender"].str.strip().astype("category")
    )  # Remove leading whitespace
    y = X.pop("salary").map({" <=50K": False, " >50K": True})

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, shuffle=True, random_state=42
    )

    model = lgb.LGBMClassifier(random_state=42).fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred = pd.Series(y_pred, name=">$50k")

    explainer = ethik.ClassificationExplainer()

    save_fig(
        explainer.plot_influence(X_test=X_test["education-num"], y_pred=y_pred),
        "header-plot-left",
    )

    save_fig(
        explainer.plot_influence(
            X_test=X_test[["education-num", "age"]], y_pred=y_pred
        ),
        "header-plot-right",
    )
