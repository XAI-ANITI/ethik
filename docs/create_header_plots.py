import os

import ethik
import json
import keras
from keras.datasets import mnist
from keras import backend as K
import lightgbm as lgb
import numpy as np
import pandas as pd
from plotly.utils import PlotlyJSONEncoder
from sklearn import model_selection

HERE = os.path.dirname(os.path.abspath(__file__))


def save_fig(fig, name):
    folder = os.path.join(HERE, "assets", "plots")
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


def setup_adult():
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

    return explainer, X_test, y_pred, y_test


def setup_mnist():
    batch_size = 128
    num_classes = 10
    epochs = 12

    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    y_pred = np.load(os.path.join(HERE, "..", "notebooks", "cache", "mnist.npy"))

    explainer = ethik.ImageClassificationExplainer()

    return explainer, x_test, y_pred, y_test


if __name__ == "__main__":
    explainer, X_test, y_pred, y_test = setup_adult()
    save_fig(
        explainer.plot_influence(X_test=X_test["education-num"], y_pred=y_pred),
        "header-plot-left",
    )

    save_fig(
        explainer.plot_influence(
            X_test=X_test[["age", "education-num", "hours-per-week"]], y_pred=y_pred
        ),
        "header-plot-right",
    )

    """
    explainer, X_test, y_pred, y_test = setup_mnist()
    save_fig(
        explainer.plot_influence(
            X_test=X_test,
            y_pred=y_pred,
            cell_width=100
        ),
        "header-plot-right",
    )
    """
