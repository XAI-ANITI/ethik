import shutil
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
DEST_FOLDER = os.path.join(HERE, "assets", "plots")


def save_fig(fig, name):
    path = os.path.join(DEST_FOLDER, f"{name}.json")

    fig.update_layout(
        plot_bgcolor="rgba(255, 255, 255, 0)", paper_bgcolor="rgba(255, 255, 255, 0)"
    )

    data = json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder))
    layout = json.loads(json.dumps(fig.layout, cls=PlotlyJSONEncoder))

    with open(path, "w") as f:
        json.dump({"data": data, "layout": layout}, f)


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
    shutil.rmtree(DEST_FOLDER)
    os.makedirs(DEST_FOLDER)

    plots = []

    explainer, X_test, y_pred, y_test = setup_adult()
    plots.append(
        dict(
            fig=explainer.plot_influence(
                X_test=X_test[["age", "education-num"]], y_pred=y_pred
            ),
            name="influence_multiple_features",
            description="Lorem ipsum dolor sit amet, consectetur adipiscing elit. In laoreet lectus vitae ipsum sollicitudin, nec venenatis lectus bibendum. Curabitur odio ligula, mollis quis nulla eu, porta congue dolor. Sed sagittis orci ut turpis suscipit consectetur. Integer libero ante, sollicitudin ac nibh ut, semper auctor dolor. Donec sit amet erat a est tincidunt accumsan id eu tellus. Proin in vulputate quam, ac congue neque. Vestibulum tempor vulputate dui, nec gravida quam. Etiam facilisis dui turpis, vel molestie ligula luctus at. Aenean consequat velit ut nunc tincidunt facilisis. Nam nunc quam, volutpat id lacus eu, efficitur consequat justo. Nunc a nibh accumsan, feugiat risus in, faucibus ante. Cras bibendum massa tellus, in sodales mauris feugiat in. Donec ligula tellus, placerat non nibh at, faucibus fermentum diam. Praesent vel iaculis nunc. Cras malesuada tristique tellus, eu dapibus felis venenatis eget.",
        )
    )

    explainer, X_test, y_pred, y_test = setup_mnist()
    plots.append(
        dict(
            fig=explainer.plot_influence(X_test=X_test, y_pred=y_pred, cell_width=130),
            name="influence_image",
            description="Lorem ipsum dolor sit amet, consectetur adipiscing elit. In laoreet lectus vitae ipsum sollicitudin, nec venenatis lectus bibendum. Curabitur odio ligula, mollis quis nulla eu, porta congue dolor. Sed sagittis orci ut turpis suscipit consectetur. Integer libero ante, sollicitudin ac nibh ut, semper auctor dolor. Donec sit amet erat a est tincidunt accumsan id eu tellus. Proin in vulputate quam, ac congue neque. Vestibulum tempor vulputate dui, nec gravida quam. Etiam facilisis dui turpis, vel molestie ligula luctus at. Aenean consequat velit ut nunc tincidunt facilisis. Nam nunc quam, volutpat id lacus eu, efficitur consequat justo. Nunc a nibh accumsan, feugiat risus in, faucibus ante. Cras bibendum massa tellus, in sodales mauris feugiat in. Donec ligula tellus, placerat non nibh at, faucibus fermentum diam. Praesent vel iaculis nunc. Cras malesuada tristique tellus, eu dapibus felis venenatis eget.",
        )
    )

    for plot in plots:
        save_fig(plot["fig"], plot["name"])

    with open(os.path.join(DEST_FOLDER, "metadata.json"), "w") as f:
        json.dump(
            dict(
                plots=[
                    dict(name=plot["name"], description=plot["description"])
                    for plot in plots
                ]
            ),
            f,
        )
