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
from sklearn import metrics, model_selection

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

    num_classes = 10

    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

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
    shutil.rmtree(DEST_FOLDER, ignore_errors=True)
    os.makedirs(DEST_FOLDER)

    plots = []

    explainer, X_test, y_pred, y_test = setup_adult()
    plots.append(
        dict(
            fig=explainer.plot_influence(X_test=X_test["age"], y_pred=y_pred),
            name="influence_age",
            description="We can plot the average output of a model relatively to the average value of a feature. Here, we plot the portion of bank customers predicted by machine learning decision rules as having an income higher than $50k with respect to their age. We can see that, on average, the older the richer until a certain limit. From about 50 year-old, the older the poorer, which probably reflects the fact that people are about to retire so companies don't especially want to invest in them.",
        )
    )

    plots.append(
        dict(
            fig=explainer.plot_influence(
                X_test=X_test[["age", "education-num"]], y_pred=y_pred
            ),
            name="influence_multiple_features",
            description="We can plot simultaneously the influence of multiple features and not only a single one as above. The x axis represents the quantiles of each variable and not its values, in order to compare the variables influence in a blink. -1 is 5%, 0 is the original mean and 1 is 95%.",
        )
    )

    plots.append(
        dict(
            fig=explainer.plot_influence_ranking(
                X_test=X_test[["age", "education-num", "hours-per-week", "gender"]],
                y_pred=y_pred,
            ),
            name="influence_ranking",
            description="We can rank the features by their influence on the model. Intuitively, a feature with no influence is an horizontal line <code>y = original mean</code>. To measure its influence, we compute the average distance to this reference line.",
        )
    )

    plots.append(
        dict(
            fig=explainer.plot_influence_comparison(
                X_test=X_test[["age", "education-num", "hours-per-week", "gender"]],
                y_pred=y_pred,
                reference=X_test.iloc[2].rename("bob"),
                compared=X_test.iloc[1].rename("mary"),
            ),
            name="influence_comparison",
            description="We can compare two individuals by looking at how the model would behave if the average individual of the dataset was different. On the left, we can see that people having Mary's age are on average 12.8% more likely to earn more than $50k a year than people with Bob's age. On the opposite, we can unfortunately see that Mary's gender is not helping her compared to Bob.",
        )
    )

    plots.append(
        dict(
            fig=explainer.plot_performance(
                X_test=X_test["age"],
                y_test=y_test,
                y_pred=y_pred > 0.5,
                metric=metrics.accuracy_score,
            ),
            name="performance_age",
            description="We can visualize how the model's performance changes with the average value of a feature. Here, we can see that the model performs worse for older people, so we are probably lacking such data samples.",
        )
    )

    plots.append(
        dict(
            fig=explainer.plot_distributions(
                feature_values=X_test["age"], targets=[27, 45], bins=60
            ),
            name="stressed_distributions",
            description='For advanced users, we can visualize the stressed distributions. As explained in <a href="https://arxiv.org/abs/1810.07924">the paper</a>, we indeed stress the probability distribution to obtain a different mean while minimizing the Kullback-Leibler divergence with the original distribution.',
        )
    )

    explainer, X_test, y_pred, y_test = setup_mnist()
    plots.append(
        dict(
            fig=explainer.plot_influence(X_test=X_test, y_pred=y_pred, cell_width=130),
            name="influence_image",
            description="For images, we can visualize the influence of the pixels on the output.",
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
