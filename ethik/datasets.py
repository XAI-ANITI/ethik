import os
import zipfile

import imageio
import pandas as pd


__all__ = [
    "load_adult",
    "load_diabetes",
    "load_heart_disease",
    "load_law_school",
    "load_wolf_or_husky",
]


def load_wolf_or_husky():
    """A set of 42 images of wolves and huskies.

    The wolves have a snowy background whereas the huskies have a grassy background.

    References
        1. https://arxiv.org/pdf/1602.04938.pdf

    """

    X = []
    y = []

    zf = zipfile.ZipFile("ethik/data/wolf_or_husky.zip")

    for file in zf.namelist():
        basename, extension = os.path.splitext(file)
        label = basename.split("/")[1]
        if extension == ".jpg":
            img = imageio.imread(zf.open(file))
            X.append(pd.np.asarray(img))
            y.append(label)

    return X, y


def load_adult():
    """https://archive.ics.uci.edu/ml/datasets/Adult"""
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
    X = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "adult.csv"),
        names=names,
        header=None,
        dtype=dtypes,
    )
    X["gender"] = (
        X["gender"].str.strip().astype("category")
    )  # Remove leading whitespace
    y = X.pop("salary").map({" <=50K": False, " >50K": True})
    return X, y


def load_law_school():
    """Law school admission data.

    The Law School Admission Council conducted a survey across 163 law schools in the United
    States. It contains information on 21,790 law students such as their entrance exam scores
    (LSAT), their grade-point average (GPA) collected prior to law school, and their first year
    average grade (FYA).

    Given this data, a school may wish to predict if an applicant will have a high FYA. The school
    would also like to make sure these predictions are not biased by an individual’s race and sex.
    However, the LSAT, GPA, and FYA scores, may be biased due to social factors.

    Example:

        >>> import ethik

        >>> X, y = ethik.datasets.load_law_school()

        >>> X.head(10)
                race     sex  LSAT  UGPA region_first  ZFYA  sander_index
        0      White  Female  39.0   3.1           GL -0.98      0.782738
        1      White  Female  36.0   3.0           GL  0.09      0.735714
        2      White    Male  30.0   3.1           MS -0.35      0.670238
        5   Hispanic    Male  39.0   2.2           NE  0.58      0.697024
        6      White  Female  37.0   3.4           GL -1.26      0.786310
        7      White  Female  30.5   3.6           GL  0.30      0.724107
        8      White    Male  36.0   3.6           GL -0.10      0.792857
        9      White    Male  37.0   2.7           NE -0.12      0.719643
        13     White  Female  37.0   2.6           GL  1.53      0.710119
        14     White    Male  31.0   3.6           GL  0.34      0.730357

        >>> y.head(10)
        0      True
        1      True
        2      True
        5      True
        6      True
        7      True
        8      True
        9     False
        13     True
        14     True
        Name: first_pf, dtype: bool

    References:
        1. https://papers.nips.cc/paper/6995-counterfactual-fairness.pdf
        2. https://github.com/mkusner/counterfactual-fairness

    """
    X = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "law_data.csv"),
        dtype={"race": "category", "region_first": "category"},
        index_col=0,
    )
    X["sex"] = X["sex"].map({1: "Female", 2: "Male"}).astype("category")
    y = X.pop("first_pf").apply(int).astype(bool)
    return X, y


def load_diabetes():
    """https://www.kaggle.com/uciml/pima-indians-diabetes-database"""
    X = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "diabetes.csv"))
    y = X.pop("Outcome").apply(int).astype(bool).rename("has_diabetes")
    return X, y


def load_heart_disease():
    """https://www.kaggle.com/ronitf/heart-disease-uci"""
    X = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "heart_disease.csv"),
        dtype={
            "sex": "category",
            "cp": "category",
            "fbs": "category",
            "restecg": "category",
            "exang": "category",
            "slope": "category",
            "ca": "category",
            "thal": "category",
        },
    )
    y = X.pop("target").apply(int).astype(bool).rename("has_heart_disease")
    return X, y
