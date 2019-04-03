# Ethik

## Introduction

Ethik is a [fair learning](https://www.microsoft.com/en-us/research/blog/machine-learning-for-fair-decisions/) framework.

## Installation

:warning: Python 3.6 or above is required :snake:

**Via [PyPI](https://pypi.org/project/ethik/)**

```shell
>>> pip install ethik
```

**Via GitHub for the latest development version**

```shell
>>> pip install git+https://github.com/MaxHalford/ethik
```

## Usage

In the following set of example we'll be using from the ["Adult" dataset](https://archive.ics.uci.edu/ml/datasets/adult). The dataset contains a binary label indicating if a person's annual income is larger than $50k per year. `ethik` analyzes a model based on the predictions it makes on a test set. Consequently you first have to split your dataset in two.

```python
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, shuffle=True, random_state=42)
```

You then want to train your model on the training set and make predictions on the test set. For this example we'll train a gradient boosting classifier from the [LightGBM library](https://lightgbm.readthedocs.io/en/latest/). We'll use a variable named `y_pred` to store the predicted probabilities associated with the `True` label.

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(random_state=42).fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:, 1]
```

We can now fit an `Explainer` using the features from the test set. This will analyze the distribution of each feature and build a set of `lambda` coefficients which can be used to explain model predictions.

```python
from ethik import Explainer

explainer = Explainer()
explainer = explainer.fit(X_test)
```

### Measuring model bias

`ethik` can be used to understand how model predictions vary as a function of one or more features. For example we may want to know how the predictions vary with respect to the `age` feature.

```python
explainer.plot_predictions(
    X=X_test,
    y_pred=y_pred,
    columns='age'
)
```

<div align="center">
    <img src="figures/age_predictions.svg" alt="Age predictions" />
</div>

```python
explainer.plot_predictions(
    X=X_test,
    y_pred=y_pred,
    columns=['age', 'education-num']
)
```

<div align="center">
    <img src="figures/age_education_predictions.svg" alt="Age and education predictions" />
</div>

### Measuring model reliability

```python
explainer.plot_metric(
    X=X_test,
    y=y_test,
    y_pred=y_pred > 0.5,
    metric=metrics.accuracy_score,
    columns='age'
)
```

<div align="center">
    <img src="figures/age_accuracy.svg" alt="Age accuracy" />
</div>

```python
explainer.plot_metric(
    X=X_test,
    y=y_test,
    y_pred=y_pred > 0.5,
    metric=metrics.accuracy_score,
    columns=['age', 'education-num']
)
```

<div align="center">
    <img src="figures/age_education_accuracy.svg" alt="Age and education accuracy" />
</div>

## Authors

This work is led by researchers from the [Toulouse Institute of Mathematics](https://www.math.univ-toulouse.fr/?lang=en), namely:

- [François Bachoc](https://www.math.univ-toulouse.fr/~fbachoc/)
- [Fabrice Gamboa](https://www.math.univ-toulouse.fr/~gamboa/newwab/Pages_Fabrice_Gamboa/Main_Page.html)
- [Max Halford](https://maxhalford.github.io/)
- [Jean-Michel Loubes](https://perso.math.univ-toulouse.fr/loubes/)
- [Laurent Risser](http://laurent.risser.free.fr/menuEng.html)

## License

[MIT license](LICENSE)
