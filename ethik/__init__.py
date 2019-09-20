"""
# Introduction

`ethik` is a Python package for performing [fair](https://www.microsoft.com/en-us/research/blog/machine-learning-for-fair-decisions/) and [explainable](https://www.wikiwand.com/en/Explainable_artificial_intelligence) machine learning.

<div align="center">
  <img src="../figures/overview.svg" width="660px" alt="overview"/>
</div>

`ethik` can be used to:

1. Determine if a predictive model is biased with respect to one or more features.
2. Understand how the performance of the model varies with respect to one or more features.
3. Visualize which parts of an image influence a model's predictions.

# Installation

**Python 3.6 or above is required!**

<!---
**Via [PyPI](https://pypi.org/project/ethik/)**

```shell
pip install ethik
```
-->

Via GitHub for the latest development version:

```shell
pip install git+https://github.com/MaxHalford/ethik
# Or through SSH:
pip install git+ssh://git@github.com/MaxHalford/ethik.git
```

For developers:

```shell
git clone https://github.com/MaxHalford/ethik
cd ethik
python setup.py develop
pip install -r requirements-dev.txt
pre-commit install # For black
```

# User guide

* [Get started](../notebooks/Binary classification.html)
* [Binary classification (Adult dataset)](../notebooks/Binary classification.html)
* [Multi-label classification (Iris dataset)](../notebooks/Multi-label classification.html)
* [Regression (Boston dataset)](../notebooks/Regression.html)
* [Images (MNIST dataset)](../notebooks/MNIST_plotly.html)
* [Comparison with Partial Dependence Plots](../notebooks/Partial Dependence Plot.html)

# API

Explore the API from the menu on the left.

# Authors

This work is led by members of the [Toulouse Institute of Mathematics](https://www.math.univ-toulouse.fr/?lang=en), namely:

- [François Bachoc](https://www.math.univ-toulouse.fr/~fbachoc/)
- [Fabrice Gamboa](https://www.math.univ-toulouse.fr/~gamboa/newwab/Pages_Fabrice_Gamboa/Main_Page.html)
- [Max Halford](https://maxhalford.github.io/)
- [Vincent Lefoulon](https://vayel.github.io/)
- [Jean-Michel Loubes](https://perso.math.univ-toulouse.fr/loubes/)
- [Laurent Risser](http://laurent.risser.free.fr/menuEng.html)

This work is financed by the [Centre National de la Recherche Scientifique (CNRS)](http://www.cnrs.fr/) and is done in the context of the [Artificial and Natural Intelligence Toulouse Institute (ANITI)](https://en.univ-toulouse.fr/aniti) project.

# License

This software is released under the [GPL license](LICENSE).
"""
from . import datasets
from .__version__ import __version__
from .classification_explainer import ClassificationExplainer
from .regression_explainer import RegressionExplainer
from .image_classification_explainer import ImageClassificationExplainer


__all__ = [
    "__version__",
    "datasets",
    "RegressionExplainer",
    "ClassificationExplainer",
    "ImageClassificationExplainer",
]
