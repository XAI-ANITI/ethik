"""
.. include:: ../README.md
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
