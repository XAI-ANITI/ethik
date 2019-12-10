"""
## Installation

<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <i class="material-icons" style="margin-right: 10px; color: red;">warning</i> Python 3.6 or above is required
</div>

**Via [PyPI](https://pypi.org/project/ethik/)**

```shell
>>> pip install ethik
```

**Via GitHub for the latest development version**

```shell
>>> pip install git+https://github.com/XAI-ANITI/ethik.git
>>> # Or through SSH:
>>> pip install git+ssh://git@github.com/XAI-ANITI/ethik.git
```

**Development installation**

```shell
>>> git clone https://github.com/MaxHalford/ethik
>>> cd ethik
>>> make install_dev
```
"""
from . import datasets
from .__version__ import __version__
from .classification_explainer import ClassificationExplainer
from .regression_explainer import RegressionExplainer
from .image_classification_explainer import ImageClassificationExplainer
from .utils import extract_category


__all__ = [
    "__version__",
    "datasets",
    "RegressionExplainer",
    "ClassificationExplainer",
    "ImageClassificationExplainer",
    "extract_category",
]
