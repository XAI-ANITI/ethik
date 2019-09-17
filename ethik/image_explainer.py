import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import explainer


def images_to_dataframe(images):
    img_shape = images[0].shape
    return pd.DataFrame(
        data=images.reshape(len(images), -1),
        columns=itertools.product(*[np.arange(n) for n in img_shape]),
    )


class ImageExplainer(explainer.Explainer):
    """An explainer specially suited for image classification.

    This has exactly the same API as `Explainer`, but expects to be provided with an array of
    images instead of a tabular dataset.

    """

    def __init__(
        self,
        alpha=0.05,
        max_iterations=5,
        n_jobs=-1,
        verbose=False,
        n_samples=1,
        sample_frac=0.8,
        conf_level=0.05,
    ):
        super().__init__(
            alpha=alpha,
            n_taus=2,
            max_iterations=max_iterations,
            n_jobs=n_jobs,
            verbose=verbose,
            n_samples=n_samples,
            sample_frac=sample_frac,
            conf_level=conf_level
        )

    def explain_bias(self, X_test, y_pred):
        self.img_shape = X_test[0].shape
        if self.img_shape[-1] == 1:
            self.img_shape = self.img_shape[:-1]
        return super().explain_bias(X_test=images_to_dataframe(X_test), y_pred=y_pred)

    def plot_bias(self, X_test, y_pred, n_cols=3, width=4):

        biases = self.explain_bias(X_test=X_test, y_pred=y_pred)

        for label, group in biases.groupby('label'):

            diffs = group.query('tau == 1')['bias'] - group.query('tau == -1')['bias'].values
            diffs = diffs.to_numpy().reshape(self.img_shape)

            break

        return go.Figure(data=go.Heatmap(
            z=diffs,
            x=list(range(28)),
            y=list(range(28)),
            colorscale='Viridis'
        ))

    def plot_metric(self, X_test, y_test, y_pred, metric):

        metrics = self.explain_metric(
            X=images_to_dataframe(images), y=y, y_pred=y_pred, metric=metric
        )

        # Create a plot if none is provided
        fig, ax = plt.subplots()

        diffs = (metrics.iloc[-1] - metrics.iloc[0]).fillna(0.0)
        img = ax.imshow(diffs.to_numpy().reshape(self.img_shape))

        fig.colorbar(img, ax=ax)

        return ax
