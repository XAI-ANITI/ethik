import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import explainer


def images_to_dataframe(images):
    img_shape = images[0].shape
    return pd.DataFrame(
        data=images.reshape(len(images), -1),
        columns=itertools.product(*[np.arange(n) for n in img_shape]),
    )


class ImageClassificationExplainer(explainer.Explainer):
    """An explainer specially suited for image classification.

    This has exactly the same API as `Explainer`, but expects to be provided with an array of
    images instead of a tabular dataset.

    """

    def __init__(self, alpha=0.05, lambda_iterations=5, n_jobs=-1, verbose=False, memoize=True):
        super().__init__(
            alpha=alpha,
            n_taus=2,
            lambda_iterations=lambda_iterations,
            n_jobs=n_jobs,
            verbose=verbose,
            memoize=memoize
        )

    def explain_bias(self, X_test, y_pred):
        self.img_shape = X_test[0].shape
        if self.img_shape[-1] == 1:
            self.img_shape = self.img_shape[:-1]
        return super().explain_bias(X_test=images_to_dataframe(X_test), y_pred=y_pred)

    def _plot(self, z_values, n_cols):
        n_plots = len(z_values)
        labels = sorted(z_values)
        n_rows = n_plots // n_cols + 1
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=list(map(str, labels)),
            shared_xaxes="all",
            shared_yaxes="all",
            horizontal_spacing=0.2 / n_cols,
            vertical_spacing=0.2 / n_rows,
        )

        # We want all the heatmaps to share the same scale
        zmin = min(np.min(z) for z in z_values.values())
        zmax = max(np.max(z) for z in z_values.values())

        # We want to make sure that 0 is at the center of the scale
        zmin, zmax = min(zmin, -zmax), max(zmax, -zmin)

        for i, label in enumerate(labels):
            fig.add_trace(
                go.Heatmap(
                    z=z_values[label][::-1],
                    x=list(range(self.img_shape[1])),
                    y=list(range(self.img_shape[0])),
                    zmin=zmin,
                    zmax=zmax,
                    colorscale="RdYlBu",
                    zsmooth="best",
                    showscale=i == 0,
                    name=label,
                    hoverinfo="x+y+z",
                    reversescale=True
                ),
                row=i // n_cols + 1,
                col=i % n_cols + 1,
            )

        for i in range(n_plots):
            fig.update_layout(
                {
                    f"xaxis{i+1}": dict(visible=False),
                    f"yaxis{i+1}": dict(scaleanchor=f"x{i+1}", visible=False),
                }
            )
        fig.update_layout(
            margin=dict(t=20, l=20, b=20),
            width=800,
            height=800,
            autosize=False,
            plot_bgcolor="white",
        )
        return fig

    def plot_bias(self, X_test, y_pred, n_cols=3):
        biases = self.explain_bias(X_test=X_test, y_pred=y_pred)
        values = {}

        for label, group in biases.groupby("label"):
            diffs = (
                group.query("tau == 1")["bias"]
                - group.query("tau == -1")["bias"].values
            )
            diffs = diffs.to_numpy().reshape(self.img_shape)
            values[label] = diffs

        return self._plot(values, n_cols=n_cols)

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
