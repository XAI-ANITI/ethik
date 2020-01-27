import itertools
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .cache_explainer import CacheExplainer
from .warnings import ConstantWarning


def images_to_dataframe(images):
    img_shape = images[0].shape
    return pd.DataFrame(
        data=images.reshape(len(images), -1),
        columns=itertools.product(*[np.arange(n) for n in img_shape]),
    )


class ImageClassificationExplainer(CacheExplainer):
    """An explainer specially suited for image classification.

    This has exactly the same API as `Explainer`, but expects to be provided with an array of
    images instead of a tabular dataset.

    TODO: add a note about n_taus being 2

    Parameters:
        alpha (float): A `float` between `0` and `0.5` which indicates by how close the `Explainer`
            should look at extreme values of a distribution. The closer to zero, the more so
            extreme values will be accounted for. The default is `0.05` which means that all values
            beyond the 5th and 95th quantiles are ignored.
        max_iterations (int): The number of iterations used when applying the Newton step
            of the optimization procedure. Default is `5`.
        tol (float): The bottom threshold for the gradient of the optimization
            procedure. When reached, the procedure stops. Otherwise, a warning
            is raised about the fact that the optimization did not converge.
            Default is `1e-4`.
        n_jobs (int): The number of jobs to use for parallel computations. See
            `joblib.Parallel()`. Default is `-1`.
        memoize (bool): Indicates whether or not memoization should be used or not. If `True`, then
            intermediate results will be stored in order to avoid recomputing results that can be
            reused by successively called methods. For example, if you call `plot_influence` followed by
            `plot_influence_ranking` and `memoize` is `True`, then the intermediate results required by
            `plot_influence` will be reused for `plot_influence_ranking`. Memoization is turned on by
            default because computations are time-consuming for images.
        disable_checks (bool): If `False` then the data will be checked. Setting this to ``True``
            may bring a significant speed up if the data has many variables.
        verbose (bool): Whether or not to show progress bars during
            computations. Default is `True`.
    """

    def __init__(
        self,
        alpha=0.05,
        max_iterations=10,
        tol=1e-4,
        n_jobs=-1,
        memoize=True,
        disable_checks=True,
        verbose=True,
    ):
        super().__init__(
            alpha=alpha,
            n_taus=2,
            max_iterations=max_iterations,
            tol=tol,
            n_jobs=n_jobs,
            memoize=memoize,
            disable_checks=disable_checks,
            verbose=verbose,
        )
        # `CacheExplainer()` will set `n_taus` to `2+1` as it requires an odd number
        # of steps, but we don't need to compute `tau == 0` here
        self.n_taus = 2

    def _set_image_shape(self, images):
        self.img_shape = images[0].shape
        if self.img_shape[-1] == 1:
            self.img_shape = self.img_shape[:-1]

    def explain_influence(self, X_test, y_pred):
        """Compute the influence of the model for the features in `X_test`.

        Args:
            X_test (np.array): An array of images, i.e. a 3d numpy array of
                dimension `(n_images, n_rows, n_cols)`.
            y_pred (pd.DataFrame or pd.Series): The model predictions
                for the samples in `X_test`. For binary classification and regression,
                `pd.Series` is expected. For multi-label classification, a
                pandas dataframe with one column per label is
                expected. The values can either be probabilities or `0/1`
                (for a one-hot-encoded output).

        Returns:
            pd.DataFrame:
                See `ethik.explainer.Explainer.explain_influence()`.
        """
        self._set_image_shape(images=X_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantWarning)
            return super().explain_influence(
                X_test=images_to_dataframe(X_test), y_pred=y_pred
            )

    def explain_performance(self, X_test, y_test, y_pred, metric):
        """Compute the change in model's performance for the features in `X_test`.

        Args:
            X_test (np.array): An array of images, i.e. a 3d numpy array of
                dimension `(n_images, n_rows, n_cols)`.
            y_test (pd.DataFrame or pd.Series): The true values
                for the samples in `X_test`. For binary classification and regression,
                a `pd.Series` is expected. For multi-label classification,
                a pandas dataframe with one column per label is
                expected. The values can either be probabilities or `0/1`
                (for a one-hot-encoded output).
            y_pred (pd.DataFrame or pd.Series): The model predictions
                for the samples in `X_test`. The format is the same as `y_test`.
            metric (callable): A scikit-learn-like metric
                `f(y_true, y_pred, sample_weight=None)`. The metric must be able
                to handle the `y` data. For instance, for `sklearn.metrics.accuracy_score()`,
                "the set of labels predicted for a sample must exactly match the
                corresponding set of labels in `y_true`".

        Returns:
            pd.DataFrame:
                See `ethik.explainer.Explainer.explain_performance()`.
        """
        self._set_image_shape(images=X_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantWarning)
            return super().explain_performance(
                X_test=images_to_dataframe(X_test),
                y_test=y_test,
                y_pred=y_pred,
                metric=metric,
            )

    def _get_fig_size(self, cell_width, n_rows, n_cols):
        if cell_width is None:
            cell_width = 800 / n_cols
        im_height, im_width = self.img_shape[:2]
        ratio = im_height / im_width
        cell_height = ratio * cell_width
        return n_cols * cell_width, n_rows * cell_height

    def plot_influence(self, X_test, y_pred, n_cols=3, cell_width=None):
        """Plot the influence of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or np.array): See `ImageClassificationExplainer.explain_influence()`.
            y_pred (pd.DataFrame or pd.Series): See `ImageClassificationExplainer.explain_influence()`.
            n_cols (int): The number of classes to render per row. Default is `3`.
            cell_width (int, optional): The width of each cell in pixels.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.
        """
        influences = self.explain_influence(X_test=X_test, y_pred=y_pred)
        z_values = {}

        for label, group in influences.groupby("label"):
            diffs = (
                group.query("tau == 1")["influence"]
                - group.query("tau == -1")["influence"].values
            )
            diffs = diffs.to_numpy().reshape(self.img_shape)

            # Handle multi-channel images
            if len(self.img_shape) == 3:
                diffs = diffs.max(axis=2)

            z_values[label] = diffs

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

        im_height, im_width = self.img_shape[:2]
        colorbar_width = 30
        colorbar_ticks_width = 27
        colorbar_x = 1.02

        for i, label in enumerate(labels):
            fig.add_trace(
                go.Heatmap(
                    z=z_values[label][::-1],
                    x=list(range(im_width)),
                    y=list(range(im_height)),
                    zmin=zmin,
                    zmax=zmax,
                    colorscale="RdBu",
                    zsmooth="best",
                    showscale=(i == 0),
                    name=label,
                    hoverinfo="x+y+z",
                    reversescale=True,
                    colorbar=dict(
                        thicknessmode="pixels",
                        thickness=colorbar_width,
                        xpad=0,
                        x=colorbar_x,
                    ),
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

        width, height = self._get_fig_size(cell_width, n_rows, n_cols)
        width += (colorbar_x - 1) * width + colorbar_width + colorbar_ticks_width

        fig.update_layout(
            margin=dict(t=20, l=20, b=20, r=20),
            width=width,
            height=height,
            autosize=False,
        )
        return fig

    def plot_performance(self, X_test, y_test, y_pred, metric):
        """Plot the performance of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or np.array): See `ImageClassificationExplainer.explain_performance()`.
            y_test (pd.DataFrame or pd.Series): See `ImageClassificationExplainer.explain_performance()`.
            y_pred (pd.DataFrame or pd.Series): See `ImageClassificationExplainer.explain_performance()`.
            metric (callable): See `ImageClassificationExplainer.explain_performance()`.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.

                TODO: explain what is represented on the image.
        """
        perf = self.explain_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        metric_name = self.get_metric_name(metric)

        mask = perf["label"] == perf["label"].iloc[0]
        diffs = (
            perf[mask].query(f"tau == 1")[metric_name].to_numpy()
            - perf[mask].query(f"tau == -1")[metric_name].to_numpy()
        )
        diffs = diffs.reshape(self.img_shape)

        # We want to make sure that 0 is at the center of the scale
        zmin, zmax = diffs.min(), diffs.max()
        zmin, zmax = min(zmin, -zmax), max(zmax, -zmin)

        height, width = self.img_shape

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=diffs[::-1],
                x=list(range(width)),
                y=list(range(height)),
                zmin=zmin,
                zmax=zmax,
                colorscale="RdBu",
                zsmooth="best",
                showscale=True,
                hoverinfo="x+y+z",
                reversescale=True,
            )
        )

        fig_width = 500
        fig.update_layout(
            margin=dict(t=20, l=20, b=20),
            width=fig_width,
            height=fig_width * height / width,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=height / width),
        )

        return fig
