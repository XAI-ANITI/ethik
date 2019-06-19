import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import explainer


def images_to_dataframe(images):
    img_shape = images[0].shape
    return pd.DataFrame(
        data=images.reshape(len(images), -1),
        columns=itertools.product(*[np.arange(n) for n in img_shape])
    )


class ImageExplainer(explainer.Explainer):
    """An explainer specially suited for image classification.

    This has exactly the same API as `Explainer`, but expects to be provided with an array of
    images instead of a tabular dataset.

    """

    def __init__(self, alpha=0.05, max_iterations=5, n_jobs=-1, verbose=False):
        super().__init__(
            alpha=alpha,
            n_taus=2,
            max_iterations=max_iterations,
            n_jobs=n_jobs,
            verbose=verbose
        )

    def fit(self, images):
        self.img_shape = images[0].shape
        if self.img_shape[-1] == 1:
            self.img_shape = self.img_shape[:-1]
        return super().fit(X=images_to_dataframe(images))

    def explain_predictions(self, images, y_pred):
        return super().explain_predictions(X=images_to_dataframe(images), y_pred=y_pred)

    def plot_predictions(self, images, y_pred, n_cols=3, width=4):

        means = self.explain_predictions(images=images, y_pred=y_pred)

        if isinstance(means.columns, pd.MultiIndex):

            # Determine the number of rows from the number of columns and the number of labels
            labels = means.columns.unique('labels')
            n_rows = math.ceil(len(labels) / n_cols)

            # Make one plot per label
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * width, n_rows * width))
            for ax in axes.ravel()[len(labels):]:
                fig.delaxes(ax)

            v_min = math.inf
            v_max = -math.inf

            for label, ax in zip(labels, axes.flat):
                label_means = means[label]
                diffs = (label_means.iloc[-1] - label_means.iloc[0]).fillna(0.)

                v_min = min(v_min, diffs.min())
                v_max = max(v_max, diffs.max())

                ax.imshow(diffs.to_numpy().reshape(self.img_shape), interpolation='none')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_title(label, fontsize=20)

            for img in plt.gca().get_images():
                img.set_clim(v_min, v_max)

            cbar = fig.colorbar(img, ax=axes.flat)
            cbar.ax.tick_params(labelsize=20)

            return fig, axes

        fig, ax = plt.subplots()
        diffs = (means.iloc[-1] - means.iloc[0]).fillna(0.)
        img = ax.imshow(diffs.to_numpy().reshape(self.img_shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        fig.colorbar(img, ax=ax)

        return ax

    def plot_metric(self, images, y, y_pred, metric):

        metrics = self.explain_metric(
            X=images_to_dataframe(images),
            y=y,
            y_pred=y_pred,
            metric=metric
        )

        # Create a plot if none is provided
        fig, ax = plt.subplots()

        diffs = (metrics.iloc[-1] - metrics.iloc[0]).fillna(0.)
        img = ax.imshow(diffs.to_numpy().reshape(self.img_shape))

        fig.colorbar(img, ax=ax)

        return ax
