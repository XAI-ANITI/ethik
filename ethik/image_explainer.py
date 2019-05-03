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
        columns=map(str, itertools.product(*[np.arange(n) for n in img_shape]))
    )


class ImageExplainer(explainer.Explainer):

    def __init__(self, alpha=0.05, max_iterations=5, n_jobs=-1, verbose=False):
        super().__init__(
            alpha=alpha,
            n_taus=3,
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

            labels = means.columns.unique('labels')
            n_rows = math.ceil(len(labels) / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * width, n_rows * width))

            for label, ax in zip(labels, axes.ravel()):
                label_means = means[label]
                diffs = (label_means.iloc[-1] - label_means.iloc[0]).fillna(0.)

                ax.imshow(diffs.to_numpy().reshape(self.img_shape), interpolation='none')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_title(label)

            # Remove unused axes
            for ax in axes.ravel()[len(labels):]:
                fig.delaxes(ax)

            return fig, axes

        ax = plt.axes()
        diffs = (means.iloc[-1] - means.iloc[0]).fillna(0.)
        ax.imshow(diffs.to_numpy().reshape(self.img_shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        return ax

    def plot_metric(self, images, y, y_pred, metric, ax=None):

        metrics = self.explain_metric(
            X=images_to_dataframe(images),
            y=y,
            y_pred=y_pred,
            metric=metric
        )

        # Create a plot if none is provided
        ax = plt.axes() if ax is None else ax

        diffs = (metrics.loc[-1] - metrics.loc[0]).fillna(0.)
        ax.imshow(diffs.to_numpy().reshape(self.img_shape))

        return ax
