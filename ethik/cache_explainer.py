import collections
import functools
import itertools
import math
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from .base_explainer import BaseExplainer
from .query import Query
from .utils import set_fig_size, to_pandas

__all__ = ["CacheExplainer"]


class CacheExplainer(BaseExplainer):
    """Explains the influence of features on model predictions and performance.

    Parameters:
        alpha (float): A `float` between `0` and `0.5` which indicates by how close the `CacheExplainer`
            should look at extreme values of a distribution. The closer to zero, the more so
            extreme values will be accounted for. The default is `0.05` which means that all values
            beyond the 5th and 95th quantiles are ignored.
        n_taus (int): The number of τ values to consider. The results will be more fine-grained the
            higher this value is. However the computation time increases linearly with `n_taus`.
            The default is `41` and corresponds to each τ being separated by it's neighbors by
            `0.05`.
        n_samples (int): The number of samples to use for the confidence interval.
            If `1`, the default, no confidence interval is computed.
        sample_frac (float): The proportion of lines in the dataset sampled to
            generate the samples for the confidence interval. If `n_samples` is
            `1`, no confidence interval is computed and the whole dataset is used.
            Default is `0.8`.
        conf_level (float): A `float` between `0` and `0.5` which indicates the
            quantile used for the confidence interval. Default is `0.05`, which
            means that the confidence interval contains the data between the 5th
            and 95th quantiles.
        max_iterations (int): The maximum number of iterations used when applying the Newton step
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
            `plot_influence` will be reused for `plot_influence_ranking`. Memoization is turned off by
            default because it can lead to unexpected behavior depending on your usage.
        verbose (bool): Whether or not to show progress bars during
            computations. Default is `True`.
    """

    def __init__(
        self,
        alpha=0.05,
        n_taus=41,
        n_samples=1,
        sample_frac=0.8,
        conf_level=0.05,
        max_iterations=15,
        tol=1e-4,
        n_jobs=1,  # Parallelism is only worth it if the dataset is "large"
        memoize=False,
        verbose=True,
    ):
        super().__init__(
            alpha=alpha,
            n_samples=n_samples,
            sample_frac=sample_frac,
            conf_level=conf_level,
            max_iterations=max_iterations,
            tol=tol,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        if not n_taus > 0:
            raise ValueError(
                f"n_taus must be a strictly positive integer, got {n_taus}"
            )

        self.n_taus = n_taus
        self.memoize = memoize
        self.metric_names = set()
        self._reset_info()

    def _reset_info(self):
        """Resets the info dataframe (for when memoization is turned off)."""
        self.info = pd.DataFrame(
            columns=[
                "group",
                "dimension",
                "feature",
                "tau",
                "target",
                "ksi",
                "label",
                "influence",
                "influence_low",
                "influence_high",
            ]
        )

    def get_metric_name(self, metric):
        """Get the name of the column in explainer's info dataframe to store the
        performance with respect of the given metric.

        Args:
            metric (callable): The metric to compute the model's performance.

        Returns:
            str: The name of the column.
        """
        name = super().get_metric_name(metric)
        if name in self.info.columns and name not in self.metric_names:
            raise ValueError(f"Cannot use {name} as a metric name")
        return name

    @property
    def features(self):
        return self.info["feature"].unique().tolist()

    def _determine_pairs_to_do(self, features, labels, multi_dim=False):
        dim = 1 if not multi_dim else len(features)
        unidim_info = self.info[self.info["dimension"] == dim]
        to_do_pairs = set(itertools.product(features, labels)) - set(
            unidim_info.groupby(["feature", "label"]).groups.keys()
        )
        to_do_map = collections.defaultdict(list)
        for feat, label in to_do_pairs:
            to_do_map[feat].append(label)
        return {feat: list(sorted(labels)) for feat, labels in to_do_map.items()}

    def _build_additional_info(self, X_test, y_pred, link_variables):
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))
        X_test = self._one_hot_encode(X_test)
        last_group = self.info["group"].max()

        if link_variables:
            # For a multi-dimensional query, it's too complicated to check if
            # it's already in the cache
            return Query.multidim_from_taus(
                X_test=X_test,
                labels=y_pred.columns,
                n_taus=self.n_taus,  #  TODO: it's a lot of points
                q=[self.alpha, 1 - self.alpha],
                first_group=0 if math.isnan(last_group) else last_group + 1,
            )

        # Check which (feature, label) pairs have to be done
        to_do_map = self._determine_pairs_to_do(
            features=X_test.columns, labels=y_pred.columns
        )
        # We need a list to keep the order of X_test
        to_do_features = list(feat for feat in X_test.columns if feat in to_do_map)
        X_test = X_test[to_do_features]

        if X_test.empty:
            return pd.DataFrame()

        return Query.unidim_from_taus(
            X_test=X_test,
            to_do_labels=to_do_map,
            n_taus=self.n_taus,
            q=[self.alpha, 1 - self.alpha],
            first_group=0 if math.isnan(last_group) else last_group + 1,
        )

    def _explain_with_cache(self, X_test, y_pred, explain, link_variables=False):
        if not self.memoize:
            self._reset_info()

        # We need dataframes for the return. To make the conversion faster in the
        # following methods, we do it now.
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))
        X_test = self._one_hot_encode(X_test)

        # TODO: when `link_variables` is `True`, the explanation is duplicated
        # (try to call the method twice)

        additional_info = self._build_additional_info(X_test, y_pred, link_variables)
        self.info = self.info.append(additional_info, ignore_index=True, sort=False)
        self.info = explain(query=self.info)

        if not link_variables:
            return self.info[
                self.info["feature"].isin(X_test.columns)
                & self.info["label"].isin(y_pred.columns)
                & (self.info["dimension"] == 1)
            ]

        ret = self.info[
            self.info["label"].isin(y_pred.columns)
            & (self.info["dimension"] == len(X_test.columns))
        ]

        # We find the groups that exactly  contain the features of `X_test`
        groups = set()
        queried_features = set(X_test.columns)
        for group, part in ret.groupby(["group"]):
            if set(part["feature"]) == queried_features:
                groups.add(group)

        return ret[ret["group"].isin(groups)]

    def explain_influence(self, X_test, y_pred, link_variables=False):
        """Compute the influence of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or pd.Series): The dataset as a pandas dataframe
                with one column per feature or a pandas series for a single feature.
            y_pred (pd.DataFrame or pd.Series): The model predictions
                for the samples in `X_test`. For binary classification and regression,
                `pd.Series` is expected. For multi-label classification, a
                pandas dataframe with one column per label is
                expected. The values can either be probabilities or `0/1`
                (for a one-hot-encoded output).
            link_variables (bool, optional): Whether to make a multidimensional
                explanation or not. Default is `False`, which means that all
                the features in `X_test` are considered independently (so the
                correlation are not taken into account).

        Returns:
            pd.DataFrame:
                A dataframe with columns `(feature, tau, target, ksi, label,
                influence, influence_low, influence_high)`. If `explainer.n_samples` is `1`,
                no confidence interval is computed and `influence = influence_low = influence_high`.
                The value of `label` is not important for regression.

        Examples:
            See more examples in `notebooks`.

            Binary classification:

            >>> X_test = pd.DataFrame([
            ...     [1, 2],
            ...     [1.1, 2.2],
            ...     [1.3, 2.3],
            ... ], columns=["x0", "x1"])
            >>> y_pred = model.predict(X_test)
            >>> y_pred
            [0, 1, 1]  # Can also be probabilities: [0.3, 0.65, 0.8]
            >>> # For readibility reasons, we give a name to the predictions
            >>> y_pred = pd.Series(y_pred, name="is_reliable")
            >>> explainer.explain_influence(X_test, y_pred)

            Regression is similar to binary classification:

            >>> X_test = pd.DataFrame([
            ...     [1, 2],
            ...     [1.1, 2.2],
            ...     [1.3, 2.3],
            ... ], columns=["x0", "x1"])
            >>> y_pred = model.predict(X_test)
            >>> y_pred
            [22, 24, 19]
            >>> # For readibility reasons, we give a name to the predictions
            >>> y_pred = pd.Series(y_pred, name="price")
            >>> explainer.explain_influence(X_test, y_pred)

            For multi-label classification, we need a dataframe to store predictions:

            >>> X_test = pd.DataFrame([
            ...     [1, 2],
            ...     [1.1, 2.2],
            ...     [1.3, 2.3],
            ... ], columns=["x0", "x1"])
            >>> y_pred = model.predict(X_test)
            >>> y_pred.columns
            ["class0", "class1", "class2"]
            >>> y_pred.iloc[0]
            [0, 1, 0] # One-hot encoded, or probabilities: [0.15, 0.6, 0.25]
            >>> explainer.explain_influence(X_test, y_pred)
        """
        return self._explain_with_cache(
            X_test,
            y_pred,
            explain=functools.partial(
                self._explain_influence, X_test=X_test, y_pred=y_pred
            ),
            link_variables=link_variables,
        )

    def explain_performance(self, X_test, y_test, y_pred, metric, link_variables=False):
        """Compute the change in model's performance for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or pd.Series): The dataset as a pandas dataframe
                with one column per feature or a pandas series for a single feature.
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
            link_variables (bool, optional): Whether to make a multidimensional
                explanation or not. Default is `False`, which means that all
                the features in `X_test` are considered independently (so the
                correlation are not taken into account).

        Returns:
            pd.DataFrame:
                A dataframe with columns `(feature, tau, target, ksi, label,
                influence, influence_low, influence_high, <metric_name>, <metric_name_low>, <metric_name_high>)`.
                If `explainer.n_samples` is `1`, no confidence interval is computed
                and `<metric_name> = <metric_name_low> = <metric_name_high>`.
                The value of `label` is not important for regression.

        Examples:
            See examples in `notebooks`.
        """
        metric_name = self.get_metric_name(metric)
        self.metric_names.add(metric_name)
        return self._explain_with_cache(
            X_test,
            y_pred,
            explain=functools.partial(
                self._explain_performance,
                X_test=X_test,
                y_pred=y_pred,
                y_test=y_test,
                metric=metric,
            ),
            link_variables=link_variables,
        )

    def rank_by_influence(self, X_test, y_pred):
        """Returns a pandas DataFrame containing the importance of each feature
        per label.

        Args:
            X_test (pd.DataFrame or pd.Series): The dataset as a pandas dataframe
                with one column per feature or a pandas series for a single feature.
            y_pred (pd.DataFrame or pd.Series): The model predictions
                for the samples in `X_test`. For binary classification and regression,
                a `pd.Series` is expected. For multi-label classification,
                a pandas dataframe with one column per label is
                expected. The values can either be probabilities or `0/1`
                (for a one-hot-encoded output).

        Returns:
            pd.DataFrame:
                A dataframe with columns `(label, feature, importance)`. The row
                `(setosa, petal length (cm), 0.282507)` means that the feature
                `petal length` of the Iris dataset has an importance of about
                30% in the prediction of the class `setosa`.

                The importance is a real number between 0 and 1. Intuitively,
                if the model influence for the feature `X` is a flat curve (the average
                model prediction is not impacted by the mean of `X`) then we
                can conclude that `X` has no importance for predictions. This
                flat curve is the baseline and satisfies \\(y = influence_{\\tau(0)}\\).
                To compute the importance of a feature, we look at the average
                distance of the influence curve to this baseline:

                $$
                I(X) = \\frac{1}{n_\\tau} \\sum_{i=1}^{n_\\tau} \\mid influence_{\\tau(i)}(X) - influence_{\\tau(0)}(X) \\mid
                $$

                The influence curve is first normalized so that the importance is
                between 0 and 1 (which may not be the case originally for regression
                problems). To normalize, we get the minimum and maximum influences
                *across all features and all classes* and then compute
                `normalized = (influence - min) / (max - min)`.

                For regression problems, there's one label only and its name
                doesn't matter (it's just to have a consistent output).
        """

        def get_importance(group, min_influence, max_influence):
            """Computes the average absolute difference in influence changes per tau increase."""
            #  Normalize influence to get an importance between 0 and 1
            # influence can be outside [0, 1] for regression
            influence = group["influence"]
            group["influence"] = (influence - min_influence) / (
                max_influence - min_influence
            )
            baseline = group.query("tau == 0").iloc[0]["influence"]
            return (group["influence"] - baseline).abs().mean()

        explanation = self.explain_influence(X_test=X_test, y_pred=y_pred)
        min_influence = explanation["influence"].min()
        max_influence = explanation["influence"].max()

        return (
            explanation.groupby(["label", "feature"])
            .apply(
                functools.partial(
                    get_importance,
                    min_influence=min_influence,
                    max_influence=max_influence,
                )
            )
            .to_frame("importance")
            .reset_index()
        )

    def rank_by_performance(self, X_test, y_test, y_pred, metric):
        """Returns a pandas DataFrame containing
        per label.

        Args:
            X_test (pd.DataFrame or pd.Series): The dataset as a pandas dataframe
                with one column per feature or a pandas series for a single feature.
            y_test (pd.DataFrame or pd.Series): The true output
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
                A dataframe with columns `(feature, min, max)`. The row
                `(age, 0.862010, 0.996360)` means that the score measured by the
                given metric (e.g. `sklearn.metrics.accuracy_score`) stays bewteen
                86.2% and 99.6% on average when we make the mean age change. With
                such information, we can find the features for which the model
                performs the worst or the best.

                For regression problems, there's one label only and its name
                doesn't matter (it's just to have a consistent output).
        """
        metric_name = self.get_metric_name(metric)

        def get_aggregates(df):
            return pd.Series(
                [df[metric_name].min(), df[metric_name].max()], index=["min", "max"]
            )

        return (
            self.explain_performance(X_test, y_test, y_pred, metric)
            .groupby("feature")
            .apply(get_aggregates)
            .reset_index()
        )

    def _plot_explanation(
        self, explanation, y_col, y_label, colors=None, yrange=None, size=None
    ):
        features = explanation["feature"].unique()

        if colors is None:
            colors = {}
        elif type(colors) is str:
            colors = {feat: colors for feat in features}

        #  There are multiple features, we plot them together with taus
        if len(features) > 1:
            fig = go.Figure()

            for i, feat in enumerate(features):
                taus = explanation.query(f'feature == "{feat}"')["tau"]
                targets = explanation.query(f'feature == "{feat}"')["target"]
                y = explanation.query(f'feature == "{feat}"')[y_col]
                fig.add_trace(
                    go.Scatter(
                        x=taus,
                        y=y,
                        mode="lines+markers",
                        hoverinfo="y",
                        name=feat,
                        customdata=list(zip(taus, targets)),
                        marker=dict(color=colors.get(feat)),
                    )
                )

            fig.update_layout(
                margin=dict(t=30, r=50, b=40),
                xaxis=dict(title="tau", nticks=5),
                yaxis=dict(title=y_label, range=yrange),
            )
            set_fig_size(fig, size)
            return fig

        #  There is only one feature, we plot it with its nominal values.
        feat = features[0]
        fig = go.Figure()
        x = explanation.query(f'feature == "{feat}"')["target"]
        y = explanation.query(f'feature == "{feat}"')[y_col]
        mean_row = explanation.query(f'feature == "{feat}" and tau == 0').iloc[0]

        if self.n_samples > 1:
            low = explanation.query(f'feature == "{feat}"')[f"{y_col}_low"]
            high = explanation.query(f'feature == "{feat}"')[f"{y_col}_high"]
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate((x, x[::-1])),
                    y=np.concatenate((low, high[::-1])),
                    name=f"{self.conf_level * 100}% - {(1 - self.conf_level) * 100}%",
                    fill="toself",
                    fillcolor=colors.get(feat),
                    line_color="rgba(0, 0, 0, 0)",
                    opacity=0.3,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                hoverinfo="x+y",
                showlegend=False,
                marker=dict(color=colors.get(feat)),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[mean_row["target"]],
                y=[mean_row[y_col]],
                text=["Dataset mean"],
                showlegend=False,
                mode="markers",
                name="Original mean",
                hoverinfo="text",
                marker=dict(symbol="x", size=9, color=colors.get(feat)),
            )
        )
        fig.update_layout(
            margin=dict(t=30, r=0, b=40),
            xaxis=dict(title=f"Average {feat}"),
            yaxis=dict(title=y_label, range=yrange),
        )
        set_fig_size(fig, size)
        return fig

    def _plot_explanation_2d(
        self, explanation, z_col, z_label, z_range=None, colorscale=None, size=None
    ):
        fx, fy = explanation["feature"].unique()
        x = np.sort(explanation[explanation["feature"] == fx]["target"].unique())
        y = np.sort(explanation[explanation["feature"] == fy]["target"].unique())

        z = explanation[explanation["feature"] == fx][z_col].values
        z = z.reshape((len(x), len(y)))

        z_min = z_max = None
        if z_range is not None:
            z_min, z_max = z_range

        fig = go.Figure()
        fig.add_heatmap(
            x=x,
            y=y,
            z=z,
            zmin=z_min,
            zmax=z_max,
            colorscale=colorscale or "Reds",
            colorbar=dict(title=z_label),
        )

        fig.update_layout(
            margin=dict(t=30, b=40),
            xaxis=dict(title=f"Average {fx}", mirror=True),
            yaxis=dict(title=f"Average {fy}", mirror=True),
        )
        set_fig_size(fig, size)
        return fig

    def _plot_ranking(
        self, ranking, score_column, title, n_features=None, colors=None, size=None
    ):
        if n_features is None:
            n_features = len(ranking)
        ascending = n_features >= 0
        ranking = ranking.sort_values(by=[score_column], ascending=ascending)
        n_features = abs(n_features)

        fig = go.Figure()
        fig.add_bar(
            x=ranking[score_column][-n_features:],
            y=ranking["feature"][-n_features:],
            orientation="h",
            hoverinfo="x",
            marker=dict(color=colors),
        )
        fig.update_layout(
            margin=dict(b=0, t=40, r=10),
            xaxis=dict(title=title, range=[0, 1], side="top", fixedrange=True),
            yaxis=dict(fixedrange=True, automargin=True),
            modebar=dict(
                orientation="v",
                color="rgba(0, 0, 0, 0)",
                activecolor="rgba(0, 0, 0, 0)",
                bgcolor="rgba(0, 0, 0, 0)",
            ),
        )
        set_fig_size(fig, size, width=500, height=100 + 60 * n_features)
        return fig

    def plot_influence(self, X_test, y_pred, colors=None, yrange=None, size=None):
        """Plot the influence of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame): See `CacheExplainer.explain_influence()`.
            y_pred (pd.DataFrame or pd.Series): See `CacheExplainer.explain_influence()`.
            colors (dict, optional): A dictionary that maps features to colors.
                Default is `None` and the colors are choosen automatically.
            yrange (list, optional): A two-item list `[low, high]`. Default is
                `None` and the range is based on the data.
            size (tuple, optional): An optional couple `(width, height)` in pixels.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.

        Examples:
            >>> explainer.plot_influence(X_test, y_pred)
            >>> explainer.plot_influence(X_test, y_pred, colors=dict(
            ...     x0="blue",
            ...     x1="red",
            ... ))
            >>> explainer.plot_influence(X_test, y_pred, yrange=[0.5, 1])
        """
        explanation = self.explain_influence(X_test, y_pred)
        labels = explanation["label"].unique()

        if len(labels) > 1:
            raise ValueError("Cannot plot multiple labels")

        return self._plot_explanation(
            explanation,
            y_col="influence",
            y_label=f"Average '{labels[0]}'",
            colors=colors,
            yrange=yrange,
            size=size,
        )

    def plot_influence_2d(
        self, X_test, y_pred, z_range=None, colorscale=None, size=None
    ):
        """Plot the influence of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame): The dataset as a pandas dataframe
                with one column per feature. Must contain exactly two columns.
            y_pred (pd.DataFrame or pd.Series): See `CacheExplainer.explain_influence()`.
            colorscale (str, optional): The colorscale used for the heatmap.
                See `plotly.graph_objs.Heatmap`. Default is `None`.
            size (tuple, optional): An optional couple `(width, height)` in pixels.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.
        """
        if len(X_test.columns) != 2:
            raise ValueError("`X_test` must contain exactly two columns.")

        explanation = self.explain_influence(X_test, y_pred, link_variables=True)
        labels = explanation["label"].unique()

        if len(labels) > 1:
            raise ValueError("Cannot plot multiple labels")

        return self._plot_explanation_2d(
            explanation,
            z_col="influence",
            z_label=f"Average '{labels[0]}'",
            z_range=z_range,
            colorscale=colorscale,
            size=size,
        )

    def plot_influence_ranking(
        self, X_test, y_pred, n_features=None, colors=None, size=None
    ):
        """Plot the ranking of the features based on their influence.

        Args:
            X_test (pd.DataFrame or np.array): See `CacheExplainer.explain_influence()`.
            y_pred (pd.DataFrame or pd.Series): See `CacheExplainer.explain_influence()`.
            n_features (int, optional): The number of features to plot. With the
                default (`None`), all of them are shown. For a positive value,
                we keep the `n_features` first features (the most impactful). For
                a negative value, we keep the `n_features` last features.
            colors (dict, optional): See `CacheExplainer.plot_influence()`.
            size (tuple, optional): An optional couple `(width, height)` in pixels.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.
        """
        ranking = self.rank_by_influence(X_test=X_test, y_pred=y_pred)
        return self._plot_ranking(
            ranking=ranking,
            score_column="importance",
            title="Importance",
            n_features=n_features,
            colors=colors,
            size=size,
        )

    def plot_performance(
        self, X_test, y_test, y_pred, metric, colors=None, yrange=None, size=None
    ):
        """Plot the performance of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or np.array): See `CacheExplainer.explain_performance()`.
            y_test (pd.DataFrame or pd.Series): See `CacheExplainer.explain_performance()`.
            y_pred (pd.DataFrame or pd.Series): See `CacheExplainer.explain_performance()`.
            metric (callable): See `CacheExplainer.explain_performance()`.
            colors (dict, optional): See `CacheExplainer.plot_influence()`.
            yrange (list, optional): See `CacheExplainer.plot_influence()`.
            size (tuple, optional): An optional couple `(width, height)` in pixels.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.
        """
        metric_name = self.get_metric_name(metric)
        explanation = self.explain_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        if yrange is None and explanation[metric_name].between(0, 1).all():
            yrange = [0, 1]

        #  The performance is the same for all labels, we remove duplicates
        label = explanation["label"].unique()[0]
        explanation = explanation[explanation["label"] == label]

        return self._plot_explanation(
            explanation,
            y_col=metric_name,
            y_label=f"Average {metric_name}",
            colors=colors,
            yrange=yrange,
            size=size,
        )

    def plot_performance_2d(
        self, X_test, y_test, y_pred, metric, z_range=None, colorscale=None, size=None
    ):
        """Plot the influence of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame): The dataset as a pandas dataframe
                with one column per feature. Must contain exactly two columns.
            y_pred (pd.DataFrame or pd.Series): See `CacheExplainer.explain_influence()`.
            colorscale (str, optional): The colorscale used for the heatmap.
                See `plotly.graph_objs.Heatmap`. Default is `None`.
            size (tuple, optional): An optional couple `(width, height)` in pixels.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.
        """
        if len(X_test.columns) != 2:
            raise ValueError("`X_test` must contain exactly two columns.")

        metric_name = self.get_metric_name(metric)
        explanation = self.explain_performance(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            metric=metric,
            link_variables=True,
        )
        if z_range is None and explanation[metric_name].between(0, 1).all():
            z_range = [0, 1]

        return self._plot_explanation_2d(
            explanation,
            z_col=metric_name,
            z_label=f"Average {metric_name}",
            z_range=z_range,
            colorscale=colorscale,
            size=size,
        )

    def plot_performance_ranking(
        self,
        X_test,
        y_test,
        y_pred,
        metric,
        criterion,
        n_features=None,
        colors=None,
        size=None,
    ):
        """Plot the performance of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or np.array): See `CacheExplainer.explain_performance()`.
            y_test (pd.DataFrame or pd.Series): See `CacheExplainer.explain_performance()`.
            y_pred (pd.DataFrame or pd.Series): See `CacheExplainer.explain_performance()`.
            metric (callable): See `CacheExplainer.explain_performance()`.
            criterion (str): Either "min" or "max" to determine whether, for a
                given feature, we keep the worst or the best performance for all
                the values taken by the mean. See `CacheExplainer.rank_by_performance()`.
            n_features (int, optional): The number of features to plot. With the
                default (`None`), all of them are shown. For a positive value,
                we keep the `n_features` first features (the most impactful). For
                a negative value, we keep the `n_features` last features.
            colors (dict, optional): See `CacheExplainer.plot_influence_ranking()`.
            size (tuple, optional): An optional couple `(width, height)` in pixels.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.
        """
        metric_name = self.get_metric_name(metric)
        ranking = self.rank_by_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        return self._plot_ranking(
            ranking=ranking,
            score_column=criterion,
            title=f"{criterion} {metric_name}",
            n_features=n_features,
            colors=colors,
            size=size,
        )
