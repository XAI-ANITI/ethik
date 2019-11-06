import collections
import functools
import itertools
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from .base_explainer import BaseExplainer
from .utils import decimal_range, to_pandas
from .warnings import ConstantWarning

__all__ = ["CacheExplainer"]


class CacheExplainer(BaseExplainer):
    """Explains the influence of features on model predictions and performance.

    Parameters:
        alpha (float): A `float` between `0` and `0.5` which indicates by how close the `Explainer`
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
            Default is `1e-3`.
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
        tol=1e-3,
        n_jobs=1,  # Parallelism is only worth it if the dataset is "large"
        memoize=False,
        verbose=True,
    ):
        super().__init__(
            alpha=alpha,
            n_taus=n_taus,
            n_samples=n_samples,
            sample_frac=sample_frac,
            conf_level=conf_level,
            max_iterations=max_iterations,
            tol=tol,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.memoize = memoize
        self.metric_names = set()
        self._reset_info()

    def _reset_info(self):
        """Resets the info dataframe (for when memoization is turned off)."""
        self.info = pd.DataFrame(
            columns=[
                "feature",
                "tau",
                "value",
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

    @property
    def taus(self):
        tau_precision = 2 / (self.n_taus - 1)
        return list(decimal_range(-1, 1, tau_precision))

    def _determine_pairs_to_do(self, features, labels):
        to_do_pairs = set(itertools.product(features, labels)) - set(
            self.info.groupby(["feature", "label"]).groups.keys()
        )
        to_do_map = collections.defaultdict(list)
        for feat, label in to_do_pairs:
            to_do_map[feat].append(label)
        return {feat: list(sorted(labels)) for feat, labels in to_do_map.items()}

    def _build_additional_info(self, X_test, y_pred):
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        # One-hot encode the categorical features
        X_test = pd.get_dummies(data=X_test, prefix_sep=self.CAT_COL_SEP)

        # Check which (feature, label) pairs have to be done
        to_do_map = self._determine_pairs_to_do(
            features=X_test.columns, labels=y_pred.columns
        )
        # We need a list to keep the order of X_test
        to_do_features = list(feat for feat in X_test.columns if feat in to_do_map)
        X_test = X_test[to_do_features]

        if X_test.empty:
            return pd.DataFrame()

        quantiles = X_test.quantile(q=[self.alpha, 1.0 - self.alpha])

        # Issue a warning if a feature doesn't have distinct quantiles
        for feature, n_unique in quantiles.nunique().to_dict().items():
            if n_unique == 1:
                warnings.warn(
                    message=f"all the values of feature {feature} are identical",
                    category=ConstantWarning,
                )

        q_mins = quantiles.loc[self.alpha].to_dict()
        q_maxs = quantiles.loc[1.0 - self.alpha].to_dict()
        means = X_test.mean().to_dict()
        info_to_complete = pd.concat(
            [
                pd.DataFrame(
                    {
                        "tau": self.taus,
                        "value": [
                            means[feature]
                            + tau
                            * (
                                max(means[feature] - q_mins[feature], 0)
                                if tau < 0
                                else max(q_maxs[feature] - means[feature], 0)
                            )
                            for tau in self.taus
                        ],
                        "feature": [feature] * len(self.taus),
                        "label": [label] * len(self.taus),
                    }
                )
                # We need to iterate over `to_do_features` to keep the order of X_test
                for feature in to_do_features
                for label in to_do_map[feature]
            ],
            ignore_index=True,
        )
        return info_to_complete

    def _explain_with_cache(self, X_test, y_pred, explain):
        if not self.memoize:
            self._reset_info()

        # We need dataframes for the return. To make the conversion faster in the
        # following methods, we do it now.
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))
        X_test = pd.get_dummies(data=X_test, prefix_sep=self.CAT_COL_SEP)

        additional_info = self._build_additional_info(X_test, y_pred)
        self.info = self.info.append(additional_info, ignore_index=True, sort=False)
        self.info = explain(query=self.info)

        return self.info[
            self.info["feature"].isin(X_test.columns)
            & self.info["label"].isin(y_pred.columns)
        ]

    def explain_influence(self, X_test, y_pred):
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

        Returns:
            pd.DataFrame:
                A dataframe with columns `(feature, tau, value, ksi, label,
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
        )

    def explain_performance(self, X_test, y_test, y_pred, metric):
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

        Returns:
            pd.DataFrame:
                A dataframe with columns `(feature, tau, value, ksi, label,
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

    def _plot_ranking(self, ranking, score_column, title, n_features=None, colors=None):
        if n_features is None:
            n_features = len(ranking)
        ascending = n_features >= 0
        ranking = ranking.sort_values(by=[score_column], ascending=ascending)
        n_features = abs(n_features)

        return go.Figure(
            data=[
                go.Bar(
                    x=ranking[score_column][-n_features:],
                    y=ranking["feature"][-n_features:],
                    orientation="h",
                    hoverinfo="x",
                    marker=dict(color=colors),
                )
            ],
            layout=go.Layout(
                margin=dict(b=0, t=40),
                xaxis=dict(
                    title=title,
                    range=[0, 1],
                    showline=True,
                    zeroline=False,
                    linecolor="black",
                    gridcolor="#eee",
                    side="top",
                    fixedrange=True,
                ),
                yaxis=dict(
                    showline=True,
                    zeroline=False,
                    fixedrange=True,
                    linecolor="black",
                    automargin=True,
                ),
                plot_bgcolor="white",
            ),
        )

    def plot_influence(self, X_test, y_pred, colors=None, yrange=None, size=None):
        """Plot the influence of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or np.array): See `Explainer.explain_influence()`.
            y_pred (pd.DataFrame or pd.Series): See `Explainer.explain_influence()`.
            colors (dict, optional): A dictionary that maps features to colors.
                Default is `None` and the colors are choosen automatically.
            yrange (list, optional): A two-item list `[low, high]`. Default is
                `None` and the range is based on the data.

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
        y_label = f"Average '{labels[0]}'"
        return self._plot_explanation(
            explanation, "influence", y_label, colors=colors, yrange=yrange, size=size
        )

    def plot_influence_ranking(self, X_test, y_pred, n_features=None, colors=None):
        """Plot the ranking of the features based on their influence.

        Args:
            X_test (pd.DataFrame or np.array): See `Explainer.explain_influence()`.
            y_pred (pd.DataFrame or pd.Series): See `Explainer.explain_influence()`.
            n_features (int, optional): The number of features to plot. With the
                default (`None`), all of them are shown. For a positive value,
                we keep the `n_features` first features (the most impactful). For
                a negative value, we keep the `n_features` last features.
            colors (dict, optional): See `Explainer.plot_influence()`.

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
        )

    def plot_performance(
        self, X_test, y_test, y_pred, metric, colors=None, yrange=None, size=None
    ):
        """Plot the performance of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or np.array): See `Explainer.explain_performance()`.
            y_test (pd.DataFrame or pd.Series): See `Explainer.explain_performance()`.
            y_pred (pd.DataFrame or pd.Series): See `Explainer.explain_performance()`.
            metric (callable): See `Explainer.explain_performance()`.
            colors (dict, optional): See `Explainer.plot_influence()`.
            yrange (list, optional): See `Explainer.plot_influence()`.

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
        if yrange is None:
            if explanation[metric_name].between(0, 1).all():
                yrange = [0, 1]

        return self._plot_explanation(
            explanation,
            metric_name,
            y_label=f"Average {metric_name}",
            colors=colors,
            yrange=yrange,
            size=size,
        )

    def plot_performance_ranking(
        self, X_test, y_test, y_pred, metric, criterion, n_features=None, colors=None
    ):
        """Plot the performance of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or np.array): See `Explainer.explain_performance()`.
            y_test (pd.DataFrame or pd.Series): See `Explainer.explain_performance()`.
            y_pred (pd.DataFrame or pd.Series): See `Explainer.explain_performance()`.
            metric (callable): See `Explainer.explain_performance()`.
            criterion (str): Either "min" or "max" to determine whether, for a
                given feature, we keep the worst or the best performance for all
                the values taken by the mean. See `Explainer.rank_by_performance()`.
            n_features (int, optional): The number of features to plot. With the
                default (`None`), all of them are shown. For a positive value,
                we keep the `n_features` first features (the most impactful). For
                a negative value, we keep the `n_features` last features.
            colors (dict, optional): See `Explainer.plot_influence_ranking()`.

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
        )
