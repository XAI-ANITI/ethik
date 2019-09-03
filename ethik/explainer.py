import collections
import decimal
import functools

import joblib
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go


__all__ = ["Explainer"]


def plot(fig, inline=False):
    if inline:
        plotly.offline.init_notebook_mode(connected=True)
        return plotly.offline.iplot(fig)
    return plotly.offline.plot(fig, auto_open=True)


def decimal_range(start: float, stop: float, step: float):
    """Like the `range` function but works for decimal values.

    This is more accurate than using `np.arange` because it doesn't introduce
    any round-off errors.

    """
    start = decimal.Decimal(str(start))
    stop = decimal.Decimal(str(stop))
    step = decimal.Decimal(str(step))
    while start <= stop:
        yield float(start)
        start += step


def to_pandas(x):
    """Converts an array-like to a Series or a DataFrame depending on the dimensionality."""

    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x

    if isinstance(x, np.ndarray):
        if x.ndim > 2:
            raise ValueError("x must have 1 or 2 dimensions")
        if x.ndim == 2:
            return pd.DataFrame(x)
        return pd.Series(x)

    return to_pandas(np.asarray(x))


def merge_dicts(dicts):
    return dict(collections.ChainMap(*dicts))


def compute_lambdas(x, target_means, max_iterations=5):
    """Finds a good lambda for a variable and a given epsilon value."""

    mean = x.mean()
    lambdas = {}

    for target_mean in target_means:

        λ = 0
        current_mean = mean

        for _ in range(max_iterations):

            # Update the sample weights and see where the mean is
            sample_weights = np.exp(λ * x)
            sample_weights = sample_weights / sum(sample_weights)
            current_mean = np.average(x, weights=sample_weights)

            # Do a Newton step using the difference between the mean and the
            # target mean
            grad = current_mean - target_mean
            hess = np.average((x - current_mean) ** 2, weights=sample_weights)
            step = grad / hess
            λ -= step

        lambdas[(x.name, target_mean)] = λ

    return lambdas


def compute_bias(y_pred, x, lambdas, sample_index):
    return [
        (
            x.name,
            y_pred.name,
            λ,
            sample_index,
            np.average(y_pred, weights=np.exp(λ * x)),
        )
        for λ in lambdas
    ]


def compute_performance(y_test, y_pred, metric, x, lambdas, sample_index):
    return [
        (x.name, λ, sample_index, metric(y_test, y_pred, sample_weight=np.exp(λ * x)))
        for λ in lambdas
    ]


def metric_to_col(metric):
    # TODO: what about lambda metrics?
    # TODO: use a prefix to avoid conflicts with other columns?
    return metric.__name__


def make_dataset_numeric(X):
    X_num = X.select_dtypes(exclude=["object", "category"])
    X_cat = X.select_dtypes(include=["object", "category"])
    cat_features = {}
    for cat_feat in X_cat:
        values = X_cat[cat_feat].unique()
        new_num_cols = [f"{cat_feat}__{v}" for v in values]
        cat_features[cat_feat] = new_num_cols
        for v, new_col in zip(values, new_num_cols):
            X_num[new_col] = (X_cat[cat_feat] == v).astype(int)
    return X_num, cat_features


def yield_masks(n_masks, n, p):
    for _ in range(n_masks):
        yield np.random.binomial(1, p, size=n).astype(bool)


class Explainer:
    """Explains the bias and reliability of model predictions.

    Parameters:
        alpha (float): A `float` between `0` and `0.5` which indicates by how close the `Explainer`
            should look at extreme values of a distribution. The closer to zero, the more so
            extreme values will be accounted for. The default is `0.05` which means that all values
            beyond the 5th and 95th quantiles are ignored.
        n_taus (int): The number of τ values to consider. The results will be more fine-grained the
            higher this value is. However the computation time increases linearly with `n_taus`.
            The default is `41` and corresponds to each τ being separated by it's neighbors by
            `0.05`.
        max_iterations (int): The maximum number of iterations used when applying the Newton step
            of the optimization procedure.

    """

    def __init__(
        self,
        alpha=0.05,
        n_taus=41,
        max_iterations=5,
        n_jobs=-1,
        verbose=False,
        n_samples=1,
        sample_frac=0.8,
        conf_level=0.05,
    ):
        if not 0 < alpha < 0.5:
            raise ValueError("alpha must be between 0 and 0.5, got " f"{alpha}")

        if not n_taus > 0:
            raise ValueError(
                "n_taus must be a strictly positive integer, got " f"{n_taus}"
            )

        if not max_iterations > 0:
            raise ValueError(
                "max_iterations must be a strictly positive "
                f"integer, got {max_iterations}"
            )

        if n_samples < 1:
            raise ValueError(f"n_samples must be strictly positive, got {n_samples}")

        if not 0 < sample_frac < 1:
            raise ValueError(f"sample_frac must be between 0 and 1, got {sample_frac}")

        if not 0 < conf_level < 0.5:
            raise ValueError(f"conf_level must be between 0 and 0.5, got {conf_level}")

        #  TODO: one column per performance metric
        self.alpha = alpha
        self.n_taus = n_taus
        self.max_iterations = max_iterations
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.metric_cols = set()
        self.n_samples = n_samples
        self.sample_frac = sample_frac if n_samples > 1 else 1
        self.conf_level = conf_level
        self.info = pd.DataFrame(
            columns=[
                "feature",
                "tau",
                "value",
                "lambda",
                "label",
                "bias",
                "bias_low",
                "bias_high",
            ]
        )
        self.cat_features = {}

    @property
    def taus(self):
        tau_precision = 2 / (self.n_taus - 1)
        return list(decimal_range(-1, 1, tau_precision))

    @property
    def features(self):
        return self.info["feature"].unique().tolist() + list(self.cat_features)

    def _fit(self, X_test, y_pred):
        """Fits the explainer to a tabular dataset.

        During a `fit` call, the following steps are taken:

        1. A list of $\tau$ values is generated using `n_taus`. The $\tau$ values range from -1 to 1.
        2. A grid of $\eps$ values is generated for each $\tau$ and for each variable. Each $\eps$ represents a shift from a variable's mean towards a particular quantile.
        3. A grid of $\lambda$ values is generated for each $\eps$. Each $\lambda$ corresponds to the optimal parameter that has to be used to weight the observations in order for the average to reach the associated $\eps$ shift.

        Parameters:
            X_test (`pandas.DataFrame` or `numpy.ndarray`)
            y_pred (`pandas.DataFrame` or `numpy.ndarray`)

        """

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        X_test = X_test[X_test.columns.difference(self.features)]
        if X_test.empty:
            return self

        X_test_num, cat_features = make_dataset_numeric(X_test)
        self.cat_features = merge_dicts([self.cat_features, cat_features])

        # Make the epsilons
        q_mins = X_test_num.quantile(q=self.alpha).to_dict()
        q_maxs = X_test_num.quantile(q=1 - self.alpha).to_dict()
        means = X_test_num.mean().to_dict()
        additional_info = pd.concat(
            [
                pd.DataFrame(
                    {
                        "tau": self.taus,
                        "value": [
                            means[col]
                            + tau
                            * (
                                (means[col] - q_mins[col])
                                if tau < 0
                                else (q_maxs[col] - means[col])
                            )
                            for tau in self.taus
                        ],
                        "feature": col,
                        "label": y,
                    }
                )
                for col in X_test_num.columns
                for y in y_pred.columns
            ],
            ignore_index=True,
        )
        self.info = self.info.append(additional_info, ignore_index=True, sort=False)

        # Find a lambda for each (column, espilon) pair
        lambdas = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            joblib.delayed(compute_lambdas)(
                x=X_test_num[col],
                target_means=part["value"].unique(),
                max_iterations=self.max_iterations,
            )
            for col, part in self.info.groupby("feature")
            if col in X_test_num
        )
        lambdas = merge_dicts(lambdas)
        self.info["lambda"] = self.info.apply(
            lambda r: lambdas.get((r["feature"], r["value"]), r["lambda"]),
            axis="columns",
        )

        return self.info

    def explain_bias(self, X_test, y_pred):
        """Returns a DataFrame containing average predictions for each (column, tau) pair.

        """
        # Coerce X_test and y_pred to DataFrames
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        self._fit(X_test, y_pred)

        X_test, _ = make_dataset_numeric(X_test)
        queried_features = X_test.columns.tolist()
        to_explain = self.info["feature"][self.info["bias"].isnull()].unique()
        X_test = X_test[X_test.columns.intersection(to_explain)]

        # Discard the features that are not relevant
        relevant = self.info.query(f"feature in {X_test.columns.tolist()}")
        if relevant.empty:
            return self.info.query(f"feature in {queried_features}")

        # Compute the average predictions for each (column, tau) pair per label
        biases = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            joblib.delayed(compute_bias)(
                y_pred=y_pred[label][mask],
                x=X_test[col][mask],
                lambdas=part["lambda"].unique(),
                sample_index=i,
            )
            for label in y_pred.columns
            for col, part in relevant.groupby("feature")
            for i, mask in enumerate(
                yield_masks(self.n_samples, len(X_test), self.sample_frac)
            )
        )
        biases = pd.DataFrame(
            [t for b in biases for t in b],
            columns=["feature", "label", "lambda", "sample_index", "bias"],
        )
        biases = biases.groupby(["feature", "label", "lambda"])["bias"].agg(
            [
                ("bias", "mean"),
                ("bias_low", functools.partial(np.quantile, q=self.conf_level)),
                ("bias_high", functools.partial(np.quantile, q=1 - self.conf_level)),
            ]
        )

        #  TODO: merge all columns in a single operation?
        for col in biases.columns:
            self.info[col] = self.info.apply(
                lambda r: biases[col].get(
                    (r["feature"], r["label"], r["lambda"]), r[col]
                ),
                axis="columns",
            )
        return self.info.query(f"feature in {queried_features}")

    def rank_by_bias(self, X_test, y_pred):
        """Returns a DataFrame containing the importance of each feature.

        """

        def get_importance(group):
            """Computes the average absolute difference in bias changes per tau increase."""
            baseline = group.query("tau == 0").iloc[0]["bias"]
            return (group["bias"] - baseline).abs().mean()

        return (
            self.explain_bias(X_test=X_test, y_pred=y_pred)
            .groupby(["label", "feature"])
            .apply(get_importance)
            .to_frame("importance")
            .reset_index()
        )

    def explain_performance(self, X_test, y_test, y_pred, metric):
        """Returns a DataFrame with metric values for each (column, tau) pair.

        Parameters:
            metric (callable): A function that evaluates the quality of a set of predictions. Must
                have the following signature: `metric(y_test, y_pred, sample_weights)`. Most
                metrics from scikit-learn will work.

        """
        metric_col = metric_to_col(metric)
        if metric_col not in self.info.columns:
            self.info[metric_col] = None
            self.info[f"{metric_col}_low"] = None
            self.info[f"{metric_col}_high"] = None
        self.metric_cols.add(metric_col)

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = to_pandas(y_pred)
        y_test = np.asarray(y_test)

        self._fit(X_test, y_pred)

        # Discard the features for which the score has already been computed
        X_test, _ = make_dataset_numeric(X_test)
        queried_features = X_test.columns.tolist()
        to_explain = self.info["feature"][self.info[metric_col].isnull()].unique()
        X_test = X_test[X_test.columns.intersection(to_explain)]
        relevant = self.info.query(f"feature in {X_test.columns.tolist()}")
        if relevant.empty:
            return self.info.query(f"feature in {queried_features}")

        # Compute the metric for each (feature, lambda) pair
        scores = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            joblib.delayed(compute_performance)(
                y_test=y_test[mask],
                y_pred=y_pred[mask],
                metric=metric,
                x=X_test[col][mask],
                lambdas=part["lambda"].unique(),
                sample_index=i,
            )
            for col, part in relevant.groupby("feature")
            for i, mask in enumerate(
                yield_masks(self.n_samples, len(X_test), self.sample_frac)
            )
        )
        scores = pd.DataFrame(
            [t for s in scores for t in s],
            columns=["feature", "lambda", "sample_index", metric_col],
        )
        scores = scores.groupby(["feature", "lambda"])[metric_col].agg(
            [
                (metric_col, "mean"),
                (
                    f"{metric_col}_low",
                    functools.partial(np.quantile, q=self.conf_level),
                ),
                (
                    f"{metric_col}_high",
                    functools.partial(np.quantile, q=1 - self.conf_level),
                ),
            ]
        )
        #  TODO: merge all columns in a single operation?
        for col in scores.columns:
            self.info[col] = self.info.apply(
                lambda r: scores[col].get((r["feature"], r["lambda"]), r[col]),
                axis="columns",
            )
        return self.info.query(f"feature in {queried_features}")

    def rank_by_performance(self, X_test, y_test, y_pred, metric):
        metric_col = metric_to_col(metric)

        def get_aggregates(df):
            return pd.Series(
                [df[metric_col].min(), df[metric_col].max()], index=["min", "max"]
            )

        return (
            self.explain_performance(X_test, y_test, y_pred, metric)
            .groupby("feature")
            .apply(get_aggregates)
            .reset_index()
        )

    def make_bias_fig(self, explanation, with_taus=False, colors=None):
        """Plots predicted means against variables values.

        If a single column is provided then the x-axis is made of the nominal
        values from that column. However if a list of columns is passed then
        the x-axis contains the tau values. This method makes use of the
        `explain_predictions` method.

        """

        if colors is None:
            colors = {}
        features = explanation["feature"].unique()
        labels = explanation["label"].unique()
        y_label = f"Proportion of {labels[0]}"  #  Single class

        if with_taus:
            fig = go.Figure()
            for feat in features:
                x = explanation.query(f'feature == "{feat}"')["tau"]
                y = explanation.query(f'feature == "{feat}"')["bias"]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers",
                        hoverinfo="x+y+text",
                        name=feat,
                        text=[
                            f"{feat} = {val}"
                            for val in explanation.query(f'feature == "{feat}"')[
                                "value"
                            ]
                        ],
                        marker=dict(color=colors.get(feat)),
                    )
                )

            fig.update_layout(
                margin=dict(t=50, r=50),
                xaxis=dict(title="tau", zeroline=False),
                yaxis=dict(title=y_label, range=[0, 1], showline=True, tickformat="%"),
                plot_bgcolor="white",
            )
            return fig

        figures = {}
        for feat in features:
            fig = go.Figure()
            x = explanation.query(f'feature == "{feat}"')["value"]
            y = explanation.query(f'feature == "{feat}"')["bias"]
            mean_row = explanation.query(f'feature == "{feat}" and tau == 0').iloc[0]

            if self.n_samples > 1:
                low = explanation.query(f'feature == "{feat}"')["bias_low"]
                high = explanation.query(f'feature == "{feat}"')["bias_high"]
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate((x, x[::-1])),
                        y=np.concatenate((low, high[::-1])),
                        name=f"{self.conf_level * 100}% - {(1 - self.conf_level) * 100}%",
                        fill="toself",
                        fillcolor="#eee",  # TODO: same color as mean line?
                        line_color="rgba(0, 0, 0, 0)",
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
                    x=[mean_row["value"]],
                    y=[mean_row["bias"]],
                    mode="markers",
                    name="Original mean",
                    hoverinfo="skip",
                    marker=dict(symbol="x", size=9),
                )
            )
            fig.update_layout(
                margin=dict(t=50, r=50),
                xaxis=dict(title=f"Mean {feat}", zeroline=False),
                yaxis=dict(title=y_label, range=[0, 1], showline=True, tickformat="%"),
                plot_bgcolor="white",
            )
            figures[feat] = fig
        return figures

    def make_performance_fig(self, explanation, metric, with_taus=False, colors=None):
        """Plots metric values against variable values.

        If a single column is provided then the x-axis is made of the nominal
        values from that column. However if a list of columns is passed then
        the x-axis contains the tau values. This method makes use of the
        `explain_metric` method.

        """

        if colors is None:
            colors = {}
        features = explanation["feature"].unique()
        metric_col = metric_to_col(metric)

        if with_taus:
            fig = go.Figure()
            for feat in features:
                x = explanation.query(f'feature == "{feat}"')["tau"]
                y = explanation.query(f'feature == "{feat}"')[metric_col]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers",
                        hoverinfo="x+y+text",
                        name=feat,
                        text=[
                            f"{feat} = {val}"
                            for val in explanation.query(f'feature == "{feat}"')[
                                "value"
                            ]
                        ],
                        marker=dict(color=colors.get(feat)),
                    )
                )
            fig.update_layout(
                margin=dict(t=50, r=50),
                xaxis=dict(title="tau", zeroline=False),
                yaxis=dict(
                    title=metric_col, range=[0, 1], showline=True, tickformat="%"
                ),
                plot_bgcolor="white",
            )
            return fig

        figures = {}
        for feat in features:
            fig = go.Figure()
            x = explanation.query(f'feature == "{feat}"')["value"]
            y = explanation.query(f'feature == "{feat}"')[metric_col]
            mean_row = explanation.query(f'feature == "{feat}" and tau == 0').iloc[0]

            if self.n_samples > 1:
                low = explanation.query(f'feature == "{feat}"')[f"{metric_col}_low"]
                high = explanation.query(f'feature == "{feat}"')[f"{metric_col}_high"]
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate((x, x[::-1])),
                        y=np.concatenate((low, high[::-1])),
                        name=f"{self.conf_level * 100}% - {(1 - self.conf_level) * 100}%",
                        fill="toself",
                        fillcolor="#eee",  # TODO: same color as mean line?
                        line_color="rgba(0, 0, 0, 0)",
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
                    x=[mean_row["value"]],
                    y=[mean_row[metric_col]],
                    mode="markers",
                    name="Original mean",
                    hoverinfo="skip",
                    marker=dict(symbol="x", size=9),
                )
            )
            fig.update_layout(
                margin=dict(t=50, r=50),
                xaxis=dict(title=f"Mean {feat}", zeroline=False),
                yaxis=dict(
                    title=metric_col, range=[0, 1], showline=True, tickformat="%"
                ),
                plot_bgcolor="white",
            )
            figures[feat] = fig
        return figures

    @classmethod
    def _make_ranking_fig(cls, ranking, score_column, title, colors=None):
        ranking = ranking.sort_values(by=[score_column])

        return go.Figure(
            data=[
                go.Bar(
                    x=ranking[score_column],
                    y=ranking["feature"],
                    orientation="h",
                    hoverinfo="x",
                    marker=dict(color=colors),
                )
            ],
            layout=go.Layout(
                margin=dict(l=200, b=0, t=40),
                xaxis=dict(
                    title=title,
                    range=[0, 1],
                    showline=True,
                    zeroline=False,
                    side="top",
                    fixedrange=True,
                ),
                yaxis=dict(showline=True, zeroline=False, fixedrange=True),
                plot_bgcolor="white",
            ),
        )

    @classmethod
    def make_bias_ranking_fig(cls, ranking, colors=None):
        return cls._make_ranking_fig(ranking, "importance", "Importance", colors=colors)

    @classmethod
    def make_performance_ranking_fig(cls, ranking, metric, criterion, colors=None):
        return cls._make_ranking_fig(
            ranking, criterion, f"{criterion} {metric_to_col(metric)}", colors=colors
        )

    def _plot(self, explanation, make_fig, inline, **fig_kwargs):
        features = explanation["feature"].unique()
        if len(features) > 1:
            return plot(
                make_fig(explanation, with_taus=True, **fig_kwargs), inline=inline
            )
        return plot(
            make_fig(explanation, with_taus=False, **fig_kwargs)[features[0]],
            inline=inline,
        )

    def plot_bias(self, X_test, y_pred, inline=False, **fig_kwargs):
        explanation = self.explain_bias(X_test=X_test, y_pred=y_pred)
        return self._plot(explanation, self.make_bias_fig, inline=inline, **fig_kwargs)

    def plot_bias_ranking(self, X_test, y_pred, inline=False, **fig_kwargs):
        ranking = self.rank_by_bias(X_test=X_test, y_pred=y_pred)
        return plot(self.make_bias_ranking_fig(ranking, **fig_kwargs), inline=inline)

    def plot_performance(
        self, X_test, y_test, y_pred, metric, inline=False, **fig_kwargs
    ):
        explanation = self.explain_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        return self._plot(
            explanation,
            functools.partial(self.make_performance_fig, metric=metric),
            inline=inline,
            **fig_kwargs,
        )

    def plot_performance_ranking(
        self, X_test, y_test, y_pred, metric, criterion, inline=False, **fig_kwargs
    ):
        ranking = self.rank_by_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        return plot(
            self.make_performance_ranking_fig(
                ranking, criterion=criterion, metric=metric, **fig_kwargs
            ),
            inline=inline,
        )
