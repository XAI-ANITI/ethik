import collections
import functools
import itertools

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy import special

from .utils import decimal_range, to_pandas

__all__ = ["Explainer"]


CAT_COL_SEP = " = "


def compute_lambdas(x, target_means, iterations=5, use_previous_lambda=False):
    """Find good lambdas for a variable and given target means.

    Args:
        x (pd.Series): The variable's values.
        target_means (iterator of floats): The means to reach by weighting the
            feature's values.
        iterations (int): The number of iterations to compute the weights for a
            given target mean. Default is `5`.

    Returns:
        dict: The keys are couples `(name of the variable, target mean)` and the
            values are the lambdas. For instance:

                {
                    ("age", 20): 1.5,
                    ("age", 21): 1.6,
                    ...
                }
    """

    mean = x.mean()
    lambdas = {}

    λ = 0

    for target_mean in target_means:

        current_mean = mean

        if not use_previous_lambda:
            λ = 0

        for _ in range(iterations):

            # Update the sample weights and see where the mean is
            sample_weights = special.softmax(λ * x)
            current_mean = np.average(x, weights=sample_weights)

            # Do a Newton step using the difference between the mean and the
            # target mean
            grad = current_mean - target_mean
            hess = np.average((x - current_mean) ** 2, weights=sample_weights)

            if hess == 0:
                λ -= 0.01 * grad
                break

            step = grad / hess
            λ -= step

        lambdas[(x.name, target_mean)] = λ

    return lambdas


def compute_bias(y_pred, x, lambdas, sample_index):
    """Compute the influence of the variable `x` on the predictions.

    Args:
        y_pred (pd.Series): The predictions.
        x (pd.Series): The variable's values.
        lambdas (iterator of floats): The lambdas for each target mean of `x`.
        sample_index (int): The index of the sample used to compute the confidence
            interval. Only used to merge the data later.

    Returns:
        list of tuples: A list of `(x's name, y_pred's name, lambda, sample_index, bias)`
            for each lambda.
    """
    return [
        (
            x.name,
            y_pred.name,
            λ,
            sample_index,
            np.average(y_pred, weights=special.softmax(λ * x)),
        )
        for λ in lambdas
    ]


def compute_performance(y_test, y_pred, metric, x, lambdas, sample_index):
    """Compute the influence of the variable `x` on the model's performance.

    Args:
        y_test (pd.Series): The true predictions.
        y_pred (pd.Series): The model's predictions.
        metric (callable): A scikit-learn-like metric with such an interface:
            `metric(y_test, y_pred, sample_weight=None)`.
        x (pd.Series): The variable's values.
        lambdas (iterator of floats): The lambdas for each target mean of `x`.
        sample_index (int): The index of the sample used to compute the confidence
            interval. Only used to merge the data later.

    Returns:
        list of tuples: A list of `(x's name, lambda, sample_index, performance)`
            for each lambda.
    """
    return [
        (
            x.name,
            λ,
            sample_index,
            metric(y_test, y_pred, sample_weight=special.softmax(λ * x))
        )
        for λ in lambdas
    ]


def metric_to_col(metric):
    """Get the name of the column in explainer's info dataframe to store the
    performance with respect of the given metric.

    Args:
        metric (callable): The metric to compute the model's performance.

    Returns:
        str: The name of the column.
    """
    return metric.__name__ + "__score"


def yield_masks(n_masks, n, p):
    """Generates a list of `n_masks` to keep a proportion `p` of `n` items.

    Args:
        n_masks (int): The number of masks to yield. It corresponds to the number
            of samples we use to compute the confidence interval.
        n (int): The number of items being filtered. It corresponds to the size
            of the dataset.
        p (float): The proportion of items to keep.

    Returns:
        generator: A generator of `n_masks` lists of `n` booleans being generated
            with a binomial distribution. As it is a probabilistic approach,
            we may get more or fewer than `p*n` items kept, but it is not a problem
            with large datasets.
    """

    if p < 0 or p > 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")

    if p < 1:
        for _ in range(n_masks):
            yield np.random.binomial(1, p, size=n).astype(bool)
    else:
        for _ in range(n_masks):
            yield np.full(shape=n, fill_value=True)


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
        lambda_iterations (int): The number of iterations used when applying the Newton step
            of the optimization procedure. Default is `5`.
        n_jobs (int): The number of jobs to use for parallel computations. See
            `joblib.Parallel()`. Default is `-1`.
        verbose (bool): Passed to `joblib.Parallel()` for parallel computations.
            Default is `False`.
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
    """

    def __init__(
        self,
        alpha=0.05,
        n_taus=41,
        lambda_iterations=5,
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

        if not lambda_iterations > 0:
            raise ValueError(
                "lambda_iterations must be a strictly positive "
                f"integer, got {lambda_iterations}"
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
        self.lambda_iterations = lambda_iterations
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.metric_names = set()
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

    def get_metric_name(self, metric):
        """Get the name of the column in explainer's info dataframe to store the
        performance with respect of the given metric.

        Args:
            metric (callable): The metric to compute the model's performance.

        Returns:
            str: The name of the column.
        """
        name = metric.__name__
        if name in self.info.columns and name not in self.metric_names:
            raise ValueError(f"Cannot use {name} as a metric name")
        return name

    @property
    def taus(self):
        tau_precision = 2 / (self.n_taus - 1)
        return list(decimal_range(-1, 1, tau_precision))

    @property
    def features(self):
        return self.info["feature"].unique().tolist()

    def _find_lambdas(self, X_test, y_pred):
        """Finds lambda values for each (feature, tau, label) triplet.

        1. A list of $\tau$ values is generated using `n_taus`. The $\tau$ values range from -1 to 1.
        2. A grid of $\eps$ values is generated for each $\tau$ and for each variable. Each $\eps$ represents a shift from a variable's mean towards a particular quantile.
        3. A grid of $\lambda$ values is generated for each $\eps$. Each $\lambda$ corresponds to the optimal parameter that has to be used to weight the observations in order for the average to reach the associated $\eps$ shift.

        Parameters:
            X_test (`pandas.DataFrame` or `numpy.ndarray`)
            y_pred (`pandas.DataFrame` or `numpy.ndarray`)

        """

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        # One-hot encode the categorical features
        cat_features = X_test.select_dtypes(include=["object", "category"]).columns
        X_test = pd.get_dummies(data=X_test, prefix_sep=CAT_COL_SEP)

        # Check which (feature, label) pairs have to be done
        to_do_pairs = set(itertools.product(X_test.columns, y_pred.columns)) - set(
            self.info.groupby(["feature", "label"]).groups.keys()
        )
        to_do_features = set(pair[0] for pair in to_do_pairs)
        X_test = X_test[to_do_features]

        if X_test.empty:
            return self

        # Make the epsilons for each (feature, label, tau) triplet
        q_mins = X_test.quantile(q=self.alpha).to_dict()
        q_maxs = X_test.quantile(q=1 - self.alpha).to_dict()
        means = X_test.mean().to_dict()
        additional_info = pd.concat(
            [
                pd.DataFrame(
                    {
                        "tau": self.taus,
                        "value": [
                            means[feature]
                            + tau
                            * (
                                (means[feature] - q_mins[feature])
                                if tau < 0
                                else (q_maxs[feature] - means[feature])
                            )
                            for tau in self.taus
                        ],
                        "feature": [feature] * len(self.taus),
                        "label": [label] * len(self.taus),
                    }
                )
                for feature, label in to_do_pairs
            ],
            ignore_index=True,
        )
        self.info = self.info.append(additional_info, ignore_index=True, sort=False)

        # Find a lambda for each (feature, espilon) pair
        lambdas = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            joblib.delayed(compute_lambdas)(
                x=X_test[feature],
                target_means=part["value"].unique(),
                iterations=self.lambda_iterations,
                use_previous_lambda=all(not feature.startswith(cat) for cat in cat_features)
            )
            for feature, part in self.info.groupby("feature")
            if feature in X_test
        )
        lambdas = dict(collections.ChainMap(*lambdas))
        self.info["lambda"] = self.info.apply(
            lambda r: lambdas.get((r["feature"], r["value"]), r["lambda"]),
            axis="columns",
        )
        self.info["lambda"] = self.info["lambda"].fillna(0.0)

        return self

    def _explain(self, X_test, y_pred, dest_col, key_cols, compute):

        # Coerce X_test and y_pred to DataFrames
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        # Find the lambda values for each (feature, tau, label) triplet
        self._find_lambdas(X_test, y_pred)

        # One-hot encode the categorical variables
        X_test = pd.get_dummies(data=X_test, prefix_sep=CAT_COL_SEP)

        # Determine which features are missing explanations; that is they have null biases for at
        # least one lambda value
        relevant = self.info[
            self.info["feature"].isin(X_test.columns)
            & self.info["label"].isin(y_pred.columns)
            & self.info[dest_col].isnull()
        ]

        if not relevant.empty:
            # `compute()` will return something like:
            # [
            #   [ # First batch
            #     (*key_cols1, sample_index1, computed_value1),
            #     (*key_cols2, sample_index2, computed_value2),
            #     ...
            #   ],
            #   ...
            # ]
            data = compute(X_test=X_test, y_pred=y_pred, relevant=relevant)
            data = pd.DataFrame(
                [line for batch in data for line in batch],
                columns=[*key_cols, "sample_index", dest_col],
            )
            # We group by the key to gather the samples and compute the confidence
            #  interval
            data = data.groupby(key_cols)[dest_col].agg(
                [
                    # Mean bias
                    (dest_col, "mean"),
                    # Lower bound on the mean bias
                    (f"{dest_col}_low", functools.partial(np.quantile, q=self.conf_level)),
                    # Upper bound on the mean bias
                    (f"{dest_col}_high", functools.partial(np.quantile, q=1 - self.conf_level)),
                ]
            )

            #  TODO: merge all columns in a single operation?
            for col in data.columns:
                self.info[col] = self.info.apply(
                    lambda r: data[col].get(tuple([r[k] for k in key_cols]), r[col]),
                    axis="columns",
                )

        return self.info[
            self.info["feature"].isin(X_test.columns)
            & self.info["label"].isin(y_pred.columns)
        ]

    def explain_bias(self, X_test, y_pred):
        def compute(X_test, y_pred, relevant):
            return joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
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

        return self._explain(
            X_test=X_test,
            y_pred=y_pred,
            dest_col="bias",
            key_cols=["feature", "label", "lambda"],
            compute=compute,
        )

    def explain_performance(self, X_test, y_test, y_pred, metric):
        metric_name = self.get_metric_name(metric)
        if metric_name not in self.info.columns:
            self.info[metric_name] = None
            self.info[f"{metric_name}_low"] = None
            self.info[f"{metric_name}_high"] = None
        self.metric_names.add(metric_name)

        y_test = np.asarray(y_test)

        def compute(X_test, y_pred, relevant):
            return joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
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

        return self._explain(
            X_test, y_pred, metric_name, ["feature", "lambda"], compute
        )

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

    def rank_by_performance(self, X_test, y_test, y_pred, metric):
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

    def _plot_explanation(self, explanation, col, y_label, colors=None, yrange=None):
        features = explanation["feature"].unique()

        if colors is None:
            colors = {}
        elif type(colors) is str:
            colors = {feat: colors for feat in features}

        #  There are multiple features, we plot them together with taus
        if len(features) > 1:
            fig = go.Figure()
            for feat in features:
                x = explanation.query(f'feature == "{feat}"')["tau"]
                y = explanation.query(f'feature == "{feat}"')[col]
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
                yaxis=dict(title=y_label, range=yrange, showline=True),
                plot_bgcolor="white",
            )
            return fig

        #  There is only one feature, we plot it with its nominal values.
        feat = features[0]
        fig = go.Figure()
        x = explanation.query(f'feature == "{feat}"')["value"]
        y = explanation.query(f'feature == "{feat}"')[col]
        mean_row = explanation.query(f'feature == "{feat}" and tau == 0').iloc[0]

        if self.n_samples > 1:
            low = explanation.query(f'feature == "{feat}"')[f"{col}_low"]
            high = explanation.query(f'feature == "{feat}"')[f"{col}_high"]
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
                y=[mean_row[col]],
                mode="markers",
                name="Original mean",
                hoverinfo="skip",
                marker=dict(symbol="x", size=9, color="black"),
            )
        )
        fig.update_layout(
            margin=dict(t=50, r=50),
            xaxis=dict(title=f"Average {feat}", zeroline=False),
            yaxis=dict(title=y_label, range=yrange, showline=True),
            plot_bgcolor="white",
        )
        return fig

    def _plot_ranking(self, ranking, score_column, title, colors=None):
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

    def plot_bias(self, X_test, y_pred, **fig_kwargs):
        explanation = self.explain_bias(X_test, y_pred)
        labels = explanation["label"].unique()
        if len(labels) > 1:
            raise ValueError("Cannot plot multiple labels")
        y_label = f'Average "{labels[0]}"'
        return self._plot_explanation(explanation, "bias", y_label, **fig_kwargs)

    def plot_bias_ranking(self, X_test, y_pred, **fig_kwargs):
        ranking = self.rank_by_bias(X_test=X_test, y_pred=y_pred)
        return self._plot_ranking(ranking, "importance", "Importance", **fig_kwargs)

    def plot_performance(self, X_test, y_test, y_pred, metric, **fig_kwargs):
        metric_name = self.get_metric_name(metric)
        explanation = self.explain_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        if fig_kwargs.get("yrange") is None:
            if explanation[metric_name].between(0, 1).all():
                fig_kwargs["yrange"] = [0, 1]

        return self._plot_explanation(
            explanation, metric_name, y_label=f"Average {metric_name}", **fig_kwargs
        )

    def plot_performance_ranking(
        self, X_test, y_test, y_pred, metric, criterion, **fig_kwargs
    ):
        metric_name = self.get_metric_name(metric)
        ranking = self.rank_by_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        return self._plot_ranking(
            ranking, criterion, f"{criterion} {metric_name}", **fig_kwargs
        )
