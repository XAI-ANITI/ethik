import collections
import functools

import joblib
import numpy as np
import pandas as pd

from .plot_mixin import PlotMixin
from .utils import decimal_range, to_pandas, metric_to_col

__all__ = ["Explainer"]


def compute_lambdas(x, target_means, iterations=5):
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

    for target_mean in target_means:

        λ = 0
        current_mean = mean

        for _ in range(iterations):

            # Stop if the goal is reached
            if current_mean == target_mean:
                break

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
            np.average(y_pred, weights=np.exp(λ * x)),
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
        (x.name, λ, sample_index, metric(y_test, y_pred, sample_weight=np.exp(λ * x)))
        for λ in lambdas
    ]


def make_dataset_numeric(X):
    """Convert a dataset with both numerical and categorical features to a
    dataset with numerical features only.

    Args:
        X (pd.DataFrame): The dataframe to convert. Categorical-feature columns
            must have either the type `object` or `category`.

    Returns:
        (pd.DataFrame, dict): A tuple with the first item being the numeric dataset,
            with the same columns for numerical features and one binary column per
            value for each categorical feature (e.g. `gender` gives `gender__male`
            and `gender_female`). The second item of the tuple is a dict mapping
            categorical features to a list of binary numerical columns.

            For instance, for `X` being:

                age | gender
                ----------
                20 | male
                23 | female
                19 | female
                30 | male

            we get:
    
                age | gender__male | gender__female
                ----------------------------------
                20 | 1 | 0
                23 | 0 | 1
                19 | 0 | 1
                30 | 1 | 0

            and:
                
                {
                    "gender": ["gender__male", "gender__female"]
                }
    """
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
    for _ in range(n_masks):
        yield np.random.binomial(1, p, size=n).astype(bool)


class Explainer(PlotMixin):
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
        lambda_iterations (int): The maximum number of iterations used when applying the Newton step
            of the optimization procedure.

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

    def _find_lambdas(self, X_test, y_pred):
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
        self.cat_features = {**self.cat_features, **cat_features}

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
                        "feature": [col] * len(self.taus),
                        "label": [y] * len(self.taus),
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
                iterations=self.lambda_iterations,
            )
            for col, part in self.info.groupby("feature")
            if col in X_test_num
        )
        lambdas = dict(collections.ChainMap(*lambdas))
        self.info["lambda"] = self.info.apply(
            lambda r: lambdas.get((r["feature"], r["value"]), r["lambda"]),
            axis="columns",
        )
        self.info["lambda"] = self.info["lambda"].fillna(0.0)

        return self.info

    def _explain(self, X_test, y_pred, dest_col, key_cols, compute):
        # Coerce X_test and y_pred to DataFrames
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        self._find_lambdas(X_test, y_pred)

        X_test, _ = make_dataset_numeric(X_test)
        queried_features = X_test.columns.tolist()
        to_explain = self.info["feature"][self.info[dest_col].isnull()].unique()
        X_test = X_test[X_test.columns.intersection(to_explain)]
        relevant = self.info[self.info["feature"].isin(X_test.columns)]

        if not relevant.empty:
            data = compute(X_test, y_pred, relevant)
            data = pd.DataFrame(
                [line for batch in data for line in batch],
                columns=[*key_cols, "sample_index", dest_col],
            )
            data = data.groupby(key_cols)[dest_col].agg(
                [
                    (dest_col, "mean"),
                    (
                        f"{dest_col}_low",
                        functools.partial(np.quantile, q=self.conf_level),
                    ),
                    (
                        f"{dest_col}_high",
                        functools.partial(np.quantile, q=1 - self.conf_level),
                    ),
                ]
            )

            #  TODO: merge all columns in a single operation?
            for col in data.columns:
                self.info[col] = self.info.apply(
                    lambda r: data[col].get(tuple([r[k] for k in key_cols]), r[col]),
                    axis="columns",
                )

        return self.info[self.info["feature"].isin(queried_features)]

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
        metric_col = metric_to_col(metric)
        if metric_col not in self.info.columns:
            self.info[metric_col] = None
            self.info[f"{metric_col}_low"] = None
            self.info[f"{metric_col}_high"] = None
        self.metric_cols.add(metric_col)

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

        return self._explain(X_test, y_pred, metric_col, ["feature", "lambda"], compute)

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

    def plot_bias(self, X_test, y_pred, **fig_kwargs):
        explanation = self.explain_bias(X_test, y_pred)
        # make_bias_fig() is inherited from PlotMixin
        return self.make_bias_fig(explanation, **fig_kwargs)

    def plot_bias_ranking(self, X_test, y_pred, **fig_kwargs):
        ranking = self.rank_by_bias(X_test=X_test, y_pred=y_pred)
        # make_bias_ranking_fig() is inherited from PlotMixin
        return self.make_bias_ranking_fig(ranking, **fig_kwargs)

    def plot_performance(self, X_test, y_test, y_pred, metric, **fig_kwargs):
        explanation = self.explain_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        # make_performance_fig() is inherited from PlotMixin
        return self.make_performance_fig(explanation, metric=metric, **fig_kwargs)

    def plot_performance_ranking(
        self, X_test, y_test, y_pred, metric, criterion, **fig_kwargs
    ):
        ranking = self.rank_by_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        # make_performance_ranking_fig() is inherited from PlotMixin
        return self.make_performance_ranking_fig(
            ranking, criterion=criterion, metric=metric, **fig_kwargs
        )
