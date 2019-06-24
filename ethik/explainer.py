import decimal

import joblib
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go


__all__ = ['Explainer']


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
            raise ValueError('x must have 1 or 2 dimensions')
        if x.ndim == 2:
            return pd.DataFrame(x)
        return pd.Series(x)

    return to_pandas(np.asarray(x))


def compute_lambdas(x, target_means, max_iterations=5):
    """Finds a good lambda for a variable and a given epsilon value."""

    mean = x.mean()
    lambdas = np.empty(len(target_means))

    for i, target_mean in enumerate(target_means):

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

        lambdas[i] = λ

    return lambdas


def compute_pred_means(y_pred, x, lambdas):
    return np.array([np.average(y_pred, weights=np.exp(λ * x)) for λ in lambdas])


def compute_metric(y_true, y_pred, metric, x, lambdas):
    return np.array([metric(y_true, y_pred, sample_weight=np.exp(λ * x)) for λ in lambdas])


class Explainer():
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

    def __init__(self, alpha=0.05, n_taus=41, max_iterations=5, n_jobs=-1, verbose=False):
        self.alpha = alpha
        self.n_taus = n_taus
        self.max_iterations = max_iterations
        self.n_jobs = n_jobs
        self.verbose = verbose

    def check_parameters(self):
        """Raises a `ValueError` if any parameter is invalid."""
        if not 0 < self.alpha < 0.5:
            raise ValueError('alpha must be between 0 and 0.5, got '
                             f'{self.alpha}')

        if not self.n_taus > 0:
            raise ValueError('n_taus must be a strictly positive integer, got '
                             f'{self.n_taus}')

        if not self.max_iterations > 0:
            raise ValueError('max_iterations must be a strictly positive '
                             f'integer, got {self.max_iterations}')

    def fit(self, X):
        """Fits the explainer to a tabular dataset.

        During a `fit` call, the following steps are taken:

        1. A list of $\tau$ values is generated using `n_taus`. The $\tau$ values range from -1 to 1.
        2. A grid of $\eps$ values is generated for each $\tau$ and for each variable. Each $\eps$ represents a shift from a variable's mean towards a particular quantile.
        3. A grid of $\lambda$ values is generated for each $\eps$. Each $\lambda$ corresponds to the optimal parameter that has to be used to weight the observations in order for the average to reach the associated $\eps$ shift.

        Parameters:
            X (`pandas.DataFrame` or `numpy.ndarray`)

        """

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Check the parameters are correct
        self.check_parameters()

        # Make the taus
        tau_precision = 2 / (self.n_taus - 1)
        self.taus = list(decimal_range(-1, 1, tau_precision))

        # Make the epsilons
        q_mins = X.quantile(q=self.alpha).to_dict()
        q_maxs = X.quantile(q=1 - self.alpha).to_dict()
        means = X.mean().to_dict()
        X_num = X.select_dtypes(exclude=['object', 'category'])
        self.info = pd.concat(
            [
                pd.DataFrame({
                    'tau': self.taus,
                    'value': [
                        means[col] + tau * (
                            (means[col] - q_mins[col])
                            if tau < 0 else
                            (q_maxs[col] - means[col])
                        )
                        for tau in self.taus
                    ],
                    'feature': col
                })
                for col in X_num.columns
            ],
            ignore_index=True
        )

        # Find a lambda for each (column, espilon) pair
        lambdas = joblib.Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )(
            joblib.delayed(compute_lambdas)(
                x=X[col],
                target_means=part['value'].to_numpy(),
                max_iterations=self.max_iterations
            )
            for col, part in self.info.groupby('feature')
        )
        self.info['lambda'] = np.concatenate(lambdas)

        return self

    @property
    def is_fitted(self):
        return hasattr(self, 'info')

    @property
    def features(self):
        if not self.is_fitted:
            raise RuntimeError('The fit method has to be called first')
        return self.info['feature'].unique().tolist()

    def explain_predictions(self, X, y_pred):
        """Returns a DataFrame containing average predictions for each (column, tau) pair.

        """
        if not self.is_fitted:
            raise RuntimeError('The fit method has to be called first')

        # Coerce X and y_pred to DataFrames
        X = pd.DataFrame(to_pandas(X))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        # Discard the features that are not relevant
        relevant = self.info.query(f'feature in {X.columns.tolist()}')

        # Compute the average predictions for each (column, tau) pair per label
        preds = joblib.Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )(
            joblib.delayed(compute_pred_means)(
                y_pred=y_pred[label],
                x=X[col],
                lambdas=part['lambda']
            )
            for label in y_pred.columns
            for col, part in relevant.groupby('feature')
        )

        # Add the label names
        relevant = pd.concat((relevant.assign(label=c) for c in y_pred.columns), ignore_index=True)

        return relevant.assign(proportion=np.concatenate(preds))

    def explain_importances(self, X, y_pred):
        """Returns a DataFrame containing the importance of each feature.

        """

        def get_importance(group):
            """Computes the average absolute difference in proportion changes per tau increase."""
            baseline = group.query('tau == 0').iloc[0]['proportion']
            return (group['proportion'] - baseline).abs().mean()

        return self.explain_predictions(X=X, y_pred=y_pred)\
                   .groupby(['label', 'feature'])\
                   .apply(get_importance)\
                   .to_frame('importance')\
                   .reset_index()

    def explain_metric(self, X, y, y_pred, metric):
        """Returns a DataFrame with metric values for each (column, tau) pair.

        Parameters:
            metric (callable): A function that evaluates the quality of a set of predictions. Must
                have the following signature: `metric(y_true, y_pred, sample_weights)`. Most
                metrics from scikit-learn will work.

        """
        if not self.is_fitted:
            raise RuntimeError('The fit method has to be called first')

        X = pd.DataFrame(to_pandas(X))

        # Discard the features that are not relevant
        relevant = self.info.query(f'feature in {X.columns.tolist()}')

        # Compute the metric for each (feature, lambda) pair
        metrics = joblib.Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )(
            joblib.delayed(compute_metric)(
                y_true=y,
                y_pred=y_pred,
                metric=metric,
                x=X[col],
                lambdas=part['lambda']
            )
            for col, part in relevant.groupby('feature')
        )

        return relevant.assign(score=np.concatenate(metrics))

    @classmethod
    def make_predictions_fig(cls, explanation, with_taus=False, colors=None):
        """Plots predicted means against variables values.

        If a single column is provided then the x-axis is made of the nominal
        values from that column. However if a list of columns is passed then
        the x-axis contains the tau values. This method makes use of the
        `explain_predictions` method.

        """

        if colors is None:
            colors = {}
        features = explanation['feature'].unique()
        labels = explanation['label'].unique()
        y_label = f'Proportion of {labels[0]}' # Single class

        if with_taus:
            traces = []
            for feat in features:
                x = explanation.query(f'feature == "{feat}"')['tau']
                y = explanation.query(f'feature == "{feat}"')['proportion']
                traces.append(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines+markers',
                    hoverinfo='x+y+text',
                    name=feat,
                    text=[
                        f'{feat} = {val}'
                        for val in explanation.query(f'feature == "{feat}"')['value']
                    ],
                    marker=dict(color=colors.get(feat)),
                ))

            return go.Figure(
                data=traces,
                layout=go.Layout(
                    margin=dict(t=50, r=50),
                    xaxis=dict(
                        title='tau',
                        zeroline=False,
                    ),
                    yaxis=dict(
                        title=y_label,
                        range=[0, 1],
                        showline=True,
                        tickformat='%',
                    ),
                ),
            )

        figures = {}
        for feat in features:
            x = explanation.query(f'feature == "{feat}"')['value']
            y = explanation.query(f'feature == "{feat}"')['proportion']
            mean_row = explanation.query(f'feature == "{feat}" and tau == 0').iloc[0]
            figures[feat] = go.Figure(
                data=[
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines+markers',
                        hoverinfo='x+y',
                        showlegend=False,
                        marker=dict(color=colors.get(feat)),
                    ),
                    go.Scatter(
                        x=[mean_row['value']],
                        y=[mean_row['proportion']],
                        mode='markers',
                        name='Original mean',
                        hoverinfo='skip',
                        marker=dict(
                            symbol='x',
                            size=9,
                        ),
                    ),
                ],
                layout=go.Layout(
                    margin=dict(t=50, r=50),
                    xaxis=dict(
                        title=f'Mean {feat}',
                        zeroline=False,
                    ),
                    yaxis=dict(
                        title=y_label,
                        range=[0, 1],
                        showline=True,
                        tickformat='%',
                    ),
                ),
            )
        return figures

    @classmethod
    def make_importances_fig(cls, explanation, colors=None):
        # TODO: handle multiple labels
        # https://plot.ly/python/axes/#subcategory-axes
        explanation = explanation.sort_values(by=["importance"])

        return go.Figure(
            data=[go.Bar(
                x=explanation["importance"],
                y=explanation["feature"],
                orientation="h",
                hoverinfo="x",
                marker=dict(
                    color=colors,
                ),
            )],
            layout=go.Layout(
                xaxis=dict(
                    title="Importance",
                    range=[0, 1],
                    showline=True,
                    zeroline=False,
                    side="top",
                ),
                yaxis=dict(
                    showline=True,
                    zeroline=False,
                ),
            )
        )

    @classmethod
    def make_metric_fig(cls, explanation, y_label='Score', with_taus=False, colors=None):
        """Plots metric values against variable values.

        If a single column is provided then the x-axis is made of the nominal
        values from that column. However if a list of columns is passed then
        the x-axis contains the tau values. This method makes use of the
        `explain_metric` method.

        """

        if colors is None:
            colors = {}
        features = explanation['feature'].unique()

        if with_taus:
            traces = []
            for feat in features:
                x = explanation.query(f'feature == "{feat}"')['tau']
                y = explanation.query(f'feature == "{feat}"')['score']
                traces.append(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines+markers',
                    hoverinfo='x+y+text',
                    name=feat,
                    text=[
                        f'{feat} = {val}'
                        for val in explanation.query(f'feature == "{feat}"')['value']
                    ],
                    marker=dict(color=colors.get(feat)),
                ))

            return go.Figure(
                data=traces,
                layout=go.Layout(
                    margin=dict(t=50, r=50),
                    xaxis=dict(
                        title='tau',
                        zeroline=False,
                    ),
                    yaxis=dict(
                        title=y_label,
                        range=[0, 1],
                        showline=True,
                        tickformat='%',
                    ),
                ),
            )

        figures = {}
        for feat in features:
            x = explanation.query(f'feature == "{feat}"')['value']
            y = explanation.query(f'feature == "{feat}"')['score']
            figures[feat] = go.Figure(
                data=[
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines+markers',
                        hoverinfo='x+y',
                        showlegend=False,
                        marker=dict(color=colors.get(feat)),
                    ),
                    go.Scatter(
                        x=[x.mean()],
                        y=explanation.query(f'feature == "{feat}" and tau == 0')['score'],
                        mode='markers',
                        name='Original mean',
                        hoverinfo='skip',
                        marker=dict(
                            symbol='x',
                            size=9,
                        ),
                    ),
                ],
                layout=go.Layout(
                    margin=dict(t=50, r=50),
                    xaxis=dict(
                        title=f'Mean {feat}',
                        zeroline=False,
                    ),
                    yaxis=dict(
                        title=y_label,
                        range=[0, 1],
                        showline=True,
                        tickformat='%',
                    ),
                ),
            )
        return figures

    def _plot(self, explanation, make_fig, inline, **fig_kwargs):
        features = explanation['feature'].unique()
        if len(features) > 1:
            return plot(
                make_fig(explanation, with_taus=True, **fig_kwargs),
                inline=inline,
            )
        return plot(
            make_fig(explanation, with_taus=False, **fig_kwargs)[features[0]],
            inline=inline,
        )

    def plot_predictions(self, X, y_pred, inline=False, **fig_kwargs):
        explanation = self.explain_predictions(X=X, y_pred=y_pred)
        return self._plot(explanation, self.make_predictions_fig, inline=inline, **fig_kwargs)

    def plot_importances(self, X, y_pred, inline=False, **fig_kwargs):
        explanation = self.explain_importances(X=X, y_pred=y_pred)
        return plot(self.make_importances_fig(explanation, **fig_kwargs), inline=inline)

    def plot_metric(self, X, y, y_pred, metric, inline=False, **fig_kwargs):
        explanation = self.explain_metric(X=X, y=y, y_pred=y_pred, metric=metric)
        return self._plot(explanation, self.make_metric_fig, inline=inline, **fig_kwargs)
