import base64
import decimal
import itertools
import os
import random
import string

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


__all__ = ['Explainer']


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


def random_id(size=20, chars=string.ascii_uppercase + string.digits):
    """Returns a random identifier.

    This is simply used for identifying a <div> when rendering HTML.

    """
    return 'i' + ''.join(random.choice(chars) for _ in range(size))


def as_list(x):
    """Converts as a list."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


class Explainer():
    """

    Parameters:
        alpha (float)
        tau_precision (float)
        max_iterations (int)

    """

    def __init__(self, alpha=0.05, n_taus=41, max_iterations=5):
        self.alpha = alpha
        self.n_taus = n_taus
        self.max_iterations = max_iterations

    def check_parameters(self):
        """Raises a `ValueError` if any parameter is invalid."""
        if not 0 < self.alpha < 0.5:
            raise ValueError(f'alpha must be between 0 and 0.5, got {self.alpha}')

        if not self.n_taus > 0:
            raise ValueError(f'n_taus must be a strictly positive integer, got {self.n_taus}')

        if not self.max_iterations > 0:
            raise ValueError('max_iterations must be a strictly positive integer, ' +
                             f'got {self.max_iterations}')

    def make_epsilons(self, X, taus):
        """Returns a DataFrame containing espilons for each (column, tau) pair.

        Attributes:
            X (pandas.DataFrame)
            taus (list of floats)

        Returns:
            pandas.DataFrame

        """

        q_mins = X.quantile(q=self.alpha).to_dict()
        q_maxs = X.quantile(q=1 - self.alpha).to_dict()
        means = X.mean().to_dict()

        return pd.DataFrame({
            col: dict(zip(
                self.taus,
                [
                    tau * (
                        (means[col] - q_mins[col])
                        if tau < 0 else
                        (q_maxs[col] - means[col])
                    )
                    for tau in taus
                ]
            ))
            for col in X.columns
        })

    def fit(self, X):
        """

        Parameters:
            X (pandas.DataFrame or numpy.ndarray)

        """

        # Check the parameters are correct
        self.check_parameters()

        # Make the taus
        tau_precision = 2 / (self.n_taus - 1)
        self.taus = list(decimal_range(-1, 1, tau_precision))

        # Make the epsilons
        X_num = X.select_dtypes(exclude=['object', 'category'])
        self.epsilons = self.make_epsilons(X_num, self.taus)

        # Find a lambda for each (column, quantile) pair
        self.lambdas = pd.DataFrame(index=self.taus, columns=X_num.columns)

        for col, tau in itertools.product(self.lambdas.columns, self.taus):
            λ = self.compute_lambda(x=X[col], eps=self.epsilons.loc[tau, col])
            self.lambdas.loc[tau, col] = λ

        return self

    def compute_lambda(self, x, eps):
        """Finds a good lambda for a variable and a given epsilon value.

        Parameters:
            x (array-like)
            eps (float)

        """

        λ = 0
        new_mean = x.mean() + eps

        for _ in range(self.max_iterations):

            # Update the sample weights and see where the mean is
            sample_weights = np.exp(λ * x)
            sample_weights = sample_weights / sum(sample_weights)
            mean = np.average(x, weights=sample_weights)

            # Do a Newton step using the difference between the mean and the
            # target mean
            grad = mean - new_mean
            hess = np.average((x - mean) ** 2, weights=sample_weights)
            step = grad / hess
            λ -= step

        return λ

    def explain_predictions(self, X, y_pred, columns):
        """Returns a DataFrame with predicted means for each (column, tau) pair.

        """
        predictions = pd.DataFrame(
            data={
                col: [
                    np.average(y_pred, weights=np.exp(λ * X[col]))
                    for tau, λ in self.lambdas[col].items()
                ]
                for col in as_list(columns)
            },
            index=self.taus
        )[columns]

        # If there a single column we can index with epsilons instead of taus
        if isinstance(predictions, pd.Series):
            mean = X[predictions.name].mean()
            predictions.index = [mean + eps for eps in self.epsilons[predictions.name]]

        return predictions

    def plot_predictions(self, X, y_pred, columns, ax=None):
        """Plots predicted means against variables values.

        If a single column is provided then the x-axis is made of the nominal
        values from that column. However if a list of columns is passed then
        the x-axis contains the tau values. This method makes use of the
        `explain_predictions` method.

        """

        predictions = self.explain_predictions(X=X, y_pred=y_pred, columns=columns)

        # Create a plot if none is provided
        ax = plt.axes() if ax is None else ax

        if isinstance(predictions, pd.Series):

            # If a single column is provided then we can plot prediction means
            # against values
            ax.plot(predictions)
            ax.set_xlabel(predictions.name)

        else:

            # If more than column is provided then we can plot scores against values
            # for each column OR plot scores against taus on one single axis
            for col in predictions.columns:
                ax.plot(predictions[col], label=col)
            ax.legend()
            ax.set_xlabel(r'$\tau$')

        # Prettify the plot
        ax.grid(True)
        ax.set_ylabel('Target mean')

        return ax

    def explain_metric(self, X, y, y_pred, metric, columns):
        """Returns a DataFrame with metric values for each (column, tau) pair.

        """
        metrics = pd.DataFrame(
            data={
                col: [
                    metric(y, y_pred, sample_weight=np.exp(λ * X[col]))
                    for tau, λ in self.lambdas[col].items()
                ]
                for col in as_list(columns)
            },
            index=self.taus
        )[columns]

        # If there a single column we can index with epsilons instead of taus
        if isinstance(metrics, pd.Series):
            mean = X[metrics.name].mean()
            metrics.index = [mean + eps for eps in self.epsilons[metrics.name]]

        return metrics

    def plot_metric(self, X, y, y_pred, metric, columns, ax=None):
        """Plots metric values against variable values.

        If a single column is provided then the x-axis is made of the nominal
        values from that column. However if a list of columns is passed then
        the x-axis contains the tau values. This method makes use of the
        `explain_metric` method.

        """

        metrics = self.explain_metric(
            X=X,
            y=y,
            y_pred=y_pred,
            metric=metric,
            columns=columns
        )

        # Create a plot if none is provided
        ax = plt.axes() if ax is None else ax

        if isinstance(metrics, pd.Series):

            # If a single column is provided then we can plot scores against
            # values
            ax.plot(metrics)
            ax.set_xlabel(metrics.name)

        else:

            # If more than column is provided then we can plot scores against values
            # for each column OR plot scores against taus on one single axis
            for col in metrics.columns:
                ax.plot(metrics[col], label=col)
            ax.legend()
            ax.set_xlabel(r'$\tau$')

        # Prettify the plot
        ax.grid(True)
        ax.set_ylabel(metric.__name__)

        return ax

    def initjs(self):

        # Load JavaScript dependencies
        display.display(display.Javascript('''
            require.config({
                paths: {
                    d3: "https://d3js.org/d3.v5.min"
                }
            });
        '''))

        # Load the JavaScript logo
        here = os.path.dirname(__file__)
        logo_path = os.path.join(here, 'resources', 'js_logo.png')
        with open(logo_path, 'rb') as f:
            logo_data = f.read()

        display.display(display.HTML(string.Template('''
            <div align="center">
                <img src="data:image/png;base64,$logo_data" />
            </div>
        ''').substitute(logo_data=base64.b64encode(logo_data).decode('utf-8'))))
