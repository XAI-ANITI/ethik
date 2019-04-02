import base64
import decimal
import itertools
import json
import os
import random
import string

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


__all__ = ['Explainer']


def decimal_range(start: float, stop: float, step: float):
    """Like the ``range`` function but works with decimal values.

    This is more accurate than using ``np.arange`` because it doesn't introduce
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


class Explainer():

    def __init__(self, alpha=0.05, tau_precision=0.05, max_iterations=5):
        self.alpha = alpha
        self.tau_precision = tau_precision
        self.max_iterations = max_iterations

    def make_epsilons(self, mean, q_min, q_max, taus):
        return [tau * ((mean - q_min) if tau < 0 else (q_max - mean)) for tau in taus]

    def fit(self, X):

        # TODO: make sure X is a pandas DataFrame

        taus = list(decimal_range(-1, 1, self.tau_precision))

        # Determine the epsilons for each variable
        X_num = X.select_dtypes(exclude=['object', 'category'])
        q_mins = X_num.quantile(q=self.alpha)
        q_maxs = X_num.quantile(q=1 - self.alpha)
        means = X_num.mean()
        self.epsilons = pd.DataFrame({
            col: dict(zip(taus, self.make_epsilons(means[col], q_mins[col], q_maxs[col], taus)))
            for col in X_num.columns
        })
        self.lambdas = pd.DataFrame(index=taus, columns=X_num.columns)

        # Find a lambda for each (column, quantile) pair
        for col, tau in itertools.product(self.lambdas.columns, taus):
            new_mean = means[col] + self.epsilons.loc[tau, col]
            λ = self.compute_lambda(x=X[col], new_mean=new_mean)
            self.lambdas.loc[tau, col] = λ

        return self

    def compute_lambda(self, x, new_mean):

        λ = 0

        for _ in range(self.max_iterations):

            # Update the sample weights and see where the mean is
            sample_weights = np.exp(λ * x)
            sample_weights = sample_weights / sum(sample_weights)
            mean = np.average(x, weights=sample_weights)

            # Do a Newton step using the difference between the mean and the target mean
            grad = mean - new_mean
            hess = np.average((x - mean) ** 2, weights=sample_weights)
            step = grad / hess
            λ -= step

        return λ

    def plot_metric(self, X, y, y_pred, metric, column=None, ax=None):

        x = X[column]

        # Evaluate the metric for each lambda
        lambdas = self.lambdas[column]
        scores = [
            metric(y, y_pred, sample_weight=np.exp(λ * x))
            for tau, λ in lambdas.items()
        ]

        mean = x.mean()
        values = [mean + eps for eps in self.epsilons[column]]

        # Plot the scores against the mean values
        ax = plt.axes() if ax is None else ax
        ax.grid(True)
        ax.plot(values, scores)

        # Prettify the plot
        ax.set_xlabel(column)
        ax.set_ylabel(metric.__name__)

        return ax

    def initjs(self):

        # Load JavaScript dependencies
        display.display(display.Javascript("""
            require.config({
                paths: {
                    d3: 'https://d3js.org/d3.v5.min'
                }
            });
        """))

        # Load the JavaScript logo
        here = os.path.dirname(__file__)
        logo_path = os.path.join(here, 'resources', 'js_logo.png')
        with open(logo_path, 'rb') as f:
            logo_data = f.read()
        logo_data = base64.b64encode(logo_data).decode('utf-8')

        return display.HTML(string.Template("""
            <div align="center">
                <img src="data:image/png;base64,$logo_data" />
            </div>
        """).substitute(logo_data=logo_data))

    def plot_fuzz(self):
        data = [10, 60, 40, 5, 30, 10]
        width = 500
        height = 200
        return display.display(display.Javascript("""
            (function(element){
                require(['circles'], function(circles) {
                    circles(element.get(0), %s, %d, %d);
                });
            })(element);
        """ % (json.dumps(data), width, height)))
