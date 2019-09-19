import colorlover as cl
import pandas as pd
from plotly.subplots import make_subplots

from .explainer import Explainer
from .utils import to_pandas

__all__ = ["ClassificationExplainer"]


class ClassificationExplainer(Explainer):
    def plot_bias(self, X_test, y_pred, colors=None, yrange=None):
        """Plot the bias for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or np.array): The dataset as a pandas dataframe
                or a numpy 2d-array of shape `(n_samples, n_features)`.
            y_pred (
        """
        if yrange is None:
            yrange = [0, 1]
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        if len(y_pred.columns) == 1:
            return super().plot_bias(
                X_test, y_pred.iloc[:, 0], colors=colors, yrange=yrange
            )

        if colors is None:
            features = X_test.columns
            #  Skip the lightest color as it is too light
            scale = cl.interp(cl.scales["10"]["qual"]["Paired"], len(features) + 1)[1:]
            colors = {feat: scale[i] for i, feat in enumerate(features)}

        labels = y_pred.columns
        plots = []
        for label in labels:
            plots.append(
                super().plot_bias(X_test, y_pred[label], colors=colors, yrange=yrange)
            )

        fig = make_subplots(rows=len(labels), cols=1, shared_xaxes=True)
        for ilabel, (label, plot) in enumerate(zip(labels, plots)):
            fig.update_layout({f"yaxis{ilabel+1}": dict(title=f"Average {label}")})
            for trace in plot["data"]:
                trace["showlegend"] = ilabel == 0 and trace["showlegend"]
                trace["legendgroup"] = trace["name"]
                fig.add_trace(trace, row=ilabel + 1, col=1)

        xlabel = "tau" if len(X_test.columns) > 1 else f"Average {X_test.columns[0]}"
        fig.update_layout(
            {f"xaxis{len(labels)}": dict(title=xlabel), "plot_bgcolor": "white"}
        )
        return fig
