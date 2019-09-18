import colorlover as cl
import pandas as pd
from plotly.subplots import make_subplots

from .explainer import Explainer
from .utils import to_pandas

__all__ = ["ClassificationExplainer"]


class ClassificationExplainer(Explainer):
    def plot_bias(self, X_test, y_pred, **fig_kwargs):
        fig_kwargs["yrange"] = fig_kwargs.get("yrange", [0, 1])
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        if len(y_pred.columns) == 1:
            return super().plot_bias(X_test, y_pred.iloc[:, 0], **fig_kwargs)

        if fig_kwargs.get("colors") is None:
            features = X_test.columns
            #  Skip the lightest color as it is too light
            scale = cl.interp(cl.scales["10"]["qual"]["Paired"], len(features) + 1)[1:]
            fig_kwargs["colors"] = {feat: scale[i] for i, feat in enumerate(features)}

        labels = y_pred.columns
        plots = []
        for label in labels:
            plots.append(super().plot_bias(X_test, y_pred[label], **fig_kwargs))

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
