import colorlover as cl
import pandas as pd
from plotly.subplots import make_subplots

from .cache_explainer import CacheExplainer
from .utils import to_pandas

__all__ = ["ClassificationExplainer"]


class ClassificationExplainer(CacheExplainer):
    def plot_influence(self, X_test, y_pred, colors=None, yrange=None, size=None):
        """Plot the influence for the features in `X_test`.

        See `ethik.explainer.Explainer.plot_influence()`.
        """
        if yrange is None:
            yrange = [0, 1]

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        if len(y_pred.columns) == 1:
            return super().plot_influence(
                X_test, y_pred.iloc[:, 0], colors=colors, yrange=yrange, size=size
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
                super().plot_influence(
                    X_test, y_pred[label], colors=colors, yrange=yrange
                )
            )

        fig = make_subplots(rows=len(labels), cols=1, shared_xaxes=True)
        for ilabel, (label, plot) in enumerate(zip(labels, plots)):
            fig.update_layout({f"yaxis{ilabel+1}": dict(title=f"Average {label}")})
            for trace in plot["data"]:
                trace["showlegend"] = ilabel == 0 and trace["showlegend"]
                trace["legendgroup"] = trace["name"]
                fig.add_trace(trace, row=ilabel + 1, col=1)

        width = height = None
        if size is not None:
            width, height = size

        fig.update_xaxes(
            nticks=5,
            showline=True,
            showgrid=True,
            zeroline=False,
            linecolor="black",
            gridcolor="#eee",
        )
        fig.update_yaxes(
            range=yrange,
            showline=True,
            showgrid=True,
            linecolor="black",
            gridcolor="#eee",
        )
        fig.update_layout(
            {
                f"xaxis{len(labels)}": dict(
                    title="tau"
                    if len(X_test.columns) > 1
                    else f"Average {X_test.columns[0]}"
                ),
                "plot_bgcolor": "white",
                "width": width,
                "height": height,
            }
        )
        return fig

    def plot_distributions(
        self,
        X_test,
        y_pred=None,
        bins=10,
        targets=None,
        colors=None,
        dataset_color="black",
        size=None,
    ):
        if y_pred is None:
            return super().plot_distributions(
                X_test=X_test,
                bins=bins,
                targets=targets,
                colors=colors,
                dataset_color=dataset_color,
                size=size,
            )

        y_pred = pd.DataFrame(to_pandas(y_pred))
        if len(y_pred.columns) == 1:
            return super().plot_distributions(
                X_test=X_test,
                y_pred=y_pred.iloc[:, 0],
                bins=bins,
                targets=targets,
                colors=colors,
                dataset_color=dataset_color,
                size=size,
            )

        labels = y_pred.columns
        plots = []
        for label in labels:
            plots.append(
                super().plot_distributions(
                    X_test,
                    y_pred[label],
                    bins=bins,
                    targets=targets,
                    colors=colors,
                    dataset_color=dataset_color,
                )
            )

        fig = make_subplots(rows=len(labels), cols=1, shared_xaxes=True)
        for ilabel, (label, plot) in enumerate(zip(labels, plots)):
            fig.update_layout(
                {
                    f"xaxis{ilabel+1}": dict(title=label),
                    f"yaxis{ilabel+1}": dict(title=f"Probability density"),
                }
            )
            for itrace, trace in enumerate(plot["data"]):
                trace["legendgroup"] = itrace
                fig.add_trace(trace, row=ilabel + 1, col=1)

        width = height = None
        if size is not None:
            width, height = size

        fig.update_xaxes(
            showline=True,
            showgrid=True,
            zeroline=False,
            linecolor="black",
            gridcolor="#eee",
        )
        fig.update_yaxes(
            showline=True, showgrid=True, linecolor="black", gridcolor="#eee"
        )
        fig.update_layout(plot_bgcolor="white", width=width, height=height)
        return fig
