import colorlover as cl
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from .cache_explainer import CacheExplainer
from .utils import set_fig_size, to_pandas

__all__ = ["ClassificationExplainer"]


class ClassificationExplainer(CacheExplainer):
    def plot_influence(
        self, X_test, y_pred, colors=None, yrange=None, size=None, constraints=None
    ):
        """Plot the influence for the features in `X_test`.

        See `ethik.cache_explainer.CacheExplainer.plot_influence()`.
        """
        if yrange is None:
            yrange = [0, 1]

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        if len(y_pred.columns) == 1:
            return super().plot_influence(
                X_test=X_test,
                y_pred=y_pred.iloc[:, 0],
                colors=colors,
                yrange=yrange,
                size=size,
                constraints=constraints,
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
                    X_test=X_test,
                    y_pred=y_pred[label],
                    colors=colors,
                    yrange=yrange,
                    constraints=constraints,
                )
            )

        fig = make_subplots(rows=len(labels), cols=1, shared_xaxes=True)
        for ilabel, (label, plot) in enumerate(zip(labels, plots)):
            fig.update_layout({f"yaxis{ilabel+1}": dict(title=f"Average {label}")})
            for trace in plot["data"]:
                trace["showlegend"] = ilabel == 0 and trace["showlegend"]
                trace["legendgroup"] = trace["name"]
                fig.add_trace(trace, row=ilabel + 1, col=1)

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
                )
            }
        )
        set_fig_size(fig, size)
        return fig

    def plot_influence_2d(
        self, X_test, y_pred, z_range=None, colorscale=None, size=None, constraints=None
    ):
        """Plot the combined influence for the features in `X_test`.

        See `ethik.cache_explainer.CacheExplainer.plot_influence_2d()`.
        """
        if z_range is None:
            z_range = [0, 1]

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        if len(y_pred.columns) == 1:
            return super().plot_influence_2d(
                X_test=X_test,
                y_pred=y_pred,
                z_range=z_range,
                colorscale=colorscale,
                size=size,
                constraints=constraints,
            )

        labels = y_pred.columns
        plots = []
        for label in labels:
            plots.append(
                super().plot_influence_2d(
                    X_test=X_test,
                    y_pred=y_pred[label],
                    z_range=z_range,
                    colorscale=colorscale,
                    size=size,
                    constraints=constraints,
                )
            )

        fig = make_subplots(
            rows=len(labels),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[p["data"][0]["colorbar"]["title"].text for p in plots],
        )
        for ilabel, (label, plot) in enumerate(zip(labels, plots)):
            fig.update_layout({f"yaxis{ilabel+1}": dict(title=plot.layout.yaxis.title)})
            heatmap, ds_mean = plot["data"]
            heatmap["showscale"] = ilabel == 0
            heatmap["colorbar"]["title"] = ""
            heatmap["hoverinfo"] = "x+y+z"
            fig.add_trace(heatmap, row=ilabel + 1, col=1)
            fig.add_trace(ds_mean, row=ilabel + 1, col=1)

        fig.update_xaxes(
            showline=True,
            showgrid=True,
            zeroline=False,
            mirror=True,
            linecolor="black",
            gridcolor="#eee",
        )
        fig.update_yaxes(
            showline=True,
            showgrid=True,
            mirror=True,
            linecolor="black",
            gridcolor="#eee",
        )
        fig.update_layout(
            {
                f"xaxis{len(labels)}": dict(title=plots[0].layout.xaxis.title),
                "title": plots[0].layout.title,
            }
        )
        set_fig_size(fig, size)
        return fig

    def plot_distributions(
        self,
        feature_values,
        y_pred=None,
        bins=None,
        show_hist=False,
        show_curve=True,
        targets=None,
        colors=None,
        dataset_color="black",
        size=None,
    ):
        """Plot the stressed distribution of `feature_values` or `y_pred` if specified
        for each mean of `feature_values` in `targets`.

        See `ethik.base_explainer.BaseExplainer.plot_distributions()`.
        """
        if y_pred is None:
            return super().plot_distributions(
                feature_values=feature_values,
                bins=bins,
                show_hist=show_hist,
                show_curve=show_curve,
                targets=targets,
                colors=colors,
                dataset_color=dataset_color,
                size=size,
            )

        y_pred = pd.DataFrame(to_pandas(y_pred))
        if len(y_pred.columns) == 1:
            return super().plot_distributions(
                feature_values=feature_values,
                y_pred=y_pred.iloc[:, 0],
                bins=bins,
                show_hist=show_hist,
                show_curve=show_curve,
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
                    feature_values=feature_values,
                    y_pred=y_pred[label],
                    bins=bins,
                    show_hist=show_hist,
                    show_curve=show_curve,
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
        set_fig_size(fig, size)
        return fig

    def plot_influence_comparison(
        self, X_test, y_pred, reference, compared, colors=None, yrange=None, size=None
    ):
        """Plot the influence of features in `X_test` on `y_pred` for the
        individual `compared` compared to `reference`. Basically, we look at how
        the model would behave if the average individual were `compared` and take
        the difference with what the output would be if the average were `reference`.

        See `ethik.base_explainer.BaseExplainer.plot_influence_comparison()`.
        """
        if yrange is None:
            yrange = [-1, 1]

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))
        features = X_test.columns
        labels = y_pred.columns

        if len(labels) == 1:
            return super().plot_influence_comparison(
                X_test,
                y_pred.iloc[:, 0],
                reference,
                compared,
                colors=colors,
                yrange=yrange,
                size=size,
            )

        if colors is None:
            #  Skip the lightest color as it is too light
            scale = cl.interp(cl.scales["10"]["qual"]["Paired"], len(features) + 1)[1:]
            colors = {feat: scale[i] for i, feat in enumerate(features)}

        plots = []
        for label in labels:
            plots.append(
                super().plot_influence_comparison(
                    X_test, y_pred[label], reference, compared, colors=colors
                )
            )

        fig = make_subplots(rows=len(labels), cols=1, shared_xaxes=True)
        title = None
        shapes = []
        for ilabel, (label, plot) in enumerate(zip(labels, plots)):
            fig.update_layout({f"yaxis{ilabel+1}": dict(title=label)})
            for trace in plot["data"]:
                shapes.append(
                    go.layout.Shape(
                        type="line",
                        x0=0,
                        y0=-0.5,
                        x1=0,
                        y1=len(features) - 0.5,
                        yref=f"y{ilabel+1}",
                        line=dict(color="black", width=1),
                    )
                )
                fig.add_trace(trace, row=ilabel + 1, col=1)

        fig.update_xaxes(
            range=yrange,
            showline=True,
            linecolor="black",
            linewidth=1,
            zeroline=False,
            showgrid=True,
            gridcolor="#eee",
            side="top",
            fixedrange=True,
            showticklabels=True,
        )
        fig.update_layout(
            showlegend=False,
            xaxis1=dict(title=plots[0].layout.xaxis.title),
            shapes=shapes,
        )
        set_fig_size(
            fig, size, width=500, height=100 + 60 * len(features) + 30 * len(labels)
        )
        return fig
