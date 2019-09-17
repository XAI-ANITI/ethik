import numpy as np
import plotly.graph_objs as go

from .utils import metric_to_col

__all__ = ["Plotter"]


class PlotMixin:
    def _make_explanation_fig(self, explanation, col, y_label, colors=None):
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
                yaxis=dict(title=y_label, range=[0, 1], showline=True, tickformat="%"),
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
                marker=dict(symbol="x", size=9),
            )
        )
        fig.update_layout(
            margin=dict(t=50, r=50),
            xaxis=dict(title=f"Mean {feat}", zeroline=False),
            yaxis=dict(title=y_label, range=[0, 1], showline=True, tickformat="%"),
            plot_bgcolor="white",
        )
        return fig

    def make_bias_fig(self, explanation, **fig_kwargs):
        labels = explanation["label"].unique()
        y_label = f'Average "{labels[0]}"'  #  Single class
        return self._make_explanation_fig(explanation, "bias", y_label, **fig_kwargs)

    def make_performance_fig(self, explanation, metric, **fig_kwargs):
        metric_col = metric_to_col(metric)
        return self._make_explanation_fig(
            explanation, metric_col, y_label=f"Average {metric.__name__}", **fig_kwargs
        )

    def _make_ranking_fig(self, ranking, score_column, title, colors=None):
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

    def make_bias_ranking_fig(self, ranking, colors=None):
        return self._make_ranking_fig(
            ranking, "importance", "Importance", colors=colors
        )

    def make_performance_ranking_fig(self, ranking, metric, criterion, colors=None):
        return self._make_ranking_fig(
            ranking, criterion, f"{criterion} {metric.__name__}", colors=colors
        )
