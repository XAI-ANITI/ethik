from .explainer import Explainer

__all__ = ["ClassificationExplainer"]


class ClassificationExplainer(Explainer):
    def plot_bias(self, X_test, y_pred, **fig_kwargs):
        fig_kwargs["yrange"] = fig_kwargs.get("yrange", [0, 1])
        return super().plot_bias(X_test, y_pred, **fig_kwargs)
