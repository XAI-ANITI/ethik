import decimal

import numpy as np
import pandas as pd
import plotly.graph_objs as go

__all__ = [
    "decimal_range",
    "extract_category",
    "join_with_overlap",
    "safe_scale",
    "to_pandas",
    "yield_masks",
]


plot_template = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="white",
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="black",
            zeroline=False,
            showgrid=True,
            gridcolor="#eee",
        ),
        yaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="black",
            zeroline=False,
            showgrid=True,
            gridcolor="#eee",
        ),
    )
)


def set_fig_size(fig, size, width=None, height=None):
    if size is not None:
        width, height = size
    fig.update_layout(width=width, height=height)


def safe_scale(x):
    # If the std is zero, the values are constant and equal to the mean, so
    # the difference is zero.
    if isinstance(x, pd.DataFrame):
        return (x - x.mean()) / x.std().replace(0, 1)
    return (x - x.mean(axis=0)) / (x.std(axis=0) or 1)


def extract_category(X, cat):
    return pd.get_dummies(X)[cat].rename(f"{X.name}={cat}")


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
            raise ValueError("x must have 1 or 2 dimensions")
        if x.ndim == 2:
            return pd.DataFrame(x)
        return pd.Series(x)

    return to_pandas(np.asarray(x))


def join_with_overlap(left, right, on):
    """Joins left with right while handling overlapping columns.

    Currently, pandas raises an error if left and right have overlapping columns. This function
    takes care of preserving the values from the columns of left that exist in right.

    """
    overlap = left[left.columns.intersection(right.columns)]
    left = left.drop(columns=overlap.columns).join(right, on=on)
    for col in overlap:
        left[col] = left[col].fillna(overlap[col])
    return left


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

    if p < 0 or p > 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")

    if p < 1:
        for _ in range(n_masks):
            yield np.random.binomial(1, p, size=n).astype(bool)
    else:
        for _ in range(n_masks):
            yield np.full(shape=n, fill_value=True)
