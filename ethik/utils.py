import decimal

import numpy as np
import pandas as pd

__all__ = ["decimal_range", "extract_category", "join_with_overlap", "to_pandas"]


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
