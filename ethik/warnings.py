__all__ = ["ConstantWarning", "ConvergenceWarning"]


class ConvergenceWarning(UserWarning):
    """Custom warning to capture convergence problems."""


class ConstantWarning(UserWarning):
    """Custom warning to capture issues in user-provided features."""
