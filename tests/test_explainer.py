import pytest

import ethik


def test_check_alpha():
    for alpha in (-1, 0, 0.5, 1):
        with pytest.raises(ValueError):
            ethik.Explainer(alpha=alpha)


def test_check_n_taus():
    for n_taus in (-1, 0):
        with pytest.raises(ValueError):
            ethik.Explainer(n_taus=n_taus)


def test_check_lambda_iterations():
    for iterations in (-1, 0):
        with pytest.raises(ValueError):
            ethik.Explainer(lambda_iterations=iterations)
