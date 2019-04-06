import unittest

import ethik


class TestExplainer(unittest.TestCase):

    def test_check_alpha(self):
        for alpha in (-1, 0, 0.5, 1):
            self.assertRaises(ValueError, ethik.Explainer(alpha=alpha).check_parameters)

    def test_check_n_taus(self):
        for n_taus in (-1, 0):
            self.assertRaises(ValueError, ethik.Explainer(n_taus=n_taus).check_parameters)

    def test_check_max_iterations(self):
        for max_iterations in (-1, 0):
            check = ethik.Explainer(max_iterations=max_iterations).check_parameters
            self.assertRaises(ValueError, check)
