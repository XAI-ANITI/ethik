import itertools
import warnings

import pandas as pd

from .utils import decimal_range
from .warnings import ConstantWarning


class Query:
    @classmethod
    def taus(cls, n):
        tau_precision = 2 / (n - 1)
        return list(decimal_range(-1, 1, tau_precision))

    @classmethod
    def target_from_tau(cls, q_min, q_max, mean, tau):
        return mean + tau * (max(mean - q_min, 0) if tau < 0 else max(q_max - mean, 0))

    @classmethod
    def unidim_from_taus(cls, X_test, to_do_labels, n_taus, q, first_group=0):
        quantiles = X_test.quantile(q=q)

        # Issue a warning if a feature doesn't have distinct quantiles
        for feature, n_unique in quantiles.nunique().to_dict().items():
            if n_unique == 1:
                warnings.warn(
                    message=f"all the values of feature {feature} are identical",
                    category=ConstantWarning,
                )

        q_mins = quantiles.loc[q[0]].to_dict()
        q_maxs = quantiles.loc[q[1]].to_dict()
        means = X_test.mean().to_dict()
        taus = cls.taus(n_taus)
        query = pd.concat(
            [
                pd.DataFrame(
                    {
                        "tau": taus,
                        "dimension": [1] * len(taus),
                        "target": [
                            cls.target_from_tau(
                                q_min=q_mins[feature],
                                q_max=q_maxs[feature],
                                mean=means[feature],
                                tau=tau,
                            )
                            for tau in taus
                        ],
                        "feature": [feature] * len(taus),
                        "label": [label] * len(taus),
                    }
                )
                for feature in X_test.columns
                for label in to_do_labels[feature]
            ],
            ignore_index=True,
        )
        query["group"] = list(range(first_group, first_group + len(query)))
        return query

    @classmethod
    def multidim_from_taus(cls, X_test, labels, n_taus, q, first_group=0):
        quantiles = X_test.quantile(q=q)

        # Issue a warning if a feature doesn't have distinct quantiles
        for feature, n_unique in quantiles.nunique().to_dict().items():
            if n_unique == 1:
                warnings.warn(
                    message=f"all the values of feature {feature} are identical",
                    category=ConstantWarning,
                )

        q_mins = quantiles.loc[q[0]].to_dict()
        q_maxs = quantiles.loc[q[1]].to_dict()
        means = X_test.mean().to_dict()
        taus = cls.taus(n_taus)
        features = X_test.columns
        targets = itertools.product(
            *[
                [
                    (
                        tau,
                        cls.target_from_tau(
                            q_min=q_mins[feature],
                            q_max=q_maxs[feature],
                            mean=means[feature],
                            tau=tau,
                        ),
                    )
                    for tau in taus
                ]
                for feature in features
            ],
            labels,
        )

        # Each element of `targets` is a tuple of length `n_features + 1`
        # The `n_features` first elements are the couples `(tau, target)` for every
        # feature
        # The last element is the output label

        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "group": [first_group + i] * len(features),
                        "dimension": [len(features)] * len(features),
                        "tau": [x[0] for x in group[:-1]],
                        "target": [x[1] for x in group[:-1]],
                        "feature": features,
                        "label": [group[-1]] * len(features),
                    }
                )
                for i, group in enumerate(targets)
            ],
            ignore_index=True,
        )
