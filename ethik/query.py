import hashlib
import itertools
import warnings

import pandas as pd

from .utils import decimal_range
from .warnings import ConstantWarning


class Query:
    @classmethod
    def create_gid(cls, features, targets):
        k = ";".join(f"{f}={t}" for f, t in zip(features, targets))
        h = hashlib.sha256()
        h.update(k.encode("utf-8"))
        return h.hexdigest()

    @classmethod
    def taus(cls, n):
        tau_precision = 2 / (n - 1)
        return list(decimal_range(-1, 1, tau_precision))

    @classmethod
    def target_from_tau(cls, q_min, q_max, mean, tau):
        return mean + tau * (max(mean - q_min, 0) if tau < 0 else max(q_max - mean, 0))

    @classmethod
    def _unidim_from_taus(cls, X_test, labels, n_taus, q, constraints=None):
        if constraints is None:
            constraints = {}

        diff_constraints = set(constraints) - set(X_test.columns)
        if diff_constraints:
            raise ValueError(
                f"Unknown features in constraints: {', '.join(diff_constraints)}"
            )

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
        groups = []

        for feature in set(X_test.columns) - set(constraints):
            for tau in taus:
                features = [feature, *constraints]
                targets = [
                    cls.target_from_tau(
                        q_min=q_mins[feature],
                        q_max=q_maxs[feature],
                        mean=means[feature],
                        tau=tau,
                    ),
                    *constraints.values(),
                ]
                gid = cls.create_gid(features, targets)

                for label in labels:
                    n_rows = 1 + len(constraints)
                    groups.append(
                        pd.DataFrame(
                            {
                                "group": [gid] * n_rows,
                                "tau": [tau, *[0] * len(constraints)],
                                "target": targets,
                                "feature": features,
                                "label": [label] * n_rows,
                            }
                        )
                    )

        query = pd.concat(groups, ignore_index=True)
        return query

    @classmethod
    def _multidim_from_taus(cls, X_test, labels, n_taus, q, constraints=None):
        if constraints is None:
            constraints = {}

        diff_constraints = set(constraints) - set(X_test.columns)
        if diff_constraints:
            raise ValueError(
                f"Unknown features in constraints: {', '.join(diff_constraints)}"
            )

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
        features = set(X_test.columns) - set(constraints)
        targets_product = [
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
        ]
        # If `constraints` is empty, `itertools.product` would return an empty generator
        if constraints:
            targets_product.append([(0, target) for target in constraints.values()])
        targets = itertools.product(*targets_product)

        # Each element of `targets` is a tuple of length `n_features + n_constraints`
        # The `n_features` first elements are the couples `(tau, target)` for every
        # feature
        # The `n_constraints` last elements are the couples `(0, target)` for every
        # constraint

        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "group": [cls.create_gid(X_test.columns, [x[1] for x in group])]
                        * len(X_test.columns),
                        "tau": [x[0] for x in group],
                        "target": [x[1] for x in group],
                        "feature": X_test.columns,
                        "label": [label] * len(X_test.columns),
                    }
                )
                for i, group in enumerate(targets)
                for label in labels
            ],
            ignore_index=True,
        )

    @classmethod
    def from_taus(
        cls, X_test, labels, n_taus, q, link_variables=False, constraints=None
    ):
        if link_variables:
            return cls._multidim_from_taus(
                X_test=X_test,
                labels=labels,
                n_taus=n_taus,
                q=q,
                constraints=constraints,
            )
        return cls._unidim_from_taus(
            X_test=X_test, labels=labels, n_taus=n_taus, q=q, constraints=constraints
        )
