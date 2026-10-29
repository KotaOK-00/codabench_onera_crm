"""Reference models used by the built-in solvers and by ``benchopt test``."""

import numpy as np


class MeanModel:
    """Predict the mean training field for every run (leaderboard floor)."""

    def fit(self, conditions, fields):
        self.mean_field = fields.mean(axis=0)
        return self

    def predict(self, conditions):
        return np.tile(self.mean_field, (len(conditions), 1))
