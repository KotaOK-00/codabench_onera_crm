"""Mean-field baseline: predicts the mean training field for every run.

The leaderboard floor: any model using the flight conditions should beat it.
"""

from benchmark_utils.base_solver import CRMSolver
from benchmark_utils.baselines import MeanModel


class Solver(CRMSolver):

    name = "Mean"

    def get_model(self, meta):
        return MeanModel()

    def fit(self, model, conditions, fields):
        model.fit(conditions, fields)
