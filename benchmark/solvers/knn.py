"""Nearest-neighbour baseline: averages the fields of the closest runs.

Looks up neighbours in the standardized ``(Mach, AoA, Pi)`` space. Strong
when the test conditions are surrounded by training runs (interpolation),
weak in Mach extrapolation.
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CRMSolver


class Solver(CRMSolver):

    name = "KNN"

    requirements = ["scikit-learn"]

    parameters = {"n_neighbors": [4]}

    def get_model(self, meta):
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                weights="distance"),
        )

    def fit(self, model, conditions, fields):
        model.fit(conditions, fields)
