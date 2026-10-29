"""Objective for the ONERA CRM wall-density competition.

Given the flight conditions ``(Mach, AoA, Pi)`` of an unseen RANS simulation
of the NASA/Boeing CRM, predict the wall density at every point of the fixed
surface mesh. A submission's model returns one field per run.

Evaluation runs on held-out subsets of whole Mach lines: ``public`` (scored
live) and ``private`` (sealed, final ranking). Metrics are emitted per subset
as ``objective_<metric>_<subset>``; the phase config maps them to leaderboard
keys and hides the private columns. See ``benchmark_utils.metrics``.

The solver always trains on the platform: ``get_objective`` hands it the
training runs and the mesh, and its ``fit`` runs before evaluation.
"""

import numpy as np
from benchopt import BaseObjective

from benchmark_utils.metrics import (check_predictions, point_weights,
                                     pooled_r2, relative_mae, run_weights,
                                     shock_grid, shock_rmae, signed_bias,
                                     slice_weights)


class Objective(BaseObjective):

    name = "onera-crm-wall-density"
    url = "https://github.com/x-datascience-datacamp/codabench_onera_crm"

    requirements = ["scikit-learn"]

    min_benchopt_version = "1.9.2"

    sampling_strategy = "run_once"

    def set_data(self, conditions, fields, geometry, components,
                 component_names, test_sets, mask=None):
        # test_sets: ``name -> (conditions, fields)``; the test fields stay
        # here so the objective owns evaluation.
        self.conditions = conditions
        self.fields = fields
        self.mesh = dict(geometry=geometry, components=components,
                         component_names=component_names)
        self.weights = point_weights(components, component_names)
        self.shock_grid = shock_grid(geometry, components, component_names)
        self.slice_weights = slice_weights(geometry)
        self.test_sets = test_sets
        n_points = len(components)
        self.mask = (np.full(n_points, 1 / n_points) if mask is None
                     else mask)

    def get_objective(self):
        return dict(conditions=self.conditions, fields=self.fields,
                    **self.mesh)

    def evaluate_result(self, model):
        scores = {}
        for name, (conditions, fields) in self.test_sets.items():
            pred = check_predictions(model.predict(conditions), fields.shape)
            w_runs = run_weights(conditions)
            rmae = relative_mae(fields, pred, self.weights)
            scores[f"rmae_{name}"] = float(np.average(rmae, weights=w_runs))
            scores[f"rmae_worst_{name}"] = float(rmae.max())
            scores[f"r2_{name}"] = pooled_r2(fields, pred)
            # 0 when no run of the subset has a shock.
            shock = shock_rmae(fields, pred, self.shock_grid)
            scores[f"shock_rmae_{name}"] = (float(shock.mean()) if len(shock)
                                            else 0.0)
            scores[f"shock_rmae_worst_{name}"] = float(shock.max(initial=0))
            # 0 when the mesh crosses no slice (simulated data).
            scores[f"slice_rmae_{name}"] = (float(np.average(
                relative_mae(fields, pred, self.slice_weights),
                weights=w_runs)) if self.slice_weights.any() else 0.0)
            scores[f"bias_{name}"] = signed_bias(fields, pred, self.mask)
        # Same bias on a fixed subset of training runs, for comparison.
        conditions, fields = self.conditions[::12], self.fields[::12]
        pred = check_predictions(model.predict(conditions), fields.shape)
        scores["bias_train"] = signed_bias(fields, pred, self.mask)
        return scores

    def get_one_result(self):
        # The mean-field floor, to validate the metrics in ``benchopt test``.
        from benchmark_utils.baselines import MeanModel
        return dict(model=MeanModel().fit(self.conditions, self.fields))
