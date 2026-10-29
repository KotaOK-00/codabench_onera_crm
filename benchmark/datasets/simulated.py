"""Synthetic CRM-like dataset: no download, runs anywhere (CI smoke tests).

Uses the same Mach split as the real data (``benchmark_utils.split``); both
the public and private subsets are present.
"""

import numpy as np
from benchopt import BaseDataset

from benchmark_utils.simulate import (COMPONENT_NAMES, make_conditions,
                                      make_fields, make_mesh)
from benchmark_utils.split import assign_split


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {"n_points": [2000], "n_aoa": [12]}
    test_config = {"n_points": 200, "n_aoa": 4}

    def get_data(self):
        rng = np.random.default_rng(self.get_seed(use_repetition=True))
        geometry, components = make_mesh(self.n_points, rng)
        conditions = make_conditions(self.n_aoa)
        fields = make_fields(conditions, geometry)
        split = assign_split(conditions[:, 0], conditions[:, 2])
        test_sets = {name: (conditions[split == name], fields[split == name])
                     for name in ("train", "public", "private")}
        train_conditions, train_fields = test_sets.pop("train")
        return dict(
            conditions=train_conditions,
            fields=train_fields,
            geometry=geometry,
            components=components,
            component_names=COMPONENT_NAMES,
            test_sets=test_sets,
        )
