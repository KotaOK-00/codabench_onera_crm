"""Submission contract for the ONERA CRM competition.

A submission is a **valid benchopt solver**: a folder with a ``submission.py``
defining ``class Solver(CRMSolver)``. All its Python code lives in that single
file (benchopt loads it by path, so sibling ``.py`` files cannot be imported);
the folder may also hold non-code files. Participants implement:

- ``get_model(self, meta)`` -> model (required). Build the (untrained) model.
- ``fit(self, model, conditions, fields)`` (required). Train the model. It
  runs on the platform before evaluation.

with

- ``conditions`` : ``(n_runs, 3)`` flight conditions ``(Mach, AoA, Pi)``;
- ``fields``     : ``(n_runs, n_points)`` wall density of each training run.

The trained model must expose::

    model.predict(conditions) -> ndarray, shape (n_runs, n_points)

``meta`` is a plain dict with the fixed surface mesh shared by every run:
``geometry`` ``(n_points, 6)`` = ``(x, y, z, nx, ny, nz)``, ``components``
``(n_points,)`` integer ids into ``component_names``, and ``submission_dir``
to read shipped non-code files (weights, configs).
"""

import inspect
from pathlib import Path

from benchopt import BaseSolver


class CRMSolver(BaseSolver):
    """Base class for all submissions (and the built-in baselines).

    numpy / scipy / scikit-learn / torch come from the competition image;
    submissions cannot install extra packages.
    """

    requirements = []

    sampling_strategy = "run_once"

    def set_objective(self, conditions, fields, **meta):
        self.conditions = conditions
        self.fields = fields
        submission_dir = Path(inspect.getfile(type(self))).parent
        self.meta = {**meta, "submission_dir": submission_dir}
        self.model = self.get_model(self.meta)

    def run(self, _):
        self.fit(self.model, self.conditions, self.fields)

    def get_result(self):
        return dict(model=self.model)

    # --- to be implemented by the submission ---------------------------

    def get_model(self, meta):
        """Build and return the (untrained) model."""
        raise NotImplementedError

    def fit(self, model, conditions, fields):
        """Train the model in place."""
