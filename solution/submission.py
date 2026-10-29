"""Sample submission: POD + Gaussian process, the starting point to improve.

A submission is a benchopt solver: a ``submission.py`` defining
``class Solver(CRMSolver)``. This baseline compresses the training fields
with a PCA (Proper Orthogonal Decomposition, POD), then regresses the POD
coefficients on the flight conditions ``(Mach, AoA, Pi)`` with a Gaussian
process. A prediction is the decoded field.

Run it locally exactly as the platform does::

    benchopt run benchmark -d Simulated -s solution/submission.py
    # real data: raw arrays in <data>/onera_crm_raw, then
    export BENCHOPT_DATA_HOME=<data>
    benchopt prepare benchmark -d ONERA-CRM
    benchopt run benchmark -d ONERA-CRM -s solution/submission.py

Ideas to improve it: normalize the fields by Pi, tune the number of modes and
the kernel, model the transonic regime separately, or use the mesh geometry
(``meta["geometry"]``) with a pointwise or graph model.
"""

from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CRMSolver


class PODGP:

    def __init__(self, n_modes):
        self.pod = PCA(n_components=n_modes, random_state=0)
        self.scaler = StandardScaler()
        kernel = (ConstantKernel() * RBF(length_scale=[1.0, 1.0, 1.0])
                  + WhiteKernel(noise_level=1e-4))
        self.gp = GaussianProcessRegressor(kernel, normalize_y=True,
                                           n_restarts_optimizer=2,
                                           random_state=0)

    def fit(self, conditions, fields):
        coefs = self.pod.fit_transform(fields)
        self.gp.fit(self.scaler.fit_transform(conditions), coefs)
        return self

    def predict(self, conditions):
        coefs = self.gp.predict(self.scaler.transform(conditions))
        return self.pod.inverse_transform(coefs.reshape(len(conditions), -1))


class Solver(CRMSolver):

    name = "submission"

    def get_model(self, meta):
        return PODGP(n_modes=10)

    def fit(self, model, conditions, fields):
        model.fit(conditions, fields)
