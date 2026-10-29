"""Competition metrics on wall density fields.

All metrics take fields of shape ``(n_runs, n_points)``. The error of a run is
a **component-weighted relative MAE**: each surface component contributes in
proportion to its weight (wing and pylon count more, as in ONERA's original
metric), independently of how many mesh points it has. Being relative per run,
it does not favour low- or high-pressure (Pi) runs.

The **shock-zone relative MAE** restricts the error to the shocks on the
upper wing surface, where a global error dilutes them. The **slice relative
MAE** restricts it to ONERA's six spanwise wing sections, the classical
chordwise distribution plots of applied aerodynamics.

Runs at ``|AoA| >= 10`` deg count half in the mean relative and slice MAE:
the CFD solver converges poorly there, so their labels are less trusted.
"""

import numpy as np
from scipy import sparse

COMPONENT_WEIGHTS = {"wing": 0.3, "pylon": 0.3, "fuselage": 0.2,
                     "nacelle": 0.2}

# ONERA's wing slices outside the engine span: (y, half-width) in metres.
SLICES = [(5.57, 0.05), (13.4, 0.1), (17.6, 0.15), (21.9, 0.1), (26.0, 0.05),
          (28.5, 0.05)]


def run_weights(conditions, high_aoa=10.0):
    """Weight of each run in averages: 0.5 at ``|AoA| >= high_aoa``, else 1."""
    return np.where(np.abs(conditions[:, 1]) >= high_aoa, 0.5, 1.0)


def slice_weights(geometry):
    """Uniform point weights on the ``SLICES`` wing sections, 0 elsewhere.

    All zero when the mesh crosses no slice (simulated data).
    """
    y = geometry[:, 1]
    hit = np.zeros(len(y), dtype=bool)
    for y0, half_width in SLICES:
        hit |= np.abs(y - y0) < half_width
    return hit / max(hit.sum(), 1)


def point_weights(components, component_names):
    """Per-point weights: each component's weight spread over its points.

    Points of components not in ``COMPONENT_WEIGHTS`` get weight 0.
    """
    missing = set(COMPONENT_WEIGHTS) - set(component_names)
    if missing:
        raise ValueError(f"components missing from the mesh: {missing}")
    w = np.zeros(len(components))
    for cid, name in enumerate(component_names):
        mask = components == cid
        if name in COMPONENT_WEIGHTS and mask.any():
            w[mask] = COMPONENT_WEIGHTS[name] / mask.sum()
    return w / w.sum()


def check_predictions(pred, shape):
    """Return ``pred`` as a float array of ``shape``, or fail loudly."""
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != shape:
        raise ValueError(f"predict() returned shape {pred.shape}, "
                         f"expected {shape} (n_runs, n_points).")
    if not np.isfinite(pred).all():
        raise ValueError(f"predict() returned {(~np.isfinite(pred)).sum()} "
                         "NaN/Inf values.")
    return pred


def relative_mae(y, yhat, weights):
    """Component-weighted relative MAE of each run, shape ``(n_runs,)``."""
    return np.abs(yhat - y) @ weights / (np.abs(y) @ weights)


def signed_bias(y, yhat, mask):
    """Signed relative error averaged with ``mask`` over points, then runs."""
    return float(((yhat / y - 1) @ mask).mean())


def pooled_r2(y, yhat):
    """R^2 over all runs and points of a subset.

    Pooled rather than per run: low-Mach fields are nearly flat, so a per-run
    R^2 divides by almost zero.
    """
    return float(1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def shock_grid(geometry, components, component_names, band=0.25,
               n_bins=50):
    """Chordwise grid on the upper wing surface, used by ``shock_zone``.

    The span is cut in strips of ``band`` metres and each strip in ``n_bins``
    bins of local chord fraction ``x / c``.
    """
    x, y, nz = geometry[:, 0], geometry[:, 1], geometry[:, 5]
    wing = components == list(component_names).index("wing")
    points = np.flatnonzero(wing & (nz > 0.3))
    strip = np.floor(y[points] / band).astype(int)
    strip -= strip.min()
    n_strips = strip.max() + 1
    x_min = np.full(n_strips, np.inf)
    x_max = np.full(n_strips, -np.inf)
    np.minimum.at(x_min, strip, x[points])
    np.maximum.at(x_max, strip, x[points])
    chord = np.maximum(x_max - x_min, 1e-9)[strip]
    xc = (x[points] - x_min[strip]) / chord
    cell = strip * n_bins + np.minimum((xc * n_bins).astype(int), n_bins - 1)
    count = np.bincount(cell, minlength=n_strips * n_bins)
    average = sparse.csr_matrix(
        (1 / count[cell], (cell, np.arange(len(points)))),
        shape=(n_strips * n_bins, len(points)))
    return dict(points=points, cell=cell, average=average,
                shape=(n_strips, n_bins),
                filled=(count > 0).reshape(n_strips, n_bins))


def shock_zone(fields, grid, threshold=0.03, halo=2):
    """Shock zone of each run on the grid points, shape ``(n_runs, n_grid)``.

    A shock is a density rise above ``threshold`` from one chord bin to the
    next, between 10% and 90% of the chord. The zone extends ``halo`` bins on
    each side.
    """
    n_runs = len(fields)
    n_bins = grid["shape"][1]
    mean = (grid["average"] @ fields[:, grid["points"]].T).T
    mean = mean.reshape(n_runs, *grid["shape"])
    ok = grid["filled"][:, 1:] & grid["filled"][:, :-1]
    rise = (mean[..., 1:] - mean[..., :-1]) / np.where(ok, mean[..., :-1], 1)
    centre = (np.arange(n_bins - 1) + 0.5) / n_bins
    hit = ok & (rise > threshold) & (centre > 0.1) & (centre < 0.9)
    hit = np.pad(hit, ((0, 0), (0, 0), (0, 1)))
    zone = hit.copy()
    for h in range(1, halo + 1):
        zone[..., h:] |= hit[..., :-h]
        zone[..., :-h] |= hit[..., h:]
    return zone.reshape(n_runs, -1)[:, grid["cell"]]


def shock_rmae(y, yhat, grid, min_points=200):
    """Relative MAE in the shock zone of each run that has one.

    The zone is the union of the shocks of the truth and of the prediction,
    so a missed, misplaced or spurious shock all count. Runs whose zone has
    fewer than ``min_points`` points are left out.
    """
    zone = shock_zone(y, grid) | shock_zone(yhat, grid)
    keep = zone.sum(axis=1) >= min_points
    y, yhat = y[:, grid["points"]][keep], yhat[:, grid["points"]][keep]
    zone = zone[keep]
    return (np.abs(yhat - y) * zone).sum(1) / (np.abs(y) * zone).sum(1)
