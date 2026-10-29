"""Synthetic stand-in for the CRM database (smoke tests and CI demo data).

Same layout as the real data: a fixed surface mesh with four components and a
grid of flight conditions ``(Mach, AoA, Pi)``. The density field scales with
Pi, depends smoothly on the flow direction, and gets a shock-like term that
switches on in the transonic range, so a nearest-neighbour model beats the
mean field and Mach extrapolation is harder than interpolation.
"""

import numpy as np

from benchmark_utils.split import split_machs

COMPONENT_NAMES = ["fuselage", "wing", "pylon", "nacelle"]
PI_VALUES = [1.0, 2.0, 4.0]
# Stand-in private lines, inside and above the training range.
PRIVATE_MACH = [0.60, 0.95]


def make_mesh(n_points, rng):
    """Random points and normals on a unit sphere, cut into 4 components."""
    xyz = rng.standard_normal((n_points, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    geometry = np.hstack([xyz, xyz]).astype(np.float32)
    components = np.digitize(xyz[:, 0], [-0.5, 0.0, 0.5])
    return geometry, components


def make_conditions(n_aoa):
    """Full grid over every Mach of the split x ``n_aoa`` AoA x 3 Pi."""
    mach = sorted(split_machs() + PRIVATE_MACH)
    aoa = np.linspace(-8.0, 10.0, n_aoa)
    grid = np.meshgrid(mach, aoa, PI_VALUES, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(np.float32)


def make_fields(conditions, geometry):
    """Density field of each run, shape ``(n_runs, n_points)``."""
    mach, aoa, pi = (conditions[:, i:i + 1] for i in range(3))
    alpha = np.deg2rad(aoa)
    normals = geometry[None, :, 3:]
    flow = np.stack([np.cos(alpha), np.sin(alpha), 0 * alpha], axis=-1)
    stagnation = (normals * flow).sum(axis=-1)
    shock = np.tanh((mach - 0.85) / 0.04) * np.sin(4 * geometry[:, 0])
    rho = 1.0 + 0.4 * mach ** 2 * stagnation + 0.15 * mach * shock
    return (pi * rho).astype(np.float32)
