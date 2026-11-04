"""Train / public / private split of the simulations, by Mach number.

Each Mach line (all its AoA x Pi runs) sits entirely on one side, except
where an entry is a ``(Mach, Pi)`` pair: that line is split by Pi. Shared by
the ``ONERA-CRM`` dataset's ``prepare`` and the ``Simulated`` dataset so both
exercise the same protocol.

Public: the transonic lines next to the hidden ones (0.75 to 0.84), and the
Mach 0.96 runs above the training range at two of the three Pi. Private:
every run left unlisted, so the private lines are only known where their
data is staged.
"""

import numpy as np

MACH_SPLIT = {
    "train": [0.30, 0.50, 0.70, 0.85, 0.86, 0.88, 0.90, 0.93],
    "public": [0.75, 0.80, 0.82, 0.84, (0.96, 1.0), (0.96, 4.0)],
}

# Training lines held out by ``ONERA-CRM[validation=True]``, as the private
# lines are: subsonic between two lines, transonic between two lines, and
# above the remaining range.
VALIDATION_MACHS = [0.50, 0.88, 0.93]


def split_machs():
    """Every Mach number listed in the split."""
    return sorted({e[0] if isinstance(e, tuple) else e
                   for entries in MACH_SPLIT.values() for e in entries})


def assign_split(mach, pi, atol=1e-6):
    """Split name of each run from its Mach and Pi: ``private`` if unlisted."""
    mach, pi = np.asarray(mach), np.asarray(pi)
    names = np.full(mach.shape, "private", dtype=object)
    for name, entries in MACH_SPLIT.items():
        for entry in entries:
            m, p = entry if isinstance(entry, tuple) else (entry, None)
            hit = np.isclose(mach, m, atol=atol)
            if p is not None:
                hit &= np.isclose(pi, p, atol=atol)
            names[hit] = name
    return names
