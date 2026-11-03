# How to participate

## The task

ONERA ran 468 RANS simulations of the NASA/Boeing **Common Research Model**
(CRM), a reference airliner geometry, for a grid of flight conditions:
13 **Mach** numbers × 12 **angles of attack** × 3 **stagnation pressures**
(Pi). The aircraft surface is the same mesh of 260,774 points in every run.

Given the flight conditions `(Mach, AoA, Pi)` of an unseen run, predict the
**air density at every surface point**. Your model outputs one field of
260,774 values per run.

You get:

- `conditions`: `(n_runs, 3)` flight conditions of the training runs;
- `fields`: `(n_runs, 260774)` their wall density;
- `meta["geometry"]`: `(260774, 6)` point coordinates and surface normals;
- `meta["components"]`, `meta["component_names"]`: the surface component
  (wing, pylon, fuselage, nacelle, ...) of each point.

## The metric

Each test run is scored by its **component-weighted relative MAE**: the mean
absolute error relative to the mean density, where the wing and the pylon
weigh 0.3 each and the fuselage and the nacelle 0.2 each, whatever their
number of points. The leaderboard shows its mean over test runs (the ranking
metric), its worst run, and the R² over all test points.

## The split

Runs are split by **Mach line**: every run of a given Mach number is on the
same side.

- **Public columns** (live feedback): held-out Mach lines.
- **Private / final ranking** (hidden): other held-out Mach lines, including
  Mach numbers outside the training range. Extrapolating to unseen flow
  regimes is the challenge. Do not overfit the public columns.

## What you submit

A single **`submission.py`** defining `class Solver(CRMSolver)`. All your
Python code must live in this one file: the platform loads it on its own, so
it cannot import other `.py` files you ship. You may add non-code files next
to it (model weights, configs) and read them from `meta["submission_dir"]` in
`get_model`. You implement two methods:

```python
class Solver(CRMSolver):
    name = "submission"

    def get_model(self, meta):
        # build your (untrained) model
        ...

    def fit(self, model, conditions, fields):
        # train it on the training runs
        ...
```

Your trained model must expose `predict(conditions) -> ndarray (n_runs,
260774)`. The platform **trains your code** on identical hardware, then
evaluates it. See `solution/submission.py` for a working baseline (POD +
Gaussian process).

## The compute

Each submission runs on its own AWS g4dn.2xlarge machine: one NVIDIA T4 GPU
(16 GB), 8 vCPUs and 32 GB of RAM, in the competition's Docker image
(`tommoral/onera-crm-compet:v1`, CUDA 12.6). Training and evaluation
together must fit in **1 hour**, or the submission fails. The three Datacamp
challenges share these machines, so a submission may wait in the queue at
busy times.

## Get the data

One command downloads all the public data from Zenodo, the 288 training
runs (about 250 MB):

```bash
pip install -r requirements.txt
benchopt prepare benchmark -d ONERA-CRM
```

The data goes to `benchmark/data/`, or to the folder `BENCHOPT_DATA_HOME`
names. The test runs are not public: they stay on the platform.

## Run it locally, same code as the server

```bash
benchopt run benchmark -d ONERA-CRM -s solution/submission.py
```

The number you get locally on the public split is the number the leaderboard
shows. The benchmark you run *is* the benchmark the server runs.

## Ideas to improve the baseline

Normalize the fields by Pi, tune the number of POD modes and the kernel,
treat the transonic regime separately, or use the geometry with a pointwise
or graph model.
