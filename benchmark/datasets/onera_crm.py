"""Real ONERA CRM dataset.

``prepare`` converts the raw files into the compact layout below, split by
Mach line (``benchmark_utils.split``); ``get_data`` only loads it::

    benchopt prepare benchmark -d ONERA-CRM

Raw files are read from ``get_data_path("onera_crm_raw")`` (override with
``data_paths`` in ``benchopt.cfg``): one or more unzipped folders, each with
``conditions.csv`` (``sim_id, Mach, AoA, Pi, ...``),
``density/rho_<sim_id>.npy`` (``rho / rho_inf`` at every wall point) and
``geometry.npz`` (``coords``,
``normals``, ``component_labels``, ``component_names``), the same in every
folder. Runs are gathered from all folders. When it holds none, ``prepare``
downloads the release zips from Zenodo and unzips them there: the train zip,
plus the restricted public and private test zips when ``ONERA_DATA_TOKEN``
is set (the worker).

Prepared layout, under ``get_data_path("onera_crm")`` (on the competition
worker the image sets ``BENCHOPT_DATA_HOME`` and the data is staged there,
read-only)::

    geometry.npy             (n_points, 6) x, y, z, nx, ny, nz
    components.npy           (n_points,) ids into component_names.json
    component_names.json
    train/{conditions.npy, rho.npy}    (n_runs, 3) Mach, AoA, Pi and
    public/{conditions.npy, rho.npy}   (n_runs, n_points) wall density
    private/{...}            only written where the raw data holds the
                             private runs (the worker)
    mask.npy                 optional (n_points,) point mask of the signed
                             bias metric; uniform when absent
"""

import json
import os
import zipfile

import numpy as np
import pandas as pd
from benchopt import BaseDataset
from benchopt.config import get_data_path

from benchmark_utils.download import download_zenodo
from benchmark_utils.split import assign_split

# Public record: train zip, read with the token of its share link while it
# is a draft. Restricted record: public and private test zips.
PUBLIC_RECORD = "23751279"
PUBLIC_TOKEN = (
    "eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJlNjQwOGMwLWEyZTYtNGNhYS1iMjk2LTk1ODAxO"
    "TczMGViOSIsImRhdGEiOnt9LCJyYW5kb20iOiI5ZTIxNmNlOTQzZTg2OTg4YmMxYmY3OTE"
    "2ZWRjNjY5YiJ9.y4b9P7mZGiHLAwEBXm28Cbx6VWqvRfxis6ms4HttEoq6adLIlbdDaTAk"
    "JKg_Q1qLRYGffq4iHFEYm0OibkNp0w"
)
PRIVATE_RECORD = "23751285"
TOKEN_ENV = "ONERA_DATA_TOKEN"


class Dataset(BaseDataset):

    name = "ONERA-CRM"

    def prepare(self):
        raw = get_data_path("onera_crm_raw")
        if not any(raw.glob("*/conditions.csv")):
            download_zenodo(PUBLIC_RECORD, ["onera_crm_train.zip"], raw,
                            PUBLIC_TOKEN)
            if os.environ.get(TOKEN_ENV):
                download_zenodo(PRIVATE_RECORD, ["onera_crm_public.zip",
                                                 "onera_crm_private.zip"],
                                raw, os.environ[TOKEN_ENV])
            for z in raw.glob("onera_crm_*.zip"):
                zipfile.ZipFile(z).extractall(raw / z.stem)
        folders = sorted(p.parent for p in raw.glob("*/conditions.csv"))
        if not folders:
            raise SystemExit(f"[ONERA-CRM] no */conditions.csv in {raw}.")
        runs = pd.concat([pd.read_csv(f / "conditions.csv").assign(folder=f)
                          for f in folders])
        runs = runs.drop_duplicates("sim_id").sort_values("sim_id")
        conditions = runs[["Mach", "AoA", "Pi"]].to_numpy(np.float32)
        fields = np.stack([np.load(f / "density" / f"rho_{sim}.npy")
                           for f, sim in zip(runs.folder, runs.sim_id)])
        geo = np.load(folders[0] / "geometry.npz")

        root = get_data_path("onera_crm")
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "geometry.npy",
                np.hstack([geo["coords"], geo["normals"]]))
        np.save(root / "components.npy", geo["component_labels"])
        (root / "component_names.json").write_text(
            json.dumps([str(n) for n in geo["component_names"]]))

        split = assign_split(conditions[:, 0], conditions[:, 2])
        for name in ("train", "public", "private"):
            mask = split == name
            if mask.any():
                (root / name).mkdir(exist_ok=True)
                np.save(root / name / "conditions.npy", conditions[mask])
                np.save(root / name / "rho.npy", fields[mask])

    def get_data(self):
        root = get_data_path("onera_crm")
        if not root.exists():
            # prepare's cache does not check its output: --force if the
            # data was deleted or BENCHOPT_DATA_HOME changed.
            raise SystemExit(
                f"[ONERA-CRM] no data at {root}. Run "
                "`benchopt prepare benchmark -d ONERA-CRM --force`."
            )

        test_sets = {
            name: (np.load(root / name / "conditions.npy"),
                   np.load(root / name / "rho.npy"))
            for name in ("train", "public", "private")
            if (root / name).exists()
        }
        conditions, fields = test_sets.pop("train")
        mask = root / "mask.npy"
        return dict(
            conditions=conditions,
            fields=fields,
            geometry=np.load(root / "geometry.npy"),
            components=np.load(root / "components.npy"),
            component_names=json.loads(
                (root / "component_names.json").read_text()),
            test_sets=test_sets,
            mask=np.load(mask) if mask.exists() else None,
        )
