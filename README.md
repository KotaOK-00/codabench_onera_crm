# ONERA CRM wall-density competition

A DataCamp challenge built on ONERA's database of 468 RANS simulations of the
NASA/Boeing Common Research Model. Given the flight conditions `(Mach, AoA,
Pi)` of an unseen run, predict the wall density at each of the 260,774 points
of the aircraft surface.

The evaluation is a [benchopt](https://benchopt.github.io) benchmark run on
Codabench: participants run locally the same code the server runs.

## Layout

| Path | Content |
|---|---|
| `benchmark/` | the benchopt benchmark: objective, datasets (`ONERA-CRM`, `Simulated`), baseline solvers (`Mean`, `KNN`) |
| `solution/submission.py` | sample submission (POD + Gaussian process) |
| `ingestion_program/`, `scoring_program/` | Codabench programs: run `benchopt run`, parse the results |
| `phases/main/input_data/config.yaml` | phase run config and leaderboard column map |
| `competition.yaml`, `pages/` | Codabench competition and pages |
| `tools/` | bundle builder, Docker image and local Docker run |

## Quick start

```bash
pip install -r requirements.txt
benchopt run benchmark -d Simulated -s solution/submission.py -s KNN
```

Real data: `BENCHOPT_DATA_HOME=<data> benchopt prepare benchmark -d ONERA-CRM`
downloads the training runs from Zenodo into `<data>/onera_crm_raw/` and
prepares them.

Build the Codabench bundle with `python tools/create_bundle.py`.
