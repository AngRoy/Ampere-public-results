# AMPERE Appliance Results

This repository publishes a reproducible evidence package for the corrected AMPERE `appliance_8ch` results.

The important correction is that `combined_output.csv` is time-major interleaved dense truth:

```text
rows 1-8   = appliances 1-8 at 0.00-0.04 s
rows 9-16  = appliances 1-8 at 0.04-0.08 s
rows 17-24 = appliances 1-8 at 0.08-0.12 s
```

Therefore the corrected appliance contract is:

| Field | Correct value |
|---|---:|
| raw values | 360,000 |
| branches | 8 |
| common time rows | 45,000 |
| sample interval | 0.04 s |
| duration | 1,800 s / 30 min |
| dwell samples | 1 |
| effective dwell | 0.04 s |
| scan cycle | 0.32 s |

The earlier appliance interpretation using `dt_s=0.32`, `dwell_s_effective=0.96`, and scan cycle `7.68 s` is superseded.

## Corrected Headline Results

| Method | Corrected dwell MAE |
|---|---:|
| Linear classical | 75.3944 W |
| Best online tree | 78.1974 W |
| DwellMLP, 3 seeds | 43.8311 +/- 2.8979 W |
| DwellObserver-T, 3 seeds | 40.6439 +/- 1.2933 W |

DwellObserver-T remains the best corrected MAE result, but the old absolute appliance numbers are not paper-facing anymore.

## Data

Raw data is not committed. Copy these files into `data/raw/`:

| File | Required SHA-256 |
|---|---|
| `combined_output.csv` | `7a46846804b2ea255b7f2cb041d9f9df2139ccbf5ee851fecdf47ebe038e7cb6` |
| `Sample_training_data.csv` | `c9fdc042ea2a3438d66d5eb1d94f65c7d2dd39fa2478518f12451985f9fd085e` |

See [data/README.md](data/README.md) for exact placement.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For neural training reruns:

```bash
pip install -r requirements-training.txt
```

## Verification

Fast verification notebooks:

```bash
python scripts/verify_publication.py --mode verify
```

This runs:

- `00_setup_and_data_check.ipynb`
- `01_appliance_40ms_contract.ipynb`
- `02_reproduce_baselines.ipynb`
- `05_verify_claims_and_plots.ipynb`

Smoke neural rerun:

```bash
python scripts/verify_publication.py --mode smoke
```

Full paper rerun:

```bash
python scripts/verify_publication.py --mode paper
```

Full neural reruns can take substantial time and are best run on CUDA.

## Repository Layout

```text
data/                  raw data instructions only
figures/               selected corrected publication figures
notebooks/             verification and rerun notebooks
results/expected/      small expected tables and manifests
scripts/               runnable pipeline and notebook verification scripts
src/ampere/            vendored AMPERE pipeline code needed for reruns
src/ampere_public/     public helper API used by notebooks
```

Generated outputs go to `runs/` and are ignored by Git.

