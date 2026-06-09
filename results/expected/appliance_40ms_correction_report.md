# Appliance 40 ms Timing Correction Report

Generated: 2026-06-09

## Understanding

`data/raw/combined_output.csv` is a time-major interleaved dense-truth file, not an 8-appliance serial stream with a 0.32 s row interval. Raw rows 1-8 are appliances 1-8 for the same 0.00-0.04 s interval, rows 9-16 are appliances 1-8 for 0.04-0.08 s, and so on. The corrected reshape is therefore `360000 / 8 = 45000` common time rows by 8 appliances. The common timeline is `dt_s = 0.04`, duration `45000 * 0.04 = 1800 s = 30 min`. For the corrected primary experiment, the observation mask is native 40 ms round-robin: one selected branch per common time row, `dwell_samples = 1`, `dwell_s_effective = 0.04`, and 8-branch scan cycle `0.32 s`. The previous appliance interpretation (`dt_s = 0.32`, `dwell_samples = 3`, `dwell_s_effective = 0.96`, scan cycle `7.68 s`) is superseded for appliance paper claims. RLC results remain separate and unchanged because the RLC corpus has explicit timestamps and verified `MCP_data` alignment.

## Dataset Citation

| Corpus | Local source | Correct interpretation | Raw fingerprint |
|---|---|---|---|
| Appliance 8-channel LTspice/component corpus | `data/raw/combined_output.csv` | Time-major interleaved dense truth, 8 branches, 45,000 common rows, 30 min, `dt_s=0.04` | SHA-256 `7a46846804b2ea255b7f2cb041d9f9df2139ccbf5ee851fecdf47ebe038e7cb6`, size `6,601,091` bytes |
| RLC sample corpus | `data/raw/Sample_training_data.csv` | Explicit timestamped 4-branch RLC sample, `dt_s=0.04`, `dwell_samples=25` | SHA-256 `c9fdc042ea2a3438d66d5eb1d94f65c7d2dd39fa2478518f12451985f9fd085e` |

The raw `Data/` files were not modified. The correction regenerated canonical outputs and downstream artifacts only.

## Artifact Root

Expected summary artifacts are committed under `results/expected/`. Recomputed artifacts are written to ignored `runs/` directories by the notebooks and scripts.

## Corrected Contract

| Field | Superseded appliance value | Corrected appliance value |
|---|---:|---:|
| Raw values | 360,000 | 360,000 |
| Branches | 8 | 8 |
| Common time rows | 45,000 | 45,000 |
| `dt_s` | 0.32 | 0.04 |
| Duration | 14,400 s / 4 h | 1,800 s / 30 min |
| `dwell_samples` | 3 | 1 |
| `dwell_s_effective` | 0.96 | 0.04 |
| Scan-cycle samples | 24 | 8 |
| Scan-cycle period | 7.68 s | 0.32 s |

## Main Result Changes

All corrected rows below use the corrected `appliance_8ch / dwell_mean_power / online_safe` test-block protocol unless stated otherwise.

| Claim / metric | Superseded result | Corrected 40 ms result | Status |
|---|---:|---:|---|
| Best classical dwell MAE | Linear `67.4753 W` | Linear `75.3944 W` | Changed |
| Best classical weighted energy error | `0.0032` | `0.0020` | Changed, still strong |
| Best online-safe tree dwell MAE | Window ExtraTrees `64.1049 W` | HistGradientBoosting `78.1974 W` | Old tree advantage invalidated |
| Best online-safe tree weighted energy error | `0.0093` | `0.0155` | Worse under corrected timing |
| DwellMLP 3-seed dwell MAE | `35.6457 +/- 0.6941 W` | `43.8311 +/- 2.8979 W` | Changed but still beats corrected classical/tree MAE |
| DwellMLP weighted energy error | `0.0145 +/- 0.0039` | `0.0218 +/- 0.0033` | Worse under corrected timing |
| DwellMLP transition MAE | `77.2218 +/- 3.2670 W` | `73.4550 +/- 4.7025 W` | Slightly better transition MAE, higher overall MAE |
| DwellMLP stable MAE | `22.6749 +/- 0.8392 W` | `21.2735 +/- 2.9527 W` | Similar/slightly better |
| DwellObserver-T 3-seed dwell MAE | `33.4993 +/- 0.3917 W` | `40.6439 +/- 1.2933 W` | Changed; corrected best neural result |
| DwellObserver-T weighted energy error | `0.0072 +/- 0.0005` | `0.0245 +/- 0.0024` | Old energy claim invalidated |
| DwellObserver-T transition MAE | `68.0309 +/- 0.9068 W` | `74.1292 +/- 3.8758 W` | Worse under corrected timing |
| DwellObserver-T stable MAE | old contract only | `18.1572 +/- 2.7250 W` | Corrected stable-region result |

Summary: the corrected neural headline is still valid directionally because DwellObserver-T remains the best corrected MAE result among the rerun methods (`40.6439 W` vs linear `75.3944 W` and online tree `78.1974 W`). The old absolute appliance numbers are not valid for the paper. The old tree-over-classical claim is invalidated: corrected online trees do not beat corrected linear interpolation. Corrected neural models reduce MAE but have worse weighted energy error than corrected linear interpolation.

## Corrected Seed Results

| Family | Seed | Dwell MAE | Weighted energy error | Transition MAE | Stable MAE |
|---|---:|---:|---:|---:|---:|
| DwellMLP | 42 | 43.5694 | 0.0200 | 70.0455 | 23.4087 |
| DwellMLP | 123 | 46.8510 | 0.0198 | 78.8197 | 22.5079 |
| DwellMLP | 2026 | 41.0730 | 0.0256 | 71.4998 | 17.9040 |
| DwellObserver-T | 42 | 40.7840 | 0.0271 | 78.4215 | 15.8943 |
| DwellObserver-T | 123 | 41.8614 | 0.0224 | 73.0800 | 21.1821 |
| DwellObserver-T | 2026 | 39.2862 | 0.0241 | 70.8859 | 17.3953 |

## Corrected Per-Branch MAE

| Branch | DwellMLP corrected MAE | DwellObserver-T corrected MAE |
|---|---:|---:|
| Branch01_Ceiling_Fan | 3.9628 | 3.6614 |
| Branch02_Tubelight | 4.8981 | 4.2885 |
| Branch03_Electric_Kettle | 107.5074 | 96.3424 |
| Branch04_Electric_Geyser | 99.4684 | 97.1606 |
| Branch05_Water_Pump | 69.5893 | 67.3920 |
| Branch06_Refrigerator | 14.5811 | 14.1373 |
| Branch07_Rice_Cooker | 9.8059 | 10.4210 |
| Branch08_Microwave_Oven | 40.8359 | 38.5859 |

The old-vs-corrected per-branch comparison is committed as `results/expected/superseded_vs_corrected_per_branch_mae.csv`.

## Generated Artifacts

Primary public entrypoint: `README.md`.

Key files:

| Artifact | Path |
|---|---|
| Correction run manifest | `results/expected/run_manifest.json` |
| Corrected canonical manifest | `runs/processed/manifest.json` |
| Corrected appliance wide parquet | `runs/processed/appliance_8ch_wide.parquet` |
| Corrected appliance long parquet | `runs/processed/appliance_8ch_long.parquet` |
| Classical leaderboard | `runs/reconstruction/classical/classical_leaderboard.csv` |
| Tree vs classical comparison | `runs/reconstruction/tree/tree_vs_classical_comparison.csv` |
| DwellMLP seed outputs | `runs/reconstruction/dwellnet_seed{42,123,2026}/` |
| DwellObserver-T seed outputs | `runs/reconstruction/dwellobserver_t_seed{42,123,2026}/` |
| Summary tables | `results/expected/` |
| Summary figures | `figures/` |

Primary figures:

| Figure | Path |
|---|---|
| Corrected MAE comparison | `figures/corrected_mae_comparison.png` |
| Corrected seed stability | `figures/corrected_seed_stability.png` |
| Corrected transition vs stable MAE | `figures/corrected_transition_stable_mae.png` |
| Corrected neural per-branch MAE | `figures/corrected_per_branch_mae.png` |
| DwellMLP old-vs-corrected per-branch MAE | `figures/dwellmlp_superseded_vs_corrected_per_branch_mae.png` |
| DwellObserver-T old-vs-corrected per-branch MAE | `figures/dwellobserver_t_superseded_vs_corrected_per_branch_mae.png` |

Per-seed training curves and prediction overlays are under:

- `runs/figures/dwellnet_seed42/`
- `runs/figures/dwellnet_seed123/`
- `runs/figures/dwellnet_seed2026/`
- `runs/figures/dwellobserver_t_seed42/`
- `runs/figures/dwellobserver_t_seed123/`
- `runs/figures/dwellobserver_t_seed2026/`

## Code And Documentation Updates

Updated the appliance timing contract in config, loaders, canonicalization, validation, scaling protocols, DwellNet dataset specs, inference bundle export defaults, runtime docs, data docs, decision log, result index, and tests. Active code/config/test paths no longer contain live appliance `0.32/0.96/7.68` contract values; remaining mentions are explicitly superseded-history text.

The corrected public scripts use corrected canonical parquet generated into `runs/processed/` and report `dwell_samples=1`, `dwell_seconds_effective=0.04`, `scan_cycle_samples=8`, `window_tokens=64`.

## Scaling And Adaptive Status

The previous appliance scaling/adaptive result rows were produced under the old appliance timing contract and are superseded for appliance paper claims. I did not rerun the full scaling/adaptive study because this correction pass focused on the primary reconstruction pipeline; the relevant synthetic adaptive/RL/scaling tests were updated to the corrected 40 ms fixture semantics and pass. If the final manuscript still uses appliance scaling/adaptive claims, those studies need a dedicated corrected rerun.

## Verification

Commands completed:

- `python -m pytest tests -q`: `263 passed, 4 warnings in 104.33s`
- Public verification script: `python scripts/verify_publication.py --mode verify`
- `uv run ruff check .`: failed on existing repo-wide lint issues (`92 errors`, mainly pre-existing `E402` path-injection patterns and unused imports in older scripts/notebooks). I did not refactor unrelated lint debt in this correction pass.
