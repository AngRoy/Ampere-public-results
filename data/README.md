# Data Placement

This repo does not publish raw data. To verify or rerun the results, copy the two source CSV files into:

```text
data/raw/combined_output.csv
data/raw/Sample_training_data.csv
```

Expected hashes:

| File | SHA-256 |
|---|---|
| `combined_output.csv` | `7a46846804b2ea255b7f2cb041d9f9df2139ccbf5ee851fecdf47ebe038e7cb6` |
| `Sample_training_data.csv` | `c9fdc042ea2a3438d66d5eb1d94f65c7d2dd39fa2478518f12451985f9fd085e` |

The verification notebooks stop immediately with a clear error if either file is missing or has a different hash.

