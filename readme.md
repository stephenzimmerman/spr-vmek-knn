# SPR VMEK KNN Kernel Size Classification

This repository contains the final SPR VMEK post-processing model and application code for predicting kernel size bucket percentages from VMEK output files.

The model is designed for **post-VMEK reporting**, not real-time sorting. It ingests raw VMEK kernel measurement files, applies machine-specific standardization, predicts screen-equivalent kernel buckets with a KNN model, and summarizes final SPR output traits.

## Final output traits

The application outputs one row per sample/file with:

| Trait | Meaning |
|---|---|
| `IPSDP` | Small discard percentage |
| `IPSSP` | Small saleable percentage |
| `IPLSP` | Large saleable percentage |
| `IPLDP` | Large discard percentage |
| `iSCLCN` | Clean seed/kernel count after valid-kernel filtering |

Screen-to-trait mapping used for training:

| Screen range | Output trait |
|---|---|
| `<=15` | `IPSDP` |
| `16–19` | `IPSSP` |
| `20–23` | `IPLSP` |
| `>=24` | `IPLDP` |

## Final model

Final model package:

```text
vmek_knn_final_optionB_plus_2021_IPLDP.joblib
```

Model specification:

| Item | Value |
|---|---|
| Algorithm | KNN |
| K | 5 |
| Distance | Euclidean |
| Scaling | `StandardScaler` |
| Model format | `.joblib` |
| Training rows | 722,439 kernels |
| Added IPLDP data | 375 older USSL APL1 screen-24 kernels |

Final model features:

```text
Area_mm2_corrected
Length_mm_corrected
Width_mm_corrected
ContLength_mm_equiv_corrected
```

## Machine correction and preprocessing

The app uses machine-specific values from the VMEK machine correction reference table:

```text
px_mm
area_corr
linear_corr = sqrt(area_corr)
```

Final preprocessing uses **Option B**:

### Legacy / Metrix pixel-style files

Required input columns:

```text
Area
Length
Width
ContLength
```

Calculations:

```text
Area_mm2_corrected = Area_px × px_mm² × area_corr
Length_mm_corrected = Length_px × px_mm × sqrt(area_corr)
Width_mm_corrected = Width_px × px_mm × sqrt(area_corr)
ContLength_mm_equiv_corrected = ContLength_px × px_mm × sqrt(area_corr)
```

A 100–1000 px area filter is applied by default to legacy pixel files.

### MX4 / mm-style files

Required input columns:

```text
Area
Length
Width
Roundness
```

Calculations:

```text
Area_mm2_corrected = Area_mm2 × area_corr
Length_mm_corrected = Length_mm × sqrt(area_corr)
Width_mm_corrected = Width_mm × sqrt(area_corr)
ContLength_mm_equiv_corrected = sqrt(4π × Area_mm2_corrected / Roundness)
```

## Applications included

### 1. Production folder-batch GUI

```text
SPR_KNN_Vmek_Model_Standalone.py
```

Use this for routine processing of folders of raw VMEK files.

It supports:

- folder import / batch processing,
- machine selection by `Site / APL`,
- built-in machine correction registry,
- optional registry update from Excel,
- sample-level output,
- optional kernel-level detail output.

Run:

```bash
python SPR_KNN_Vmek_Model_Standalone.py
```

### 2. Old vs new comparison app

```text
VMEK_Bucket_Predictor_Final_Comparison.py
```

Use this for leadership/coworker demonstrations comparing:

1. old area-threshold method, and  
2. final KNN model.

The old method uses corrected area with historical thresholds:

```json
{
  "small_discard_lower": 8,
  "small_discard_upper": 34,
  "saleable": 51,
  "large_discard_lower": 68,
  "large_discard_upper": 87
}
```

Run:

```bash
python VMEK_Bucket_Predictor_Final_Comparison.py
```

## Recommended repository contents

```text
spr-vmek-knn/
├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   └── vmek_knn_final_optionB_plus_2021_IPLDP.joblib
├── app/
│   ├── SPR_KNN_Vmek_Model_Standalone.py
│   └── VMEK_Bucket_Predictor_Final_Comparison.py
├── reference/
│   └── vmek-machine-correction-reference_final_names.xlsx
├── validation/
│   ├── vmek_final_model_holdout_validation.csv
│   ├── vmek_final_model_holdout_confusion.csv
│   └── vmek_knn_final_optionB_plus_2021_IPLDP_profile.json
└── docs/
    ├── spr-vmek-classification-summary.md
    └── VMEK_project_notes.md
```

## Files not recommended for GitHub

Do **not** commit raw kernel-level training or production data unless the repository is private and approved for that data.

Keep these out of the repo by default:

```text
vmek_final_training_table_optionB_plus_2021_IPLDP.csv
vmek_consolidated_elk_farm.csv
vmek-consolidated-quality-lab.csv
vmek-consolidated-slater-r-d.xlsx
vmek-training.csv
raw VMEK production folders
```

These are large and may contain proprietary operational data.

## Installation

Create a Python environment and install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Basic usage

1. Launch the standalone GUI:

```bash
python app/SPR_KNN_Vmek_Model_Standalone.py
```

2. Select:
   - `Site / APL`,
   - raw VMEK data folder,
   - output folder,
   - final `.joblib` model.

3. Click **Process folder**.

Main outputs:

```text
batch_spr_traits.csv
batch_processing_log.csv
```

## Validation summary

Cross-machine holdout validation from final model build:

| Held-out machine | Accuracy | Balanced accuracy | Mean output MAE |
|---|---:|---:|---:|
| Elk Farm SPR | 70.28% | 59.99% | 16.06 pp |
| Quality Lab241 | 74.51% | 64.31% | 12.85 pp |
| USSL APL1 | 70.36% | 54.53% | 14.72 pp |

For SPR reporting, the primary practical metrics are overall kernel accuracy and sample-level output trait MAE. Balanced accuracy is retained as a warning metric for sparse classes, especially `IPLDP`.

## Notes

- The model uses top-down VMEK 2D measurements.
- Physical screen sorting is based on minimum cross-section, so the IPSSP/IPLSP boundary remains the hardest region.
- `IPLDP` remains the sparsest class even after adding older screen-24 data.
- The machine registry is built into the apps, but can be updated from the machine correction reference workbook when needed.
