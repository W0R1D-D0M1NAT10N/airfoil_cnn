# Results and evidence notes

This audit compares the supplied 18-page manuscript ([source notes](research/README.md)) with the repository snapshot at commit `4562f00076a64516211418cf9820767f208608e8` on `newmain`. It distinguishes saved-table calculations, manuscript reports, and missing evidence. Documentation and extracted figures do not supersede the original source. The full manuscript PDF is not distributed here.

## 2D metrics that can be checked

The committed `airfoil_data_clean.csv` contains 70,339 rows across 1,579 distinct `file_path` values. `airfoil_predictions_vs_true.csv` contains 13,907 rows across 313 profiles. Every saved prediction's `(airfoil_name, AoA, true_values)` matches a label in the clean table. That checks label correspondence, not whether the samples were excluded from training.

Using `y = true_values` and `p = pred_values`:

```text
MAE  = mean(abs(y - p))
MSE  = mean((y - p)^2)
RMSE = sqrt(MSE)
R²   = 1 - sum((y - p)^2) / sum((y - mean(y))^2)
```

| Raw 2D metric | Recomputed from saved CSV | Manuscript Table 1 (p. 18) |
| --- | ---: | ---: |
| N | 13,907 | 13,907 |
| MAE | 0.3252833052 | 0.3253 |
| MSE | 0.1201576807 | 0.1202 |
| RMSE | 0.3466376793 | 0.3466 |
| R² | 0.2275717498 | 0.2276 |

The raw metrics agree to the reported precision. The saved CSV does not encode the training run, checkpoint hash, or calibration split.

### Calibration: numerical agreement and an evaluation gap

The manuscript (p. 12) says affine calibration used data disjoint from evaluation. Neither calibration code, fitted parameters, nor split assignments are committed. The prediction scripts export raw model outputs.

As a diagnostic, fitting ordinary least squares **on all of the same saved prediction/label pairs used for scoring** gives:

```text
CL_cal = 1.3252635567 * CL_raw + 0.1204197231
```

| Same-sample fitted metric | Calculated diagnostic | Manuscript calibrated Table 1 |
| --- | ---: | ---: |
| MAE | 0.0522338494 | 0.05223 |
| MSE | 0.0060747726 | 0.0061 |
| RMSE | 0.0779408275 | 0.0779 |
| R² | 0.9609485974 | 0.9610 |

These agree to the table's precision. This does **not** establish how the historical calibration was performed or prove data leakage. It shows that the table can be numerically reproduced by an in-sample fit, while its stronger independent-evaluation claim cannot be verified from the available artifacts. The diagnostic must not be presented as a fresh held-out score.

## 3D metrics that remain manuscript reports

The manuscript describes 63 cases across five families: SECH (13), Sinc (13), sphere-disk/UFO (11), cosine (13), and Gaussian/UFOv4 (13). Its text (p. 4) reports calibrated Pearson `r = 0.9977` and `R² = 0.9955`. These are rounded to approximately 0.998 and 0.996 in the summary.

Table 2 (p. 18) reports different aggregate quantities:

| Table 2 summary | Raw | Calibrated |
| --- | ---: | ---: |
| MAE | 0.6886 | 0.2757 |
| MSE | 0.7961 | 0.1235 |
| RMSE | 0.8461 | 0.3322 |
| R² | 0.9527 | 0.9924 |

The row is labeled **Overall Average** with `N = 13`, although the family counts sum to 63. Its MAE/MSE/RMSE/R² values agree, to rounding, with unweighted arithmetic means of the five printed family metrics. A mean of family R² values is a different statistic from pooled/global R²; a mean of RMSE values is likewise not pooled RMSE. The text's global score might describe a different aggregation, but without the paired predictions it cannot be reconciled. Both reports are retained rather than choosing one as authoritative.

The checked-in `3DModel/flight_data.csv` instead has:

| Mesh | CSV rows | AoA range |
| --- | ---: | --- |
| `ufov4.obj` (Gaussian) | 16 | 0–15° |
| `sinc.obj` | 16 | 0–15° |
| `sech.obj` | 13 | 0–12° |
| **Total** | **45** | Three meshes |

Restricting those rows to 0–12° gives 39 cases, not 63. Sphere-disk and cosine meshes are present in `models/`, but their CFD labels are absent. There are no committed 3D prediction CSVs, trained weights, calibration artifacts, or eight-seed outputs. Consequently no 3D MAE, R², Pearson correlation, or error bars were independently recomputed here.

## Architecture and training differences

| Item | Manuscript description | Checked-in evidence |
| --- | --- | --- |
| Predicted targets | Several overview/method passages say lift and drag | Both models have one output trained on `cl`; p. 12 also says only lift was used |
| CNN input size | 200 × 60 pixels | GPU trainer resizes to 100 × 30; older versions use 256 × 256 or source size |
| CNN head | Methods/caption: 512 → 128 → 1; Figure 1 labels show 1024 → 256 → 1 | GPU trainer: 1024 → 256 → 1; older/prediction classes: 64 → 32 → 1 |
| Saved CNN checkpoint | No uniquely identified checkpoint in manuscript | Archive metadata has `head.0.weight` shape `(512, 48016)`, `head.3.weight` `(128, 512)`, `head.5.weight` `(1, 128)` |
| CNN validation | Five-fold, family-wise validation | Scripts take only `next(...)` from five-way GroupKFold and group by exact filename, not broader family |
| CNN preprocessing | Held-out evaluation described | AoA scaler is fitted before the split; original scaler and split CSVs are missing |
| CNN checkpoint selection | Selected by validation performance | GPU script overwrites the same checkpoint each epoch without tracking best validation loss |
| 3D training duration | 1,000 epochs | `3DModel/train.py`: 100 epochs |
| 3D dropout/AoA | Dropout 0.2, normalized AoA | No dropout layers in `model.py`; dataset passes raw CSV AoA |
| 3D validation | Random held-out split, eight runs, metadata tracked | Trainer uses all CSV rows, no validation loader or seed loop; CSV has no split/seed columns |
| 3D inference sweep | 0–12° | `predict.py` main uses `range(0, 10)`, or 0–9° |

The checkpoint tensor dimensions were inspected from ZIP/pickle **opcodes**, without unpickling or executing the model. A flattened convolution output of 48,000 plus 16 angle features is consistent with a 200 × 60 input after two poolings, but tensor shape alone does not establish the historical preprocessing. The prediction classes also reverse the declared height/width relative to PIL's 100 × 30 resize. These differences mean the current inference scripts are not verified loaders for the included weights.

The 3D model is **PointNet-style**: it uses shared point features and symmetric pooling, but does not implement the full original PointNet architecture. The 2D and 3D workflows are separate models; no automated two-stage coupling or optimization loop is implemented.

## Simulation and data provenance

- **2D operating conditions:** manuscript p. 9 specifies `Re = 1e6`, Mach 0.262, and 2,000 maximum XFOIL iterations. Both committed generation scripts specify `Re = 6e6`, Mach 0.265, and `ITER 5000`. The common settings include Ncrit 9, VACC 0.00001, and a 0–12° sweep in 0.25° steps. The saved label table contains no solver-setting columns or polar headers, so its generating configuration cannot be resolved from the table alone.
- **Raw profile count:** the manuscript lists 1,647 starting airfoils; the repository contains 1,648 `.dat` files. The clean table's 1,579 profiles and 70,339 rows do match the manuscript. The extra coordinate file's origin is not established by this audit.
- **3D setup:** the manuscript reports `Re = 6e6`, Mach 0.262, `U = 100 m/s`, viscosity `1.8e-5 Pa·s`, incompressible k-omega SST, residual target below `1e-5`, mesh checks, and nominal 6 m span/~2 m height. Solver case files, density/reference-area settings, convergence histories, and mesh-refinement outputs are absent. These settings and the claimed convergence checks therefore remain manuscript reports.
- **Drag scaling:** at 12°, Figure 5 shows roughly `CD = 0.16–0.19`, while the CSV stores SECH 3.51, Sinc 4.37, and Gaussian 4.12. A normalization or reference-area difference might explain the scales, but no conversion or CFD reference-area record is provided. Do not silently rescale the CSV or infer a validated lift-to-drag ratio from those mixed sources. The CSV's lift values are also large; their physical interpretation depends on recovering the solver's coefficient normalization.
- **Geometry normalization:** manuscript passages mention matched volume/frontal area as well as common overall dimensions. The example Blender script does not implement a complete five-family matched-volume/matched-area procedure. Network max-norm scaling is an input preprocessing step, not evidence that the physical CFD bodies had equal volume or reference area.
- **Figure provenance:** original plots remain unchanged, including annotations. Figure 6's green calibrated curve differs from the text's orange description. Its error bars are attributed to eight training runs by the caption, but the available trainer does not implement that protocol.
- **Repository provenance:** the manuscript Appendix links `W0R1D-D0M1NAT10N/Machine_Learning`. This review uses the user-selected `airfoil_cnn` repository, not that other repository or its possible artifacts.

## Scope of verification

The documentation update ran the standard-library [CSV audit](../tools/audit_saved_results.py), checked prediction-label correspondence, inspected source and checkpoint metadata, checked documentation links, and visually reviewed extracted figures and the new diagram. Original-file identity was checked against the base commit, allowing only exact relocations of six learning scripts.

No training, model inference, XFOIL execution, CFD rerun, wind-tunnel experiment, eight-seed study, or timing benchmark was performed. Manuscript statements about being orders of magnitude faster are not supported by a committed timing study. High `R²` against solver labels does not establish experimental accuracy, out-of-family generalization, reliability near stall, or robustness across Reynolds/Mach conditions.
