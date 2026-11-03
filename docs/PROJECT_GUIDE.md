# Project guide

This is a research-code snapshot with multiple experimental versions. The guide identifies the primary workflows and the setup still needed for reproduction. It does not imply that every script is ready to run from a fresh clone.

## Repository layout

```text
airfoil_cnn/
├── README.md                         Portfolio overview and qualified results
├── airfoils/                         1,648 original .dat coordinate files
├── airfoil_data_clean.csv            70,339 XFOIL-labeled rows
├── airfoil_predictions_vs_true.csv   13,907 saved 2D prediction/label pairs
├── airfoil_cnn.pth                   Original 2D state dictionary
├── generate_airfoil_data*.py         XFOIL batch and single-profile runners
├── data_csv_cleanup.py              Duplicate cleanup and sorting
├── airfoilNN*.py                     Original 2D training/prediction variants
├── 3DModel/
│   ├── model.py                     PointNetLiftRegressor
│   ├── flightpod_dataset.py         OBJ loading, normalization, point sampling
│   ├── train.py                     Training and checkpoint export
│   ├── predict.py                   Mesh/AoA prediction, CSV and plotting
│   ├── flight_data.csv              45 rows for three meshes
│   ├── flight_data_transposed.csv   Same data arranged for plotting
│   ├── data/                        Training OBJ/STL meshes and materials
│   ├── models/                      Additional prediction meshes
│   ├── transpose_csv.py             Table reshaping utility
│   ├── smooth_func.py               Radius-function plotting experiment
│   └── test3d.py                    Manual mesh-loading probe
├── blender/model1.py                Example revolved geometry and render
├── experiments/learning/            Original non-aerodynamic learning exercises
├── tools/audit_saved_results.py      Standard-library CSV evidence audit
└── docs/
    ├── RESULTS.md                   Metrics, discrepancies, evidence boundaries
    ├── assets/                      Manuscript figures and architecture diagram
    └── research/                    Manuscript source notes (full PDF not included)
```

Six independent learning scripts were relocated into `experiments/learning/` with their contents unchanged. Original aerodynamic paths, data, meshes, weights, and `readme.txt` files remain in place. There is no common command-line interface or packaged library.

## Primary workflow: 2D

1. **Coordinates and solver labels:** `generate_airfoil_data.py` calls an external `xfoil` process, sweeps 0–12 degrees in 0.25-degree steps, records solver logs/status, and writes `airfoil_data.csv`. It uses a four-worker pool. `generate_airfoil_data_onefoil.py` runs one profile with two positional arguments: a `.dat` path and an airfoil name; it expects `Xfoil/bin/xfoil`. Both contain local/external executable assumptions.
2. **Cleanup:** `data_csv_cleanup.py` removes exact duplicate `(file_path, aoa, cl)` rows and sorts the result. Training scripts additionally deduplicate by `(image_path, aoa)`.
3. **Images:** the CNN requires PNG renders corresponding to the `.dat` filenames. These image files and a complete batch rasterization workflow are not committed; coordinates alone are not the direct CNN input.
4. **Training:** `airfoilNN_nvidia.py` reads the labels and matching images, scales AoA with `StandardScaler`, and uses the first split of `GroupKFold(n_splits=5)` grouped by filename. It trains a two-branch CNN, saves split lists, and writes the checkpoint and scaler. It does not run all five folds.
5. **Prediction:** `airfoilNN_predict.py` loads a validation-name list, a scaler or fallback fit, and a checkpoint. It exports `airfoil_predictions_vs_true.csv` and a scatter plot. Its model definition does not match the saved checkpoint or GPU trainer, so it requires reconciliation before use. `airfoilNN_pred.py` is another prediction/scatter variant with a fallback-scaler ordering issue (`np` is used before import when the scaler is absent).

### Training variants

| Script | Image handling | Head hidden widths | Training choice |
| --- | --- | --- | --- |
| `airfoilNN_nvidia.py` | Resize to 100 × 30 pixels (width × height) | 1024 → 256 | 1,000-epoch loop; Adam 0.0015, StepLR; physics penalty disabled |
| `airfoilNN.py` | Resize to 256 × 256 | 64 → 32 | CPU, 100 epochs, active thin-airfoil-inspired penalty |
| `airfoilNN8020.py` | Resize to 100 × 30 | 64 → 32 | First grouped split despite the filename; 100 epochs, physics penalty |
| `airfoilNN_linux.py` | Keeps source image size | 64 → 32 | Contains Windows absolute paths despite its name; 100 epochs, physics penalty |
| `airfoilNN_predict.py`, `airfoilNN_pred.py` | Resize to 100 × 30 | 64 → 32 | Inference definitions use height 100/width 30, reversed relative to PIL's resized array |

All use one lift output. A physics-inspired penalty in some older scripts is an experimental training choice, not a guarantee that the network obeys fluid mechanics. The committed checkpoint has head tensors consistent with **512 → 128 → 1**, and therefore does not match any of these checked-in 2D definitions exactly. See [architecture evidence](RESULTS.md#architecture-and-training-differences).

## Primary workflow: 3D

1. **Geometry:** `blender/model1.py` creates an elliptical radius profile, revolves it about the Z axis, adds material/lighting/camera, and renders it. It is an example generator, not the full five-family generation procedure described in the manuscript. Its original note specifies Blender 3.6.23. Running it deletes objects in the current Blender scene; use a disposable scene.
2. **Mesh formats:** PyTorch3D reads OBJ files in this workflow. The original notes suggest `assimp export input.stl output.obj` for conversion. Both formats are retained where available.
3. **Reference data:** `flight_data.csv` uses columns `data,aoa,cl,cd,cm`. `ufov4.obj` and `sinc.obj` each have 16 rows at 0–15 degrees; `sech.obj` has 13 at 0–12 degrees. These CSVs do not store train/validation assignments or run seeds.
4. **Dataset:** `FlightPodMeshDataset` scales mesh vertices by `1 / max(norm(vertex))`, then samples 1,024 points. Each item returns points, the CSV angle, and lift. AoA is passed directly, without standardization. Surface sampling is stochastic.
5. **Model:** `PointNetLiftRegressor` uses a shared `3 → 512 → 512` MLP with ReLU, max pooling over the points, concatenation of one AoA value, then `513 → 128 → 1` regression. There is no dropout or PointNet transform network in this implementation.
6. **Training:** `train.py` trains on all CSV rows for 100 epochs with batches of four, Adam at 0.001, and MSE loss. It selects a best checkpoint by training loss, saves every 50 epochs, and exports a final state dictionary. There is no validation loader or multi-seed training loop.
7. **Inference:** `predict.py` loads `saved_models/model_weights.pth`, samples and normalizes a mesh, and predicts lift. Its main function scans `models/` and defaults to **0–9 degrees** (`range(0, 10)`). The x-axis limit of 15 degrees does not extend that sweep. It can write `lift_predictions.csv` and plots. No affine calibration is applied.

`3DModel/test3d.py` is an interactive mesh-loading check, not an automated test suite. `smooth_func.py` is a radius-plot experiment; its exponent/support differ from the manuscript's Gaussian-like equation. `transpose_csv.py` reshapes the CFD table without changing its meaning.

## Running the original workflows

### Inspect the saved data now

From the repository root, with Python 3.9 or later:

```sh
python tools/audit_saved_results.py
```

This reads committed CSVs and prints counts, raw 2D metrics, an explicitly **in-sample** affine-fit diagnostic, and artifact availability. It uses only Python's standard library and does not load weights, train, run inference, or invoke a solver. It works from other working directories as well.

### Dependencies used by the original code

| Workflow | Imports / external tools |
| --- | --- |
| 2D training and prediction | PyTorch, NumPy, pandas, Pillow, scikit-learn, Matplotlib; GPU variant also imports tqdm and joblib |
| XFOIL data generation | XFOIL executable; batch runner imports pandas and tqdm |
| 3D training and prediction | PyTorch, PyTorch3D, pandas; predictor imports NumPy and optionally Matplotlib |
| Geometry generation | Blender's bundled Python (`bpy`, `bmesh`); optionally Assimp for STL conversion |
| Learning exercises only | Some also import torchvision and download MNIST |

No tested dependency lockfile is available. This update adds no dependency or installation requirement to the existing workflows. PyTorch3D and CUDA compatibility must be matched to the machine used for reproduction.

### Before attempting 2D execution

- Supply the missing PNG image collection, `train_image_paths.csv`, `val_image_paths.csv`, and `aoa_scaler.joblib` or the original preprocessing record. The split lists and scaler can be generated by the GPU script, but their historical versions are absent.
- Update the absolute data/image directories in the selected script. The GPU variant uses local Windows directories or an `images` directory on other systems.
- Reconcile image dimensions, model widths, and checkpoint tensor shapes. Do not assume the included `airfoil_cnn.pth` loads into the current prediction class.
- The scripts run data loading/training at module scope. The GPU trainer also uses 32 DataLoader workers without a Windows main guard, requiring attention on Windows. It materializes an image for each sample, which can consume significant memory.
- Training overwrites `airfoil_cnn.pth` each epoch; prediction overwrites the saved prediction CSV. Use a separate working copy to preserve the historical evidence.

### Before attempting 3D execution

Use a compatible PyTorch/PyTorch3D environment and run from `3DModel/`, since the original scripts resolve paths there:

```sh
cd 3DModel
python train.py
python predict.py
```

These are **original workflow commands, not commands validated by this documentation update**. Training must first produce `saved_models/model_weights.pth`, or that file must be recovered from an earlier run; no 3D checkpoint is committed. The checked-in trainer supplies no held-out evaluation, so its training loss is not a test score. The comparison plotting code also uses a combined format string that fails for some long color names (for example `orangeo-`), which can interrupt export when processing enough meshes.

For table reshaping only, run from `3DModel/` and write to a new output filename:

```sh
python transpose_csv.py flight_data.csv flight_data_reformatted.csv
```

## Reproduction priorities

Recovering the original run artifacts would be more useful than restructuring the model code: matching preprocessing/scaler and weights, persisted train/calibration/test assignments, all five 3D reference tables, eight-seed prediction outputs, and CFD setup/reference-area records. Independent evaluation and a measured timing comparison should follow. These are follow-up tasks, not completed features.
