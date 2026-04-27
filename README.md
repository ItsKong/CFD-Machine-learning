# Computational Fluid Dynamic Machine Learning

This project builds machine-learning surrogate models for RAE2822 airfoil CFD
results. The notebook flow now stays focused on one story:

```text
Kaggle baseline -> dense SU2 dataset validation -> neural-network comparison
```

## Current Scope

The current Cp modeling task is:

```text
fixed RAE2822 geometry + AoA -> predicted Cp distribution
```

There is also an AoA x Mach SU2 extension dataset in the repository. The next
natural extension is:

```text
fixed RAE2822 geometry + AoA + Freestream Mach -> predicted Cp distribution
```

The repository is intentionally focused on reproducible notebook experiments,
metrics, and report outputs. Optional SU2 dataset generation is kept separate
from the Kaggle baseline so the two data sources can be compared cleanly.

## Project Structure

```text
cfd_ml/
  paths.py          shared project paths
  data.py           dataset loading helpers
  cp.py             Cp surrogate model feature helpers
  evaluation.py     shared scoring helpers
  tuning.py         hyperparameter search helpers
notebooks/
  01_cp_surrogate_model.ipynb
  02_su2_vs_kaggle_comparison.ipynb
  03_neural_network_comparison.ipynb
scripts/
  compare_rae2822_dense_to_kaggle.py
  generate_dense_rae2822_su2_dataset.py
  inspect_datasets.py
models/
  README.md         note about tracked artifacts
reports/
  figures/          generated plots for reports
  metrics/          model evaluation tables
```

## Dataset

The training data comes from Kaggle:

- [Postprocessing CFD Data](https://www.kaggle.com/code/shtrausslearning/postprocessing-cfd-data/notebook)
- [External CFD Aerodynamics Dataset](https://www.kaggle.com/datasets/shtrausslearning/external-cfd-aero/data)

Place the downloaded dataset in this structure:

```text
data/
  rae2822/
    angvar_sa/
      0/surface_flow.csv
      1/surface_flow.csv
      ...
      20/surface_flow.csv
```

The `data/` directory is intentionally ignored by git.

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Quick Checks

Inspect the Kaggle baseline and dense SU2 extension datasets:

```bash
python3 scripts/inspect_datasets.py
```

## Optional Dense SU2 Dataset

The original Kaggle Cp dataset has 21 independent AoA cases. To study whether
denser AoA coverage improves generalization, generate a separate SU2 dataset
instead of mixing new labels directly into the Kaggle dataset.

Start with a fast pilot case:

```bash
python3 scripts/generate_dense_rae2822_su2_dataset.py \
  --aoa-start 10 --aoa-stop 10 --aoa-step 1 \
  --max-iter 50 \
  --output-dir /tmp/rae2822_dense_pilot/angvar_sa \
  --allow-overwrite
```

Then run overlapping integer AoA cases and compare them against Kaggle:

```bash
python3 scripts/generate_dense_rae2822_su2_dataset.py \
  --aoa-start 0 --aoa-stop 20 --aoa-step 5

python3 scripts/compare_rae2822_dense_to_kaggle.py
```

If the overlap comparison is acceptable, run the denser half-degree dataset:

```bash
python3 scripts/generate_dense_rae2822_su2_dataset.py --aoa-step 0.5
```

The generator copies the Kaggle RAE2822 config and mesh, changes `AOA`, and
uses `v84_compatible` mode by default because SU2 v8.4 rejects the original
`JST + MUSCL_FLOW=YES` combination. It also rewrites SU2 v8.4 surface output
into a Kaggle-like `surface_flow.csv` with `Pressure_Coefficient` and the same
18 columns as the original dataset. The output is intentionally separate:

```text
data/rae2822_dense_su2/angvar_sa/
```

Use the comparison report to decide whether the generated labels are close
enough to discuss as an extension experiment. The current generated dense set
contains 41 AoA cases from 0 to 20 degrees in 0.5-degree increments. Each case
has 192 surface points and the same 18 columns as the Kaggle `surface_flow.csv`
files. The overlap comparison against Kaggle integer AoA cases gives
`R2 > 0.998` for every overlapping AoA.

## Notebook Flow

Open the final project notebooks in this order:

```bash
jupyter notebook notebooks/01_cp_surrogate_model.ipynb
jupyter notebook notebooks/02_su2_vs_kaggle_comparison.ipynb
jupyter notebook notebooks/03_neural_network_comparison.ipynb
```

- `01` builds the Kaggle baseline Cp surrogate models.
- `02` validates that the dense SU2 dataset is close enough to Kaggle for an
  extension experiment.
- `03` compares the neural network against the stronger classical baseline.

For the AoA x Mach extension, the reusable helpers now live in `cfd_ml/cp.py`:

- `load_cp_aoa_mach_modeling_data()`
- `split_cp_features(..., feature_columns=CP_AOA_MACH_FEATURES, group_column=CP_AOA_MACH_GROUP)`

Use `CP_AOA_MACH_GROUP` so validation holds out entire `(Mach, AoA)` cases
instead of leaking points from the same CFD case across train and test folds.

## Results Snapshot

The Cp surrogate task uses Leave-One-AoA-Out validation so each test fold is an
unseen angle of attack. The current result is:

```text
Random forest tuned RMSE: 0.0621
Neural network RMSE:     0.0976
Winner:                  Random forest tuned
```

This is a useful ML conclusion: for this small tabular CFD dataset, the tuned
tree ensemble generalizes better than the fixed neural-network architecture.

## Report Outputs

The Cp notebooks and SU2 comparison script export report tables under:

```text
reports/metrics/cp_model_comparison_summary.csv
reports/metrics/cp_tuned_model_comparison_summary.csv
reports/metrics/cp_nn_comparison_summary.csv
reports/metrics/cp_nn_vs_random_forest_summary.csv
reports/metrics/rae2822_dense_su2_vs_kaggle_summary.csv
```

Keep new final-project notebooks in `notebooks/` and move reusable code into
`cfd_ml/` so the notebooks stay readable.

## Report Outline

A complete project report can follow this structure:

```text
1. Problem definition
2. Dataset description
3. Feature engineering
4. Validation strategy
5. Baseline model comparison
6. Hyperparameter tuning
7. Neural-network comparison
8. Results and discussion
9. Error analysis
10. Limitations and future work
```

## Next ML Improvements

Keep the next improvements small and incremental:

```text
1. Cp baseline comparison with clear validation reports.
2. Dense SU2 validation and discussion of data-source shift.
3. Cp neural-network comparison against the classical baselines.
4. Error analysis by AoA and representative Cp curves.
```
