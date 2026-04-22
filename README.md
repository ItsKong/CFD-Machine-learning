# Computational Fluid Dynamic Machine Learning

This project builds a machine-learning surrogate model for RAE2822 airfoil CFD
results. The current neural network predicts surface pressure coefficient
(`Cp`) from airfoil surface coordinates (`x`, `y`) and angle of attack (`AoA`).

## Current Scope

The current Cp demo is:

```text
fixed RAE2822 geometry + AoA -> predicted Cp distribution
```

The project also includes a Part B response model:

```text
AoA, max camber, camber position, thickness -> CL, CD, CL/CD
```

The next ML work can use that response model in a design optimizer.

## Project Structure

```text
cfd_ml/
  paths.py          shared project paths
  cp.py             Cp surrogate model feature helpers
  tradeoff.py       CL/CD response model helpers
notebooks/
  00_exploration/  older experimental notebooks
  01_cp_surrogate_model.ipynb
  02_neural_network_comparison.ipynb
  03_tradeoff_response_model.ipynb
gui/
  cfd_model.py      Cp model loading and prediction
  classes.py        Tk desktop UI fallback
scripts/
  inspect_datasets.py
  smoke_predict.py
  train_tradeoff_response_model.py
models/
  cp/               trained Cp model, scalers, base geometry
  tradeoff/         trained N2412 CL/CD response model
reports/
  figures/          generated plots for reports
  metrics/          model evaluation tables
batch_calculation/
  su2_batch_runner.py
run_web_gui.py      recommended browser demo
run_gui.py          Tk desktop demo
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

The extracted Kaggle archive also contains an N2412 trade-off table used for
the next project phase:

```text
archive/tradestudies/n2412/n2412_optimisation.csv
```

That table supports the Part B response model:

```text
AoA, max camber, camber position, thickness -> CL, CD, CL/CD
```

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run The Browser GUI

This is the recommended way to use the project locally:

```bash
python run_web_gui.py
```

Then open:

```text
http://127.0.0.1:8000
```

Enter an angle of attack and press `Predict` to plot the predicted pressure
coefficient on the upper and lower airfoil surfaces.

## Quick Checks

Inspect the datasets:

```bash
python scripts/inspect_datasets.py
```

Run a prediction smoke test:

```bash
python scripts/smoke_predict.py
```

## Run The Tk GUI

The Tk version is kept as a lightweight desktop fallback:

```bash
python run_gui.py
```

## Train Or Experiment

Open the final-project notebooks after installing the dependencies:

```bash
jupyter notebook notebooks/01_cp_surrogate_model.ipynb
jupyter notebook notebooks/02_neural_network_comparison.ipynb
jupyter notebook notebooks/03_tradeoff_response_model.ipynb
```

The Cp notebooks read from `data/rae2822/angvar_sa/`, train Cp surrogate
models, and can export:

```text
models/cp/cfd_model.keras
models/cp/scaler_X.pkl
models/cp/scaler_y.pkl
models/cp/base_geometry.csv
```

The Part B response model can also be trained from the command line:

```bash
python scripts/train_tradeoff_response_model.py
```

It exports:

```text
models/tradeoff/response_model.joblib
models/tradeoff/response_model_metadata.json
reports/metrics/tradeoff_response_summary.csv
reports/metrics/tradeoff_response_scores_by_fold.csv
reports/metrics/tradeoff_response_predictions.csv
reports/metrics/tradeoff_response_ranking.csv
```

Keep new final-project notebooks in `notebooks/` and move reusable code into
`cfd_ml/` so the notebooks stay readable.

## SU2 Batch Cases

`batch_calculation/su2_batch_runner.py` can generate additional SU2 cases for
new AoA values. Update `SU2_BIN` for your local SU2 installation before running
it.

## Next ML Improvements

Keep the next improvements small and incremental:

```text
1. Cp baseline comparison with clear validation reports.
2. Cp neural-network comparison against the classical baselines.
3. N2412 trade-off response model for CL, CD, and CL/CD.
4. Design optimizer that searches for high CL/CD under user constraints.
```
