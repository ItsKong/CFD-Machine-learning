from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from cfd_ml.data import load_tradeoff_data


TRADEOFF_FEATURES = ["aoa", "maxcamber", "maxcamberposition", "thickness"]
TRADEOFF_TARGETS = ["cl", "cd", "cl_cd"]
TARGET_LABELS = {
    "cl": "CL",
    "cd": "CD",
    "cl_cd": "CL/CD",
}


def load_tradeoff_modeling_data() -> pd.DataFrame:
    raw_df = load_tradeoff_data().copy()
    modeling_df = raw_df[["id", "design", *TRADEOFF_FEATURES, "cl", "cd"]].copy()

    numeric_columns = [*TRADEOFF_FEATURES, "cl", "cd"]
    modeling_df[numeric_columns] = modeling_df[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    modeling_df = modeling_df.dropna(subset=numeric_columns).reset_index(drop=True)
    modeling_df["cl_cd"] = modeling_df["cl"] / modeling_df["cd"]

    return modeling_df


def split_tradeoff_features(data: pd.DataFrame):
    X = data[TRADEOFF_FEATURES]
    y = data[TRADEOFF_TARGETS]
    return X, y


def build_tradeoff_models() -> dict[str, object]:
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial ridge d2": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=220,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
        ),
    }


def evaluate_tradeoff_models(
    models: Mapping[str, object],
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    fold_predictions = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        for model_name, model in models.items():
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            y_pred = pd.DataFrame(
                fold_model.predict(X_test),
                columns=y.columns,
                index=y_test.index,
            )

            for target in y.columns:
                fold_scores.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "target": target,
                        "target_label": TARGET_LABELS.get(target, target),
                        "MAE": mean_absolute_error(y_test[target], y_pred[target]),
                        "RMSE": np.sqrt(
                            mean_squared_error(y_test[target], y_pred[target])
                        ),
                        "R2": r2_score(y_test[target], y_pred[target]),
                    }
                )

            prediction_frame = X_test.copy()
            prediction_frame["fold"] = fold
            prediction_frame["model"] = model_name
            prediction_frame["row_index"] = y_test.index
            for target in y.columns:
                prediction_frame[f"{target}_true"] = y_test[target].to_numpy()
                prediction_frame[f"{target}_pred"] = y_pred[target].to_numpy()
            fold_predictions.append(prediction_frame)

    return (
        pd.DataFrame(fold_scores),
        pd.concat(fold_predictions, ignore_index=True),
    )


def summarize_tradeoff_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["model", "target", "target_label"])
        .agg(
            MAE_mean=("MAE", "mean"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            RMSE_std=("RMSE", "std"),
        )
        .reset_index()
        .sort_values(["target", "RMSE_mean"])
    )


def rank_tradeoff_models(
    summary: pd.DataFrame,
    ranking_target: str = "cl_cd",
) -> pd.DataFrame:
    return (
        summary[summary["target"] == ranking_target]
        .sort_values("RMSE_mean")
        .reset_index(drop=True)
    )


def fit_tradeoff_model(model: object, X: pd.DataFrame, y: pd.DataFrame):
    fitted_model = clone(model)
    fitted_model.fit(X, y)
    return fitted_model


def predict_tradeoff_response(
    model: object,
    designs: pd.DataFrame | Mapping[str, float] | Sequence[Mapping[str, float]],
) -> pd.DataFrame:
    if isinstance(designs, pd.DataFrame):
        design_frame = designs.copy()
    elif isinstance(designs, Mapping):
        design_frame = pd.DataFrame([designs])
    else:
        design_frame = pd.DataFrame(list(designs))

    missing_features = [
        feature for feature in TRADEOFF_FEATURES if feature not in design_frame.columns
    ]
    if missing_features:
        raise ValueError(f"Missing trade-off features: {missing_features}")

    X_design = design_frame[TRADEOFF_FEATURES]
    predictions = pd.DataFrame(
        model.predict(X_design),
        columns=[f"{target}_pred" for target in TRADEOFF_TARGETS],
        index=design_frame.index,
    )
    return pd.concat(
        [design_frame.reset_index(drop=True), predictions.reset_index(drop=True)],
        axis=1,
    )
