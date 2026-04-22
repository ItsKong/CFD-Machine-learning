from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def summarize_regression_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("model")
        .agg(
            MAE_mean=("MAE", "mean"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            RMSE_std=("RMSE", "std"),
        )
        .sort_values("RMSE_mean")
    )
