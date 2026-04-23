from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class SearchSpec:
    estimator: object
    param_distributions: dict[str, list[Any]]
    n_iter: int = 20


def build_cp_tuning_specs(random_state: int = 42) -> dict[str, SearchSpec]:
    return {
        "Polynomial ridge tuned": SearchSpec(
            estimator=make_pipeline(
                PolynomialFeatures(include_bias=False),
                StandardScaler(),
                Ridge(),
            ),
            param_distributions={
                "polynomialfeatures__degree": [1, 2, 3],
                "ridge__alpha": [
                    0.001,
                    0.003,
                    0.01,
                    0.03,
                    0.1,
                    0.3,
                    1.0,
                    3.0,
                    10.0,
                    30.0,
                    100.0,
                ],
            },
            n_iter=18,
        ),
        "Random forest tuned": SearchSpec(
            estimator=RandomForestRegressor(random_state=random_state, n_jobs=-1),
            param_distributions={
                "n_estimators": [120, 200, 300, 500],
                "max_depth": [None, 8, 12, 20],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 5, 10],
                "max_features": [0.6, 0.8, 1.0, "sqrt"],
            },
            n_iter=24,
        ),
        "Gradient boosting tuned": SearchSpec(
            estimator=GradientBoostingRegressor(random_state=random_state),
            param_distributions={
                "n_estimators": [100, 160, 220, 320],
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "max_depth": [2, 3, 4],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.75, 0.9, 1.0],
            },
            n_iter=24,
        ),
    }


def tune_randomized_search_specs(
    specs: dict[str, SearchSpec],
    X,
    y,
    *,
    cv,
    groups=None,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
    search_n_jobs: int = 1,
    verbose: int = 1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    best_models = {}

    cv_splits = _count_cv_splits(cv, X, y, groups)

    for model_name, spec in specs.items():
        if verbose:
            print(f"Tuning {model_name}...")

        search = RandomizedSearchCV(
            estimator=spec.estimator,
            param_distributions=spec.param_distributions,
            n_iter=spec.n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=search_n_jobs,
            refit=True,
            return_train_score=True,
        )
        fit_kwargs = {"groups": groups} if groups is not None else {}
        search.fit(X, y, **fit_kwargs)

        best_models[model_name] = search.best_estimator_
        rows.append(
            {
                "model": model_name,
                "best_score": search.best_score_,
                "best_rmse": _score_to_rmse(search.best_score_, scoring),
                "best_params": search.best_params_,
                "n_iter": spec.n_iter,
                "cv_splits": cv_splits,
                "scoring": scoring,
            }
        )

    return pd.DataFrame(rows).sort_values("best_rmse"), best_models


def encode_params_for_csv(
    frame: pd.DataFrame,
    column: str = "best_params",
) -> pd.DataFrame:
    csv_frame = frame.copy()
    csv_frame[column] = csv_frame[column].apply(params_to_json)
    return csv_frame


def params_to_json(params: dict[str, Any]) -> str:
    return json.dumps(_json_safe(params), sort_keys=True)


def _score_to_rmse(score: float, scoring: str) -> float:
    if scoring == "neg_root_mean_squared_error":
        return -score
    return np.nan


def _count_cv_splits(cv, X, y, groups) -> int | None:
    if isinstance(cv, int):
        return cv
    if not hasattr(cv, "get_n_splits"):
        return None
    try:
        return cv.get_n_splits(X, y, groups)
    except TypeError:
        return cv.get_n_splits(X, y)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
