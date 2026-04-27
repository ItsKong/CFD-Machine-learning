from __future__ import annotations

from pathlib import Path

import pandas as pd

from cfd_ml.data import load_rae2822_surface_data


CP_FEATURES = ["x", "y", "sin_AoA", "cos_AoA"]
CP_AOA_MACH_FEATURES = [*CP_FEATURES, "Freestream_Mach"]
CP_TARGET = "Pressure_Coefficient"
CP_GROUP = "AoA"
CP_AOA_MACH_GROUP_COLUMNS = ["Freestream_Mach", "AoA"]
CP_AOA_MACH_GROUP = "case_id"


def build_case_id(data: pd.DataFrame, group_columns: list[str] | tuple[str, ...]) -> pd.Series:
    return data[group_columns].astype(str).agg("__".join, axis=1)


def load_cp_modeling_data(
    data_dir: Path | None = None,
    feature_columns: list[str] | tuple[str, ...] | None = None,
    group_column: str = CP_GROUP,
) -> pd.DataFrame:
    raw_df = load_rae2822_surface_data(data_dir) if data_dir is not None else load_rae2822_surface_data()
    feature_columns = list(feature_columns or CP_FEATURES)
    columns = [group_column, *feature_columns, CP_TARGET]
    return raw_df[columns].dropna().reset_index(drop=True)


def load_cp_aoa_mach_modeling_data(data_dir: Path | None = None) -> pd.DataFrame:
    raw_df = load_rae2822_surface_data(data_dir) if data_dir is not None else load_rae2822_surface_data()
    working_df = raw_df.copy()
    working_df[CP_AOA_MACH_GROUP] = build_case_id(working_df, CP_AOA_MACH_GROUP_COLUMNS)
    columns = [*CP_AOA_MACH_GROUP_COLUMNS, CP_AOA_MACH_GROUP, *CP_FEATURES, "Freestream_Mach", CP_TARGET]
    return working_df[columns].dropna().reset_index(drop=True)


def split_cp_features(
    data: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...] | None = None,
    group_column: str = CP_GROUP,
):
    feature_columns = list(feature_columns or CP_FEATURES)
    X = data[feature_columns]
    y = data[CP_TARGET]
    groups = data[group_column]
    return X, y, groups
