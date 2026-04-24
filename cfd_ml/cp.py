from __future__ import annotations

import pandas as pd

from cfd_ml.data import load_rae2822_surface_data


CP_FEATURES = ["x", "y", "sin_AoA", "cos_AoA", "Mach", "AoA"]
CP_TARGET = "Pressure_Coefficient"
CP_GROUP = "CaseID"


def load_cp_modeling_data() -> pd.DataFrame:
    raw_df = load_rae2822_surface_data()
    raw_df['CaseID'] = raw_df['AoA'].astype(str) + "_" + raw_df['Mach'].astype(str)
    columns = [CP_GROUP, *CP_FEATURES, CP_TARGET]
    return raw_df[columns].dropna().reset_index(drop=True)


def split_cp_features(data: pd.DataFrame):
    X = data[CP_FEATURES]
    y = data[CP_TARGET]
    groups = data[CP_GROUP]
    return X, y, groups

