from pathlib import Path

import numpy as np
import pandas as pd

from cfd_ml.paths import RAE2822_CASE_DIR


def load_rae2822_surface_data(data_dir: Path = RAE2822_CASE_DIR) -> pd.DataFrame:
    frames = []
    case_paths = sorted(
        (path for path in data_dir.iterdir() if path.is_dir()),
        key=lambda path: float(path.name),
    )

    for case_path in case_paths:
        surface_path = case_path / "surface_flow.csv"
        if not surface_path.exists():
            continue

        frame = pd.read_csv(surface_path)
        frame["AoA"] = float(case_path.name)
        frame["sin_AoA"] = np.sin(np.deg2rad(frame["AoA"]))
        frame["cos_AoA"] = np.cos(np.deg2rad(frame["AoA"]))
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No surface_flow.csv files found under {data_dir}")

    return pd.concat(frames, ignore_index=True)
