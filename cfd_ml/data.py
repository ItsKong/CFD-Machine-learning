from pathlib import Path

import numpy as np
import pandas as pd

from cfd_ml.paths import RAE2822_CASE_DIR, TRADEOFF_DATA_PATH


def _parse_case_name(name: str) -> tuple[float, float]:
    """Parse 'aoa_{aoa}_mach_{mach}' folder names, e.g. 'aoa_0.00_mach_0.68'."""
    parts = name.split("_")
    # Expected tokens: ['aoa', '<aoa>', 'mach', '<mach>']
    if len(parts) != 4 or parts[0] != "aoa" or parts[2] != "mach":
        raise ValueError(
            f"Unexpected case folder name: '{name}'. "
            "Expected format: 'aoa_<float>_mach_<float>'."
        )
    return float(parts[1]), float(parts[3])


def load_rae2822_surface_data(data_dir: Path = RAE2822_CASE_DIR) -> pd.DataFrame:
    frames = []
    case_paths = sorted(
        (path for path in data_dir.iterdir() if path.is_dir()),
        key=lambda path: _parse_case_name(path.name),  # sorts by (aoa, mach)
    )

    for case_path in case_paths:
        surface_path = case_path / "surface_flow.csv"
        if not surface_path.exists():
            continue

        aoa, mach = _parse_case_name(case_path.name)

        frame = pd.read_csv(surface_path)
        frame["AoA"] = aoa
        frame["Mach"] = mach
        frame["sin_AoA"] = np.sin(np.deg2rad(aoa))
        frame["cos_AoA"] = np.cos(np.deg2rad(aoa))
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No surface_flow.csv files found under {data_dir}")

    return pd.concat(frames, ignore_index=True)


def load_tradeoff_data(path: Path = TRADEOFF_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")