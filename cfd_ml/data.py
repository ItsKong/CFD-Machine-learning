from pathlib import Path

import numpy as np
import pandas as pd

from cfd_ml.paths import AOA_MACH_RAE2822_CASE_DIR, RAE2822_CASE_DIR


def _prefixed_float(name: str, prefix: str) -> float:
    if not name.startswith(prefix):
        raise ValueError(f"Expected {name!r} to start with {prefix!r}")
    return float(name[len(prefix):])


def _add_aoa_features(frame: pd.DataFrame, aoa: float) -> pd.DataFrame:
    frame["AoA"] = aoa
    frame["sin_AoA"] = np.sin(np.deg2rad(frame["AoA"]))
    frame["cos_AoA"] = np.cos(np.deg2rad(frame["AoA"]))
    return frame


def load_rae2822_surface_data(data_dir: Path = RAE2822_CASE_DIR) -> pd.DataFrame:
    case_paths = [path for path in data_dir.iterdir() if path.is_dir()]
    if not case_paths:
        raise FileNotFoundError(f"No case folders found under {data_dir}")

    if all(path.name.startswith("mach_") for path in case_paths):
        return load_rae2822_aoa_mach_surface_data(data_dir)

    frames = []
    case_paths = sorted(
        case_paths,
        key=lambda path: float(path.name),
    )

    for case_path in case_paths:
        surface_path = case_path / "surface_flow.csv"
        if not surface_path.exists():
            continue

        frame = pd.read_csv(surface_path)
        frames.append(_add_aoa_features(frame, float(case_path.name)))

    if not frames:
        raise FileNotFoundError(f"No surface_flow.csv files found under {data_dir}")

    return pd.concat(frames, ignore_index=True)


def load_rae2822_aoa_mach_surface_data(data_dir: Path = AOA_MACH_RAE2822_CASE_DIR) -> pd.DataFrame:
    frames = []
    mach_paths = sorted(
        (path for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("mach_")),
        key=lambda path: _prefixed_float(path.name, "mach_"),
    )

    for mach_path in mach_paths:
        freestream_mach = _prefixed_float(mach_path.name, "mach_")
        aoa_paths = sorted(
            (path for path in mach_path.iterdir() if path.is_dir() and path.name.startswith("aoa_")),
            key=lambda path: _prefixed_float(path.name, "aoa_"),
        )

        for aoa_path in aoa_paths:
            surface_path = aoa_path / "surface_flow.csv"
            if not surface_path.exists():
                continue

            frame = pd.read_csv(surface_path)
            frame["Freestream_Mach"] = freestream_mach
            frames.append(_add_aoa_features(frame, _prefixed_float(aoa_path.name, "aoa_")))

    if not frames:
        raise FileNotFoundError(f"No nested surface_flow.csv files found under {data_dir}")

    return pd.concat(frames, ignore_index=True)
