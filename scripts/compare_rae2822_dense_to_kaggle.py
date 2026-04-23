import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DIR = PROJECT_ROOT / "data" / "rae2822" / "angvar_sa"
DENSE_DIR = PROJECT_ROOT / "data" / "rae2822_dense_su2" / "angvar_sa"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
GAMMA = 1.4

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare generated dense SU2 cases against overlapping Kaggle RAE2822 cases."
    )
    parser.add_argument("--original-dir", type=Path, default=ORIGINAL_DIR)
    parser.add_argument("--dense-dir", type=Path, default=DENSE_DIR)
    parser.add_argument(
        "--summary",
        type=Path,
        default=METRICS_DIR / "rae2822_dense_su2_vs_kaggle_summary.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=FIGURES_DIR / "rae2822_dense_su2_vs_kaggle_cp.png",
    )
    return parser.parse_args()


def fit_cp_normalization(original_dir: Path) -> tuple[float, float]:
    pieces = []
    for surface_csv in original_dir.glob("*/surface_flow.csv"):
        frame = pd.read_csv(surface_csv, skipinitialspace=True)
        required = {"Pressure", "Pressure_Coefficient"}
        if required.issubset(frame.columns):
            pieces.append(frame[["Pressure", "Pressure_Coefficient"]])

    if not pieces:
        raise FileNotFoundError("No original cases with Pressure and Pressure_Coefficient found")

    data = pd.concat(pieces, ignore_index=True)
    cp = data["Pressure_Coefficient"].to_numpy(dtype=float)
    pressure = data["Pressure"].to_numpy(dtype=float)
    design = np.column_stack([cp, np.ones_like(cp)])
    q_inf, p_inf = np.linalg.lstsq(design, pressure, rcond=None)[0]
    return float(q_inf), float(p_inf)


def pressure_from_conservative(frame: pd.DataFrame) -> np.ndarray:
    rho = frame["Density"].to_numpy(dtype=float)
    mom_x = frame["Momentum_x"].to_numpy(dtype=float)
    mom_y = frame["Momentum_y"].to_numpy(dtype=float)
    energy = frame["Energy"].to_numpy(dtype=float)
    kinetic = 0.5 * (mom_x * mom_x + mom_y * mom_y) / rho
    return (GAMMA - 1.0) * (energy - kinetic)


def generated_cp(frame: pd.DataFrame, q_inf: float, p_inf: float) -> tuple[np.ndarray, str]:
    if "Pressure_Coefficient" in frame.columns:
        return frame["Pressure_Coefficient"].to_numpy(dtype=float), "native"

    pressure = pressure_from_conservative(frame)
    return (pressure - p_inf) / q_inf, "derived_from_conservative"


def compare_case(aoa: float, original_csv: Path, dense_csv: Path, q_inf: float, p_inf: float) -> dict[str, object]:
    original = pd.read_csv(original_csv, skipinitialspace=True)
    dense = pd.read_csv(dense_csv, skipinitialspace=True)

    row = {
        "AoA": aoa,
        "original_csv": str(original_csv),
        "dense_csv": str(dense_csv),
        "row_count_original": len(original),
        "row_count_dense": len(dense),
        "pointid_match": False,
        "xy_match": False,
        "cp_source": None,
        "cp_mae": None,
        "cp_rmse": None,
        "cp_max_abs": None,
        "cp_r2": None,
    }

    if len(original) != len(dense):
        return row

    row["pointid_match"] = original["PointID"].tolist() == dense["PointID"].tolist()
    row["xy_match"] = np.allclose(
        original[["x", "y"]].to_numpy(dtype=float),
        dense[["x", "y"]].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    if not row["pointid_match"] or not row["xy_match"]:
        return row

    original_cp = original["Pressure_Coefficient"].to_numpy(dtype=float)
    dense_cp, cp_source = generated_cp(dense, q_inf, p_inf)
    diff = dense_cp - original_cp
    denominator = float(np.sum((original_cp - original_cp.mean()) ** 2))

    row["cp_source"] = cp_source
    row["cp_mae"] = float(np.mean(np.abs(diff)))
    row["cp_rmse"] = float(np.sqrt(np.mean(diff * diff)))
    row["cp_max_abs"] = float(np.max(np.abs(diff)))
    row["cp_r2"] = float(1.0 - np.sum(diff * diff) / denominator) if denominator else None
    return row


def overlapping_aoa_cases(original_dir: Path, dense_dir: Path) -> list[tuple[float, Path, Path]]:
    cases = []
    for original_csv in sorted(original_dir.glob("*/surface_flow.csv"), key=lambda path: float(path.parent.name)):
        aoa = float(original_csv.parent.name)
        dense_csv = dense_dir / f"{aoa:g}" / "surface_flow.csv"
        if dense_csv.exists():
            cases.append((aoa, original_csv, dense_csv))
    return cases


def plot_comparison(rows: list[dict[str, object]], q_inf: float, p_inf: float, figure_path: Path) -> None:
    import matplotlib.pyplot as plt

    valid_rows = [
        row for row in rows
        if row["pointid_match"] and row["xy_match"] and row["cp_rmse"] is not None
    ]
    if not valid_rows:
        return

    fig, axes = plt.subplots(
        len(valid_rows),
        2,
        figsize=(13, max(3.5, 2.8 * len(valid_rows))),
        constrained_layout=True,
        squeeze=False,
    )

    for idx, row in enumerate(valid_rows):
        aoa = float(row["AoA"])
        original = pd.read_csv(Path(row["original_csv"]), skipinitialspace=True)
        dense = pd.read_csv(Path(row["dense_csv"]), skipinitialspace=True)
        original_cp = original["Pressure_Coefficient"].to_numpy(dtype=float)
        dense_cp, cp_source = generated_cp(dense, q_inf, p_inf)
        diff = dense_cp - original_cp
        order = np.arange(len(original))

        axes[idx, 0].plot(order, original_cp, label="Kaggle", linewidth=1.4)
        axes[idx, 0].plot(order, dense_cp, label=f"Dense SU2 ({cp_source})", linewidth=1.2)
        axes[idx, 0].invert_yaxis()
        axes[idx, 0].set_title(f"AoA {aoa:g}: Cp")
        axes[idx, 0].set_xlabel("surface point index")
        axes[idx, 0].set_ylabel("Cp")
        axes[idx, 0].grid(True, alpha=0.25)
        axes[idx, 0].legend()

        axes[idx, 1].plot(original["x"], diff, linewidth=1.2)
        axes[idx, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[idx, 1].set_title(f"AoA {aoa:g}: Dense SU2 minus Kaggle")
        axes[idx, 1].set_xlabel("x")
        axes[idx, 1].set_ylabel("delta Cp")
        axes[idx, 1].grid(True, alpha=0.25)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cases = overlapping_aoa_cases(args.original_dir, args.dense_dir)
    if not cases:
        print("No overlapping AoA cases found. Generate integer AoA cases first.")
        return 1

    q_inf, p_inf = fit_cp_normalization(args.original_dir)
    rows = [
        compare_case(aoa, original_csv, dense_csv, q_inf, p_inf)
        for aoa, original_csv, dense_csv in cases
    ]

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.summary, index=False)
    plot_comparison(rows, q_inf, p_inf, args.figure)

    print(pd.DataFrame(rows)[["AoA", "cp_source", "cp_rmse", "cp_r2"]].to_string(index=False))
    print()
    print(f"Summary: {args.summary}")
    print(f"Figure: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
