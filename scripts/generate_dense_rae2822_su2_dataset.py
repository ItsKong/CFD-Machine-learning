import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CASE_DIR = PROJECT_ROOT / "data" / "rae2822" / "angvar_sa"
DEFAULT_TEMPLATE_CASE = ORIGINAL_CASE_DIR / "10"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "rae2822_aoa_mach_su2" / "extended_sa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an RAE2822 AoA x Mach dataset by copying the Kaggle SU2 "
            "case setup and changing the angle of attack, Mach number, and "
            "only the minimum compatibility edits needed by newer SU2 versions."
        )
    )
    parser.add_argument("--aoa-start", type=float, default=0.0)
    parser.add_argument("--aoa-stop", type=float, default=20.0)
    parser.add_argument("--aoa-step", type=float, default=0.5)
    parser.add_argument("--mach-start", type=float, default=0.68)
    parser.add_argument("--mach-stop", type=float, default=0.78)
    parser.add_argument("--mach-step", type=float, default=0.02)

    parser.add_argument(
        "--template-case",
        type=Path,
        default=DEFAULT_TEMPLATE_CASE,
        help=f"Kaggle case folder used as config/mesh template. Default: {DEFAULT_TEMPLATE_CASE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Generated dataset case root. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--su2-bin",
        default=os.environ.get("SU2_BIN"),
        help="Path to SU2_CFD. Defaults to SU2_BIN or known local paths.",
    )
    parser.add_argument(
        "--mode",
        choices=["original", "v84_compatible"],
        default="v84_compatible",
        help=(
            "original changes only AOA. v84_compatible also disables MUSCL_FLOW "
            "and MUSCL_ADJFLOW because SU2 v8.4 rejects JST+MUSCL."
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Optional ITER override for quick pilot runs. Omit for full 5000-iteration Kaggle setting.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of SU2 cases to run in parallel. Use 1 for safest reproducibility.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write case folders/configs but do not run SU2.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Delete existing generated case folders before preparing/running.",
    )
    return parser.parse_args()


def find_su2_bin(explicit: Optional[str]) -> Optional[str]:
    candidates = [
        explicit,
        shutil.which("SU2_CFD"),
        "/Users/pun/SU2/bin/SU2_CFD",
        "/Users/pun/dev/SU2/build/SU2_CFD/src/SU2_CFD",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return None


def aoa_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--aoa-step must be positive")

    values = []
    value = start
    epsilon = step / 1000.0
    while value <= stop + epsilon:
        values.append(round(value, 10))
        value += step
    return values


def mach_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--mach-step must be positive")

    values = []
    value = start
    epsilon = step / 1000.0
    while value <= stop + epsilon:
        values.append(round(value, 10))
        value += step
    return values


def build_case_grid(aoa_list: list[float], mach_list: list[float]) -> list[tuple[float, float]]:
    return [(mach, aoa) for mach in mach_list for aoa in aoa_list]


def mach_folder_name(mach: float) -> str:
    return f"mach_{mach:.2f}"


def aoa_folder_name(aoa: float) -> str:
    return f"aoa_{aoa:g}"


def case_label(mach: float, aoa: float) -> str:
    return f"{mach_folder_name(mach)}/{aoa_folder_name(aoa)}"


def set_value(cfg: str, key: str, value: str) -> str:
    pattern = rf"^({re.escape(key)}\s*=\s*).*$"
    new_cfg, count = re.subn(pattern, rf"\g<1>{value}", cfg, flags=re.MULTILINE)
    if count == 0:
        raise KeyError(f"Key {key!r} not found in template config")
    return new_cfg


def set_or_append_value(cfg: str, key: str, value: str) -> str:
    pattern = rf"^({re.escape(key)}\s*=\s*).*$"
    new_cfg, count = re.subn(pattern, rf"\g<1>{value}", cfg, flags=re.MULTILINE)
    if count:
        return new_cfg

    output_marker = "% Output files"
    insertion = f"{key}= {value}\n"
    if output_marker in cfg:
        return cfg.replace(output_marker, insertion + output_marker, 1)
    return cfg.rstrip() + "\n" + insertion


def prepare_case(
    mach: float,
    aoa: float,
    template_case: Path,
    output_dir: Path,
    mode: str,
    max_iter: Optional[int],
    allow_overwrite: bool,
) -> Path:
    template_cfg = template_case / "turb_SA_RAE2822.cfg"
    template_mesh = template_case / "mesh_RAE2822_turb.su2"
    if not template_cfg.exists():
        raise FileNotFoundError(template_cfg)
    if not template_mesh.exists():
        raise FileNotFoundError(template_mesh)

    case_dir = output_dir / mach_folder_name(mach) / aoa_folder_name(aoa)

    if case_dir.exists() and allow_overwrite:
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    cfg = template_cfg.read_text()
    cfg = set_value(cfg, "AOA", str(aoa))
    cfg = set_value(cfg, "MACH_NUMBER", f"{mach:.2f}")
    cfg = set_or_append_value(cfg, "VOLUME_OUTPUT", "(COORDINATES, SOLUTION, PRIMITIVE)")

    if mode == "v84_compatible":
        cfg = set_value(cfg, "MUSCL_FLOW", "NO")
        cfg = set_value(cfg, "MUSCL_ADJFLOW", "NO")
        cfg = set_value(cfg, "OUTPUT_FILES", "(RESTART, SURFACE_PARAVIEW_ASCII, SURFACE_CSV)")

    if max_iter is not None:
        cfg = set_value(cfg, "ITER", str(max_iter))

    (case_dir / "config.cfg").write_text(cfg)

    mesh_dst = case_dir / template_mesh.name
    if not mesh_dst.exists():
        try:
            mesh_dst.symlink_to(template_mesh.resolve())
        except OSError:
            shutil.copy(template_mesh, mesh_dst)

    return case_dir


def write_case_metadata(case_dir: Path, mach: float, aoa: float, status: str, **extra: object) -> None:
    metadata = {
        "case": case_label(mach, aoa),
        "case_dir": str(case_dir),
        "AoA": aoa,
        "Freestream_Mach": mach,
        "status": status,
    }
    metadata.update(extra)
    (case_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_ascii_vtu_point_data(vtu_path: Path) -> dict[str, object]:
    import numpy as np

    root = ET.parse(vtu_path).getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"Could not find Piece node in {vtu_path}")

    n_points = int(piece.attrib["NumberOfPoints"])
    point_data: dict[str, object] = {}

    points_array = piece.find("./Points/DataArray")
    if points_array is None or points_array.text is None:
        raise ValueError(f"Could not find point coordinates in {vtu_path}")
    n_components = int(points_array.attrib.get("NumberOfComponents", "1"))
    coordinates = np.fromstring(points_array.text, sep=" ", dtype=float).reshape(n_points, n_components)
    point_data["x"] = coordinates[:, 0]
    point_data["y"] = coordinates[:, 1]

    point_data_node = piece.find("./PointData")
    if point_data_node is None:
        raise ValueError(f"Could not find PointData node in {vtu_path}")

    for data_array in point_data_node.findall("./DataArray"):
        name = data_array.attrib.get("Name")
        if not name or data_array.text is None:
            continue
        n_components = int(data_array.attrib.get("NumberOfComponents", "1"))
        values = np.fromstring(data_array.text, sep=" ", dtype=float)
        if n_components == 1:
            point_data[name] = values
        else:
            point_data[name] = values.reshape(n_points, n_components)

    return point_data


def parse_legacy_vtk_point_data(vtk_path: Path) -> dict[str, object]:
    import numpy as np

    tokens = vtk_path.read_text().split()

    points_idx = tokens.index("POINTS")
    n_points = int(tokens[points_idx + 1])
    point_values_start = points_idx + 3
    point_values_end = point_values_start + n_points * 3
    coordinates = np.array(tokens[point_values_start:point_values_end], dtype=float).reshape(n_points, 3)

    point_data: dict[str, object] = {
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
    }

    idx = tokens.index("POINT_DATA") + 2
    while idx < len(tokens):
        section = tokens[idx]
        if section == "SCALARS":
            name = tokens[idx + 1]
            maybe_components = tokens[idx + 3]
            n_components = int(maybe_components) if maybe_components.isdigit() else 1
            idx += 4 if maybe_components.isdigit() else 3
            if tokens[idx] == "LOOKUP_TABLE":
                idx += 2
            value_count = n_points * n_components
            values = np.array(tokens[idx:idx + value_count], dtype=float)
            point_data[name] = values if n_components == 1 else values.reshape(n_points, n_components)
            idx += value_count
        elif section == "VECTORS":
            name = tokens[idx + 1]
            idx += 3
            value_count = n_points * 3
            point_data[name] = np.array(tokens[idx:idx + value_count], dtype=float).reshape(n_points, 3)
            idx += value_count
        else:
            idx += 1

    return point_data


def parse_surface_point_data(case_dir: Path) -> dict[str, object]:
    surface_vtu = case_dir / "surface_flow.vtu"
    surface_vtk = case_dir / "surface_flow.vtk"
    if surface_vtu.exists():
        return parse_ascii_vtu_point_data(surface_vtu)
    if surface_vtk.exists():
        return parse_legacy_vtk_point_data(surface_vtk)
    raise FileNotFoundError(f"No surface_flow.vtu or surface_flow.vtk found in {case_dir}")


def normalize_surface_csv_from_vtu(case_dir: Path) -> bool:
    import pandas as pd

    surface_csv = case_dir / "surface_flow.csv"
    if not surface_csv.exists():
        return False

    raw_csv = case_dir / "surface_flow_su2_raw.csv"
    if not raw_csv.exists():
        shutil.copy(surface_csv, raw_csv)

    raw = pd.read_csv(surface_csv, skipinitialspace=True)
    point_data = parse_surface_point_data(case_dir)

    normalized = pd.DataFrame(
        {
            "PointID": raw["PointID"].to_numpy(),
            "x": raw["x"].to_numpy(),
            "y": raw["y"].to_numpy(),
            "Density": raw["Density"].to_numpy(),
            "Momentum_x": raw["Momentum_x"].to_numpy(),
            "Momentum_y": raw["Momentum_y"].to_numpy(),
            "Energy": raw["Energy"].to_numpy(),
            "Nu_Tilde": raw["Nu_Tilde"].to_numpy(),
            "Pressure": point_data["Pressure"],
            "Temperature": point_data["Temperature"],
            "Mach": point_data["Mach"],
            "Pressure_Coefficient": point_data["Pressure_Coefficient"],
            "Laminar_Viscosity": point_data["Laminar_Viscosity"],
            "Skin_Friction_Coefficient_x": point_data["Skin_Friction_Coefficient"][:, 0],
            "Skin_Friction_Coefficient_y": point_data["Skin_Friction_Coefficient"][:, 1],
            "Heat_Flux": point_data["Heat_Flux"],
            "Y_Plus": point_data["Y_Plus"],
            "Eddy_Viscosity": point_data["Eddy_Viscosity"],
        }
    )
    normalized.to_csv(surface_csv, index=False, float_format="%.15e")
    return True


def has_kaggle_like_surface_csv(surface_csv: Path) -> bool:
    import pandas as pd

    if not surface_csv.exists():
        return False
    columns = pd.read_csv(surface_csv, nrows=0).columns.tolist()
    return "Pressure_Coefficient" in columns and "Pressure" in columns


def run_case(case_dir: Path, mach: float, aoa: float, su2_bin: str, allow_overwrite: bool) -> dict[str, object]:
    surface_csv = case_dir / "surface_flow.csv"
    label = case_label(mach, aoa)
    if surface_csv.exists() and not allow_overwrite:
        row = {
            "case": label,
            "case_dir": str(case_dir),
            "AoA": aoa,
            "Freestream_Mach": mach,
            "status": "SKIPPED_EXISTS",
            "surface_flow_csv": True,
            "kaggle_like_surface_csv": has_kaggle_like_surface_csv(surface_csv),
        }
        write_case_metadata(case_dir, mach, aoa, row["status"], surface_flow_csv=True, kaggle_like_surface_csv=row["kaggle_like_surface_csv"])
        return row

    with (case_dir / "run.log").open("w") as log:
        proc = subprocess.run(
            [su2_bin, "config.cfg"],
            cwd=case_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    normalized = False
    postprocess_error = None
    if proc.returncode == 0 and surface_csv.exists():
        try:
            normalized = normalize_surface_csv_from_vtu(case_dir)
        except Exception as exc:  # noqa: BLE001 - keep batch runs moving and report the failed case.
            postprocess_error = str(exc)
            (case_dir / "postprocess_error.txt").write_text(postprocess_error + "\n")

    kaggle_like = has_kaggle_like_surface_csv(surface_csv)
    if proc.returncode != 0:
        status = f"FAILED_{proc.returncode}"
    elif not surface_csv.exists():
        status = "FAILED_NO_SURFACE_CSV"
    elif not kaggle_like:
        status = "FAILED_POSTPROCESS"
    else:
        status = "OK"
    row = {
        "case": label,
        "case_dir": str(case_dir),
        "AoA": aoa,
        "Freestream_Mach": mach,
        "status": status,
        "surface_flow_csv": surface_csv.exists(),
        "normalized_from_vtu": normalized,
        "kaggle_like_surface_csv": kaggle_like,
        "postprocess_error": postprocess_error,
    }
    write_case_metadata(
        case_dir,
        mach,
        aoa,
        status,
        surface_flow_csv=row["surface_flow_csv"],
        normalized_from_vtu=normalized,
        kaggle_like_surface_csv=kaggle_like,
        postprocess_error=postprocess_error,
    )
    return row


def write_metadata(
    args: argparse.Namespace,
    su2_bin: Optional[str],
    aoa_list: list[float],
    mach_list: list[float],
    case_count: int,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "rae2822_aoa_mach_su2",
        "source_template_case": str(args.template_case),
        "output_dir": str(args.output_dir),
        "aoa_start": args.aoa_start,
        "aoa_stop": args.aoa_stop,
        "aoa_step": args.aoa_step,
        "aoa_values": aoa_list,
        "aoa_count": len(aoa_list),
        "mach_start": args.mach_start,
        "mach_stop": args.mach_stop,
        "mach_step": args.mach_step,
        "mach_values": mach_list,
        "mach_count": len(mach_list),
        "case_count": case_count,
        "mode": args.mode,
        "max_iter_override": args.max_iter,
        "su2_bin": su2_bin,
        "notes": [
            "Cases copy the Kaggle RAE2822 config and mesh.",
            "AOA and MACH_NUMBER are changed for each generated case.",
            "Case folders use nested layout: mach_xx/aoa_xx.",
            "VOLUME_OUTPUT is set to COORDINATES, SOLUTION, PRIMITIVE so surface_flow.csv keeps Kaggle-like Cp and primitive columns.",
            "SU2 v8.4 writes conservative-only SURFACE_CSV for this case, so the script rewrites surface_flow.csv from SURFACE_PARAVIEW_ASCII.",
            "v84_compatible mode sets MUSCL_FLOW=NO and MUSCL_ADJFLOW=NO because SU2 v8.4 rejects JST+MUSCL.",
            "Keep this dataset separate from Kaggle until overlap cases are compared.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    aoa_list = aoa_values(args.aoa_start, args.aoa_stop, args.aoa_step)
    mach_list = mach_values(args.mach_start, args.mach_stop, args.mach_step)
    case_grid = build_case_grid(aoa_list, mach_list)

    su2_bin = find_su2_bin(args.su2_bin)

    if not args.prepare_only and not su2_bin:
        raise FileNotFoundError("Could not find SU2_CFD. Set SU2_BIN or pass --su2-bin.")

    print("Dense RAE2822 SU2 dataset")
    print(f"  AoA count: {len(aoa_list)}")
    print(f"  AoA range: {aoa_list[0]:g} to {aoa_list[-1]:g} step {args.aoa_step:g}")
    print(f"  Mach count: {len(mach_list)}")
    print(f"  Mach range: {mach_list[0]:.2f} to {mach_list[-1]:.2f} step {args.mach_step:.2f}")
    print(f"  Total cases: {len(case_grid)}")
    print(f"  Template: {args.template_case}")
    print(f"  Output: {args.output_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  SU2_CFD: {su2_bin or 'not needed for --prepare-only'}")
    print()


    case_records = [
        {
            "AoA": aoa,
            "Freestream_Mach": mach,
            "case_dir": prepare_case(
                mach,
                aoa,
                args.template_case,
                args.output_dir,
                args.mode,
                args.max_iter,
                args.allow_overwrite,
            ),
        }
        for mach, aoa in case_grid
    ]

    for record in case_records:
        write_case_metadata(record["case_dir"], record["Freestream_Mach"], record["AoA"], "PREPARED")

    write_metadata(args, su2_bin, aoa_list, mach_list, len(case_records))

    if args.prepare_only:
        print("Prepared case folders only. No SU2 runs were launched.")
        return 0

    assert su2_bin is not None
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_case = {
            executor.submit(
                run_case,
                record["case_dir"],
                record["Freestream_Mach"],
                record["AoA"],
                su2_bin,
                args.allow_overwrite,
            ): record
            for record in case_records
        }
        for future in as_completed(future_to_case):
            row = future.result()
            rows.append(row)
            print(f"{row['case']}: {row['status']}")

    rows.sort(key=lambda row: (float(row["Freestream_Mach"]), float(row["AoA"])))
    summary_path = args.output_dir / "generation_summary.csv"
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(summary_path, index=False)
    except ImportError:
        summary_path.write_text(json.dumps(rows, indent=2) + "\n")

    print()
    print(f"Generation summary: {summary_path}")
    return 0 if all(row["status"] in {"OK", "SKIPPED_EXISTS"} for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
