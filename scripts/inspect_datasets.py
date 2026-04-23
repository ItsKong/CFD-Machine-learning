import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cfd_ml.data import load_rae2822_surface_data
from cfd_ml.paths import DENSE_RAE2822_CASE_DIR, RAE2822_CASE_DIR


def describe_dataset(title, data_dir):
    frame = load_rae2822_surface_data(data_dir=data_dir)
    print(title)
    print(f"  path: {data_dir}")
    print(f"  rows: {len(frame)}")
    print(f"  cases: {frame['AoA'].nunique()}")
    print(f"  AoA range: {frame['AoA'].min()} to {frame['AoA'].max()}")
    print(f"  columns: {', '.join(frame.columns)}")


def main():
    describe_dataset("RAE2822 Kaggle baseline", RAE2822_CASE_DIR)
    print()
    describe_dataset("RAE2822 dense SU2 extension", DENSE_RAE2822_CASE_DIR)


if __name__ == "__main__":
    main()
