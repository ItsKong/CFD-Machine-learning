import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cfd_ml.data import load_rae2822_surface_data, load_tradeoff_data


def main():
    rae = load_rae2822_surface_data()
    print("RAE2822 surface data")
    print(f"  rows: {len(rae)}")
    print(f"  cases: {rae['AoA'].nunique()}")
    print(f"  AoA range: {rae['AoA'].min()} to {rae['AoA'].max()}")
    print(f"  columns: {', '.join(rae.columns)}")

    tradeoff = load_tradeoff_data()
    print()
    print("N2412 trade-off data")
    print(f"  rows: {len(tradeoff)}")
    print(f"  columns: {', '.join(tradeoff.columns)}")
    print(f"  AoA range: {tradeoff['aoa'].min()} to {tradeoff['aoa'].max()}")
    print(f"  CL range: {tradeoff['cl'].min():.4f} to {tradeoff['cl'].max():.4f}")
    print(f"  CD range: {tradeoff['cd'].min():.4f} to {tradeoff['cd'].max():.4f}")


if __name__ == "__main__":
    main()
