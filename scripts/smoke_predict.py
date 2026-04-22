import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gui.cfd_model import CP_Predict_Model


def main():
    model = CP_Predict_Model()
    for aoa in [0, 5, 10, 15, 20]:
        result = model.predict_airfoil_cp(aoa)
        cp = result["Cp_predicted"]
        print(
            f"AoA={aoa:>2}: "
            f"rows={len(result)}, "
            f"mean={cp.mean(): .6f}, "
            f"min={cp.min(): .6f}, "
            f"max={cp.max(): .6f}"
        )


if __name__ == "__main__":
    main()
