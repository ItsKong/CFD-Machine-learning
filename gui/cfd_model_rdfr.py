import os
import warnings
from pathlib import Path

# --- Path Logic ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" # <-- Change "models" if your folder is named something else!
# ------------------

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

class CP_Predict_Model_RDFR:
    """Predict surface Cp for the fixed RAE2822 base geometry using Random Forest."""

    def __init__(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            
            # Load only the Random Forest model (no scalers)
            model_path = MODELS_DIR / "random_forest_cp_surrogate.pkl"
            self.__model = joblib.load(model_path)
            
        self.__geom_data = pd.read_csv(MODELS_DIR / "base_geometry.csv")
        print("Random Forest model loaded successfully!")

    def predict_airfoil_cp(self, target_aoa: float, target_mach: float):
        # Start with base geometry
        x_new = self.__geom_data[["x", "y"]].copy()

        # Add the engineered/input features
        x_new["sin_AoA"] = np.sin(np.deg2rad(target_aoa))
        x_new["cos_AoA"] = np.cos(np.deg2rad(target_aoa))
        x_new["Mach"] = target_mach

        features = ["x", "y", "sin_AoA", "cos_AoA", "Mach"]

        # Predict directly (no scaling required)
        # We pass .values to ensure we are sending a numpy array to sklearn
        cp_predicted = self.__model.predict(x_new[features]) 

        # Build the final output dataframe
        result = x_new[["x", "y"]].copy()
        result["Cp_predicted"] = cp_predicted

        return result


if __name__ == "__main__":
    # Define your test conditions here
    predict_aoa = 2.31  # Example AoA
    predict_mach = 0.75 # Example Mach number

    model = CP_Predict_Model_RDFR()

    # Pass both AoA and Mach to the prediction function
    final_results = model.predict_airfoil_cp(predict_aoa, predict_mach)

    print(final_results.head())