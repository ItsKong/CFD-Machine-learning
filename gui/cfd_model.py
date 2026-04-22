import json
import os
import tempfile
import warnings
import zipfile
from pathlib import Path

from cfd_ml.paths import CP_MODEL_DIR, PROJECT_ROOT


os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.exceptions import InconsistentVersionWarning


class CP_Predict_Model:
    """Predict surface Cp for the fixed RAE2822 base geometry."""

    def __init__(self):
        self.__model = self.load_model()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            self.__scaler_X = joblib.load(CP_MODEL_DIR / "scaler_X.pkl")
            self.__scaler_y = joblib.load(CP_MODEL_DIR / "scaler_y.pkl")
        self.__geom_data = pd.read_csv(CP_MODEL_DIR / "base_geometry.csv")

    def load_model(self):
        model_path = CP_MODEL_DIR / "cfd_model.keras"
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except TypeError as exc:
            if "quantization_config" not in str(exc):
                raise
            patched_model_path = self._create_keras_compat_copy(model_path)
            try:
                model = tf.keras.models.load_model(patched_model_path, compile=False)
            finally:
                patched_model_path.unlink(missing_ok=True)

        print("Model loaded successfully!")
        return model

    def _create_keras_compat_copy(self, model_path: Path) -> Path:
        temp_file = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        with zipfile.ZipFile(model_path, "r") as source:
            with zipfile.ZipFile(temp_path, "w") as target:
                for item in source.infolist():
                    data = source.read(item.filename)
                    if item.filename == "config.json":
                        config = json.loads(data)
                        self._strip_unsupported_keras_fields(config)
                        data = json.dumps(config).encode("utf-8")
                    target.writestr(item, data)

        return temp_path

    def _strip_unsupported_keras_fields(self, value):
        if isinstance(value, dict):
            value.pop("quantization_config", None)
            for child in value.values():
                self._strip_unsupported_keras_fields(child)
        elif isinstance(value, list):
            for child in value:
                self._strip_unsupported_keras_fields(child)

    def predict_airfoil_cp(self, target_aoa: float):
        x_new = self.__geom_data[["x", "y"]].copy()

        x_new["sin_AoA"] = np.sin(np.deg2rad(target_aoa))
        x_new["cos_AoA"] = np.cos(np.deg2rad(target_aoa))

        features = ["x", "y", "sin_AoA", "cos_AoA"]

        x_new_scaled = self.__scaler_X.transform(x_new[features].values)
        cp_predicted_scaled = self.__model.predict(x_new_scaled, verbose=0)
        cp_predicted = self.__scaler_y.inverse_transform(cp_predicted_scaled).ravel()

        result = x_new[["x", "y"]].copy()
        result["Cp_predicted"] = cp_predicted

        return result


if __name__ == "__main__":
    predict_aoa = 0

    model = CP_Predict_Model()

    final_results = model.predict_airfoil_cp(predict_aoa)

    print(final_results.head())
