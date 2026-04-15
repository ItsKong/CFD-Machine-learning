import tensorflow as tf
import pandas as pd
import joblib
import numpy as np


class CP_Predict_Model():

    def __init__(self):
        self.__model = self.load_model()

    def load_model(self):
        # Point this to the directory containing 'saved_model.pb'
        model_path = '../cfd_model.keras'

        # Load the model
        model = tf.keras.models.load_model(model_path)

        print("Model loaded successfully!")
        # Optional: View the architecture to remind yourself of the expected input shapes
        model.summary()
        return model

    def predict_airfoil_cp(self, target_aoa: float):
        print("Loading scalers...")
        scaler_X = joblib.load('../scaler_X.pkl')
        scaler_y = joblib.load('../scaler_y.pkl')

        geom_data = pd.read_csv('../base_geometry.csv')

        X_new = geom_data[['x', 'y']].copy()

        X_new['sin_AoA'] = np.sin(np.deg2rad(target_aoa))
        X_new['cos_AoA'] = np.cos(np.deg2rad(target_aoa))

        features = ['x', 'y', 'sin_AoA', 'cos_AoA']

        X_new_s = scaler_X.transform(X_new[features])
        Cp_predicted_s = self.__model.predict(X_new_s, verbose=0)
        Cp_predicted = scaler_y.inverse_transform(Cp_predicted_s)

        result = X_new[['x', 'y']].copy()
        result['Cp_predicted'] = Cp_predicted

        return result


if __name__ == "__main__":
    predict_aoa = 0

    model = CP_Predict_Model()

    final_results = model.predict_airfoil_cp(predict_aoa)

    print(final_results.head())
