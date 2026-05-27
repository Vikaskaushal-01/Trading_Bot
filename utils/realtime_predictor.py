import os
import pandas as pd

from joblib import load

MODEL_FILE = "models/market_model.pkl"

SCALER_FILE = "models/scaler.pkl"


def predict_stock_price(
    change_percent,
    volume
):

    # CHECK MODEL
    if not os.path.exists(MODEL_FILE):

        print("Model not found")

        return None

    if not os.path.exists(SCALER_FILE):

        print("Scaler not found")

        return None

    try:

        # LOAD MODEL
        model = load(MODEL_FILE)

        # LOAD SCALER
        scaler = load(SCALER_FILE)

        # INPUT DATA
        input_df = pd.DataFrame([
            {
                "change_percent":
                change_percent,

                "volume":
                volume
            }
        ])

        # SCALE
        scaled_input = scaler.transform(
            input_df
        )

        # PREDICT
        prediction = model.predict(
            scaled_input
        )[0]

        return round(
            float(prediction),
            2
        )

    except Exception as e:

        print(e)

        return None