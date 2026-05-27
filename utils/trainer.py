import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from joblib import dump

TRAIN_FILE = "data/trained_market_data.json"

MODEL_FILE = "models/market_model.pkl"

SCALER_FILE = "models/scaler.pkl"


def train_market_model():

    if not os.path.exists(TRAIN_FILE):

        print("No training file")

        return None

    try:

        df = pd.read_json(TRAIN_FILE)

    except Exception as e:

        print(e)

        return None

    if len(df) < 10:

        print("Not enough training samples")

        return None

    X = df[
        [
            "change_percent",
            "volume"
        ]
    ]

    y = df["price"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    dump(
        model,
        MODEL_FILE
    )

    dump(
        scaler,
        SCALER_FILE
    )

    return {
        "samples": len(df),
        "mae": round(mae, 2)
    }