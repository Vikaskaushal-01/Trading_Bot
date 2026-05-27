import json
import pandas as pd
import os

RAW_FILE = "data/market_data.json"

TRAIN_FILE = "data/trained_market_data.json"


def process_market_data():

    if not os.path.exists(RAW_FILE):

        print("No raw data found")

        return None

    try:

        with open(RAW_FILE, "r") as f:

            raw_data = json.load(f)

    except Exception as e:

        print(e)

        return None

    if len(raw_data) == 0:

        print("Raw dataset empty")

        return None

    df = pd.DataFrame(raw_data)

    # REMOVE NULLS
    df = df.dropna()

    # REMOVE DUPLICATES
    df = df.drop_duplicates()

    # KEEP IMPORTANT FEATURES
    processed = df[
        [
            "price",
            "change_percent",
            "volume"
        ]
    ]

    processed = processed[
        processed["price"] > 0
    ]

    if len(processed) == 0:

        print("Processed dataset empty")

        return None

    os.makedirs(
        "data",
        exist_ok=True
    )

    processed.to_json(
        TRAIN_FILE,
        orient="records",
        indent=4
    )

    print(
        f"Processed {len(processed)} rows"
    )

    return processed