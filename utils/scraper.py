import yfinance as yf
import json
import os

from datetime import datetime

RAW_FILE = "data/market_data.json"

stocks = {

    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "GOOGL": "Google",

    "RELIANCE.NS": "Reliance",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "BHARTIARTL.NS": "Bharti Airtel"

}


def load_existing_data():

    if not os.path.exists(RAW_FILE):

        return []

    try:

        with open(RAW_FILE, "r") as f:

            return json.load(f)

    except:

        return []


def save_data(new_data):

    os.makedirs(
        "data",
        exist_ok=True
    )

    existing = load_existing_data()

    existing.extend(new_data)

    # REMOVE DUPLICATES
    unique = []

    seen = set()

    for item in existing:

        key = (
            item["stock"],
            item["price"],
            item["change_percent"]
        )

        if key not in seen:

            seen.add(key)

            unique.append(item)

    # KEEP LAST 5000
    unique = unique[-5000:]

    with open(RAW_FILE, "w") as f:

        json.dump(
            unique,
            f,
            indent=4
        )

    print(
        f"Saved {len(unique)} records"
    )


def fetch_market_data():

    market_data = []

    for ticker, name in stocks.items():

        try:

            stock = yf.Ticker(ticker)

            hist = stock.history(
                period="1mo",
                interval="30m"
            )
            if len(hist) < 2:

                continue

            current_price = float(
                hist["Close"].iloc[-1]
            )

            previous_price = float(
                hist["Close"].iloc[-2]
            )

            volume = float(
                hist["Volume"].iloc[-1]
            )

            change_percent = (
                (
                    current_price
                    - previous_price
                )
                / previous_price
            ) * 100

            market_data.append({

                "stock": name,

                "price": round(
                    current_price,
                    2
                ),

                "change_percent": round(
                    change_percent,
                    2
                ),

                "volume": round(
                    volume,
                    2
                ),

                "timestamp": str(
                    datetime.now()
                )

            })

        except Exception as e:

            print(
                f"{name} error:",
                e
            )

    save_data(market_data)

    print(
        f"Fetched {len(market_data)} rows"
    )

    return market_data