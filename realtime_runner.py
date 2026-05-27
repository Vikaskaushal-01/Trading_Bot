import time
import schedule

from utils.scraper import fetch_market_data
from utils.data_processor import process_market_data
from utils.trainer import train_market_model


def run_pipeline():

    print(
        "Fetching Market Data..."
    )

    fetch_market_data()

    print(
        "Processing Market Data..."
    )

    process_market_data()

    print(
        "Training AI Model..."
    )

    result = train_market_model()

    print(result)


# EVERY 10 MIN
schedule.every(10).minutes.do(
    run_pipeline
)

run_pipeline()

while True:

    schedule.run_pending()

    time.sleep(1)