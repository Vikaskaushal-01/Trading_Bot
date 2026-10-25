"""Central configuration: paths, asset universe and trading constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"

LIVE_ACCOUNT_FILE = DATA_DIR / "trading_memory.json"
DEMO_ACCOUNT_FILE = DATA_DIR / "demo_memory.json"
CHAT_FILE = DATA_DIR / "chat_history.json"

STARTING_BALANCE = 500.0
COMMISSION_RATE = 0.001  # 0.1% per trade

# symbol -> (display name, category, native currency)
UNIVERSE = {
    "AAPL": ("Apple", "US Tech", "USD"),
    "MSFT": ("Microsoft", "US Tech", "USD"),
    "NVDA": ("NVIDIA", "US Tech", "USD"),
    "GOOGL": ("Google", "US Tech", "USD"),
    "AMZN": ("Amazon", "US Tech", "USD"),
    "META": ("Meta", "US Tech", "USD"),
    "TSLA": ("Tesla", "US Tech", "USD"),
    "BTC-USD": ("Bitcoin", "Crypto", "USD"),
    "ETH-USD": ("Ethereum", "Crypto", "USD"),
    "RELIANCE.NS": ("Reliance Industries", "India", "INR"),
    "TCS.NS": ("TCS", "India", "INR"),
    "INFY.NS": ("Infosys", "India", "INR"),
    "HDFCBANK.NS": ("HDFC Bank", "India", "INR"),
    "ICICIBANK.NS": ("ICICI Bank", "India", "INR"),
    "BHARTIARTL.NS": ("Bharti Airtel", "India", "INR"),
}

FEATURED = ["AAPL", "NVDA", "MSFT", "TSLA", "BTC-USD"]

CURRENCY_SYMBOLS = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥"}

PERIODS = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}


def asset_name(symbol: str) -> str:
    return UNIVERSE.get(symbol, (symbol,))[0]
