import streamlit as st
import json
import os
import yfinance as yf

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- MEMORY ---------------- #

TRADING_MEMORY = "data/trading_memory.json"
DEMO_MEMORY = "data/demo_memory.json"

def load_json(file_path, default_data):

    if not os.path.exists("data"):

        os.makedirs("data")

    if not os.path.exists(file_path):

        with open(file_path, "w") as f:

            json.dump(default_data, f, indent=4)

    with open(file_path, "r") as f:

        return json.load(f)

trading_memory = load_json(
    TRADING_MEMORY,
    {
        "wallet": 500,
        "transactions": []
    }
)

demo_memory = load_json(
    DEMO_MEMORY,
    {
        "demo_wallet": 500,
        "demo_transactions": []
    }
)

# ---------------- TOP 5 TRENDING STOCKS ---------------- #

stocks = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Microsoft": "MSFT",
    "Bitcoin": "BTC-USD"
}

# ---------------- CSS ---------------- #

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #0d1117;
    color: white;
}

.block-container {
    padding-top: 1rem;
    max-width: 1450px;
}

.hero {
    background:
    linear-gradient(
        135deg,
        #111827,
        #1f2937,
        #0f172a
    );

    padding: 60px;
    border-radius: 30px;
    margin-bottom: 30px;
    border: 1px solid #30363d;
}

.hero-title {
    font-size: 60px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    font-size: 22px;
    color: #cbd5e1;
    margin-top: 10px;
    line-height: 1.7;
}

.wallet-card {
    background:
    linear-gradient(
        135deg,
        #161b22,
        #1e293b
    );

    padding: 28px;
    border-radius: 24px;
    text-align: center;
    border: 1px solid #30363d;
}

.wallet-title {
    font-size: 20px;
    color: #94a3b8;
}

.wallet-value {
    color: #00c896;
    font-size: 42px;
    font-weight: bold;
}

.feature-card {
    background:
    linear-gradient(
        135deg,
        #161b22,
        #111827
    );

    padding: 30px;
    border-radius: 24px;
    border: 1px solid #30363d;
    text-align: center;
    transition: 0.3s;
}

.feature-card:hover {
    transform: translateY(-5px);
    border: 1px solid #00c896;
}

.feature-icon {
    font-size: 42px;
    margin-bottom: 12px;
}

.feature-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 10px;
}

.feature-text {
    color: #94a3b8;
    line-height: 1.6;
}

.section-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 20px;
    color: white;
}

.stButton button {
    width: 100%;
    border-radius: 14px;
    background:
    linear-gradient(
        135deg,
        #00c896,
        #00a67d
    );

    color: white;
    border: none;
    padding: 13px;
    font-size: 16px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 50px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ---------------- #

st.markdown("""
<div class="hero">

<div class="hero-title">
📈 AI Trading Platform
</div>

<div class="hero-subtitle">
Trade smarter with AI-powered predictions,
real-time stock tracking, persistent portfolios,
and intelligent trading insights.
</div>

</div>
""", unsafe_allow_html=True)

# ---------------- WALLET SECTION ---------------- #

c1, c2 = st.columns(2)

with c1:

    st.markdown(f"""
    <div class="wallet-card">

    <div class="wallet-title">
    💰 Main Wallet
    </div>

    <div class="wallet-value">
    ${round(trading_memory['wallet'],2)}
    </div>

    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""
    <div class="wallet-card">

    <div class="wallet-title">
    🧪 Demo Wallet
    </div>

    <div class="wallet-value">
    ${round(demo_memory['demo_wallet'],2)}
    </div>

    </div>
    """, unsafe_allow_html=True)

# ---------------- TRENDING STOCKS ---------------- #


st.write("")

st.markdown("""
<div class="section-title">
🔥_Top Trending Stocks
</div>
""", unsafe_allow_html=True)

trend_cols = st.columns(5)

stock_items = list(stocks.items())

CARD_HEIGHT = 320

for i, (name, ticker) in enumerate(stock_items):

    with trend_cols[i]:

        try:

            stock = yf.Ticker(ticker)

            hist = stock.history(period="5d")

            current_price = float(
                hist["Close"].iloc[-1]
            )

            previous_price = float(
                hist["Close"].iloc[-2]
            )

            percent = (
                (
                    current_price
                    - previous_price
                )
                / previous_price
            ) * 100

            percent_color = (
                "#00ff99"
                if percent >= 0
                else "#ff4d4d"
            )

            arrow = (
                "📈"
                if percent >= 0
                else "📉"
            )

            display_price = (
                f"${current_price:,.2f}"
            )

            st.markdown(f"""
            <div style="
                background:#0f172a;
                border-radius:22px;
                border:1px solid #1e293b;
                height:{CARD_HEIGHT}px;
                width:100%;
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
                text-align:center;
                padding:20px;
                box-sizing:border-box;
                margin-bottom:22px;
            ">

            <div style="
                color:white;
                font-size:28px;
                font-weight:700;
                margin-bottom:25px;
                min-height:40px;
                display:flex;
                align-items:center;
                justify-content:center;
            ">
            {name}
            </div>

            <div style="
                color:#00d4aa;
                font-size:32px;
                font-weight:800;
                margin-bottom:25px;
                word-break:break-word;
                line-height:1.2;
            ">
            {display_price}
            </div>

            <div style="
                color:{percent_color};
                font-size:24px;
                font-weight:700;
            ">
            {arrow} {round(percent,2)}%
            </div>

            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                "<div style='margin-top:12px;'></div>",
                unsafe_allow_html=True
            )

            st.link_button(
                f"🛒 Buy {name}",
                f"http://localhost:8501/trading?stock={ticker}",
                use_container_width=True
            )

        except Exception:

            st.markdown(f"""
            <div style="
                background:#0f172a;
                border-radius:22px;
                border:1px solid #1e293b;
                height:{CARD_HEIGHT}px;
                width:100%;
                display:flex;
                align-items:center;
                justify-content:center;
                text-align:center;
                color:#ff4d4d;
                font-size:20px;
                font-weight:700;
                padding:20px;
                box-sizing:border-box;
                margin-bottom:22px;
            ">
            {name} unavailable
            </div>
            """, unsafe_allow_html=True)

# ---------------- MARKET LINKS ---------------- #

st.write("")

st.markdown("""
<div class="section-title">
🌐 Market References
</div>
""", unsafe_allow_html=True)

r1, r2, r3 = st.columns(3)

with r1:

    st.link_button(
        "Yahoo Finance",
        "https://finance.yahoo.com"
    )

with r2:

    st.link_button(
        "TradingView",
        "https://www.tradingview.com"
    )

with r3:

    st.link_button(
        "MoneyControl",
        "https://www.moneycontrol.com"
    )

# ---------------- FOOTER ---------------- #

st.markdown("""
<div class="footer">

Built with ❤️ using Streamlit + AI/ML

</div>
""", unsafe_allow_html=True)