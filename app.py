import streamlit as st
import json
import os

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
    padding-bottom: 2rem;
    max-width: 1400px;
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
    margin-bottom: 40px;
    border: 1px solid #30363d;
}

.hero-title {
    font-size: 60px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 20px;
    color: #cbd5e1;
    line-height: 1.7;
}

.wallet-card {
    background:
    linear-gradient(
        135deg,
        #161b22,
        #1e293b
    );

    padding: 30px;
    border-radius: 25px;
    text-align: center;
    border: 1px solid #30363d;
    transition: 0.3s;
}

.wallet-card:hover {
    transform: translateY(-5px);
}

.wallet-title {
    font-size: 20px;
    color: #94a3b8;
}

.wallet-value {
    font-size: 42px;
    font-weight: bold;
    color: #00c896;
}

.feature-card {
    background:
    linear-gradient(
        135deg,
        #161b22,
        #111827
    );

    padding: 25px;
    border-radius: 22px;
    border: 1px solid #30363d;
    text-align: center;
    transition: 0.3s;
    height: 100%;
}

.feature-card:hover {
    transform: translateY(-6px);
    border: 1px solid #00c896;
}

.feature-icon {
    font-size: 40px;
    margin-bottom: 10px;
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

.stock-card {
    background:
    linear-gradient(
        135deg,
        #161b22,
        #111827
    );

    padding: 18px;
    border-radius: 18px;
    border: 1px solid #30363d;
    text-align: center;
    margin-bottom: 15px;
    transition: 0.3s;
}

.stock-card:hover {
    transform: scale(1.03);
    border: 1px solid #00c896;
}

.stock-name {
    font-size: 17px;
    font-weight: 600;
}

.section-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 20px;
}

.stButton button {
    width: 100%;
    border-radius: 14px;
    background: linear-gradient(
        135deg,
        #00c896,
        #00a67d
    );

    color: white;
    border: none;
    padding: 15px;
    font-size: 17px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.02);
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 50px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ---------------- #

st.markdown("""
<div class="hero">

<div class="hero-title">
📈 AI Trading Platform
</div>

<div class="hero-subtitle">
Trade smarter with AI-powered predictions,
real-time market tracking, persistent portfolios,
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

st.write("")

# ---------------- FEATURES ---------------- #

st.markdown("""
<div class="section-title">
🔥 Platform Features
</div>
""", unsafe_allow_html=True)

f1, f2, f3 = st.columns(3)

with f1:

    st.markdown("""
    <div class="feature-card">

    <div class="feature-icon">
    📈
    </div>

    <div class="feature-title">
    Real-Time Charts
    </div>

    <div class="feature-text">
    Monitor live stock movements with dynamic market charts and indicators.
    </div>

    </div>
    """, unsafe_allow_html=True)

with f2:

    st.markdown("""
    <div class="feature-card">

    <div class="feature-icon">
    🤖
    </div>

    <div class="feature-title">
    AI Predictions
    </div>

    <div class="feature-text">
    Machine learning powered stock predictions using LSTM neural networks.
    </div>

    </div>
    """, unsafe_allow_html=True)

with f3:

    st.markdown("""
    <div class="feature-card">

    <div class="feature-icon">
    💬
    </div>

    <div class="feature-title">
    AI Chatbot
    </div>

    <div class="feature-text">
    Ask trading questions and get intelligent market insights instantly.
    </div>

    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- STOCKS ---------------- #

st.markdown("""
<div class="section-title">
📊 Supported Assets
</div>
""", unsafe_allow_html=True)

stocks = [
    "🍎 Apple",
    "🚗 Tesla",
    "🪟 Microsoft",
    "🌐 Google",
    "💻 NVIDIA",
    "₿ Bitcoin",
    "🏢 Reliance",
    "🏦 HDFC Bank",
    "🏦 ICICI Bank",
    "📡 Airtel"
]

cols = st.columns(5)

for i, stock in enumerate(stocks):

    with cols[i % 5]:

        st.markdown(f"""
        <div class="stock-card">

        <div class="stock-name">
        {stock}
        </div>

        </div>
        """, unsafe_allow_html=True)

st.write("")

# ---------------- NAVIGATION ---------------- #

st.markdown("""
<div class="section-title">
🚀 Start Trading
</div>
""", unsafe_allow_html=True)

n1, n2 = st.columns(2)

with n1:

    st.page_link(
        "pages/trading.py",
        label="📈 Open Trading Dashboard"
    )

with n2:

    st.page_link(
        "pages/demo.py",
        label="🧪 Open Demo Trading"
    )

st.write("")

# ---------------- REFERENCES ---------------- #

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