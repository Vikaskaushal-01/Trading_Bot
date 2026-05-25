import streamlit as st
import json
import os

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide"
)

# ---------------- MEMORY ---------------- #

MEMORY_FILE = "data/trading_memory.json"

def load_memory():

    if not os.path.exists(MEMORY_FILE):

        os.makedirs("data", exist_ok=True)

        default_data = {
            "wallet": 500,
            "transactions": []
        }

        with open(MEMORY_FILE, "w") as f:
            json.dump(default_data, f)

    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

memory = load_memory()

# ---------------- SESSION ---------------- #

if "wallet" not in st.session_state:
    st.session_state.wallet = memory["wallet"]

if "demo_wallet" not in st.session_state:
    st.session_state.demo_wallet = 500

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.main-title{
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:white;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#b0b0b0;
    margin-bottom:40px;
}

.card{
    background:#161b22;
    padding:25px;
    border-radius:18px;
    text-align:center;
    border:1px solid #30363d;
    transition:0.3s;
}

.card:hover{
    transform:translateY(-5px);
}

.wallet{
    background:#0f172a;
    padding:20px;
    border-radius:18px;
    text-align:center;
    border:1px solid #1e293b;
}

.stock-card{
    background:#161b22;
    padding:12px;
    border-radius:14px;
    text-align:center;
    margin-bottom:10px;
}

.stButton button{
    width:100%;
    border:none;
    border-radius:12px;
    padding:12px;
    background:#00c896;
    color:white;
    font-size:16px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown("""
<div class='main-title'>
📈 AI Trading Platform
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='subtitle'>
Trade smarter with AI predictions, live charts, and portfolio tracking
</div>
""", unsafe_allow_html=True)

# ---------------- WALLETS ---------------- #

c1, c2 = st.columns(2)

with c1:

    st.markdown(f"""
    <div class='wallet'>
        <h2>💰 Main Wallet</h2>
        <h1>${round(st.session_state.wallet,2)}</h1>
    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""
    <div class='wallet'>
        <h2>🧪 Demo Wallet</h2>
        <h1>${round(st.session_state.demo_wallet,2)}</h1>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- FEATURES ---------------- #

st.subheader("🔥 Features")

f1, f2, f3 = st.columns(3)

with f1:

    st.markdown("""
    <div class='card'>
        <h3>📈 Real-Time Charts</h3>
        <p>Track live stock market prices with dynamic graphs.</p>
    </div>
    """, unsafe_allow_html=True)

with f2:

    st.markdown("""
    <div class='card'>
        <h3>🤖 AI Predictions</h3>
        <p>Get AI-based stock price predictions using ML models.</p>
    </div>
    """, unsafe_allow_html=True)

with f3:

    st.markdown("""
    <div class='card'>
        <h3>💬 AI Chatbot</h3>
        <p>Ask trading-related questions and market analysis.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- STOCKS ---------------- #

st.subheader("📊 Available Stocks")

stocks = [
    "Apple (AAPL)",
    "Tesla (TSLA)",
    "Microsoft (MSFT)",
    "Google (GOOGL)",
    "NVIDIA (NVDA)",
    "Bitcoin (BTC-USD)",
    "Reliance Industries",
    "HDFC Bank",
    "ICICI Bank",
    "Bharti Airtel"
]

cols = st.columns(5)

for i, stock in enumerate(stocks):

    with cols[i % 5]:

        st.markdown(f"""
        <div class='stock-card'>
            {stock}
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ---------------- NAVIGATION ---------------- #

st.subheader("🚀 Start Trading")

b1, b2 = st.columns(2)

with b1:

    st.page_link(
        "pages/trading.py",
        label="📈 Open Trading Dashboard"
    )

with b2:

    st.page_link(
        "pages/demo.py",
        label="🧪 Open Demo Trading"
    )

st.write("")

# ---------------- REFERENCES ---------------- #

st.subheader("🌐 Market Links")

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

st.write("")
st.write("")

st.markdown("""
<center>
Built with ❤️ using Streamlit + AI/ML
</center>
""", unsafe_allow_html=True)