import streamlit as st

st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SESSION ---------------- #

if "wallet" not in st.session_state:
    st.session_state.wallet = 500

if "demo_wallet" not in st.session_state:
    st.session_state.demo_wallet = 500

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0e1117;
    color: white;
}

.hero {
    padding: 40px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #111827,
        #1f2937
    );
    margin-bottom: 30px;
}

.feature-card {
    padding: 25px;
    border-radius: 20px;
    background: #161b22;
    border: 1px solid #30363d;
    transition: 0.3s;
}

.feature-card:hover {
    transform: scale(1.02);
}

.wallet-box {
    background: #161b22;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #30363d;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    background: #00c896;
    color: white;
    border: none;
    padding: 12px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ---------------- #

st.markdown("""
<div class="hero">

<h1>🚀 AI Trading Platform</h1>

<p>
Advanced AI-powered trading platform with:
</p>

<ul>
<li>📈 Real-Time Market Charts</li>
<li>🤖 AI Predictions</li>
<li>💹 Persistent Trading Portfolio</li>
<li>📊 Live Profit/Loss Tracking</li>
<li>🇮🇳 Indian + US Stocks</li>
<li>🧪 Demo Trading System</li>
<li>💬 AI Trading Chatbot</li>
</ul>

</div>
""", unsafe_allow_html=True)

# ---------------- WALLET DISPLAY ---------------- #

col1, col2 = st.columns(2)

with col1:

    st.markdown(f"""
    <div class="wallet-box">
        <h2>💰 Main Wallet</h2>
        <h1>${round(st.session_state.wallet,2)}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="wallet-box">
        <h2>🧪 Demo Wallet</h2>
        <h1>${round(st.session_state.demo_wallet,2)}</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- FEATURES ---------------- #

st.subheader("🔥 Platform Features")

f1, f2, f3 = st.columns(3)

with f1:

    st.markdown("""
    <div class="feature-card">
        <h3>📈 Real-Time Trading</h3>
        <p>
        Track live stock prices with dynamic
        charts and technical indicators.
        </p>
    </div>
    """, unsafe_allow_html=True)

with f2:

    st.markdown("""
    <div class="feature-card">
        <h3>🤖 AI Predictions</h3>
        <p>
        AI-based stock prediction system using
        machine learning and LSTM models.
        </p>
    </div>
    """, unsafe_allow_html=True)

with f3:

    st.markdown("""
    <div class="feature-card">
        <h3>💬 AI Trading Chatbot</h3>
        <p>
        Ask trading and investment questions
        with real-time AI assistance.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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

st.write(stocks)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- NAVIGATION ---------------- #

st.subheader("🚀 Start Trading")

c1, c2 = st.columns(2)

with c1:

    st.page_link(
        "pages/trading.py",
        label="📈 Open Trading Dashboard",
        icon="📈"
    )

with c2:

    st.page_link(
        "pages/demo.py",
        label="🧪 Open Demo Trading",
        icon="🧪"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- REFERENCES ---------------- #

st.subheader("🌐 Market References")

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

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<center>

Made with ❤️ using Streamlit + AI/ML

</center>
""", unsafe_allow_html=True)