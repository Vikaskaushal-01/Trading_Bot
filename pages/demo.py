import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
import time

from utils.data_loader import load_data
from utils.indicators import add_indicators

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Demo Trading",
    layout="wide"
)

# ---------------- MEMORY FILE ---------------- #

MEMORY_FILE = "data/demo_memory.json"

# ---------------- LOAD MEMORY ---------------- #

def load_memory():

    if not os.path.exists("data"):

        os.makedirs("data")

    if not os.path.exists(MEMORY_FILE):

        default_data = {
            "demo_wallet": 500,
            "demo_transactions": []
        }

        with open(MEMORY_FILE, "w") as f:

            json.dump(default_data, f, indent=4)

    with open(MEMORY_FILE, "r") as f:

        return json.load(f)

# ---------------- SAVE MEMORY ---------------- #

def save_memory(data):

    with open(MEMORY_FILE, "w") as f:

        json.dump(data, f, indent=4)

memory = load_memory()

# ---------------- SESSION ---------------- #

if "demo_wallet" not in st.session_state:

    st.session_state.demo_wallet = (
        memory["demo_wallet"]
    )

if "demo_transactions" not in st.session_state:

    st.session_state.demo_transactions = (
        memory["demo_transactions"]
    )

# ---------------- TITLE ---------------- #

st.title("🧪 Demo Trading Platform")

# ---------------- STOCKS ---------------- #

stock_options = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "NVIDIA": "NVDA",
    "Bitcoin": "BTC-USD",
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS"
}

# ---------------- WALLET ---------------- #

c1, c2 = st.columns([8,2])

with c2:

    st.metric(
        "Demo Wallet",
        f"${round(st.session_state.demo_wallet,2)}"
    )

# ---------------- SELECT STOCK ---------------- #

selected_stock = st.selectbox(
    "Select Demo Stock",
    list(stock_options.keys())
)

symbol = stock_options[selected_stock]

# ---------------- LOAD DATA ---------------- #

df = load_data(symbol)

df = add_indicators(df)

current_price = float(
    df["Close"].iloc[-1]
)

previous_price = float(
    df["Close"].iloc[-2]
)

price_change = (
    current_price - previous_price
)

price_percent = (
    price_change / previous_price
) * 100

# ---------------- METRICS ---------------- #

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Current Price",
        f"${round(current_price,2)}",
        f"{round(price_percent,2)}%"
    )

with m2:

    st.metric(
        "RSI",
        round(df["RSI"].iloc[-1],2)
    )

with m3:

    st.metric(
        "MACD",
        round(df["MACD"].iloc[-1],2)
    )

# ---------------- CHART ---------------- #

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Price",
        line=dict(width=3)
    )
)

fig.update_layout(
    template="plotly_dark",
    height=600,
    title=f"{selected_stock} Demo Market Chart",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------- BUY STOCK ---------------- #

st.subheader("💹 Demo Buy Stocks")

investment_amount = st.number_input(
    "Demo Investment Amount ($)",
    min_value=1,
    max_value=max(
        1,
        int(st.session_state.demo_wallet)
    ),
    step=1
)

if st.button("Buy Demo Stock"):

    quantity = (
        investment_amount / current_price
    )

    st.session_state.demo_wallet -= (
        investment_amount
    )

    trade = {
        "Asset": selected_stock,
        "Symbol": symbol,
        "Buy Price": current_price,
        "Investment": investment_amount,
        "Quantity": quantity
    }

    st.session_state.demo_transactions.append(
        trade
    )

    memory["demo_wallet"] = (
        st.session_state.demo_wallet
    )

    memory["demo_transactions"] = (
        st.session_state.demo_transactions
    )

    save_memory(memory)

    st.success(
        "Demo Stock Purchased"
    )

# ---------------- PORTFOLIO ---------------- #

st.subheader("📊 Demo Portfolio")

updated_transactions = []

portfolio_value = 0

initial_investment_total = 0

for trade in st.session_state.demo_transactions:

    latest_df = load_data(
        trade["Symbol"]
    )

    latest_price = float(
        latest_df["Close"].iloc[-1]
    )

    pnl_percent = (
        (latest_price - trade["Buy Price"])
        / trade["Buy Price"]
    ) * 100

    current_value = (
        trade["Quantity"] * latest_price
    )

    profit_loss_amount = (
        current_value
        - trade["Investment"]
    )

    portfolio_value += current_value

    initial_investment_total += (
        trade["Investment"]
    )

    updated_transactions.append({
        "Asset": trade["Asset"],
        "Buy Price": round(
            trade["Buy Price"],2
        ),
        "Current Price": round(
            latest_price,2
        ),
        "Investment": round(
            trade["Investment"],2
        ),
        "Current Value": round(
            current_value,2
        ),
        "Profit/Loss %": round(
            pnl_percent,2
        ),
        "Profit/Loss Amount": round(
            profit_loss_amount,2
        )
    })

portfolio_df = pd.DataFrame(
    updated_transactions
)

st.dataframe(
    portfolio_df,
    use_container_width=True
)

# ---------------- LIVE WALLET ---------------- #

live_wallet = (
    st.session_state.demo_wallet
    + portfolio_value
)

total_profit_loss = (
    portfolio_value
    - initial_investment_total
)

profit_percent = 0

if initial_investment_total > 0:

    profit_percent = (
        total_profit_loss
        / initial_investment_total
    ) * 100

# ---------------- METRICS ---------------- #

x1, x2, x3 = st.columns(3)

with x1:

    st.metric(
        "Live Demo Wallet",
        f"${round(live_wallet,2)}"
    )

with x2:

    st.metric(
        "Total Profit/Loss",
        f"${round(total_profit_loss,2)}"
    )

with x3:

    st.metric(
        "Profit/Loss %",
        f"{round(profit_percent,2)}%"
    )

# ---------------- AUTO REFRESH ---------------- #

time.sleep(5)

st.rerun()