import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
import time

from utils.data_loader import load_data
from utils.indicators import add_indicators
from utils.predictor import train_model
from utils.chatbot import ask_bot

MEMORY_FILE = "data/trading_memory.json"

# ---------------- MEMORY ---------------- #

def load_memory():

    if not os.path.exists(MEMORY_FILE):

        default_data = {
            "wallet": 500,
            "transactions": []
        }

        with open(MEMORY_FILE, "w") as f:
            json.dump(default_data, f)

    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(data):

    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=4)

memory = load_memory()

# ---------------- SESSION ---------------- #

if "wallet" not in st.session_state:
    st.session_state.wallet = memory["wallet"]

if "transactions" not in st.session_state:
    st.session_state.transactions = memory["transactions"]

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- PAGE ---------------- #

st.set_page_config(
    page_title="Advanced Trading Platform",
    layout="wide"
)

st.title("📈 AI Trading Platform")

# ---------------- WALLET ---------------- #

col1, col2 = st.columns([8,2])

with col2:

    st.metric(
        "Wallet",
        f"${round(st.session_state.wallet,2)}"
    )

    with st.expander("➕ Add Money"):

        add_amount = st.number_input(
            "Add Funds",
            min_value=1,
            step=1
        )

        if st.button("Add Funds"):

            st.session_state.wallet += add_amount

            memory["wallet"] = st.session_state.wallet

            save_memory(memory)

            st.success("Funds Added Successfully")

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

selected_stock = st.selectbox(
    "Select Stock",
    list(stock_options.keys())
)

symbol = stock_options[selected_stock]

# ---------------- DATA ---------------- #

df = load_data(symbol)

df = add_indicators(df)

current_price = float(df["Close"].iloc[-1])

previous_price = float(df["Close"].iloc[-2])

change_percent = (
    (current_price - previous_price)
    / previous_price
) * 100

# ---------------- METRICS ---------------- #

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Current Price",
        f"${round(current_price,2)}",
        f"{round(change_percent,2)}%"
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

# ---------------- DYNAMIC GRAPH ---------------- #

st.subheader("📊 Dynamic Stock Graph")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name=symbol,
        line=dict(width=3)
    )
)

fig.update_layout(
    template="plotly_dark",
    height=600,
    title=f"{selected_stock} Live Market Graph"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------- BUY ---------------- #

st.subheader("💹 Buy Stocks")

investment = st.number_input(
    "Investment Amount",
    min_value=1,
    max_value=int(st.session_state.wallet),
    step=1
)

if st.button("Buy Stock"):

    quantity = investment / current_price

    st.session_state.wallet -= investment

    trade = {
        "Asset": selected_stock,
        "Symbol": symbol,
        "Buy Price": current_price,
        "Quantity": quantity,
        "Investment": investment
    }

    st.session_state.transactions.append(trade)

    memory["wallet"] = st.session_state.wallet
    memory["transactions"] = st.session_state.transactions

    save_memory(memory)

    st.success("Stock Purchased Successfully")

# ---------------- SELL ---------------- #

st.subheader("📉 Sell Stocks")

owned_assets = [
    t["Asset"]
    for t in st.session_state.transactions
]

if owned_assets:

    sell_asset = st.selectbox(
        "Select Asset To Sell",
        owned_assets
    )

    if st.button("Sell Stock"):

        for trade in st.session_state.transactions:

            if trade["Asset"] == sell_asset:

                latest_df = load_data(trade["Symbol"])

                latest_price = float(
                    latest_df["Close"].iloc[-1]
                )

                sell_value = (
                    trade["Quantity"] * latest_price
                )

                st.session_state.wallet += sell_value

                st.session_state.transactions.remove(trade)

                memory["wallet"] = st.session_state.wallet
                memory["transactions"] = st.session_state.transactions

                save_memory(memory)

                st.success(
                    f"{sell_asset} Sold Successfully"
                )

                break

# ---------------- PORTFOLIO ---------------- #

st.subheader("📋 Portfolio History")

portfolio = []

portfolio_value = 0

initial_investment_total = 0

for trade in st.session_state.transactions:

    latest_df = load_data(trade["Symbol"])

    latest_price = float(
        latest_df["Close"].iloc[-1]
    )

    current_value = (
        trade["Quantity"] * latest_price
    )

    investment_value = trade["Investment"]

    pnl_percent = (
        (latest_price - trade["Buy Price"])
        / trade["Buy Price"]
    ) * 100

    profit_amount = (
        current_value - investment_value
    )

    portfolio_value += current_value

    initial_investment_total += investment_value

    portfolio.append({
        "Asset": trade["Asset"],
        "Buy Price": round(trade["Buy Price"],2),
        "Current Price": round(latest_price,2),
        "Investment": round(investment_value,2),
        "Current Value": round(current_value,2),
        "Profit/Loss %": round(pnl_percent,2),
        "Profit/Loss Amount": round(profit_amount,2)
    })

portfolio_df = pd.DataFrame(portfolio)

st.dataframe(
    portfolio_df,
    use_container_width=True
)

# ---------------- TOTAL PROFIT ---------------- #

st.metric(
    "Total Profit/Loss",
    f"${round(total_profit,2)}"
)

# ---------------- AI PREDICTION ---------------- #

st.subheader("🤖 AI Prediction")

model, scaled_data = train_model(df)

predicted_price = current_price * 1.02

prediction_percent = (
    (predicted_price - current_price)
    / current_price
) * 100

st.metric(
    "Predicted Next Price",
    f"${round(predicted_price,2)}",
    f"{round(prediction_percent,2)}%"
)

# ---------------- CHATBOT ---------------- #

st.subheader("💬 AI Trading Chatbot")

user_input = st.chat_input(
    "Ask trading-related questions..."
)

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    response = ask_bot(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

# ---------------- AUTO REFRESH ---------------- #

time.sleep(5)

st.rerun()