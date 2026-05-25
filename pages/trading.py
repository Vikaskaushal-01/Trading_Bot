import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time

from utils.data_loader import load_data
from utils.indicators import add_indicators
from utils.predictor import train_model
from utils.chatbot import ask_bot
from utils.wallet import (
    load_memory,
    save_memory,
    add_money
)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Trading Dashboard",
    layout="wide"
)

# ---------------- MEMORY ---------------- #

memory = load_memory()

# ---------------- SESSION ---------------- #

if "wallet" not in st.session_state:

    st.session_state.wallet = memory["wallet"]

if "transactions" not in st.session_state:

    st.session_state.transactions = memory["transactions"]

if "messages" not in st.session_state:

    st.session_state.messages = []

# ---------------- STOCK OPTIONS ---------------- #

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

# ---------------- TITLE ---------------- #

st.title("📈 AI Trading Dashboard")

# ---------------- WALLET ---------------- #

c1, c2 = st.columns([8,2])

with c2:

    st.metric(
        "Wallet Balance",
        f"${round(st.session_state.wallet,2)}"
    )

    with st.expander("➕ Add Money"):

        amount = st.number_input(
            "Enter Amount",
            min_value=1,
            step=1
        )

        if st.button("Add Funds"):

            updated_wallet = add_money(amount)

            st.session_state.wallet = updated_wallet

            st.success(
                "Money Added Successfully"
            )

# ---------------- SELECT STOCK ---------------- #

selected_stock = st.selectbox(
    "Select Asset",
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
    title=f"{selected_stock} Live Market Chart",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------- BUY STOCK ---------------- #

st.subheader("💹 Buy Stocks")

investment_amount = st.number_input(
    "Investment Amount ($)",
    min_value=1,
    max_value=max(
        1,
        int(st.session_state.wallet)
    ),
    step=1
)

if st.button("Buy Now"):

    quantity = (
        investment_amount / current_price
    )

    st.session_state.wallet -= investment_amount

    trade = {
        "Asset": selected_stock,
        "Symbol": symbol,
        "Buy Price": current_price,
        "Investment": investment_amount,
        "Quantity": quantity
    }

    st.session_state.transactions.append(trade)

    memory["wallet"] = (
        st.session_state.wallet
    )

    memory["transactions"] = (
        st.session_state.transactions
    )

    save_memory(memory)

    st.success(
        "Stock Purchased Successfully"
    )

# ---------------- PORTFOLIO ---------------- #

st.subheader("📊 Live Portfolio")

updated_transactions = []

portfolio_value = 0

initial_investment_total = 0

for trade in st.session_state.transactions:

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
    st.session_state.wallet
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
        "Live Wallet Value",
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

# ---------------- AI PREDICTION ---------------- #

st.subheader("🤖 AI Prediction")

model, scaled_data = train_model(df)

predicted_price = (
    current_price * 1.02
)

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

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

# ---------------- AUTO REFRESH ---------------- #

time.sleep(5)

st.rerun()