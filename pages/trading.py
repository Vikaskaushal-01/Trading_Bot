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

buy_col1, buy_col2 = st.columns([3,1])

with buy_col1:

    if st.button(
        "🛒 Buy Now",
        use_container_width=True
    ):

        if investment_amount <= st.session_state.wallet:

            quantity = (
                investment_amount
                / current_price
            )

            st.session_state.wallet -= (
                investment_amount
            )

            trade = {
                "Asset": selected_stock,
                "Symbol": symbol,
                "Buy Price": current_price,
                "Investment": investment_amount,
                "Quantity": quantity
            }

            st.session_state.transactions.append(
                trade
            )

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

            st.rerun()

# ---------------- PORTFOLIO ---------------- #

st.subheader("📊 Live Portfolio")

portfolio_value = 0

initial_investment_total = 0

updated_transactions = []

for index, trade in enumerate(
    st.session_state.transactions
):

    latest_df = load_data(
        trade["Symbol"]
    )

    latest_price = float(
        latest_df["Close"].iloc[-1]
    )

    current_value = (
        trade["Quantity"]
        * latest_price
    )

    pnl_percent = (
        (
            latest_price
            - trade["Buy Price"]
        )
        / trade["Buy Price"]
    ) * 100

    profit_amount = (
        current_value
        - trade["Investment"]
    )

    portfolio_value += current_value

    initial_investment_total += (
        trade["Investment"]
    )

    # ---------------- PORTFOLIO CARD ---------------- #

    card_col1, card_col2 = st.columns([5,1])

    with card_col1:

        st.markdown(f"""
        <div style="
            background:#111827;
            padding:20px;
            border-radius:18px;
            border:1px solid #30363d;
            margin-bottom:15px;
        ">

        <h3 style="color:white;">
        {trade['Asset']}
        </h3>

        <p style="color:#9ca3af;">
        Buy Price: ${round(trade['Buy Price'],2)}
        </p>

        <p style="color:#9ca3af;">
        Current Price: ${round(latest_price,2)}
        </p>

        <p style="color:#9ca3af;">
        Investment: ${round(trade['Investment'],2)}
        </p>

        <p style="color:#00d4aa;">
        Current Value: ${round(current_value,2)}
        </p>

        <p style="
            color:{
                '#00ff99'
                if pnl_percent >= 0
                else '#ff4d4d'
            };
            font-weight:bold;
        ">
        Profit/Loss:
        {round(pnl_percent,2)}%
        </p>

        </div>
        """, unsafe_allow_html=True)

    with card_col2:

        st.write("")
        st.write("")
        st.write("")

        if st.button(
            f"💰 Sell",
            key=f"sell_{index}",
            use_container_width=True
        ):

            # ADD CURRENT VALUE BACK
            st.session_state.wallet += (
                current_value
            )

            # REMOVE STOCK
            st.session_state.transactions.pop(
                index
            )

            # SAVE MEMORY
            memory["wallet"] = (
                st.session_state.wallet
            )

            memory["transactions"] = (
                st.session_state.transactions
            )

            save_memory(memory)

            st.success(
                "Stock Sold Successfully"
            )

            st.rerun()

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
            profit_amount,2
        )
    })

# ---------------- LIVE METRICS ---------------- #

live_wallet = (
    st.session_state.wallet
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

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.metric(
        "💰 Available Wallet",
        f"${round(st.session_state.wallet,2)}"
    )

with metric2:

    st.metric(
        "📈 Portfolio Value",
        f"${round(portfolio_value,2)}"
    )

with metric3:

    st.metric(
        "📊 Total Profit/Loss",
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

# SHOW ONLY REAL CHAT MESSAGES
if len(st.session_state.messages) > 0:

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

# USER INPUT
user_input = st.chat_input(
    "Ask trading-related questions..."
)

if user_input:

    # USER MESSAGE
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # AI RESPONSE
    response = ask_bot(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()

# ---------------- AUTO REFRESH ---------------- #

time.sleep(5)

st.rerun()