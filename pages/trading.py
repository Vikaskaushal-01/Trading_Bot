import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
import time

from utils.data_loader import load_data
from utils.indicators import add_indicators
from utils.realtime_predictor import predict_stock_price
from utils.scraper import fetch_market_data
from utils.chatbot import ask_bot

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Live Trading",
    layout="wide"
)

# ---------------- MEMORY ---------------- #

MEMORY_FILE = "data/trading_memory.json"


def load_memory():

    os.makedirs(
        "data",
        exist_ok=True
    )

    default_data = {

        "wallet": 500,

        "transactions": [],

        "messages": []

    }

    # CREATE FILE
    if not os.path.exists(MEMORY_FILE):

        with open(MEMORY_FILE, "w") as f:

            json.dump(
                default_data,
                f,
                indent=4
            )

    try:

        with open(MEMORY_FILE, "r") as f:

            memory = json.load(f)

    except:

        memory = default_data

    # FIX MISSING KEYS
    for key in default_data:

        if key not in memory:

            memory[key] = default_data[key]

    # SAVE UPDATED MEMORY
    with open(MEMORY_FILE, "w") as f:

        json.dump(
            memory,
            f,
            indent=4
        )

    return memory


def save_memory(data):

    with open(MEMORY_FILE, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )


memory = load_memory()

# ---------------- SESSION ---------------- #

st.session_state.wallet = (
    memory["wallet"]
)

st.session_state.transactions = (
    memory["transactions"]
)

st.session_state.messages = (
    memory["messages"]
)

# ---------------- STOCK OPTIONS ---------------- #

stock_options = {

    "Apple": "AAPL",

    "Tesla": "TSLA",

    "Microsoft": "MSFT",

    "NVIDIA": "NVDA",

    "Google": "GOOGL",

    "Bitcoin": "BTC-USD",

    "Reliance Industries": "RELIANCE.NS",

    "HDFC Bank": "HDFCBANK.NS",

    "ICICI Bank": "ICICIBANK.NS",

    "Bharti Airtel": "BHARTIARTL.NS"

}

# ---------------- TITLE ---------------- #

st.title("📈 AI Live Trading Platform")

# ---------------- WALLET ---------------- #

wallet_col1, wallet_col2 = st.columns([8,2])

with wallet_col2:

    st.markdown(f"""
    <div style="
        background:#111827;
        padding:25px;
        border-radius:18px;
        border:1px solid #30363d;
    ">

    <h3 style="color:white;">
    Wallet Balance
    </h3>

    <h1 style="color:#00d4aa;">
    ${round(st.session_state.wallet,2)}
    </h1>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    add_money = st.number_input(
        "Add Funds",
        min_value=1,
        step=1
    )

    if st.button("➕ Add Money"):

        st.session_state.wallet += (
            add_money
        )

        memory["wallet"] = (
            st.session_state.wallet
        )

        save_memory(memory)

        st.success(
            "Money Added"
        )

        st.rerun()

# ---------------- STOCK SELECT ---------------- #

selected_stock = st.selectbox(
    "Select Stock",
    list(stock_options.keys())
)

symbol = stock_options[selected_stock]

# ---------------- LOAD DATA ---------------- #

df = load_data(symbol)

df = add_indicators(df)

# ---------------- CURRENT DATA ---------------- #

current_price = float(
    df["Close"].iloc[-1]
)

previous_price = float(
    df["Close"].iloc[-2]
)

volume = float(
    df["Volume"].iloc[-1]
)

price_percent = (
    (
        current_price
        - previous_price
    )
    / previous_price
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

# ---------------- GRAPH ---------------- #

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["Close"],

        mode="lines",

        name="Price",

        line=dict(
            width=3
        )

    )

)

fig.update_layout(

    template="plotly_dark",

    height=600,

    title=f"{selected_stock} Live Market Chart",

    xaxis_title="Time",

    yaxis_title="Price"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------- MARKET FEED ---------------- #

st.subheader("🌍 Real-Time Market Feed")

market_data = fetch_market_data()

if len(market_data) > 0:

    market_df = pd.DataFrame(
        market_data[-10:]
    )

    st.dataframe(
        market_df,
        use_container_width=True
    )

# ---------------- BUY SECTION ---------------- #

st.subheader("💹 Buy Stocks")

investment = st.number_input(

    "Investment Amount",

    min_value=1,

    max_value=max(
        1,
        int(st.session_state.wallet)
    ),

    step=1

)

if st.button("🛒 Buy Stock"):

    quantity = (
        investment
        / current_price
    )

    st.session_state.wallet -= (
        investment
    )

    trade = {

        "Asset": selected_stock,

        "Symbol": symbol,

        "Buy Price": current_price,

        "Investment": investment,

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

initial_investment = 0

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

    portfolio_value += (
        current_value
    )

    initial_investment += (
        trade["Investment"]
    )

    c1, c2 = st.columns([5,1])

    with c1:

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
        Buy Price:
        ${round(trade['Buy Price'],2)}
        </p>

        <p style="color:#9ca3af;">
        Current Price:
        ${round(latest_price,2)}
        </p>

        <p style="color:#00d4aa;">
        Current Value:
        ${round(current_value,2)}
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

    with c2:

        st.write("")
        st.write("")
        st.write("")

        if st.button(
            "💰 Sell",
            key=f"sell_{index}"
        ):

            st.session_state.wallet += (
                current_value
            )

            st.session_state.transactions.pop(
                index
            )

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

# ---------------- LIVE STATS ---------------- #

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

    total_profit = (
        portfolio_value
        - initial_investment
    )

    st.metric(
        "📊 Total Profit/Loss",
        f"${round(total_profit,2)}"
    )

# ---------------- AI PREDICTION ---------------- #

st.subheader("🤖 AI Prediction")

predicted_price = predict_stock_price(
    price_percent,
    volume
)

if predicted_price:

    prediction_percent = (
        (
            predicted_price
            - current_price
        )
        / current_price
    ) * 100

    prediction_color = (
        "green"
        if prediction_percent >= 0
        else "red"
    )

    st.markdown(f"""
    <div style="
        background:#111827;
        padding:25px;
        border-radius:18px;
        border:1px solid #30363d;
    ">

    <h2 style="color:white;">
    AI Predicted Price
    </h2>

    <h1 style="
        color:#00d4aa;
        font-size:48px;
    ">
    ${round(predicted_price,2)}
    </h1>

    <h3 style="
        color:{prediction_color};
    ">
    {round(prediction_percent,2)}%
    </h3>

    </div>
    """, unsafe_allow_html=True)

else:

    st.warning(
        "AI model is training..."
    )

# ---------------- CHATBOT ---------------- #

st.markdown("""
<div style="
    background:#111827;
    padding:25px;
    border-radius:18px;
    border:1px solid #30363d;
    margin-top:30px;
">

<h1 style="
    color:white;
    margin-bottom:25px;
">
💬 AI Trading Chatbot
</h1>

</div>
""", unsafe_allow_html=True)

# CHAT INPUT
user_input = st.chat_input(
    placeholder="Ask trading-related questions..."
)

# USER MESSAGE
if user_input:

    st.session_state.messages.append({

        "role": "user",

        "content": user_input

    })

    # AI RESPONSE
    response = ask_bot(
        user_input
    )

    st.session_state.messages.append({

        "role": "assistant",

        "content": response

    })

    # SAVE MEMORY
    memory["messages"] = (
        st.session_state.messages
    )

    save_memory(memory)

    st.rerun()

# DISPLAY CHAT
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(
            msg["content"]
        )

# ---------------- AUTO REFRESH ---------------- #

time.sleep(10)

st.rerun()