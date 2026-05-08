import streamlit as st
import plotly.graph_objects as go
from utils.data_loader import load_data
from utils.indicators import add_indicators
from utils.predictor import train_model
from utils.chatbot import ask_bot
from utils.wallet import add_money

st.set_page_config(page_title="Trading Dashboard", layout="wide")

# Initialize session state
if "wallet" not in st.session_state:
    st.session_state.wallet = 500

if "transactions" not in st.session_state:
    st.session_state.transactions = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Page Title
st.title("📈 AI Trading Dashboard")

# Wallet Section
col1, col2 = st.columns([8, 2])

with col2:
    st.metric("Wallet Balance", f"${st.session_state.wallet}+")

    with st.expander("➕ Add Money"):

        amount_to_add = st.number_input(
            "Enter Amount",
            min_value=1,
            step=1
        )

        if st.button("Add Funds"):

            add_money(amount_to_add)

            st.success("Money Added Successfully")

# Stock Selection
symbol = st.selectbox(
    "Select Asset",
    ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA", "BTC-USD"]
)

# Load Stock Data
df = load_data(symbol)

# Add Indicators
df = add_indicators(df)

# Create Chart
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Price"
    )
)

fig.update_layout(
    template="plotly_dark",
    height=500,
    title=f"{symbol} Live Market Chart"
)

# Display Chart
st.plotly_chart(fig, use_container_width=True)

# Investment Section
st.subheader("💹 Investment Simulator")

investment_amount = st.number_input(
    "Investment Amount",
    min_value=1,
    max_value=int(st.session_state.wallet),
    step=1
)

if st.button("Invest Now"):

    st.session_state.wallet -= investment_amount

    st.session_state.transactions.append({
        "Asset": symbol,
        "Amount": investment_amount
    })

    st.success("Investment Successful")

# Transactions Table
st.subheader("📋 Live Investment Details")

st.dataframe(
    st.session_state.transactions,
    use_container_width=True
)

# AI Prediction Section
st.subheader("🤖 AI Prediction")

model, scaled = train_model(df)

latest_price = float(df["Close"].iloc[-1])

predicted_price = latest_price * 1.02

st.metric(
    "Predicted Next Price",
    f"${round(predicted_price, 2)}"
)

# Chatbot Section
st.subheader("💬 AI Trading Chatbot")

user_question = st.chat_input(
    "Ask trading-related questions..."
)

if user_question:

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    bot_response = ask_bot(user_question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response
    })

# Display Chat Messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])