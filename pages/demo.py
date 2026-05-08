import streamlit as st
import plotly.graph_objects as go
from utils.data_loader import load_data

st.set_page_config(layout="wide")

if "demo_wallet" not in st.session_state:
    st.session_state.demo_wallet=500

if "demo_transactions" not in st.session_state:
    st.session_state.demo_transactions=[]

st.title("🧪 Demo Trading")

c1,c2=st.columns([8,2])

with c2:
    st.metric("Demo Wallet",f"${st.session_state.demo_wallet}")

symbol=st.selectbox("Select Demo Asset",["AAPL","TSLA","BTC-USD"])

df=load_data(symbol)

fig=go.Figure()

fig.add_trace(go.Scatter(x=df['Date'],y=df['Close'],mode='lines'))

fig.update_layout(height=500,template='plotly_dark')

st.plotly_chart(fig,use_container_width=True)

amount=st.number_input("Demo Investment",min_value=1,max_value=int(st.session_state.demo_wallet))

if st.button("Demo Buy"):
    st.session_state.demo_wallet-=amount

    st.session_state.demo_transactions.append({
        "Asset":symbol,
        "Amount":amount
    })

    st.success("Demo Trade Executed")

st.dataframe(st.session_state.demo_transactions,use_container_width=True)