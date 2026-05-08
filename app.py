import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="AI Trading Bot",layout="wide")

if "wallet" not in st.session_state:
    st.session_state.wallet=500

if "transactions" not in st.session_state:
    st.session_state.transactions=[]

st.markdown("""
<style>
.main{background:#0d1117;color:white}
.stButton button{background:#00c896;color:white;border-radius:10px;border:none;padding:10px 25px}
.hero{padding:50px;border-radius:20px;background:linear-gradient(135deg,#111827,#1f2937)}
.metric-card{padding:20px;border-radius:20px;background:#161b22}
</style>
""",unsafe_allow_html=True)

with st.sidebar:
    selected=option_menu(menu_title="AI Trading Bot",options=["Home","Trading","Demo"],icons=["house","graph-up","cpu"],default_index=0)

if selected=="Home":
    st.markdown("""
    <div class='hero'>
    <h1>AI Trading Bot Platform</h1>
    <p>Trade smarter using AI-powered insights, technical indicators, real-time charts, and predictive analysis.</p>
    </div>
    """,unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)

    with c1:
        st.markdown("<div class='metric-card'><h3>AI Predictions</h3><p>LSTM + Technical Analysis</p></div>",unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='metric-card'><h3>Real-Time Charts</h3><p>Live Financial Visualization</p></div>",unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='metric-card'><h3>Demo Trading</h3><p>Practice Without Risk</p></div>",unsafe_allow_html=True)

    st.subheader("Features")
    st.write("✔ AI Chatbot for Trading Questions")
    st.write("✔ Real-Time Market Data")
    st.write("✔ Portfolio Management")
    st.write("✔ Demo Wallet")
    st.write("✔ Technical Indicators")

    st.link_button("Yahoo Finance","https://finance.yahoo.com")
    st.link_button("TradingView","https://www.tradingview.com")

elif selected=="Trading":
    st.switch_page("pages/trading.py")

elif selected=="Demo":
    st.switch_page("pages/demo.py")