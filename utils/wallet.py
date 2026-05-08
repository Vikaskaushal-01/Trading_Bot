import streamlit as st

if "wallet" not in st.session_state:
    st.session_state.wallet=500

def add_money(amount):
    st.session_state.wallet+=amount