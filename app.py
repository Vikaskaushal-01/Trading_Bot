import streamlit as st

from utils.chatbot import llm_available
from utils.features import FEATURE_GROUPS
from utils.market import get_history, get_live_price, get_quotes
from utils.predictor import DEFAULT_SETTINGS, HISTORY, HORIZONS, MODELS
from utils.ui import inject_css

st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

nav = st.navigation({
    "Overview": [st.Page("app_pages/dashboard.py", title="Dashboard", icon=":material/space_dashboard:", default=True)],
    "Trade": [
        st.Page("app_pages/trading.py", title="Live Trading", icon=":material/candlestick_chart:", url_path="trading"),
        st.Page("app_pages/demo.py", title="Demo Trading", icon=":material/science:", url_path="demo"),
    ],
    "Research": [
        st.Page("app_pages/ai_lab.py", title="AI Lab", icon=":material/psychology:", url_path="ai-lab"),
        st.Page("app_pages/backtest.py", title="Backtester", icon=":material/query_stats:"),
        st.Page("app_pages/assistant.py", title="AI Assistant", icon=":material/smart_toy:"),
    ],
})

with st.sidebar:
    st.markdown(
        '<div style="font-size:22px;font-weight:800;margin:4px 0 2px">📈 Trade<span style="color:#10e3a5">AI</span></div>'
        '<div class="muted">Paper trading · ML signals · Backtests</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    with st.expander("🧠 AI model settings", expanded=False):
        d = DEFAULT_SETTINGS
        st.selectbox("Forecast horizon", list(HORIZONS), index=list(HORIZONS).index(d["horizon"]), key="ai_horizon",
                     help="How far ahead the model predicts. Longer horizons are smoother but react slower.")
        st.selectbox("Model", list(MODELS), index=list(MODELS).index(d["model"]), key="ai_model",
                     help="\n\n".join(f"**{k}** — {v}" for k, v in MODELS.items()))
        st.selectbox("Training history", list(HISTORY), index=list(HISTORY).index(d["history"]), key="ai_history",
                     help="More history = more examples, but older market regimes.")
        st.multiselect("Feature groups", FEATURE_GROUPS, default=list(d["groups"]), key="ai_groups",
                       help="Which families of indicators the model may learn from.")
        st.slider("Required edge for a signal", 0.01, 0.15, d["edge"], 0.01, key="ai_edge", format="%.2f",
                  help="BUY when P(up) beats the asset's usual up-rate by this much; SELL when it falls short by this much.")
        st.caption("Models retrain automatically and are cached for 30 min per setting.")
    st.caption("🟢 AI assistant online" if llm_available() else "🟡 AI assistant offline (set GROQ_API_KEY)")
    if st.button("↻ Refresh market data", width="stretch"):
        for fn in (get_quotes, get_history, get_live_price):
            fn.clear()
        st.rerun()

nav.run()
