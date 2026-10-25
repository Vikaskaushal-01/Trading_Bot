import streamlit as st

from utils.chat_view import chat_panel
from utils.config import UNIVERSE
from utils.ui import disclaimer, hero

hero(
    '💬 AI <span class="hl">Assistant</span>',
    "Ask about your balances, holdings, today's movers, strategies or any asset. "
    "Answers are grounded in your live platform data.",
    badge="Powered by Groq",
)

focus = st.selectbox(
    "Focus asset (adds its AI signal & technicals to the assistant's context)",
    [None] + list(UNIVERSE),
    format_func=lambda s: "None" if s is None else f"{UNIVERSE[s][0]} · {s}",
)
chat_panel(selected=focus, key="assistant", height=560)
disclaimer()
