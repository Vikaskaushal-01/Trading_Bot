"""Reusable chat panel backed by utils.chatbot."""

import streamlit as st

from utils.chatbot import ask_bot, llm_available
from utils.ui import assistant_context, load_chat, md_escape, save_chat

SUGGESTIONS = [
    "What's my balance?",
    "Show my portfolio",
    "Top gainers today",
    "Should I buy the selected asset?",
    "How can I reduce risk in my portfolio?",
    "Explain RSI and MACD simply",
]


def chat_panel(selected: str | None = None, key: str = "main", height: int = 520):
    if "chat" not in st.session_state:
        st.session_state.chat = load_chat()
    messages = st.session_state.chat

    top_l, top_r = st.columns([4, 1])
    with top_l:
        status = "🟢 AI model connected" if llm_available() else "🟡 Offline mode — add GROQ_API_KEY to .env"
        st.caption(status)
    with top_r:
        if st.button("Clear chat", key=f"clear_{key}", width="stretch"):
            st.session_state.chat = []
            save_chat([])
            st.rerun()

    def _pick():
        st.session_state[f"q_{key}"] = st.session_state[f"sugg_{key}"]
        st.session_state[f"sugg_{key}"] = None

    st.pills("Try asking", SUGGESTIONS, key=f"sugg_{key}", on_change=_pick, label_visibility="collapsed")
    prompt = st.session_state.pop(f"q_{key}", None)
    box = st.container(height=height, border=True)
    with box:
        if not messages:
            st.markdown('<div class="muted">Ask anything about your portfolio, the market or a strategy.</div>', unsafe_allow_html=True)
        for m in messages:
            with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
                st.markdown(md_escape(m["content"]))

    typed = st.chat_input("Ask the AI trading assistant…", key=f"input_{key}")
    question = typed or prompt
    if not question:
        return

    with box:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(md_escape(question))
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                ctx = assistant_context(selected)
            reply = ask_bot(question, ctx, messages)
            if isinstance(reply, str):
                st.markdown(md_escape(reply))
            else:
                streamed = st.write_stream(md_escape(chunk) for chunk in reply)
                reply = streamed.replace("\\$", "$")

    messages += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
    save_chat(messages)
    st.rerun()
