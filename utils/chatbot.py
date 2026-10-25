"""AI trading assistant: instant answers from platform data, Groq LLM for everything else."""

import json
import os
import re

from dotenv import load_dotenv

from utils.config import ROOT

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are the AI assistant of an AI paper-trading platform.

You are given LIVE PLATFORM DATA below (account balances, holdings, market quotes,
the currently selected asset and its AI/technical signals). Always ground answers in
that data first, quoting actual numbers. Never claim you cannot see the user's data.

Scope: stocks, crypto, trading, investing, portfolio management, technical analysis,
market data and risk management. Politely steer unrelated questions back to trading.

Be concise and practical; use short bullet points and markdown tables where helpful.
Remind the user that signals are educational, not financial advice, when recommending trades.
"""

_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            return None
        from groq import Groq
        _client = Groq(api_key=key)
    return _client


def llm_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _has(q: str, *words) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", q) for w in words)


def _quote_lines(quotes):
    return "\n".join(
        f"- **{x['name']}** ({x['symbol']}): {x['price_fmt']} ({x['change_pct']:+.2f}%)" for x in quotes
    )


def local_answer(question: str, ctx: dict) -> str | None:
    """Answer simple data lookups instantly; return None to defer to the LLM."""
    q = question.lower().strip()
    quotes = ctx.get("quotes", [])
    accounts = ctx.get("accounts", {})

    if _has(q, "balance", "wallet", "cash") and len(q.split()) <= 8:
        lines = [
            f"- **{name}** — cash ${a['cash']:,.2f}, net worth ${a['equity']:,.2f} "
            f"(P&L ${a['total_pnl']:+,.2f})"
            for name, a in accounts.items()
        ]
        return "Here are your account balances:\n\n" + "\n".join(lines)

    if _has(q, "portfolio", "holdings", "positions") and len(q.split()) <= 8:
        parts = []
        for name, a in accounts.items():
            if not a["positions"]:
                parts.append(f"**{name}:** no open positions.")
                continue
            rows = "\n".join(
                f"| {p['name']} | {p['qty']:.4f} | ${p['value']:,.2f} | {p['pnl_pct']:+.2f}% |"
                for p in a["positions"]
            )
            parts.append(f"**{name}**\n\n| Asset | Qty | Value | P&L |\n|---|---|---|---|\n{rows}")
        return "\n\n".join(parts)

    if not quotes:
        return None

    if _has(q, "gainers", "top gainers", "winners") and len(q.split()) <= 6:
        up = sorted([x for x in quotes if x["change_pct"] > 0], key=lambda x: -x["change_pct"])
        return ("Today's gainers:\n\n" + _quote_lines(up)) if up else "No assets are up today."

    if _has(q, "losers", "top losers", "decliners") and len(q.split()) <= 6:
        down = sorted([x for x in quotes if x["change_pct"] < 0], key=lambda x: x["change_pct"])
        return ("Today's losers:\n\n" + _quote_lines(down)) if down else "No assets are down today."

    if _has(q, "trending", "market overview", "watchlist") and len(q.split()) <= 6:
        return "Market snapshot:\n\n" + _quote_lines(sorted(quotes, key=lambda x: -x["change_pct"]))

    return None


def stream_answer(question: str, ctx: dict, history: list):
    """Yield the assistant reply in chunks (Groq streaming)."""
    client = _get_client()
    if client is None:
        yield (
            "⚠️ The AI model isn't configured. Add `GROQ_API_KEY` to your `.env` file to enable "
            "free-form answers. Meanwhile I can answer: *balance*, *portfolio*, *gainers*, *losers*, *trending*."
        )
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nLIVE PLATFORM DATA:\n" + json.dumps(ctx, indent=1, default=str)}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
    messages.append({"role": "user", "content": question})

    try:
        stream = client.chat.completions.create(
            model=os.getenv("MODEL_NAME") or DEFAULT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=900,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"\n\n⚠️ Assistant error: {e}"


def ask_bot(question: str, ctx: dict, history: list):
    """Return either a ready string (instant answer) or a generator for streaming."""
    return local_answer(question, ctx) or stream_answer(question, ctx, history)
