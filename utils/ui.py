"""Shared look & feel: CSS theme, HTML components, Plotly charts and the chat panel."""

import json
import html

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.config import (
    CHAT_FILE, DEMO_ACCOUNT_FILE, FEATURED, LIVE_ACCOUNT_FILE, UNIVERSE,
)
from utils.market import fmt_price, get_quotes, usd_price
from utils.portfolio import Account

GREEN, RED, ACCENT, CYAN, MUTED = "#10e3a5", "#ff5c7a", "#10e3a5", "#38bdf8", "#8b97ad"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
  --bg: #070b14; --surface: #0f1522; --surface-2: #141c2d; --border: #1f2a3f;
  --text: #e8edf7; --muted: #8b97ad; --green: #10e3a5; --red: #ff5c7a; --cyan: #38bdf8;
}
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(16,227,165,.08), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(56,189,248,.08), transparent 60%),
    var(--bg);
}
.block-container { padding-top: 2rem; max-width: 1400px; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #0a101c; border-right: 1px solid var(--border); }

.hero {
  position: relative; overflow: hidden;
  padding: 36px 40px; border-radius: 24px; margin-bottom: 24px;
  background: linear-gradient(135deg, #0f1a2e 0%, #0b1322 55%, #0d1f24 100%);
  border: 1px solid var(--border);
}
.hero:after {
  content: ""; position: absolute; right: -80px; top: -80px; width: 320px; height: 320px;
  background: radial-gradient(circle, rgba(16,227,165,.25), transparent 70%);
}
.hero-badge {
  display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 600;
  letter-spacing: .06em; text-transform: uppercase; color: var(--green);
  background: rgba(16,227,165,.1); border: 1px solid rgba(16,227,165,.3); margin-bottom: 14px;
}
.hero h1 { font-size: 40px; font-weight: 800; margin: 0; color: var(--text); letter-spacing: -.02em; }
.hero .hl { background: linear-gradient(90deg, var(--green), var(--cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: var(--muted); font-size: 17px; margin: 10px 0 0; max-width: 720px; line-height: 1.6; }

.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-bottom: 18px; }
.kpi {
  background: linear-gradient(180deg, var(--surface-2), var(--surface));
  border: 1px solid var(--border); border-radius: 18px; padding: 18px 20px;
}
.kpi-label { color: var(--muted); font-size: 13px; font-weight: 500; display: flex; gap: 6px; align-items: center; }
.kpi-value { color: var(--text); font-size: 26px; font-weight: 700; margin-top: 6px; font-family: 'JetBrains Mono', monospace; letter-spacing: -.02em; }
.kpi-delta { font-size: 13px; font-weight: 600; margin-top: 4px; }

.qcard {
  background: linear-gradient(180deg, var(--surface-2), var(--surface));
  border: 1px solid var(--border); border-radius: 18px; padding: 16px 18px 10px;
  transition: border-color .2s, transform .2s;
}
.qcard:hover { border-color: rgba(16,227,165,.5); transform: translateY(-2px); }
.qcard-top { display: flex; justify-content: space-between; align-items: baseline; }
.qcard-name { color: var(--text); font-weight: 600; font-size: 15px; }
.qcard-sym { color: var(--muted); font-size: 12px; font-family: 'JetBrains Mono', monospace; }
.qcard-price { color: var(--text); font-weight: 700; font-size: 22px; margin-top: 8px; font-family: 'JetBrains Mono', monospace; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 8px; font-size: 12px; font-weight: 700; }
.pill.up { color: var(--green); background: rgba(16,227,165,.12); }
.pill.down { color: var(--red); background: rgba(255,92,122,.12); }

.section { color: var(--text); font-size: 20px; font-weight: 700; margin: 26px 0 12px; display: flex; align-items: center; gap: 10px; }
.section small { color: var(--muted); font-weight: 500; font-size: 13px; }

.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; padding: 18px 20px; }
.signal { display:inline-block; font-weight: 800; font-size: 28px; letter-spacing: .04em; padding: 6px 18px; border-radius: 12px; }
.signal.BUY { color: var(--green); background: rgba(16,227,165,.12); border: 1px solid rgba(16,227,165,.35); }
.signal.SELL { color: var(--red); background: rgba(255,92,122,.12); border: 1px solid rgba(255,92,122,.35); }
.signal.HOLD { color: #fbbf24; background: rgba(251,191,36,.10); border: 1px solid rgba(251,191,36,.35); }
.meter { height: 8px; border-radius: 99px; background: linear-gradient(90deg, var(--red), #fbbf24, var(--green)); position: relative; margin: 14px 0 6px; }
.meter i { position: absolute; top: -5px; width: 4px; height: 18px; border-radius: 3px; background: #fff; box-shadow: 0 0 8px rgba(255,255,255,.7); }
.muted { color: var(--muted); font-size: 13px; }
.disclaimer { color: var(--muted); font-size: 12px; text-align: center; margin-top: 40px; padding: 18px; border-top: 1px solid var(--border); }

div[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 14px 18px; }
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius: 12px; font-weight: 600; border: 1px solid var(--border); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(135deg, #10e3a5, #0ea5e9); color: #04111a; border: none;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }
div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; }
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title_html: str, subtitle: str, badge: str = ""):
    badge_html = f'<div class="hero-badge">{badge}</div>' if badge else ""
    st.markdown(f'<div class="hero">{badge_html}<h1>{title_html}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(title: str, note: str = ""):
    st.markdown(f'<div class="section">{title} <small>{note}</small></div>', unsafe_allow_html=True)


def signed_money(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def md_escape(text: str) -> str:
    """Stop Streamlit markdown from treating $...$ as LaTeX."""
    return text.replace("$", r"\$")


def color_for(x: float) -> str:
    return GREEN if x >= 0 else RED


def kpis(items: list[dict]):
    """items: {label, value, delta?(str), delta_value?(float), icon?}"""
    cells = []
    for it in items:
        delta = ""
        if it.get("delta"):
            c = color_for(it.get("delta_value", 0))
            delta = f'<div class="kpi-delta" style="color:{c}">{it["delta"]}</div>'
        cells.append(
            f'<div class="kpi"><div class="kpi-label">{it.get("icon", "")} {it["label"]}</div>'
            f'<div class="kpi-value">{it["value"]}</div>{delta}</div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def sparkline(values: list, width=220, height=46) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1
    step = width / (len(values) - 1)
    pts = [(i * step, height - 4 - (v - lo) / rng * (height - 8)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    color = color_for(values[-1] - values[0])
    gid = f"g{abs(hash(line)) % 10**8}"
    area = f"0,{height} {line} {width},{height}"
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{color}" stop-opacity=".35"/><stop offset="1" stop-color="{color}" stop-opacity="0"/>'
        f'</linearGradient></defs><polygon points="{area}" fill="url(#{gid})"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/></svg>'
    )


def quote_card(q: dict):
    cls = "up" if q["change_pct"] >= 0 else "down"
    arrow = "▲" if q["change_pct"] >= 0 else "▼"
    st.markdown(
        f'<div class="qcard"><div class="qcard-top"><span class="qcard-name">{html.escape(q["name"])}</span>'
        f'<span class="qcard-sym">{q["symbol"]}</span></div>'
        f'<div class="qcard-price">{fmt_price(q["price"], q["currency"])}</div>'
        f'<span class="pill {cls}">{arrow} {q["change_pct"]:+.2f}%</span>'
        f'{sparkline(q["spark"])}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c9d3e3"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1a2336", zeroline=False, showspikes=True, spikecolor="#33415c", spikethickness=1)
    fig.update_yaxes(gridcolor="#1a2336", zeroline=False)
    return fig


def price_chart(df, name: str, style: str, overlays: list, currency: str, levels: list | None = None) -> go.Figure:
    """levels: [{y, label, color}] horizontal price lines (cost basis, stops, targets, limit orders)."""
    show_rsi, show_macd = "RSI" in overlays, "MACD" in overlays
    rows = 2 + show_rsi + show_macd
    heights = [0.58, 0.14] + [0.14] * (show_rsi + show_macd)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=heights)

    if style == "Candles":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=name,
            increasing=dict(line=dict(color=GREEN), fillcolor=GREEN),
            decreasing=dict(line=dict(color=RED), fillcolor=RED),
        ), row=1, col=1)
    else:
        c = color_for(df["Close"].iloc[-1] - df["Close"].iloc[0])
        fill = "rgba(16,227,165,0.10)" if c == GREEN else "rgba(255,92,122,0.10)"
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=name, line=dict(color=c, width=2.2),
                                 fill="tozeroy", fillcolor=fill), row=1, col=1)
        fig.update_yaxes(range=[df["Low"].min() * 0.98, df["High"].max() * 1.02], row=1, col=1)

    if "EMA 20" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20", line=dict(color="#fbbf24", width=1.3)), row=1, col=1)
    if "EMA 50" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50", line=dict(color=CYAN, width=1.3)), row=1, col=1)
    if "SMA 200" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA 200", line=dict(color="#c084fc", width=1.3)), row=1, col=1)
    if "Bollinger" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB upper", line=dict(color="#64748b", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB lower", line=dict(color="#64748b", width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(100,116,139,0.08)"), row=1, col=1)

    vol_colors = [GREEN if c >= o else RED for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, opacity=0.5, showlegend=False), row=2, col=1)

    r = 3
    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#a78bfa", width=1.5)), row=r, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(167,139,250,0.07)", line_width=0, row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1, title_text="RSI")
        r += 1
    if show_macd:
        hist_colors = [GREEN if v >= 0 else RED for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=hist_colors, opacity=0.6, name="MACD hist", showlegend=False), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color=CYAN, width=1.3)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="#fbbf24", width=1.3)), row=r, col=1)
        fig.update_yaxes(title_text="MACD", row=r, col=1)

    lo, hi = df["Low"].min(), df["High"].max()
    for lvl in levels or []:
        if lvl["y"] and lo * 0.6 <= lvl["y"] <= hi * 1.6:  # skip lines far outside the visible range
            fig.add_hline(y=lvl["y"], line=dict(color=lvl["color"], width=1.2, dash="dash"), row=1, col=1,
                          annotation_text=f'{lvl["label"]} {lvl["y"]:,.2f}', annotation_position="top left",
                          annotation_font=dict(color=lvl["color"], size=11))

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text=currency, row=1, col=1)
    # Hide weekend gaps for exchange-traded assets.
    if not name.endswith("-USD") and len(df) > 5 and (df.index.dayofweek < 5).all():
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return style_fig(fig, height=420 + 130 * (rows - 1))


# ---------------- accounts & assistant context ---------------- #

def accounts() -> dict[str, Account]:
    return {
        "Live": Account(LIVE_ACCOUNT_FILE, "Live"),
        "Demo": Account(DEMO_ACCOUNT_FILE, "Demo"),
    }


def held_symbols(accts: dict) -> list:
    syms = set()
    for a in accts.values():
        syms.update(a.positions)
        syms.update(a.watchlist)
        syms.update(o["symbol"] for o in a.orders)
    return sorted(syms)


def all_quotes(extra=()) -> dict:
    syms = tuple(sorted(set(UNIVERSE) | set(extra)))
    return get_quotes(syms)


def usd_prices(quotes: dict) -> dict:
    return {s: usd_price(s, q["price"]) for s, q in quotes.items()}


def assistant_context(selected: str | None = None) -> dict:
    accts = accounts()
    quotes = all_quotes(held_symbols(accts))
    prices = usd_prices(quotes)
    ctx = {
        "accounts": {},
        "quotes": [
            {"symbol": s, "name": q["name"], "price_fmt": fmt_price(q["price"], q["currency"]),
             "change_pct": round(q["change_pct"], 2)}
            for s, q in quotes.items()
        ],
        "featured": FEATURED,
        "selected_asset": selected,
    }
    for name, a in accts.items():
        v = a.valuation(prices)
        ctx["accounts"][name] = {
            "cash": round(v["cash"], 2), "equity": round(v["equity"], 2),
            "total_pnl": round(v["total_pnl"], 2), "realized_pnl": round(v["realized"], 2),
            "positions": [
                {"name": r["name"], "symbol": r["symbol"], "qty": r["qty"], "value": round(r["value"], 2),
                 "pnl_pct": round(r["pnl_pct"], 2)} for r in v["rows"]
            ],
            "recent_trades": a.history[-5:],
        }
    if selected:
        from utils.predictor import describe, predict_symbol
        from utils.indicators import add_indicators, technical_rating
        from utils.market import get_history
        pred = predict_symbol(selected)
        if pred:
            ctx["selected_ai_signal"] = {
                "horizon": pred["settings"]["horizon"], "model": pred["settings"]["model"],
                "signal": pred["signal"], "prob_up_calibrated": round(pred["prob_up"], 3),
                "typical_up_rate": round(pred["base_rate"], 3),
                "reliability": pred["reliability"], "oos_roc_auc": round(pred["auc"], 3),
                "expected_return_median": round(pred["ret_q50"], 4),
                "expected_return_10_90": [round(pred["ret_q10"], 4), round(pred["ret_q90"], 4)],
                "similar_setups_up_rate": round(pred["analog_up"], 2),
                "composite": pred["composite"]["label"], "composite_score": pred["composite"]["score"],
                "top_drivers": [describe(f)[1] for f in pred["relevance"].index[:5]],
            }
        hist = get_history(selected, "1y")
        if len(hist) > 30:
            df = add_indicators(hist)
            rating = technical_rating(df)
            last = df.iloc[-1]
            ctx["selected_technicals"] = {
                "rating": rating["label"], "score": rating["score"],
                "RSI": round(float(last["RSI"]), 1), "MACD": round(float(last["MACD"]), 3),
                "close": round(float(last["Close"]), 2),
            }
    return ctx


def load_chat() -> list:
    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_chat(messages: list):
    CHAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(messages[-100:], f, indent=2)


def disclaimer():
    st.markdown(
        '<div class="disclaimer">Paper trading only — no real money is moved. '
        'AI signals are statistical estimates for education, not financial advice.</div>',
        unsafe_allow_html=True,
    )
