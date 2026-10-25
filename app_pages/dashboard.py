import pandas as pd
import streamlit as st

from utils.config import FEATURED, UNIVERSE
from utils.predictor import get_settings, predict_symbol
from utils.ui import (
    accounts, all_quotes, disclaimer, held_symbols, hero, kpis, quote_card, section, usd_prices,
)
from utils.market import fmt_price, fx_to_usd

hero(
    'Trade smarter with <span class="hl">AI-powered</span> insights',
    "Live prices across US tech, crypto and Indian markets, machine-learning signals, "
    "strategy backtesting and an AI assistant that knows your portfolio.",
    badge="AI Trading Platform",
)

accts = accounts()
quotes = all_quotes(held_symbols(accts))
prices = usd_prices(quotes)

# Fire any stop-loss / take-profit / limit orders that triggered since the last visit.
_native = {s: q["price"] for s, q in quotes.items()}
_fx = {s: fx_to_usd(q["currency"]) for s, q in quotes.items()}
for _acct in accts.values():
    for _event in _acct.process(_native, _fx):
        st.toast(f"{_acct.label}: {_event}", icon="⚡")
vals = {n: a.valuation(prices) for n, a in accts.items()}

if not quotes:
    st.error("Market data is unavailable right now. Check your internet connection and refresh.")
    st.stop()

up = sum(q["change_pct"] > 0 for q in quotes.values())
live, demo = vals["Live"], vals["Demo"]
kpis([
    {"icon": "🏦", "label": "Live net worth", "value": f"${live['equity']:,.2f}",
     "delta": f"{live['total_pnl']:+,.2f} ({live['total_pnl_pct']:+.2f}%)", "delta_value": live["total_pnl"]},
    {"icon": "🧪", "label": "Demo net worth", "value": f"${demo['equity']:,.2f}",
     "delta": f"{demo['total_pnl']:+,.2f} ({demo['total_pnl_pct']:+.2f}%)", "delta_value": demo["total_pnl"]},
    {"icon": "💵", "label": "Cash available", "value": f"${live['cash'] + demo['cash']:,.2f}"},
    {"icon": "🌡️", "label": "Market breadth", "value": f"{up}/{len(quotes)} up",
     "delta": "risk-on" if up >= len(quotes) / 2 else "risk-off", "delta_value": up - len(quotes) / 2},
])


def go_trade(symbol: str, page: str = "app_pages/trading.py"):
    st.session_state.pending_symbol = symbol
    st.switch_page(page)


# ---------------- featured ---------------- #
section("🔥 Featured markets", "click to trade")
cols = st.columns(len(FEATURED))
for col, sym in zip(cols, FEATURED):
    with col:
        if sym in quotes:
            quote_card(quotes[sym])
            b1, b2 = st.columns(2)
            if b1.button("Trade", key=f"t_{sym}", type="primary", width="stretch"):
                go_trade(sym)
            if b2.button("Demo", key=f"d_{sym}", width="stretch"):
                go_trade(sym, "app_pages/demo.py")
        else:
            st.warning(f"{sym} unavailable")

# ---------------- movers ---------------- #
section("⚡ Today's movers")
ranked = sorted(quotes.values(), key=lambda q: q["change_pct"], reverse=True)
g, l = st.columns(2)


def mover_list(items, title):
    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #1a2336">'
        f'<span><b>{q["name"]}</b> <span class="muted">{q["symbol"]}</span></span>'
        f'<span style="font-family:JetBrains Mono">{fmt_price(q["price"], q["currency"])} '
        f'<span class="pill {"up" if q["change_pct"] >= 0 else "down"}">{q["change_pct"]:+.2f}%</span></span></div>'
        for q in items
    )
    st.markdown(f'<div class="panel"><div class="muted" style="margin-bottom:6px">{title}</div>{rows}</div>',
                unsafe_allow_html=True)


with g:
    mover_list(ranked[:5], "🚀 Top gainers")
with l:
    mover_list(ranked[::-1][:5], "🔻 Top losers")

# ---------------- full market table ---------------- #
section("🌍 Market overview")
categories = ["All"] + sorted({c for _, c, _ in UNIVERSE.values()})
cat = st.segmented_control("Category", categories, default="All", label_visibility="collapsed") or "All"
rows = [
    {"Asset": q["name"], "Symbol": s, "Market": UNIVERSE.get(s, ("", "Watchlist"))[1],
     "Price": fmt_price(q["price"], q["currency"]), "Change %": q["change_pct"],
     "Volume": q["volume"], "30-day trend": q["spark"]}
    for s, q in quotes.items()
]
table = pd.DataFrame(rows)
if cat != "All":
    table = table[table["Market"] == cat]
st.dataframe(
    table.sort_values("Change %", ascending=False), hide_index=True, width="stretch",
    column_config={
        "Change %": st.column_config.NumberColumn(format="%+.2f%%"),
        "Volume": st.column_config.NumberColumn(format="compact"),
        "30-day trend": st.column_config.AreaChartColumn(width="medium"),
    },
)

# ---------------- AI scanner ---------------- #
settings = get_settings()
section("🤖 AI signal scanner", f"{settings['model']} · {settings['horizon']} horizon · change in the sidebar")
if st.button("Scan all markets", type="primary") or st.session_state.get("scan_done"):
    st.session_state.scan_done = True
    progress = st.progress(0.0, text="Training models…")
    results = []
    for i, sym in enumerate(UNIVERSE):
        p = predict_symbol(sym, settings)
        progress.progress((i + 1) / len(UNIVERSE), text=f"Analysed {UNIVERSE[sym][0]}")
        if p:
            results.append({
                "Asset": UNIVERSE[sym][0], "Symbol": sym, "Signal": p["signal"],
                "Composite": p["composite"]["score"], "View": p["composite"]["label"],
                "P(up)": p["prob_up"] * 100, "Edge vs usual": (p["prob_up"] - p["base_rate"]) * 100,
                "Expected move": p["ret_q50"] * 100, "Reliability": p["reliability"], "AUC": p["auc"],
            })
    progress.empty()
    scan = pd.DataFrame(results).sort_values("Composite", ascending=False)
    st.dataframe(
        scan, hide_index=True, width="stretch",
        column_config={
            "Composite": st.column_config.ProgressColumn("Composite score", format="%+d", min_value=-100, max_value=100),
            "P(up)": st.column_config.NumberColumn("P(up)", format="%.1f%%"),
            "Edge vs usual": st.column_config.NumberColumn(format="%+.1f pts"),
            "Expected move": st.column_config.NumberColumn(f"Median {settings['horizon']} move", format="%+.2f%%"),
            "AUC": st.column_config.NumberColumn("OOS AUC", format="%.3f"),
        },
    )
    st.caption("Composite blends the AI (weighted by its reliability), the technical rating and historical analogs. "
               "Open an asset in the AI Lab for the full diagnosis.")
else:
    st.caption(f"Trains the model on all {len(UNIVERSE)} assets (~1–2 min first time, then cached for 30 min) "
               "and ranks them by composite score.")

section("🔗 Market references")
r1, r2, r3, r4 = st.columns(4)
r1.link_button("Yahoo Finance", "https://finance.yahoo.com", width="stretch")
r2.link_button("TradingView", "https://www.tradingview.com", width="stretch")
r3.link_button("Moneycontrol", "https://www.moneycontrol.com", width="stretch")
r4.link_button("CoinMarketCap", "https://coinmarketcap.com", width="stretch")

disclaimer()
