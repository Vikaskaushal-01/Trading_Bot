import plotly.graph_objects as go
import streamlit as st

from utils.backtest import STRATEGIES, run_backtest
from utils.config import UNIVERSE
from utils.market import get_history
from utils.predictor import get_settings, levels, predict_symbol
from utils.ui import GREEN, RED, disclaimer, hero, kpis, section, style_fig

hero(
    '🧪 Strategy <span class="hl">Backtester</span>',
    "Test trading strategies — including the AI model — on years of real price history, with stop-losses, "
    "take-profits and trailing stops, and compare them against simply buying and holding.",
    badge="Research lab",
)

c1, c2, c3, c4 = st.columns([1.6, 1.6, 0.8, 1])
symbol = c1.selectbox("Asset", list(UNIVERSE), format_func=lambda s: f"{UNIVERSE[s][0]} · {s}")
strategy = c2.selectbox("Strategy", list(STRATEGIES))
period = c3.selectbox("History", ["1y", "2y", "5y", "10y"], index=2)
capital = c4.number_input("Starting capital ($)", min_value=100.0, value=10_000.0, step=1000.0)
st.caption(STRATEGIES[strategy])

# ---------------- strategy parameters ---------------- #
params = {}
p1, p2, p3 = st.columns(3)
if strategy == "SMA Crossover":
    params["fast"] = p1.slider("Fast SMA", 5, 100, 20)
    params["slow"] = p2.slider("Slow SMA", 20, 250, 50)
    if params["fast"] >= params["slow"]:
        st.warning("The fast SMA should be shorter than the slow SMA.")
elif strategy == "RSI Mean Reversion":
    params["lower"] = p1.slider("Buy below RSI", 10, 45, 30)
    params["upper"] = p2.slider("Sell above RSI", 55, 90, 70)
elif strategy == "Donchian Breakout":
    params["entry"] = p1.slider("Entry: N-day high", 10, 100, 20)
    params["exit"] = p2.slider("Exit: M-day low", 5, 60, 10)
elif strategy == "AI Model":
    s = get_settings()
    with st.spinner("Training AI model (purged walk-forward)…"):
        pred = predict_symbol(symbol, s)
    if not pred:
        st.error("Not enough history to train the AI model for this asset.")
        st.stop()
    edge = p1.slider("Required edge (pts)", 1, 15, int(round(s["edge"] * 100))) / 100
    buy_level, sell_level = levels(pred, edge)
    params.update({"proba": pred["oos_proba"], "buy_level": buy_level, "sell_level": sell_level})
    p2.markdown(
        f'<div class="panel" style="font-size:13px">{s["model"]} · {s["horizon"]} horizon · reliability '
        f'<b>{pred["reliability"]}</b> (AUC {pred["auc"]:.3f})<br>BUY ≥ {buy_level * 100:.1f}% · SELL ≤ {sell_level * 100:.1f}%</div>',
        unsafe_allow_html=True)
    st.caption("Uses only out-of-sample probabilities, so the first training block (~1/6 of the model's history) is flat. "
               "Model settings live in the sidebar.")
fee_bps = p3.slider("Fee per trade (bps)", 0, 50, 10)

# ---------------- risk controls ---------------- #
with st.expander("🛡️ Risk controls (applied on top of the strategy)", expanded=True):
    r1, r2, r3, r4 = st.columns(4)
    stop_pct = r1.number_input("Stop-loss % (0 = off)", 0.0, 50.0, 0.0, 0.5,
                               help="Exit if price falls this far below entry (checked on the daily low; gaps fill at the open).")
    take_pct = r2.number_input("Take-profit % (0 = off)", 0.0, 200.0, 0.0, 1.0,
                               help="Exit if price rises this far above entry (checked on the daily high).")
    trail_pct = r3.number_input("Trailing stop % (0 = off)", 0.0, 50.0, 0.0, 0.5,
                                help="Exit if price falls this far below its highest close since entry.")
    cooldown = r4.number_input("Re-entry cooldown (days)", 0, 60, 5,
                               help="After a stop/target exit, wait this many days before re-entering if the strategy is still long. "
                                    "0 = wait for the strategy's own next entry signal.")

hist = get_history(symbol, period=period)
if len(hist) < 60:
    st.error("Not enough history to backtest this asset.")
    st.stop()

res = run_backtest(hist, strategy, params, fee=fee_bps / 10_000,
                   stop_pct=stop_pct or None, take_pct=take_pct or None, trail_pct=trail_pct or None,
                   cooldown=int(cooldown))
s_, h = res["strategy"], res["hold"]
pct = lambda v: "—" if v is None else f"{v * 100:.1f}%"

kpis([
    {"icon": "💰", "label": "Final value", "value": f"${capital * res['equity'].iloc[-1]:,.0f}",
     "delta": f"{s_['Total Return'] * 100:+.1f}% vs {h['Total Return'] * 100:+.1f}% hold",
     "delta_value": s_["Total Return"] - h["Total Return"]},
    {"icon": "📈", "label": "CAGR", "value": f"{s_['CAGR'] * 100:+.1f}%",
     "delta": f"hold {h['CAGR'] * 100:+.1f}%", "delta_value": s_["CAGR"] - h["CAGR"]},
    {"icon": "⚖️", "label": "Sharpe · Sortino", "value": f"{s_['Sharpe']:.2f} · {s_['Sortino']:.2f}",
     "delta": f"hold {h['Sharpe']:.2f} · {h['Sortino']:.2f}", "delta_value": s_["Sharpe"] - h["Sharpe"]},
    {"icon": "📉", "label": "Max drawdown", "value": f"{s_['Max Drawdown'] * 100:.1f}%",
     "delta": f"hold {h['Max Drawdown'] * 100:.1f}%", "delta_value": s_["Max Drawdown"] - h["Max Drawdown"]},
    {"icon": "🛡️", "label": "Calmar ratio", "value": f"{s_['Calmar']:.2f}",
     "delta": f"hold {h['Calmar']:.2f}", "delta_value": s_["Calmar"] - h["Calmar"]},
])
pf = s_["Profit Factor"]
kpis([
    {"icon": "🔁", "label": "Round trips", "value": f"{s_['Trades']}", "delta": f"invested {s_['Exposure'] * 100:.0f}% of the time"},
    {"icon": "🎯", "label": "Win rate", "value": pct(s_["Win Rate"])},
    {"icon": "💹", "label": "Profit factor", "value": "—" if pf is None else ("∞" if pf == float("inf") else f"{pf:.2f}"),
     "delta": "gross wins ÷ gross losses" if pf is not None else "", "delta_value": (pf or 1) - 1},
    {"icon": "⬆️", "label": "Avg win", "value": pct(s_["Avg Win"])},
    {"icon": "⬇️", "label": "Avg loss", "value": pct(s_["Avg Loss"])},
])

section("Equity curve")
fig = go.Figure()
fig.add_trace(go.Scatter(x=res["equity"].index, y=capital * res["equity"], name=strategy,
                         line=dict(color=GREEN, width=2.2), fill="tozeroy", fillcolor="rgba(16,227,165,0.07)"))
fig.add_trace(go.Scatter(x=res["hold_equity"].index, y=capital * res["hold_equity"], name="Buy & hold",
                         line=dict(color="#64748b", width=2, dash="dot")))
fig.update_yaxes(tickprefix="$")
st.plotly_chart(style_fig(fig, 420), width="stretch")

trades = res["trades"]
left, right = st.columns([2, 1])
with left:
    section("Trade signals on price")
    df = res["df"]
    fig = go.Figure(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#94a3b8", width=1.5)))
    if not trades.empty:
        styles = {"BUY": (GREEN, "triangle-up"), "Signal": (RED, "triangle-down"), "Stop-loss": ("#f97316", "x"),
                  "Trailing stop": ("#fbbf24", "x"), "Take-profit": ("#38bdf8", "star")}
        buys = trades[trades["action"] == "BUY"]
        fig.add_trace(go.Scatter(x=buys["date"], y=buys["price"], mode="markers", name="BUY",
                                 marker=dict(color=GREEN, size=10, symbol="triangle-up")))
        for reason, grp in trades[trades["action"] == "SELL"].groupby("reason"):
            color, marker = styles.get(reason, (RED, "triangle-down"))
            fig.add_trace(go.Scatter(x=grp["date"], y=grp["price"], mode="markers",
                                     name="SELL" if reason == "Signal" else reason,
                                     marker=dict(color=color, size=10, symbol=marker)))
    st.plotly_chart(style_fig(fig, 360), width="stretch")
with right:
    section("Drawdown")
    dd = (res["equity"] / res["equity"].cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=RED, width=1),
                               fillcolor="rgba(255,92,122,0.15)", name="Drawdown"))
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(style_fig(fig, 360), width="stretch")

if len(res["trips"]):
    section("Trade return distribution")
    fig = go.Figure(go.Histogram(x=res["trips"] * 100, nbinsx=30, marker_color="#38bdf8", opacity=0.85))
    fig.add_vline(x=0, line=dict(color="#94a3b8", dash="dot"))
    fig.update_xaxes(title="Return per round trip %")
    st.plotly_chart(style_fig(fig, 280), width="stretch")

with st.expander(f"Trade log ({len(trades)} fills)"):
    st.dataframe(trades, hide_index=True, width="stretch")

disclaimer()
