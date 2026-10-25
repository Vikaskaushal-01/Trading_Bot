import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.config import UNIVERSE
from utils.features import FEATURE_INFO
from utils.market import fmt_price, get_currency
from utils.predictor import MODELS, describe, get_settings, predict_symbol, reliability
from utils.ui import GREEN, RED, color_for, disclaimer, hero, kpis, section, style_fig

AMBER = "#fbbf24"

hero(
    '🧠 AI <span class="hl">Lab</span>',
    "Look inside the prediction model: how it was validated, what it is reacting to, "
    "how reliable it has been, and what similar setups did in the past.",
    badge="Model diagnostics",
)

settings = get_settings()
options = list(UNIVERSE)
default = st.session_state.pop("lab_symbol", None)
if default and default not in options:
    options.append(default)
if default:
    st.session_state.lab_sym = default

c1, c2 = st.columns([2, 3])
symbol = c1.selectbox("Asset", options, key="lab_sym",
                      format_func=lambda s: f"{UNIVERSE.get(s, (s,))[0]} · {s}")
c2.markdown(
    f'<div class="panel" style="margin-top:4px;font-size:13px">⚙️ <b>{settings["model"]}</b> · horizon '
    f'<b>{settings["horizon"]}</b> · {settings["history"]} of data · groups: {", ".join(settings["groups"])} · '
    f'edge ±{settings["edge"] * 100:.0f} pts <span class="muted">— change in the sidebar</span></div>',
    unsafe_allow_html=True,
)

with st.spinner("Training & validating the model (purged walk-forward, 5 folds)…"):
    pred = predict_symbol(symbol, settings)
if not pred:
    st.error("Not enough clean history to train a model for this asset.")
    st.stop()

cur = get_currency(symbol)
comp = pred["composite"]
kpis([
    {"icon": "🎯", "label": f"Signal ({settings['horizon']})", "value": pred["signal"],
     "delta": f"P(up) {pred['prob_up'] * 100:.1f}% vs usual {pred['base_rate'] * 100:.1f}%",
     "delta_value": pred["prob_up"] - pred["base_rate"]},
    {"icon": "🧪", "label": "Reliability (OOS ROC-AUC)", "value": f"{pred['auc']:.3f}",
     "delta": pred["reliability"], "delta_value": pred["auc"] - 0.52},
    {"icon": "📍", "label": "Median price target", "value": fmt_price(pred["target_mid"], cur),
     "delta": f"{pred['ret_q50'] * 100:+.1f}% · 80% band {pred['ret_q10'] * 100:+.1f}% … {pred['ret_q90'] * 100:+.1f}%",
     "delta_value": pred["ret_q50"]},
    {"icon": "🧭", "label": "Composite score", "value": f"{comp['score']:+d}", "delta": comp["label"],
     "delta_value": comp["score"]},
    {"icon": "📚", "label": "Training samples", "value": f"{pred['n_samples']:,}",
     "delta": f"{len(pred['features'])} features · data to {pred['latest_date']:%d %b %Y}", "delta_value": 0},
])

if pred["reliability"] in ("No edge", "Weak"):
    st.warning(
        f"**Reliability: {pred['reliability']}.** On days it had never seen, this model ranked up- vs down-moves "
        f"with ROC-AUC {pred['auc']:.3f} (0.5 = coin flip). Its probability has been calibrated back towards the "
        "usual up-rate, and the composite score gives it less weight. Try a different horizon, model or feature set — "
        "or lean on risk management rather than prediction for this asset."
    )

# ---------------- composite breakdown ---------------- #
section("🧭 Signal consensus", "how the composite score is built")
w = comp["weights"]
parts = [("AI model", comp["ai"], w["ai"]), ("Technical rating", comp["tech"], w["tech"]), ("Historical analogs", comp["analog"], w["analog"])]
fig = go.Figure(go.Bar(
    y=[p[0] for p in parts], x=[p[1] for p in parts], orientation="h",
    marker_color=[GREEN if p[1] > 0 else RED for p in parts],
    text=[f"{p[1]:+.0f} × weight {p[2] * 100:.0f}%" for p in parts], textposition="auto",
))
fig.update_xaxes(range=[-100, 100], zeroline=True, zerolinecolor="#475569")
st.plotly_chart(style_fig(fig, 220), width="stretch", config={"displayModeBar": False})

# ---------------- probability history ---------------- #
section("📈 Out-of-sample probability history", "every point was predicted by a model that had not seen that day")
p = pred["oos_proba"]
close = pred["close"].reindex(p.index)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.04)
fig.add_trace(go.Scatter(x=close.index, y=close, name="Price", line=dict(color="#cbd5e1", width=1.4)), row=1, col=1)
buys, sells = p >= pred["buy_level"], p <= pred["sell_level"]
fig.add_trace(go.Scatter(x=close.index[buys], y=close[buys], mode="markers", name="BUY signal",
                         marker=dict(color=GREEN, size=5)), row=1, col=1)
fig.add_trace(go.Scatter(x=close.index[sells], y=close[sells], mode="markers", name="SELL signal",
                         marker=dict(color=RED, size=5)), row=1, col=1)
fig.add_trace(go.Scatter(x=p.index, y=p * 100, name="P(up)", line=dict(color="#a78bfa", width=1.3)), row=2, col=1)
for lvl, col, lab in ((pred["buy_level"], GREEN, "BUY level"), (pred["sell_level"], RED, "SELL level"),
                      (pred["base_rate"], "#64748b", "usual up-rate")):
    fig.add_hline(y=lvl * 100, line=dict(color=col, dash="dot", width=1), row=2, col=1,
                  annotation_text=lab, annotation_font=dict(color=col, size=10))
fig.update_yaxes(ticksuffix="%", row=2, col=1)
st.plotly_chart(style_fig(fig, 520), width="stretch")

# ---------------- validation charts ---------------- #
left, right = st.columns(2)
with left:
    section("🪜 Returns by AI conviction")
    b = pred["buckets"].reset_index()
    fig = go.Figure(go.Bar(x=b["bucket"].astype(str), y=b["avg_return"] * 100,
                           marker_color=[GREEN if v >= 0 else RED for v in b["avg_return"]],
                           text=[f"{h * 100:.0f}% up<br>n={n}" for h, n in zip(b["hit_rate"], b["days"])],
                           textposition="outside"))
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(style_fig(fig, 340), width="stretch")
    st.caption(f"Average {settings['horizon']} forward return for each fifth of days, sorted by the model's probability. "
               "A model with real skill shows a staircase rising to the right.")
with right:
    section("🎯 Calibration")
    cal = pred["calibration"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Perfect", line=dict(color="#475569", dash="dot")))
    fig.add_trace(go.Scatter(x=cal["pred"] * 100, y=cal["actual"] * 100, mode="markers+lines", name="Model",
                             marker=dict(size=[min(28, 6 + n / 15) for n in cal["n"]], color="#38bdf8"),
                             text=[f"n={n}" for n in cal["n"]]))
    fig.update_xaxes(title="Predicted P(up) %", range=[0, 100])
    fig.update_yaxes(title="Actual up-rate %", range=[0, 100])
    st.plotly_chart(style_fig(fig, 340), width="stretch")
    st.caption(f"Brier score {pred['brier']:.4f} (lower is better; {pred['base_rate'] * (1 - pred['base_rate']):.4f} "
               "= always predicting the usual up-rate). Bubble size = number of days.")

# ---------------- trading the signal ---------------- #
section("💹 Trading the signal (out-of-sample)", "start invested · exit on SELL · re-enter on BUY · 0.1% fee")
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred["equity"].index, y=(pred["equity"] - 1) * 100, name="AI signals",
                         line=dict(color=GREEN, width=2.2)))
fig.add_trace(go.Scatter(x=pred["hold_equity"].index, y=(pred["hold_equity"] - 1) * 100, name="Buy & hold",
                         line=dict(color="#64748b", width=2, dash="dot")))
fig.update_yaxes(ticksuffix="%")
st.plotly_chart(style_fig(fig, 340), width="stretch")
fmt_pct = lambda v: "—" if v is None else f"{v * 100:.1f}%"
kpis([
    {"label": "Strategy return", "value": f"{pred['strategy_return'] * 100:+.1f}%",
     "delta": f"hold {pred['hold_return'] * 100:+.1f}%", "delta_value": pred["strategy_return"] - pred["hold_return"]},
    {"label": "Sharpe · Max DD", "value": f"{pred['sharpe']:.2f} · {pred['max_drawdown'] * 100:.0f}%"},
    {"label": "BUY signals right", "value": fmt_pct(pred["buy_precision"]),
     "delta": f"{pred['buy_count']} days · avg {fmt_pct(pred['buy_avg_ret'])} fwd", "delta_value": (pred["buy_avg_ret"] or 0) - pred["avg_ret_all"]},
    {"label": "SELL signals right", "value": fmt_pct(pred["sell_precision"]),
     "delta": f"{pred['sell_count']} days · avg {fmt_pct(pred['sell_avg_ret'])} fwd", "delta_value": pred["avg_ret_all"] - (pred["sell_avg_ret"] or 0)},
    {"label": "Time invested", "value": f"{pred['exposure'] * 100:.0f}%"},
])

# ---------------- features ---------------- #
section("🔬 Feature dashboard", f"{len(pred['features'])} inputs, ranked by relevance")
rows = []
for f in pred["relevance"].index:
    group, desc = describe(f)
    rows.append({"Feature": f, "Meaning": desc, "Group": group, "Relevance": pred["relevance"][f],
                 "Current": pred["latest_values"][f], "Percentile": pred["percentiles"][f] * 100})
feat_df = pd.DataFrame(rows)
groups = ["All"] + sorted(feat_df["Group"].unique())
g = st.segmented_control("Group", groups, default="All", key="lab_group", label_visibility="collapsed") or "All"
view = feat_df if g == "All" else feat_df[feat_df["Group"] == g]
st.dataframe(view, hide_index=True, width="stretch", height=min(38 * len(view) + 40, 560), column_config={
    "Relevance": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=float(feat_df["Relevance"].max())),
    "Current": st.column_config.NumberColumn(format="%.4f"),
    "Percentile": st.column_config.ProgressColumn("Percentile vs history", format="%.0f%%", min_value=0, max_value=100,
                                                  help="Where today's value sits in this asset's history (0% = lowest ever, 100% = highest)."),
})
extreme = feat_df[(feat_df["Percentile"] <= 5) | (feat_df["Percentile"] >= 95)].head(6)
if len(extreme):
    st.caption("Unusual readings today: " + " · ".join(
        f"**{r.Meaning}** at the {r.Percentile:.0f}th percentile" for r in extreme.itertuples()))

# ---------------- analogs ---------------- #
section("🕰️ Historical analogs", "the 30 past days whose setup most resembles today")
a1, a2 = st.columns([1.5, 1])
with a1:
    ar = pred["analog_ret"] * 100
    fig = go.Figure(go.Histogram(x=ar, nbinsx=15, marker_color=GREEN, opacity=0.85))
    fig.add_vline(x=0, line=dict(color="#94a3b8", dash="dot"))
    fig.add_vline(x=float(ar.median()), line=dict(color=AMBER), annotation_text="median",
                  annotation_font=dict(color=AMBER))
    fig.update_xaxes(title=f"{settings['horizon']} forward return %")
    st.plotly_chart(style_fig(fig, 300), width="stretch")
with a2:
    st.markdown(
        f'<div class="panel"><div class="muted">After the 30 most similar setups</div>'
        f'<div style="font-size:30px;font-weight:800;color:{color_for(pred["analog_up"] - pred["base_rate"])}">'
        f'{pred["analog_up"] * 100:.0f}% rose</div>'
        f'<div class="muted">vs {pred["base_rate"] * 100:.0f}% of all {settings["horizon"]} periods</div>'
        f'<div style="margin-top:10px">Median move <b>{pred["analog_median"] * 100:+.2f}%</b><br>'
        f'Best {ar.max():+.1f}% · worst {ar.min():+.1f}%</div>'
        f'<div class="muted" style="margin-top:10px;font-size:12px">Matched on the 12 most relevant features. '
        f'Most recent analogs: {", ".join(d.strftime("%b %Y") for d in sorted(pred["analog_dates"])[-4:])}</div></div>',
        unsafe_allow_html=True,
    )

# ---------------- model comparison ---------------- #
section("🏁 Model comparison", f"same data, horizon {settings['horizon']}")
if st.button("Compare all models", type="primary") or st.session_state.get(f"cmp_{symbol}"):
    st.session_state[f"cmp_{symbol}"] = True
    out = []
    bar = st.progress(0.0)
    for i, m in enumerate(MODELS):
        r = predict_symbol(symbol, {**settings, "model": m})
        bar.progress((i + 1) / len(MODELS), text=f"Trained {m}")
        if r:
            out.append({"Model": m, "ROC-AUC": r["auc"], "Reliability": reliability(r["auc"])[1],
                        "Brier": r["brier"], "Signal now": r["signal"], "P(up)": r["prob_up"] * 100,
                        "Strategy": r["strategy_return"] * 100, "Buy & hold": r["hold_return"] * 100,
                        "Sharpe": r["sharpe"]})
    bar.empty()
    cmp = pd.DataFrame(out).sort_values("ROC-AUC", ascending=False)
    st.dataframe(cmp, hide_index=True, width="stretch", column_config={
        "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        "Brier": st.column_config.NumberColumn(format="%.4f"),
        "P(up)": st.column_config.NumberColumn(format="%.1f%%"),
        "Strategy": st.column_config.NumberColumn("Strategy return", format="%+.1f%%"),
        "Buy & hold": st.column_config.NumberColumn(format="%+.1f%%"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
    })
    best = cmp.iloc[0]
    st.caption(f"Best out-of-sample ranking: **{best['Model']}** (AUC {best['ROC-AUC']:.3f}). "
               "Differences under ~0.02 AUC are usually noise — prefer the simpler or ensemble model then.")
else:
    st.caption(f"Trains all {len(MODELS)} models on {symbol} and ranks them by out-of-sample ROC-AUC (~20–30 s first time).")

with st.expander("📖 How this model works"):
    st.markdown(f"""
- **Target** — will the close be higher **{settings['horizon']}** from now?
- **Inputs** — {len(FEATURE_INFO)} candidate features in 7 groups (momentum, trend, volatility, volume, price structure,
  market context incl. benchmark & VIX, calendar). Features with missing data for an asset (e.g. volume for FX) are dropped automatically.
- **Validation** — *purged walk-forward*: the history is split into 6 consecutive blocks; each of the last 5 is predicted by a model
  trained only on earlier blocks, with a gap of {settings['horizon']} between them so overlapping returns can't leak.
- **Calibration** — raw probabilities are re-mapped (Platt scaling) so "60%" really meant ~60% historically. With no skill they collapse to the usual up-rate.
- **Signal** — BUY when calibrated P(up) ≥ usual up-rate + {settings['edge'] * 100:.0f} pts, SELL when ≤ usual − {settings['edge'] * 100:.0f} pts.
- **Price range** — separate quantile-regression models estimate the 10th/50th/90th percentile return.
- **Analogs** — nearest neighbours on the 12 most relevant (standardised) features.
""")

disclaimer()
