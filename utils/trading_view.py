"""Trading terminal shared by the Live and Demo pages."""

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.chat_view import chat_panel
from utils.config import COMMISSION_RATE, PERIODS, STARTING_BALANCE, UNIVERSE, asset_name
from utils.indicators import add_indicators, technical_rating
from utils.market import (
    fmt_price, fx_to_usd, get_currency, get_history, get_live_price, get_quotes, validate_symbol,
)
from utils.predictor import describe, predict_symbol
from utils.ui import (
    GREEN, RED, accounts, color_for, disclaimer, hero, kpis, md_escape, price_chart,
    signed_money, style_fig,
)

PIE_COLORS = ["#334155", "#10e3a5", "#38bdf8", "#a78bfa", "#fbbf24", "#f472b6", "#fb923c", "#22d3ee"]
OVERLAYS = ["EMA 20", "EMA 50", "SMA 200", "Bollinger", "RSI", "MACD"]
AMBER = "#fbbf24"


# ---------------- helpers ---------------- #

def _symbol_picker(acct, key: str) -> str:
    pending = st.session_state.pop("pending_symbol", None) or st.query_params.get("stock")
    if "stock" in st.query_params:
        del st.query_params["stock"]

    options = list(dict.fromkeys(list(UNIVERSE) + acct.watchlist + list(acct.positions)))
    if pending:
        if pending not in options:
            options.append(pending)
        st.session_state[key] = pending
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]

    def label(s):
        name = acct.positions.get(s, {}).get("name") or asset_name(s)
        return f"{name}  ·  {s}" + ("  ★" if s in acct.watchlist else "")

    return st.selectbox("Asset", options, key=key, format_func=label)


def _add_ticker(acct, key: str):
    with st.popover("➕ Ticker", width="stretch", help="Trade any Yahoo Finance symbol"):
        st.caption("Any Yahoo Finance symbol, e.g. `AMD`, `SPY`, `SOL-USD`, `TATAMOTORS.NS`.")
        t = st.text_input("Ticker", key=f"new_{key}").strip().upper()
        if st.button("Add & open", key=f"add_{key}", type="primary", width="stretch") and t:
            if validate_symbol(t):
                if t not in acct.watchlist:
                    acct.toggle_watch(t)
                st.session_state.pending_symbol = t
                st.rerun()
            else:
                st.error(f"No market data found for {t}.")


def _process_triggers(acct):
    """Fill limit orders / fire stops for every symbol the account has rules on."""
    syms = set(acct.positions) | {o["symbol"] for o in acct.orders}
    watched = {s for s in syms if s in acct.positions and any(
        acct.positions[s].get(k) for k in ("stop", "target", "trail_pct"))} | {o["symbol"] for o in acct.orders}
    if not watched:
        return
    native = {s: p for s in watched if (p := get_live_price(s)) is not None}
    fx = {s: fx_to_usd(get_currency(s)) for s in watched}
    for event in acct.process(native, fx):
        st.toast(event, icon="⚡")


def _price_header(symbol: str, currency: str):
    q = get_quotes((symbol,)).get(symbol)
    live = get_live_price(symbol)
    if not q or live is None:
        st.warning("Price data temporarily unavailable.")
        return
    prev = q["price"] - q["change"]
    chg_pct = (live / prev - 1) * 100 if prev else 0
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin:4px 0 8px">'
        f'<span style="font-size:22px;font-weight:700">{q["name"]}</span>'
        f'<span style="font-size:34px;font-weight:800;font-family:JetBrains Mono">{fmt_price(live, currency)}</span>'
        f'<span style="color:{color_for(chg_pct)};font-weight:700;font-size:17px">{live - prev:+,.2f} ({chg_pct:+.2f}%)</span>'
        f'<span class="muted">today · vol {q["volume"]:,.0f}</span></div>',
        unsafe_allow_html=True,
    )


def _meter(pos_0_to_1: float, tick: float | None = None) -> str:
    tick_html = (f'<b style="position:absolute;top:-3px;left:calc({tick * 100:.1f}% - 1px);width:2px;height:14px;'
                 f'background:#94a3b8"></b>') if tick is not None else ""
    return f'<div class="meter">{tick_html}<i style="left:calc({pos_0_to_1 * 100:.1f}% - 2px)"></i></div>'


def _row(label: str, value: str, color: str = "#e8edf7") -> str:
    return (f'<div style="display:flex;justify-content:space-between;font-size:13px;padding:3px 0">'
            f'<span class="muted">{label}</span><span style="color:{color};font-weight:600">{value}</span></div>')


# ---------------- side panel cards ---------------- #

def _ai_card(pred: dict | None, currency: str):
    if not pred:
        st.info("Not enough history for an AI signal (needs ~2 years of daily data).")
        return
    s = pred["settings"]
    comp = pred["composite"]
    comp_color = GREEN if comp["score"] >= 12 else RED if comp["score"] <= -12 else AMBER
    rel_color = {"Strong": GREEN, "Moderate": "#38bdf8", "Weak": AMBER, "No edge": RED}[pred["reliability"]]
    prob, base = pred["prob_up"], pred["base_rate"]

    st.markdown(
        f'<div class="panel"><div class="muted">🤖 AI signal · {s["horizon"]} ahead · {s["model"]}</div>'
        f'<div style="display:flex;align-items:center;gap:12px;margin:10px 0">'
        f'<span class="signal {pred["signal"]}">{pred["signal"]}</span>'
        f'<span style="font-size:12px;padding:3px 9px;border-radius:8px;border:1px solid {rel_color};color:{rel_color}">'
        f'Reliability: {pred["reliability"]}</span></div>'
        f'<div>P(up in {s["horizon"]}): <b>{prob * 100:.1f}%</b> '
        f'<span class="muted">vs usual {base * 100:.1f}%</span></div>{_meter(prob, base)}'
        f'<div class="muted" style="display:flex;justify-content:space-between;font-size:12px"><span>Bearish</span>'
        f'<span>│ = typical up-rate</span><span>Bullish</span></div>'
        f'<hr style="border-color:#1f2a3f;margin:12px 0">'
        f'<div class="muted">Expected price in {s["horizon"]} (10–90% range)</div>'
        f'<div style="font-family:JetBrains Mono;font-weight:700;font-size:18px">{fmt_price(pred["target_mid"], currency)}'
        f' <span style="font-size:13px;color:{color_for(pred["ret_q50"])}">({pred["ret_q50"] * 100:+.1f}%)</span></div>'
        f'<div class="muted" style="font-family:JetBrains Mono">{fmt_price(pred["target_low"], currency)} – '
        f'{fmt_price(pred["target_high"], currency)}</div>'
        f'<hr style="border-color:#1f2a3f;margin:12px 0">'
        + _row("Out-of-sample ROC-AUC", f'{pred["auc"]:.3f}', rel_color)
        + _row("BUY-signal hit rate", f'{pred["buy_precision"] * 100:.0f}% of {pred["buy_count"]}'
               if pred["buy_precision"] is not None else "no past signals")
        + _row("Similar past setups rose", f'{pred["analog_up"] * 100:.0f}% (median {pred["analog_median"] * 100:+.1f}%)',
               color_for(pred["analog_up"] - base))
        + f'<hr style="border-color:#1f2a3f;margin:12px 0"><div class="muted">Composite score (AI + technicals + analogs)</div>'
        f'<div style="font-size:22px;font-weight:800;color:{comp_color}">{comp["label"]} '
        f'<span class="muted">({comp["score"]:+d})</span></div>{_meter((comp["score"] + 100) / 200)}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Open full analysis in AI Lab →", key=f"lab_{pred['symbol']}", width="stretch"):
        st.session_state.lab_symbol = pred["symbol"]
        st.switch_page("app_pages/ai_lab.py")


def _tech_card(df: pd.DataFrame):
    rating = technical_rating(df)
    color = GREEN if rating["score"] > 10 else RED if rating["score"] < -10 else AMBER
    rows = "".join(
        _row(label, "Bullish" if v > 0 else "Bearish" if v < 0 else "Neutral",
             GREEN if v > 0 else RED if v < 0 else AMBER)
        for v, label in rating["checks"]
    )
    last = df.iloc[-1]
    st.markdown(
        f'<div class="panel"><div class="muted">📐 Technical rating</div>'
        f'<div style="font-size:24px;font-weight:800;color:{color};margin:6px 0">{rating["label"]} '
        f'<span class="muted">({rating["score"]:+d})</span></div>{_meter((rating["score"] + 100) / 200)}'
        f'<div style="display:flex;gap:18px;margin:10px 0 6px;font-size:13px">'
        f'<span>RSI <b>{last["RSI"]:.1f}</b></span><span>MACD <b>{last["MACD"]:.2f}</b></span>'
        f'<span>ATR <b>{last["ATR"] / last["Close"] * 100:.2f}%</b></span></div>{rows}</div>',
        unsafe_allow_html=True,
    )


# ---------------- order ticket ---------------- #

def _risk_panel(key: str, symbol: str, atr_pct: float, equity: float, max_amt: float, amt_key: str, pred):
    """Stop / target / trailing / sizing inputs. Returns (sl_pct, tp_pct, trail_pct) — None when off."""
    with st.expander("🛡️ Risk management", expanded=True):
        default_sl = round(min(max(2 * atr_pct, 0.5), 30.0), 1)
        c1, c2 = st.columns(2)
        use_sl = c1.toggle("Stop-loss", value=True, key=f"use_sl_{key}")
        use_tp = c2.toggle("Take-profit", value=True, key=f"use_tp_{key}")
        sl_pct = st.number_input("Stop distance (%)", 0.5, 50.0, default_sl, 0.5, key=f"sl_{key}_{symbol}",
                                 help=f"Default = 2 × ATR ({atr_pct:.2f}% daily) — outside normal daily noise.",
                                 disabled=not use_sl)
        rr = st.slider("Reward : risk", 1.0, 5.0, 2.0, 0.5, key=f"rr_{key}", disabled=not use_tp,
                       help="Take-profit distance as a multiple of the stop distance.")
        use_trail = st.toggle("Trailing stop", value=False, key=f"use_trail_{key}",
                              help="Stop follows the highest price since entry, locking in gains.")
        trail_pct = st.number_input("Trail distance (%)", 0.5, 50.0, round(min(max(3 * atr_pct, 1.0), 30.0), 1), 0.5,
                                    key=f"trail_{key}_{symbol}", disabled=not use_trail)

        st.markdown('<div class="muted" style="margin-top:6px">Position sizing</div>', unsafe_allow_html=True)
        risk_pct = st.slider("Risk per trade (% of net worth)", 0.25, 5.0, 1.0, 0.25, key=f"risk_{key}",
                             help="How much you lose if the stop is hit. Pros typically risk 0.5–2%.")
        stop_for_size = sl_pct if use_sl else (trail_pct if use_trail else None)
        if stop_for_size:
            suggested = min(math.floor(equity * risk_pct / stop_for_size * 100) / 100, max_amt)

            def use_suggested():
                st.session_state[amt_key] = max(suggested, 0.01)

            st.button(f"Use suggested size: ${suggested:,.2f}", key=f"size_{key}", on_click=use_suggested,
                      width="stretch", disabled=suggested < 0.01)
        else:
            st.caption("Enable a stop to get a risk-based size.")

        if pred and use_tp and use_sl:
            p, b = pred["prob_up"], rr
            kelly = p - (1 - p) / b
            if kelly > 0:
                st.caption(md_escape(f"½-Kelly with the AI's {p * 100:.0f}% and {rr:.1f}R: ≤ {kelly / 2 * 100:.1f}% of net worth "
                                     f"(${equity * kelly / 2:,.0f})."))
            else:
                st.caption(f"Kelly check: at {p * 100:.0f}% and {rr:.1f}R the odds don't justify this trade size.")
    return (sl_pct if use_sl else None), (sl_pct * rr if use_tp and use_sl else None), (trail_pct if use_trail else None)


def _order_ticket(acct, symbol, name, currency, key, equity, atr_pct, pred):
    native = get_live_price(symbol)
    if native is None:
        st.warning("Trading unavailable: no live price.")
        return
    fx = fx_to_usd(currency)
    px = native * fx
    pos = acct.positions.get(symbol)

    st.markdown('<div class="muted" style="margin-bottom:6px">🧾 Order ticket</div>', unsafe_allow_html=True)
    buy_tab, sell_tab = st.tabs(["Buy", "Sell"])

    with buy_tab:
        max_amt = max(math.floor(acct.cash / (1 + COMMISSION_RATE) * 100) / 100, 0.0)
        if max_amt < 1:
            st.info("No cash available. Add funds in the **Funds** tab.")
        else:
            otype = st.segmented_control("Order type", ["Market", "Limit"], default="Market",
                                         key=f"otype_{key}", width="stretch") or "Market"
            entry = native
            if otype == "Limit":
                entry = st.number_input(f"Limit price ({currency})", min_value=0.0001, value=float(f"{native * 0.98:.4g}"),
                                        key=f"lim_{key}_{symbol}", help="Fills when the price drops to this level or lower.")

            amt_key = f"buy_amt_{key}"
            st.session_state[amt_key] = min(max(st.session_state.get(amt_key, 100.0), 1.0), max_amt)

            def preset():
                frac = {"25%": .25, "50%": .5, "75%": .75, "Max": 1.0}[st.session_state[f"bp_{key}"]]
                st.session_state[amt_key] = max(math.floor(max_amt * frac * 100) / 100, 1.0)
                st.session_state[f"bp_{key}"] = None

            st.number_input("Amount (USD)", min_value=1.0, max_value=max_amt, step=10.0, key=amt_key)
            st.segmented_control("Preset", ["25%", "50%", "75%", "Max"], key=f"bp_{key}", on_change=preset,
                                 label_visibility="collapsed", width="stretch")
            sl, tp, trail = _risk_panel(key, symbol, atr_pct, equity, max_amt, amt_key, pred)

            amount = st.session_state[amt_key]
            fee = amount * COMMISSION_RATE
            lines = [f"≈ {amount / (entry * fx):,.6f} units at {fmt_price(entry, currency)} · fee ${fee:,.2f}"]
            exits = {"ref": entry}
            if sl:
                exits["stop"] = entry * (1 - sl / 100)
                lines.append(f"Stop {fmt_price(exits['stop'], currency)} (−{sl:.1f}%) → risk ${amount * sl / 100 + 2 * fee:,.2f}")
            if tp:
                exits["target"] = entry * (1 + tp / 100)
                lines.append(f"Target {fmt_price(exits['target'], currency)} (+{tp:.1f}%) → reward ${amount * tp / 100 - 2 * fee:,.2f}")
            if trail:
                exits["trail_pct"] = trail
                lines.append(f"Trailing stop {trail:.1f}% below the running high")
            if amount > equity * 0.25:
                lines.append(f"⚠️ This is {amount / equity * 100:.0f}% of your net worth in one asset.")
            st.caption(md_escape("  \n".join(lines)))

            label = f"Buy {name}" if otype == "Market" else f"Place limit buy @ {fmt_price(entry, currency)}"
            if st.button(label, type="primary", key=f"buy_{key}", width="stretch"):
                if otype == "Market":
                    ok, msg = acct.buy(symbol, amount, px, name, exits=exits)
                else:
                    ok, msg = acct.place_limit("BUY", symbol, name, entry, currency, amount=amount, exits=exits)
                (st.toast if ok else st.error)(("✅ " if ok else "") + msg)
                if ok:
                    st.rerun()

    with sell_tab:
        if not pos:
            st.info(f"You don't hold any {name}.")
        else:
            held = pos["qty"]
            st.caption(md_escape(f"You hold **{held:,.6f}** (≈ ${held * px:,.2f}) · avg cost ${pos['cost'] / held:,.2f}"))
            otype = st.segmented_control("Order type", ["Market", "Limit"], default="Market",
                                         key=f"sotype_{key}", width="stretch") or "Market"
            exit_px = native
            if otype == "Limit":
                exit_px = st.number_input(f"Limit price ({currency})", min_value=0.0001, value=float(f"{native * 1.03:.4g}"),
                                          key=f"slim_{key}_{symbol}", help="Fills when the price rises to this level or higher.")
            pct = st.slider("Portion to sell", 1, 100, 100, format="%d%%", key=f"sell_pct_{key}")
            qty = held * pct / 100
            proceeds = qty * exit_px * fx
            pnl = proceeds * (1 - COMMISSION_RATE) - qty * pos["cost"] / held
            st.caption(md_escape(f"Proceeds ≈ ${proceeds:,.2f} · est. P&L ")
                       + f"<span style='color:{color_for(pnl)}'>{md_escape(signed_money(pnl))}</span>",
                       unsafe_allow_html=True)
            label = f"Sell {pct}% of {name}" if otype == "Market" else f"Place limit sell @ {fmt_price(exit_px, currency)}"
            if st.button(label, type="primary", key=f"sell_{key}", width="stretch"):
                if otype == "Market":
                    ok, msg = acct.sell(symbol, qty, px)
                else:
                    ok, msg = acct.place_limit("SELL", symbol, name, exit_px, currency, qty=qty)
                (st.toast if ok else st.error)(("✅ " if ok else "") + msg)
                if ok:
                    st.rerun()

    starred = symbol in acct.watchlist
    if st.button("★ Remove from watchlist" if starred else "☆ Add to watchlist", key=f"watch_{key}", width="stretch"):
        acct.toggle_watch(symbol)
        st.rerun()


# ---------------- tabs ---------------- #

def _positions_tab(acct, val: dict, key: str):
    if not val["rows"]:
        st.info("No open positions yet — place your first order above.")
        return
    df = pd.DataFrame(val["rows"])
    df["currency"] = df["symbol"].map(get_currency)
    df["rules"] = df.apply(lambda r: " · ".join(filter(None, [
        f"SL {r['stop']:,.2f}" if r["stop"] else "", f"TP {r['target']:,.2f}" if r["target"] else "",
        f"Trail {r['trail_pct']:.1f}%" if r["trail_pct"] else ""])) or "—", axis=1)
    left, right = st.columns([2.2, 1])
    with left:
        st.dataframe(
            df[["name", "qty", "avg_price", "price", "value", "pnl", "pnl_pct", "rules"]],
            hide_index=True, width="stretch",
            column_config={
                "name": "Asset",
                "qty": st.column_config.NumberColumn("Qty", format="%.6f"),
                "avg_price": st.column_config.NumberColumn("Avg cost", format="$%.2f"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "value": st.column_config.NumberColumn("Value", format="$%.2f"),
                "pnl": st.column_config.NumberColumn("P&L", format="$%+.2f"),
                "pnl_pct": st.column_config.NumberColumn("P&L %", format="%+.2f%%"),
                "rules": st.column_config.TextColumn("Exit rules (native ccy)"),
            },
        )
        target = st.selectbox("Manage position", df["symbol"], key=f"close_sel_{key}",
                              format_func=lambda s: f"{asset_name(s)} ({s})")
        pos = acct.positions[target]
        cur = get_currency(target)
        native_now = get_live_price(target) or 0.0
        with st.form(f"exits_{key}_{target}", border=True):
            st.caption(f"Exit rules in {cur} · current price {fmt_price(native_now, cur)} · 0 = off")
            c1, c2, c3 = st.columns(3)
            stop = c1.number_input("Stop-loss price", min_value=0.0, value=float(pos.get("stop") or 0.0), format="%.4f")
            tgt = c2.number_input("Take-profit price", min_value=0.0, value=float(pos.get("target") or 0.0), format="%.4f")
            trail = c3.number_input("Trailing stop %", min_value=0.0, max_value=50.0, value=float(pos.get("trail_pct") or 0.0))
            b1, b2 = st.columns(2)
            if b1.form_submit_button("Save exit rules", type="primary", width="stretch"):
                if stop and native_now and stop >= native_now:
                    st.error("Stop must be below the current price.")
                elif tgt and native_now and tgt <= native_now:
                    st.error("Target must be above the current price.")
                else:
                    ok, msg = acct.set_exits(target, stop or None, tgt or None, trail or None, ref=native_now)
                    st.toast(msg)
                    st.rerun()
            if b2.form_submit_button("Close position now", width="stretch"):
                price = df.set_index("symbol").at[target, "price"]
                ok, msg = acct.sell(target, pos["qty"], price, note="Manual close")
                st.toast(("✅ " if ok else "") + msg)
                st.rerun()
    with right:
        labels = ["Cash"] + df["name"].tolist()
        values = [val["cash"]] + df["value"].tolist()
        fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, sort=False,
                               marker=dict(colors=PIE_COLORS, line=dict(color="#070b14", width=2))))
        fig.update_traces(textinfo="percent", hovertemplate="%{label}: $%{value:,.2f}<extra></extra>")
        fig.add_annotation(text=f"<b>${val['equity']:,.0f}</b><br><span style='font-size:11px'>net worth</span>",
                           showarrow=False, font=dict(size=16))
        st.plotly_chart(style_fig(fig, 300), width="stretch", config={"displayModeBar": False})
        largest = df["value"].max() / val["equity"] * 100 if val["equity"] else 0
        protected = df["stop"].notna().sum() + df["trail_pct"].notna().sum()
        st.markdown(
            '<div class="panel">'
            + _row("Largest position", f"{largest:.0f}% of net worth", RED if largest > 40 else AMBER if largest > 25 else GREEN)
            + _row("Positions with a stop", f"{min(protected, len(df))}/{len(df)}", GREEN if protected >= len(df) else AMBER)
            + _row("Cash buffer", f"{val['cash'] / val['equity'] * 100:.0f}%" if val["equity"] else "—")
            + "</div>", unsafe_allow_html=True)


def _orders_tab(acct, key: str):
    if not acct.orders:
        st.info("No pending orders. Choose **Limit** in the order ticket to queue one.")
        return
    df = pd.DataFrame(acct.orders)
    df["size"] = df.apply(lambda r: f"${r['amount']:,.2f}" if r["side"] == "BUY" else f"{r['qty']:,.6f} units", axis=1)
    df["now"] = df["symbol"].map(lambda s: get_live_price(s))
    df["distance"] = (df["limit"] / df["now"] - 1) * 100
    st.dataframe(
        df[["time", "side", "name", "limit", "currency", "now", "distance", "size"]], hide_index=True, width="stretch",
        column_config={
            "time": "Placed", "side": "Side", "name": "Asset",
            "limit": st.column_config.NumberColumn("Limit", format="%.4f"), "currency": "Ccy",
            "now": st.column_config.NumberColumn("Now", format="%.4f"),
            "distance": st.column_config.NumberColumn("Distance", format="%+.2f%%"),
            "size": "Size",
        },
    )
    c1, c2 = st.columns([3, 1])
    oid = c1.selectbox("Order", df["id"], key=f"ord_sel_{key}", label_visibility="collapsed",
                       format_func=lambda i: " · ".join(str(df.set_index("id").at[i, c]) for c in ("side", "name", "limit")))
    if c2.button("Cancel order", key=f"cancel_{key}", width="stretch"):
        acct.cancel_order(oid)
        st.toast("Order cancelled")
        st.rerun()
    st.caption("Orders are checked against live prices every time the app refreshes (turn on live updates to keep them monitored).")


def _history_tab(acct):
    if not acct.history:
        st.info("No activity yet.")
        return
    df = pd.DataFrame(acct.history[::-1])
    if "note" not in df:
        df["note"] = ""
    st.dataframe(
        df[["time", "side", "name", "symbol", "qty", "price", "amount", "fee", "pnl", "note"]], hide_index=True, width="stretch",
        column_config={
            "time": "Time", "side": "Type", "symbol": "Symbol", "name": "Asset",
            "qty": st.column_config.NumberColumn("Qty", format="%.6f"),
            "price": st.column_config.NumberColumn("Price (USD)", format="$%.2f"),
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "fee": st.column_config.NumberColumn("Fee", format="$%.2f"),
            "pnl": st.column_config.NumberColumn("Realized P&L", format="$%+.2f"),
            "note": "Trigger",
        },
    )
    sells = df[df["side"] == "SELL"]
    if len(sells):
        wins = (sells["pnl"] > 0).sum()
        st.caption(f"Closed trades: {len(sells)} · win rate {wins / len(sells) * 100:.0f}% · "
                   f"realized {md_escape(signed_money(sells['pnl'].sum()))} · fees paid {md_escape(f'${df.fee.sum():,.2f}')}")
    st.download_button("⬇ Export CSV", df.to_csv(index=False), f"{acct.label.lower()}_history.csv", "text/csv")


def _funds_tab(acct, key: str, is_demo: bool):
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.form(f"dep_{key}", border=True):
            amt = st.number_input("Deposit (USD)", min_value=1.0, value=100.0, step=50.0)
            if st.form_submit_button("Deposit", type="primary", width="stretch"):
                ok, msg = acct.deposit(amt)
                st.toast(msg)
                st.rerun()
    with c2:
        with st.form(f"wd_{key}", border=True):
            amt = st.number_input("Withdraw (USD)", min_value=1.0, value=50.0, step=50.0)
            if st.form_submit_button("Withdraw", width="stretch"):
                ok, msg = acct.withdraw(amt)
                (st.toast if ok else st.error)(msg)
                if ok:
                    st.rerun()
    with c3:
        with st.form(f"reset_{key}", border=True):
            bal = st.number_input("Reset with balance", min_value=10.0, value=STARTING_BALANCE, step=100.0)
            confirm = st.checkbox("I understand all positions, orders & history will be erased")
            if st.form_submit_button("Reset account", width="stretch"):
                if confirm:
                    acct.reset(bal)
                    st.toast("Account reset")
                    st.rerun()
                else:
                    st.error("Tick the confirmation box first.")


def _ai_tab(pred, currency: str):
    if not pred:
        st.info("AI insights unavailable for this asset.")
        return
    s = pred["settings"]
    c1, c2 = st.columns([1.4, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred["equity"].index, y=(pred["equity"] - 1) * 100, name="Follow AI signals",
                                 line=dict(color=GREEN, width=2)))
        fig.add_trace(go.Scatter(x=pred["hold_equity"].index, y=(pred["hold_equity"] - 1) * 100, name="Buy & hold",
                                 line=dict(color="#64748b", width=2, dash="dot")))
        fig.update_yaxes(ticksuffix="%")
        fig.update_layout(title="Out-of-sample: trading the signals vs buy & hold")
        st.plotly_chart(style_fig(fig, 330), width="stretch")
    with c2:
        b = pred["buckets"].reset_index()
        fig = go.Figure(go.Bar(x=b["bucket"].astype(str), y=b["avg_return"] * 100,
                               marker_color=[GREEN if v >= 0 else RED for v in b["avg_return"]],
                               text=[f"{h * 100:.0f}% up" for h in b["hit_rate"]], textposition="outside"))
        fig.update_yaxes(ticksuffix="%")
        fig.update_layout(title=f"Avg {s['horizon']} return by AI conviction")
        st.plotly_chart(style_fig(fig, 330), width="stretch")
    st.caption(
        "A useful model shows returns rising from *Very bearish* to *Very bullish*. "
        f"Signals: {pred['buy_count']} BUY / {pred['sell_count']} SELL days in the test period · "
        f"strategy {pred['strategy_return'] * 100:+.0f}% vs hold {pred['hold_return'] * 100:+.0f}% · "
        f"Sharpe {pred['sharpe']:.2f} · max drawdown {pred['max_drawdown'] * 100:.0f}%."
    )

    st.markdown('<div class="muted" style="margin:8px 0">What the model is looking at right now (top drivers)</div>',
                unsafe_allow_html=True)
    rows = []
    for f in pred["relevance"].index[:10]:
        group, desc = describe(f)
        rows.append({"Feature": desc, "Group": group, "Relevance": pred["relevance"][f],
                     "Current value": pred["latest_values"][f], "Percentile vs history": pred["percentiles"][f] * 100})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Relevance": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=float(pred["relevance"].max())),
        "Current value": st.column_config.NumberColumn(format="%.4f"),
        "Percentile vs history": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
    })


# ---------------- page ---------------- #

def render(account_name: str):
    is_demo = account_name == "Demo"
    key = account_name.lower()
    acct = accounts()[account_name]
    _process_triggers(acct)

    hero(
        '🧪 Demo <span class="hl">Trading</span>' if is_demo else '📈 Live <span class="hl">Trading</span>',
        "Practice strategies risk-free with a separate sandbox balance." if is_demo
        else "Real-time prices, AI signals and a full trading terminal for your main paper account.",
        badge=f"{account_name} account",
    )

    # ---- selection row ---- #
    c1, c2, c3, c4 = st.columns([1.9, 0.8, 2.1, 1])
    with c1:
        symbol = _symbol_picker(acct, f"sym_{key}")
    with c2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _add_ticker(acct, key)
    with c3:
        period = st.segmented_control("Range", list(PERIODS), default="1Y", key=f"period_{key}") or "1Y"
    with c4:
        style = st.segmented_control("Chart", ["Candles", "Line"], default="Candles", key=f"style_{key}") or "Candles"

    name = acct.positions.get(symbol, {}).get("name") or asset_name(symbol)
    currency = get_currency(symbol)
    fx = fx_to_usd(currency)

    # ---- account KPIs ---- #
    syms = tuple(sorted(set(acct.positions) | {symbol}))
    quotes = get_quotes(syms)
    prices = {s: (get_live_price(s) or q["price"]) * fx_to_usd(q["currency"]) for s, q in quotes.items()}
    val = acct.valuation(prices)
    kpis([
        {"icon": "💵", "label": "Cash", "value": f"${val['cash']:,.2f}",
         "delta": f"{len(acct.orders)} pending order(s)" if acct.orders else "", "delta_value": 0},
        {"icon": "📦", "label": "Holdings", "value": f"${val['holdings']:,.2f}",
         "delta": f"{signed_money(val['unrealized'])} unrealized", "delta_value": val["unrealized"]},
        {"icon": "🏦", "label": "Net worth", "value": f"${val['equity']:,.2f}"},
        {"icon": "📊", "label": "Total P&L", "value": signed_money(val["total_pnl"]),
         "delta": f"{val['total_pnl_pct']:+.2f}% on ${val['deposited']:,.0f} deposited", "delta_value": val["total_pnl"]},
        {"icon": "✅", "label": "Realized P&L", "value": signed_money(val["realized"])},
    ])

    # ---- live price header (also keeps stops/limits monitored) ---- #
    live = st.toggle("Live updates every 15s (prices, stops & limit orders)", value=True, key=f"live_{key}")

    @st.fragment(run_every=15 if live else None)
    def live_strip():
        if live:
            fresh = accounts()[account_name]
            n_before = len(fresh.history)
            _process_triggers(fresh)
            if len(fresh.history) != n_before:
                st.rerun(scope="app")
        _price_header(symbol, currency)

    live_strip()

    # ---- data ---- #
    hist = get_history(symbol, period=PERIODS[period])
    full = get_history(symbol, period="5y")
    if hist.empty or full.empty:
        st.error(f"Couldn't load market data for {symbol}. Try again shortly.")
        return
    full_ind = add_indicators(full)
    ind = full_ind.loc[hist.index.min():]
    last = full_ind.iloc[-1]
    atr_pct = float(last["ATR"] / last["Close"] * 100) if pd.notna(last.get("ATR")) else 2.0

    with st.spinner("Training AI model… (first run per asset takes a few seconds)"):
        pred = predict_symbol(symbol)

    # ---- price levels drawn on the chart ---- #
    levels = []
    pos = acct.positions.get(symbol)
    if pos:
        levels.append({"y": pos["cost"] / pos["qty"] / fx, "label": "Avg cost", "color": "#94a3b8"})
        if pos.get("stop"):
            levels.append({"y": pos["stop"], "label": "Stop", "color": RED})
        if pos.get("target"):
            levels.append({"y": pos["target"], "label": "Target", "color": GREEN})
        if pos.get("trail_pct") and pos.get("high"):
            levels.append({"y": pos["high"] * (1 - pos["trail_pct"] / 100), "label": "Trail", "color": AMBER})
    for o in acct.orders:
        if o["symbol"] == symbol:
            levels.append({"y": o["limit"], "label": f"Limit {o['side'].lower()}", "color": "#38bdf8"})
    if pred:
        levels.append({"y": pred["target_mid"], "label": f"AI {pred['settings']['horizon']}", "color": "#a78bfa"})

    left, right = st.columns([2.4, 1], gap="medium")
    with left:
        overlays = st.pills("Indicators", OVERLAYS, selection_mode="multi", default=["EMA 20", "EMA 50", "RSI"],
                            key=f"ov_{key}", label_visibility="collapsed")
        st.plotly_chart(price_chart(ind, name, style, overlays or [], currency, levels=levels), width="stretch",
                        config={"displaylogo": False, "scrollZoom": True})
        ai_col, tech_col = st.columns(2)
        with ai_col:
            _ai_card(pred, currency)
        with tech_col:
            _tech_card(full_ind)
    with right:
        _order_ticket(acct, symbol, name, currency, key, val["equity"], atr_pct, pred)

    # ---- tabs ---- #
    tabs = st.tabs(["📊 Positions", f"⏳ Orders ({len(acct.orders)})", "🧾 History", "💳 Funds", "🤖 AI insights", "💬 Ask AI"])
    with tabs[0]:
        _positions_tab(acct, val, key)
    with tabs[1]:
        _orders_tab(acct, key)
    with tabs[2]:
        _history_tab(acct)
    with tabs[3]:
        _funds_tab(acct, key, is_demo)
    with tabs[4]:
        _ai_tab(pred, currency)
    with tabs[5]:
        chat_panel(selected=symbol, key=key, height=380)

    disclaimer()
