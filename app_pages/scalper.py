"""Live Scalper — streaming TradingView prices (ticking every second) and BUY / SELL / WAIT signals with odds."""

import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from utils import journal
from utils.config import DATA_DIR
from utils.scalper import MarketSession, ScalpParams, build, hourly_edge, optimize, reliability_label, score
from utils.tv_feed import PRESETS, TIMEFRAMES, get_intraday
from utils.tv_stream import get_stream
from utils.ui import CYAN, GREEN, MUTED, RED, disclaimer, hero, kpis, sparkline, style_fig

AMBER = "#fbbf24"
TIMEZONES = ["Asia/Kolkata", "America/New_York", "Europe/London", "Asia/Dubai", "Asia/Singapore", "UTC"]
CADENCES = {"Every candle close (most reliable)": None, "Every 30 seconds": 30, "Every 15 seconds": 15}
SETTINGS_FILE = DATA_DIR / "scalper_settings.json"
DEFAULTS = {
    # Tuned on ~17 days of SPREADEX:SPX (Nov 2026), choosing on the first half of the walk-forward replay and
    # checking on the unseen second half: 3m candles, 15-minute hold, trend filter, 2% edge was the only family
    # that made money in both halves. 1m direction showed no skill (ROC-AUC ≈ 0.50) at any hold.
    "sc_preset": next(iter(PRESETS)), "sc_custom": "FOREXCOM:SPXUSD", "sc_tf": "3m", "sc_tz": "Asia/Kolkata",
    "sc_cadence": next(iter(CADENCES)), "sc_live": True, "sc_sound": True,
    "sc_exit": "Close after X minutes", "sc_hold_min": 15, "sc_mode": "Points", "sc_tp": 8.0, "sc_sl": 6.0, "sc_tp_atr": 1.0, "sc_sl_atr": 1.5,
    "sc_auto_spread": True, "sc_spread": 0.6, "sc_hz": 20, "sc_lot": 0.01, "sc_ppv": 1.0,
    "sc_edge": 0.02, "sc_k": 150, "sc_trend": True, "sc_hconf": False, "sc_ml": True, "sc_skill": True,
    "sc_hours_on": False, "sc_h_start": 19, "sc_h_end": 2, "sc_nshow": 160,
    "sc_max_trades": 0, "sc_max_loss": 0.0, "sc_target": 0.0,
}


# ------------------------------------------------------------------ settings persist across days
CHOICES = {"sc_preset": list(PRESETS) + ["Custom TradingView symbol…"], "sc_tf": list(TIMEFRAMES),
           "sc_tz": TIMEZONES, "sc_cadence": list(CADENCES), "sc_mode": ["Points", "ATR ×"],
           "sc_exit": ["Close after X minutes", "Take-profit / Stop-loss"]}


def _load_settings():
    """Fill any setting missing from this session (first visit, or after visiting another page) from disk."""
    missing = [k_ for k_ in DEFAULTS if k_ not in st.session_state]
    if not missing:
        return
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        saved = {}
    for k_ in missing:
        v = saved.get(k_, DEFAULTS[k_])
        if k_ in CHOICES and v not in CHOICES[k_]:
            v = DEFAULTS[k_]
        st.session_state[k_] = v


def _save_settings():
    cur = {k_: st.session_state.get(k_, v) for k_, v in DEFAULTS.items()}
    if cur != st.session_state.get("sc_saved"):
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(cur, indent=1), encoding="utf-8")
            st.session_state["sc_saved"] = cur
        except OSError:
            pass


_load_settings()

hero(
    'Live <span class="hl">Scalper</span>',
    "Streaming TradingView prices that tick every second, and a fresh BUY / SELL / WAIT as soon as each candle "
    "closes (or every 15–30 s). The odds come from simulating your exact TP, SL and spread on every past candle "
    "and comparing today's picture with the most similar setups.",
    badge="TradingView live stream · scalping",
)

# ------------------------------------------------------------------ settings UI
with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns([2.1, 0.8, 1.2, 1.7, 1.2])
    preset = c1.selectbox("Market", list(PRESETS) + ["Custom TradingView symbol…"], key="sc_preset")
    if preset == "Custom TradingView symbol…":
        tv_symbol = c1.text_input("TradingView symbol (EXCHANGE:TICKER)", key="sc_custom").strip().upper()
        yf_ticker = ""
        if ":" not in tv_symbol:
            st.warning("Type a TradingView symbol like **SPREADEX:SPX** (exchange, colon, ticker).")
            st.stop()
    else:
        tv_symbol, yf_ticker = PRESETS[preset]
    tf = c2.selectbox("Timeframe", list(TIMEFRAMES), key="sc_tf",
                      help="1m = a new closed candle, and a new signal, every minute.")
    tz_name = c3.selectbox("Show times in", TIMEZONES, key="sc_tz")
    cadence_label = c4.selectbox("New signal", list(CADENCES), key="sc_cadence",
                                 help="Candle close = decided on a finished candle (doesn't change afterwards). "
                                      "15 / 30 s = re-decided on the forming candle; faster but provisional — "
                                      "it can flip before the candle closes.")
    with c5:
        live_on = st.toggle("Live", key="sc_live", help="Stream prices and refresh signals automatically.")
        sound_on = st.toggle("Sound alerts", key="sc_sound", help="Beep when a new BUY or SELL appears.")

    c1, c2, c3, c4, c5, c6, c7 = st.columns([1.3, 1, 1, 1.2, 1, 0.9, 1])
    exit_label = c1.radio("How you exit", ["Close after X minutes", "Take-profit / Stop-loss"], key="sc_exit",
                          help="Close after X minutes: the % is the chance the trade is in profit (after spread) "
                               "when you close it X minutes later.\n\nTake-profit / Stop-loss: the % is the chance "
                               "TP is hit before SL.")
    time_exit = exit_label == "Close after X minutes"
    mode = "Points"
    tp = sl = 1.0
    horizon = 10
    if time_exit:
        hold_min = c2.number_input("Close after (minutes)", 1, 240, step=1, key="sc_hold_min",
                                   help="How long you keep each trade open. The BUY / SELL % is the chance of being "
                                        "in profit at that moment.")
        c3.markdown(f'<div class="muted" style="margin-top:30px">≈ {max(1, -(-hold_min // int(TIMEFRAMES[tf])))} '
                    f'× {tf} candle(s)</div>', unsafe_allow_html=True)
    else:
        hold_min = st.session_state.get("sc_hold_min", 3)
        mode = c1.radio("Targets in", ["Points", "ATR ×"], key="sc_mode", horizontal=True,
                        help="ATR × scales TP/SL with volatility: wider when the market moves fast, tighter when quiet.")
    if time_exit:
        pass
    elif mode == "Points":
        tp = c2.number_input("Take-profit", 0.1, 10_000.0, step=0.5, key="sc_tp", help="Points, after spread.")
        sl = c3.number_input("Stop-loss", 0.1, 10_000.0, step=0.5, key="sc_sl", help="Points, after spread.")
    else:
        tp = c2.number_input("Take-profit × ATR", 0.1, 20.0, step=0.25, key="sc_tp_atr")
        sl = c3.number_input("Stop-loss × ATR", 0.1, 20.0, step=0.25, key="sc_sl_atr")
    with c4:
        auto_spread = st.checkbox("Live spread", key="sc_auto_spread",
                                  help="Use the real bid/ask gap streamed from TradingView. Your broker's may differ.")
        manual_spread = st.number_input("Spread (points)", 0.0, 1_000.0, step=0.1, key="sc_spread",
                                        disabled=auto_spread, label_visibility="collapsed")
    if not time_exit:
        horizon = c5.number_input("Max hold (candles)", 2, 120, step=1, key="sc_hz",
                                  help="Close at market if neither TP nor SL is hit within this many candles.")
    lot = c6.number_input("Lot", 0.01, 100.0, step=0.01, key="sc_lot", format="%.2f")
    ppv = c7.number_input("$/point/lot", 0.01, 1_000.0, step=0.5, key="sc_ppv",
                          help="Contract value per 1.00 lot — check your broker. SPX500 CFDs are often $1 or $10.")

    with st.expander("Filters, risk limits & advanced"):
        a1, a2, a3, a4 = st.columns(4)
        min_edge = a1.slider("Required edge over break-even", 0.0, 0.20, step=0.01, key="sc_edge",
                             help="Only signal when the success rate beats break-even by at least this much.")
        k = a2.slider("Similar setups compared (k)", 50, 500, step=25, key="sc_k")
        trend_filter = a3.toggle("Trade with the trend only", key="sc_trend",
                                 help="BUY only above EMA 200 and the 15-minute EMA 20; SELL only below both.")
        high_conf = a4.toggle("High-confidence only", key="sc_hconf",
                              help="Also require the 90% lower confidence bound of the success rate to beat break-even.")
        m1, m2, _, _ = st.columns(4)
        use_ml = m1.toggle("Use ML model too", key="sc_ml",
                           help="Blend a gradient-boosting model with the similar-setups estimate. Rebuilt every "
                                "15 minutes (takes ~5 s).")
        require_skill = m2.toggle("Only signal when the % proved accurate", key="sc_skill",
                                  help="Strongly recommended. No BUY / SELL unless, in the replay, the percentages "
                                       "actually separated winners from losers (ROC-AUC ≥ 0.53).")
        b1, b2, b3, b4 = st.columns(4)
        hours_on = b1.toggle("Trading hours only", key="sc_hours_on",
                             help="Signals only inside these hours (your time zone). US session ≈ 19:00–02:30 IST.")
        h_start = b2.selectbox("From", list(range(24)), key="sc_h_start", format_func=lambda h: f"{h:02d}:00",
                               disabled=not hours_on)
        h_end = b3.selectbox("To", list(range(24)), key="sc_h_end", format_func=lambda h: f"{h:02d}:00",
                             disabled=not hours_on)
        n_show = b4.slider("Candles on chart", 60, 600, step=20, key="sc_nshow")
        r1, r2, r3 = st.columns(3)
        max_trades = r1.number_input("Max trades per day (0 = off)", 0, 500, step=1, key="sc_max_trades")
        max_loss = r2.number_input("Stop for the day at loss of $ (0 = off)", 0.0, 1e6, step=0.5, key="sc_max_loss")
        target = r3.number_input("Stop for the day at profit of $ (0 = off)", 0.0, 1e6, step=0.5, key="sc_target")
        st.caption("Daily limits count the trades you log in the journal below.")

_save_settings()

TZ = ZoneInfo(tz_name)
res_min = int(TIMEFRAMES[tf])
cadence = CADENCES[cadence_label]
provisional = cadence is not None
hold_candles = max(1, -(-int(hold_min) // res_min))
hold_txt = f"{int(hold_min)} min"
money = lambda pts: pts * lot * ppv  # noqa: E731


def utc_session() -> tuple | None:
    if not hours_on or h_start == h_end:
        return None
    off = int(datetime.now(TZ).utcoffset().total_seconds() // 60)
    return ((h_start * 60 - off) % 1440, (h_end * 60 - off) % 1440)


def local(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_localize("UTC").tz_convert(tz_name)


def local_index(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).tz_localize("UTC").tz_convert(tz_name).tz_localize(None)


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def usd(x: float, d: int = 3) -> str:
    return f"{'+' if x >= 0 else '−'}${abs(x):,.{d}f}"


stream = get_stream(tv_symbol, TIMEFRAMES[tf])


def live_quote() -> dict:
    """Last price, bid, ask from the stream (None if not streaming)."""
    q = stream.quote_snapshot()
    bid, ask = q.get("bid"), q.get("ask")
    # Show the bid/ask mid-price: some feeds (e.g. SPREADEX) update their "last" price less often than bid/ask.
    mid = (bid + ask) / 2 if bid and ask else q.get("lp")
    return {"lp": mid, "last": q.get("lp"), "bid": bid, "ask": ask, "spread": (ask - bid) if bid and ask else None,
            "ch": q.get("ch"), "chp": q.get("chp"), "desc": q.get("description") or tv_symbol}


# ------------------------------------------------------------------ 1-second live ticker
@st.fragment(run_every=1 if live_on else None)
def ticker():
    q = live_quote()
    now = datetime.now(TZ)
    secs_left = res_min * 60 - int(time.time()) % (res_min * 60)
    if q["lp"] is None:
        state = "⏳ connecting to TradingView…" if stream.status in ("connecting", "loading") else \
            f"🔴 {stream.status} — {stream.error or 'retrying'}"
        st.markdown(f'<div class="panel"><b>{now:%H:%M:%S}</b> · {state}</div>', unsafe_allow_html=True)
        return

    prev = st.session_state.get("sc_last_lp")
    if prev != q["lp"]:
        st.session_state["sc_dir"] = prev is None or q["lp"] >= prev
        st.session_state["sc_last_lp"] = q["lp"]
    up = st.session_state.get("sc_dir", True)
    color, arrow = (GREEN, "▲") if up else (RED, "▼")
    tick_age = time.time() - stream.last_tick if stream.last_tick else 999
    status = ("🟢 LIVE", GREEN) if stream.healthy and tick_age < 20 else \
        ("🟡 QUIET — no ticks for a while (market closed?)", AMBER) if stream.healthy else (f"🔴 {stream.status.upper()}", RED)
    ticks = [p for _, p in stream.tick_history()][-240:]
    spark = sparkline(ticks[:: max(1, len(ticks) // 120)], width=300, height=54) if len(ticks) > 5 else ""
    ch = q["ch"] or 0.0
    ny = datetime.now(ZoneInfo("America/New_York"))
    ny_m = ny.hour * 60 + ny.minute
    if ny.weekday() >= 5:
        cash = "US market closed (weekend)"
    elif ny_m < 570:
        left_ = 570 - ny_m
        cash = f"US market opens in <b>{left_ // 60}h {left_ % 60:02d}m</b>"
    elif ny_m < 960:
        left_ = 960 - ny_m
        cash = f"🔔 US market <b>OPEN</b> · closes in {left_ // 60}h {left_ % 60:02d}m"
    else:
        cash = "US market closed · overnight session"
    spread_txt = f"{q['spread']:.2f}" if q["spread"] is not None else "—"
    st.markdown(
        f'<div class="panel" style="display:flex;flex-wrap:wrap;gap:28px;align-items:center">'
        f'<div><div class="muted">{q["desc"]} · {tv_symbol}</div>'
        f'<div style="font:700 40px JetBrains Mono,monospace;color:{color}">{arrow} {q["lp"]:,.2f}</div>'
        f'<div style="color:{GREEN if ch >= 0 else RED};font-weight:600">{ch:+,.2f} ({(q["chp"] or 0):+.2f}%) today</div></div>'
        f'<div style="font-family:JetBrains Mono,monospace;line-height:1.7">'
        f'<div><span class="muted">BID</span> <b style="color:{RED}">{(q["bid"] or 0):,.2f}</b></div>'
        f'<div><span class="muted">ASK</span> <b style="color:{GREEN}">{(q["ask"] or 0):,.2f}</b></div>'
        f'<div><span class="muted">SPREAD</span> <b>{spread_txt}</b></div>'
        f'<div class="muted" style="font-size:11px">big price = mid of bid/ask</div></div>'
        f'<div style="min-width:220px;flex:1">{spark}<div class="muted">last {len(ticks)} ticks</div></div>'
        f'<div style="text-align:right"><div style="font:700 30px JetBrains Mono,monospace">{now:%H:%M:%S}</div>'
        f'<div class="muted">{tz_name}</div>'
        f'<div style="margin-top:4px">⏱ {tf} candle closes in <b>{secs_left // 60}:{secs_left % 60:02d}</b></div>'
        f'<div style="margin-top:2px">{cash}</div>'
        f'<div style="color:{status[1]};font-weight:700;margin-top:2px">{status[0]}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # follow the trades you logged: close them at TP / SL with the live bid/ask
    for t in journal.update(tv_symbol, q["bid"], q["ask"]):
        st.toast(f"{t['side']} closed at {t['status']}: {t['points']:+.2f} pts ({usd(t['usd'])})",
                 icon="✅" if t["points"] > 0 else "❌")
    for t in [t for t in journal.load() if t["status"] == "open" and t["symbol"] == tv_symbol]:
        px = q["bid"] if t["side"] == "BUY" else q["ask"]
        if px is None:
            continue
        pts = (px - t["entry"]) if t["side"] == "BUY" else (t["entry"] - px)
        c1, c2, c3 = st.columns([5, 1, 1])
        c1.markdown(
            f'<div class="panel" style="padding:10px 16px">📌 <b>{t["side"]}</b> {t["lot"]:.2f} lot @ '
            f'{t["entry"]:,.2f} · '
            + (f'closes in {max(0, int(t["close_at"] - time.time()))}s · ' if t.get("close_at") else "")
            + (f'TP {t["tp"]:,.2f} · SL {t["sl"]:,.2f} · ' if t.get("tp") is not None else "")
            + f'now {px:,.2f} → '
            f'<b style="color:{GREEN if pts >= 0 else RED}">{pts:+.2f} pts ({usd(pts * t["lot"] * t["ppv"])})</b></div>',
            unsafe_allow_html=True)
        if c2.button("Close", key=f"close_{t['id']}", width="stretch", help="Mark as closed at the current price"):
            journal.close_now(t["id"], q["bid"], q["ask"])
            st.rerun(scope="fragment")
        if c3.button("Remove", key=f"del_{t['id']}", width="stretch", help="Delete — you didn't really take it"):
            journal.delete(t["id"])
            st.rerun(scope="fragment")


ticker()


# ------------------------------------------------------------------ data + analysis
@st.cache_resource(max_entries=8, show_spinner=False)
def run_build(_df: pd.DataFrame, key: tuple, params: ScalpParams, sess: MarketSession) -> dict:
    """Labels + models + replay. Rebuilt every 15 minutes or when a setting that changes the labels changes."""
    return build(_df, params, sess)


@st.cache_resource(max_entries=32, show_spinner=False)
def run_score(_df: pd.DataFrame, _model: dict, key: tuple, params: ScalpParams, provisional: bool) -> dict:
    return score(_df, _model, params, use_last_bar=provisional)


def market_session() -> MarketSession:
    return MarketSession.from_info(stream.info or {}, tv_symbol)


def label_spread(live: float | None) -> float:
    """Spread used for the labels: follows the live spread but ignores tiny flickers (avoids constant rebuilds)."""
    if not auto_spread or live is None:
        return manual_spread
    key_ = f"sc_lab_spread_{tv_symbol}"
    cur = st.session_state.get(key_)
    if cur is None or abs(live - cur) > 0.15:
        st.session_state[key_] = cur = round(live, 2)
    return cur


def get_candles() -> tuple[pd.DataFrame, str, bool]:
    df, _ = stream.snapshot()
    if stream.history_ready and len(df) > 300:
        return df, "TradingView live stream", stream.healthy
    if time.time() - stream.started < 25:
        return pd.DataFrame(), "loading", False
    feed = get_intraday(tv_symbol, TIMEFRAMES[tf], 5000, yf_ticker, int(time.time() // 60))
    return feed["df"], feed["source"], feed["live"]


def beep(side: str):
    freq = 880 if side == "BUY" else 440
    components.html(
        f"<script>try{{const a=new (window.AudioContext||window.webkitAudioContext)();"
        f"[0,0.25].forEach(t=>{{const o=a.createOscillator(),g=a.createGain();o.frequency.value={freq};"
        f"o.connect(g);g.connect(a.destination);g.gain.setValueAtTime(0.25,a.currentTime+t);"
        f"o.start(a.currentTime+t);o.stop(a.currentTime+t+0.18);}});}}catch(e){{}}</script>",
        height=0)


@st.fragment(run_every=3 if live_on else None)
def signals():
    df, source, is_live = get_candles()
    if df.empty:
        st.info("⏳ Loading candle history from TradingView… (first load takes a few seconds)")
        return
    q = live_quote()
    spread = label_spread(q["spread"])
    params = ScalpParams(
        tp=tp, sl=sl, spread=spread, horizon=int(horizon), k=k, min_edge=min_edge,
        tp_mode="points" if mode == "Points" else "atr", trend_filter=trend_filter,
        session=utc_session(), high_conf=high_conf,
        exit_mode="time" if time_exit else "tp_sl", hold=hold_candles,
        use_ml=use_ml, require_skill=require_skill,
    )
    sess = market_session()
    bucket = int(time.time() // cadence) if provisional else df.index[-2]
    try:
        build_key = (tv_symbol, tf, params.label_key(), sess, int(time.time() // 900))
        with st.spinner("Re-training on the latest candles… (every 15 min, a few seconds)"):
            model = run_build(df, build_key, params, sess)
        res = run_score(df, model, (build_key, bucket, len(df), params.decision_key()), params, provisional)
    except ValueError as exc:
        st.warning(str(exc))
        return
    data, qi = res["data"], res["q"]
    bar_time = data.index[qi]
    price = q["lp"] if q["lp"] is not None else float(df["Close"].iat[-1])
    bid = q["bid"] if q["bid"] is not None else price
    ask = q["ask"] if q["ask"] is not None else price + spread

    # ---- daily risk limits from the journal
    trades_all = journal.load()
    today = journal.today_stats([t for t in trades_all if t["symbol"] == tv_symbol], TZ)
    halt = ""
    if max_trades and today["trades"] >= max_trades:
        halt = f"You've taken {today['trades']} trades today (limit {max_trades})."
    elif max_loss and today["usd"] <= -max_loss:
        halt = f"Today's loss {usd(today['usd'], 2)} hit your limit of ${max_loss:,.2f}."
    elif target and today["usd"] >= target:
        halt = f"Today's profit {usd(today['usd'], 2)} reached your target of ${target:,.2f}. Well done — stop here."
    signal = "WAIT" if halt else res["signal"]

    # ---- log + alerts when the signal changes
    log = st.session_state.setdefault("sc_log", [])
    key_ = (tv_symbol, tf, str(bucket))
    if not log or log[-1]["key"] != key_:
        prev_sig = log[-1]["Signal"] if log else "WAIT"
        log.append({"key": key_, "Time": datetime.now(TZ).strftime("%d %b %H:%M:%S"), "Market": f"{tv_symbol} {tf}",
                    "Signal": signal, "Price": round(price, 2), "BUY success": pct(res["p_long"]),
                    "SELL success": pct(res["p_short"]), "Why": halt or res["why"]})
        del log[:-500]
        if signal != "WAIT" and signal != prev_sig:
            st.toast(f"{signal} {tv_symbol} @ {price:,.2f}", icon="🟢" if signal == "BUY" else "🔴")
            if sound_on:
                beep(signal)

    st.caption(
        f"{'🟢' if is_live else '🟡'} {source} · {len(df):,} candles of history · decision candle "
        f"{local(bar_time):%H:%M} · analysed {datetime.now(TZ):%H:%M:%S} · spread used {spread:.2f} pts"
    )
    if not is_live:
        st.error("⚠️ **Not live right now** — TradingView isn't streaming, so these candles may be delayed. "
                 "Don't trade off this until the badge turns green again (it reconnects automatically).")

    # ---- signal card + KPIs
    tp_pts, sl_pts = res["tp_pts"], res["sl_pts"]
    if time_exit:
        entry = ask if signal == "BUY" else bid if signal == "SELL" else None
        tp_px = sl_px = None
    elif signal == "BUY":
        entry, tp_px, sl_px = ask, ask + tp_pts, ask - sl_pts
    elif signal == "SELL":
        entry, tp_px, sl_px = bid, bid - tp_pts, bid + sl_pts
    else:
        entry = tp_px = sl_px = None
    p_side = res["p_long"] if signal == "BUY" else res["p_short"]
    lcb_side = res["lcb_long"] if signal == "BUY" else res["lcb_short"]
    conf = "High" if lcb_side >= params.breakeven else "Medium"

    rel, rel_txt = reliability_label(res["accuracy"])
    need = (params.breakeven + params.min_edge) * 100
    if signal in ("BUY", "SELL"):
        when = (f"and close it at about <b>{(datetime.now(TZ) + pd.Timedelta(minutes=int(hold_min))):%H:%M}</b>"
                if time_exit else f"with TP <b>{tp_px:,.2f}</b> and SL <b>{sl_px:,.2f}</b>")
        answer = (f"👉 <b>{signal} now</b> at about <b>{entry:,.2f}</b> {when}. Real chance of profit "
                  f"<b>{pct(p_side)}</b> (you need {need:.0f}%+).")
        tone = GREEN if signal == "BUY" else RED
    else:
        answer = (f"👉 <b>Don't trade right now.</b> {halt or res['why']} Real chance of profit: BUY "
                  f"<b>{pct(res['p_long'])}</b> · SELL <b>{pct(res['p_short'])}</b> — a signal needs {need:.0f}%+.")
        tone = AMBER
    st.markdown(
        f'<div class="panel" style="border-left:5px solid {tone};margin-bottom:12px">'
        f'<div style="font-size:18px;line-height:1.6">{answer}</div>'
        f'<div class="muted" style="margin-top:6px">How accurate are these %? <b>{rel}</b> — {rel_txt} '
        f'Why don\'t BUY + SELL add up to 100%? You pay the {spread:.2f}-point spread on every trade, so after small '
        f'moves both sides lose. Next decision in {res_min * 60 - int(time.time()) % (res_min * 60)} s.</div></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1])
    with left:
        cls = {"BUY": "BUY", "SELL": "SELL"}.get(signal, "HOLD")
        label = {"BUY": "▲ BUY", "SELL": "▼ SELL"}.get(signal, "🛑 STOP FOR TODAY" if halt else "⏸ WAIT")
        head = (f"⏳ <b>Provisional</b> — forming {tf} candle, re-decided every {cadence}s" if provisional
                else f"Decided on the closed {tf} candle of {local(bar_time):%H:%M}")
        if entry is not None and time_exit:
            close_at = datetime.now(TZ) + pd.Timedelta(minutes=int(hold_min))
            exp_side = res["exp_long"] if signal == "BUY" else res["exp_short"]
            detail = (f"Entry ≈ <b>{entry:,.2f}</b> ({'ask' if signal == 'BUY' else 'bid'}) · "
                      f"close at ≈ <b>{close_at:%H:%M:%S}</b> (after {hold_txt})<br>"
                      f"Chance of profit at close <b>{pct(p_side)}</b> · average result {exp_side:+.2f} pts "
                      f"({usd(money(exp_side))}) · confidence <b>{conf}</b>")
        elif entry is not None:
            detail = (f"Entry ≈ <b>{entry:,.2f}</b> ({'ask' if signal == 'BUY' else 'bid'}) · "
                      f"TP <b style='color:{GREEN}'>{tp_px:,.2f}</b> · SL <b style='color:{RED}'>{sl_px:,.2f}</b><br>"
                      f"Win {usd(money(tp_pts))} / lose {usd(-money(sl_pts))} at {lot:.2f} lot · "
                      f"success {pct(p_side)} · confidence <b>{conf}</b>")
        else:
            detail = halt or res["why"]

        def bar(name, p_, exp_, color):
            be = params.breakeven * 100
            return (
                f'<div style="margin-top:12px"><div style="display:flex;justify-content:space-between">'
                f'<span style="font-weight:700;color:{color}">{name}</span>'
                f'<span class="muted">{"profit after " + hold_txt if time_exit else "success"} '
                f'<b style="color:#e8edf7">{p_ * 100:.1f}%</b> · avg {exp_:+.2f} pts '
                f'({usd(money(exp_))})</span></div>'
                f'<div style="position:relative;height:10px;border-radius:99px;background:#1a2336;margin-top:6px">'
                f'<div style="width:{max(2, min(100, p_ * 100))}%;height:100%;border-radius:99px;background:{color}"></div>'
                f'<i style="position:absolute;left:{be}%;top:-4px;width:2px;height:18px;background:#fff"></i>'
                f'<i style="position:absolute;left:{min(99, be + params.min_edge * 100)}%;top:-4px;width:2px;'
                f'height:18px;background:{AMBER}"></i></div></div>'
            )

        st.markdown(
            f'<div class="panel"><div class="muted">{head} · {tz_name}</div>'
            f'<div style="margin:10px 0"><span class="signal {cls}">{label}</span></div>'
            f'<div style="color:#e8edf7;line-height:1.6">{detail}</div>'
            f'{bar("BUY", res["p_long"], res["exp_long"], GREEN)}{bar("SELL", res["p_short"], res["exp_short"], RED)}'
            f'<div class="muted" style="margin-top:10px">White = break-even {params.breakeven * 100:.1f}% · '
            f'amber = needed for a signal ({(params.breakeven + params.min_edge) * 100:.1f}%). '
            f'From the {params.k} most similar of {res["n_hist"]:,} past candles. Usual rates: BUY '
            f'{pct(res["base_long"])}, SELL {pct(res["base_short"])}.</div></div>',
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2)
        for col, side in ((b1, "BUY"), (b2, "SELL")):
            px = ask if side == "BUY" else bid
            if time_exit:
                tpx = slx = None
            else:
                tpx = px + tp_pts if side == "BUY" else px - tp_pts
                slx = px - sl_pts if side == "BUY" else px + sl_pts
            if col.button(f"📒 I took a {side} @ {px:,.2f}", key=f"take_{side}", width="stretch",
                          type="primary" if side == signal else "secondary",
                          help="Log the trade you just placed at your broker; the journal tracks it live "
                               + (f"and closes it after {hold_txt}." if time_exit else "to TP / SL.")):
                journal.open_trade(tv_symbol, side, px, tpx, slx, lot, ppv,
                                   res["p_long"] if side == "BUY" else res["p_short"],
                                   close_after=int(hold_min) * 60 if time_exit else None)
                st.toast(f"Logged {side} @ {px:,.2f}")
    with right:
        trend = "Up" if res["ctx"]["trend_up"] else "Down" if res["ctx"]["trend_down"] else "Mixed"
        kpis([
            {"icon": "🟢", "label": f"BUY profit after {hold_txt}" if time_exit else "BUY success", "value": pct(res["p_long"]),
             "delta": f"model says {res['raw_long'] * 100:.0f}%" + ("" if rel in ("Good", "Fair") else " (not proven)")
                      + f" · avg {res['exp_long']:+.2f} pts", "delta_value": res["exp_long"]},
            {"icon": "🔴", "label": f"SELL profit after {hold_txt}" if time_exit else "SELL success", "value": pct(res["p_short"]),
             "delta": f"model says {res['raw_short'] * 100:.0f}%" + ("" if rel in ("Good", "Fair") else " (not proven)")
                      + f" · avg {res['exp_short']:+.2f} pts", "delta_value": res["exp_short"]},
            {"icon": "⏳", "label": "Exit", "value": f"after {hold_txt}", "delta": "needs > 50% and avg > 0",
             "delta_value": 0} if time_exit else
            {"icon": "🎯", "label": "TP / SL now", "value": f"{tp_pts:.2f} / {sl_pts:.2f}",
             "delta": f"break-even {params.breakeven * 100:.1f}%", "delta_value": 0},
            {"icon": "🌊", "label": "ATR (volatility)", "value": f"{data['ATR'].iat[qi]:.2f} pts",
             "delta": f"{data['vol_regime'].iat[qi]:.2f}× normal", "delta_value": data["vol_regime"].iat[qi] - 1},
            {"icon": "📒", "label": "Today (journal)", "value": usd(today["usd"], 2),
             "delta": f"{today['closed']} closed · win {pct(today['win_rate'])}", "delta_value": today["usd"]},
            {"icon": "🧭", "label": "Trend", "value": trend, "delta": "EMA 200 + 15m EMA 20",
             "delta_value": {"Up": 1, "Down": -1}.get(trend, 0)},
        ])

    tabs = st.tabs(["📈 Chart", "🔎 Indicators", "🧪 Did it work?", "🕐 Best hours", "⚙️ Optimizer",
                    "📒 Journal", "📜 Signal log"])
    with tabs[0]:
        chart(res, price, entry, tp_px, sl_px, params)
    with tabs[1]:
        est = pd.DataFrame([
            {"Estimate": "Similar past setups (k-NN)", "BUY %": res["knn_long"] * 100, "SELL %": res["knn_short"] * 100},
            {"Estimate": "ML model (gradient boosting)" + ("" if use_ml else " — off"),
             "BUY %": res["ml_long"] * 100, "SELL %": res["ml_short"] * 100},
            {"Estimate": "Combined model estimate", "BUY %": res["raw_long"] * 100, "SELL %": res["raw_short"] * 100},
            {"Estimate": "✅ Accuracy-corrected (shown above)", "BUY %": res["p_long"] * 100,
             "SELL %": res["p_short"] * 100},
        ]).round(1)
        st.markdown("**How the percentage is made**")
        st.dataframe(est, hide_index=True, width="stretch")
        st.caption("Accuracy-corrected = what actually happened in the replay when the model said that number. "
                   f"Market day: {sess.tz}, starts {sess.start_min // 60:02d}:{sess.start_min % 60:02d} local"
                   + (" · US cash-session inputs on" if sess.us_index else "") + ".")
        rows = [{"Indicator": n, "Reading": r, "Bias": "🟢 bullish" if v > 0 else "🔴 bearish" if v < 0 else "⚪ neutral"}
                for n, r, v in res["reasons"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"The odds use {len(res['features'])} inputs: " + ", ".join(res["features"]) + ".")
    with tabs[2]:
        replay(res, params)
    with tabs[3]:
        best_hours(res, params)
    with tabs[4]:
        optimizer(df, params)
    with tabs[5]:
        journal_tab(trades_all)
    with tabs[6]:
        st.dataframe(pd.DataFrame([{k_: v for k_, v in r.items() if k_ != "key"} for r in log[::-1]]),
                     hide_index=True, width="stretch", height=360)


def chart(res, price, entry, tp_px, sl_px, params):
    d = res["data"].iloc[-n_show:]
    idx = local_index(d.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=idx, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price",
                                 increasing_line_color=GREEN, decreasing_line_color=RED), 1, 1)
    for col, color, dash in [("EMA9", CYAN, None), ("EMA21", AMBER, None), ("EMA50", "#a78bfa", None),
                             ("VWAP", "#e8edf7", "dot"), ("BB_upper", MUTED, "dash"), ("BB_lower", MUTED, "dash")]:
        fig.add_trace(go.Scatter(x=idx, y=d[col], name=col.replace("_", " "), mode="lines",
                                 line=dict(color=color, width=1.2, dash=dash)), 1, 1)
    last = d.iloc[-1]
    lo, hi = d["Low"].min(), d["High"].max()
    for col, label in [("PDH", "Yesterday high"), ("PDL", "Yesterday low")]:
        if pd.notna(last.get(col)) and lo - (hi - lo) * 0.2 < last[col] < hi + (hi - lo) * 0.2:
            fig.add_hline(y=last[col], line=dict(color="#f472b6", width=1, dash="dot"), row=1, col=1,
                          annotation_text=label, annotation_position="top left")
    tr = res["trades"]
    if not tr.empty:
        tr = tr[tr["time"] >= d.index[0]]
        for side, sym, color in [("BUY", "triangle-up", GREEN), ("SELL", "triangle-down", RED)]:
            s = tr[tr["side"] == side]
            if len(s):
                fig.add_trace(go.Scatter(
                    x=local_index(s["time"]), y=s["entry"], mode="markers", name=f"Past {side} ({len(s)})",
                    marker=dict(symbol=sym, size=11, color=color, line=dict(color="#fff", width=1)),
                    text=[f"{r} {p:+.2f} pts" for r, p in zip(s["result"], s["points"])],
                    hovertemplate="%{text}<extra></extra>"), 1, 1)
    fig.add_hline(y=price, line=dict(color="#e8edf7", width=1), row=1, col=1,
                  annotation_text=f"Live {price:,.2f}", annotation_position="top right")
    if entry is not None:
        for y, label, color in [(entry, "Entry", CYAN), (tp_px, "TP", GREEN), (sl_px, "SL", RED)]:
            if y is None:
                continue
            fig.add_hline(y=y, line=dict(color=color, width=1.4, dash="dash"), row=1, col=1,
                          annotation_text=f"{label} {y:,.2f}", annotation_position="top right")
    wf = res["walk_forward"]
    wf = wf[wf.index >= d.index[0]]
    if len(wf):
        x = local_index(wf.index)
        fig.add_trace(go.Scatter(x=x, y=wf["p_long"] * 100, name="BUY success %", line=dict(color=GREEN, width=1.4)), 2, 1)
        fig.add_trace(go.Scatter(x=x, y=wf["p_short"] * 100, name="SELL success %", line=dict(color=RED, width=1.4)), 2, 1)
        fig.add_hline(y=params.breakeven * 100, line=dict(color="#fff", width=1, dash="dot"), row=2, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, uirevision=f"{tv_symbol}{tf}")
    st.plotly_chart(style_fig(fig, height=640), width="stretch", config={"displaylogo": False}, key="sc_chart")


def replay(res, params):
    acc = res["accuracy"]
    rel, rel_txt = reliability_label(acc)
    st.markdown(f"**Accuracy check — {rel}.** {rel_txt}")
    if not acc["table"].empty:
        st.caption("Each row: when the model said this range, how often did that trade really make money? Good "
                   "percentages have 'actually won %' close to 'avg predicted %' and rising from top to bottom.")
        st.dataframe(acc["table"].round(1), hide_index=True, width="stretch")
    s = res["stats"]
    st.caption(f"Walk-forward replay of the last {params.validate_frac:.0%} of history: each decision only used "
               "candles whose outcome was already known, same filters, one trade at a time, spread included.")
    if not s.get("trades"):
        st.info("The signal never fired in the replay window with these settings — it found no setup that beat "
                "break-even by your required edge. That's the bot staying out of a market with no clear edge. "
                "Try the Optimizer tab, ATR targets, or a smaller required edge.")
        return
    ok = s["win_rate"] > params.breakeven and s["net_points"] > 0
    kpis([
        {"icon": "🧪", "label": "Trades", "value": f"{s['trades']}",
         "delta": f"BUY {s['BUY_n']} · SELL {s['SELL_n']} · {s['per_day']:.1f}/day", "delta_value": 0},
        {"icon": "✅", "label": "Win rate", "value": pct(s["win_rate"]),
         "delta": f"needs > {params.breakeven * 100:.1f}%", "delta_value": s["win_rate"] - params.breakeven},
        {"icon": "📈", "label": "Net", "value": f"{s['net_points']:+.2f} pts",
         "delta": f"{usd(money(s['net_points']), 2)} at your lot", "delta_value": s["net_points"]},
        {"icon": "⚖️", "label": "Profit factor", "value": f"{s['profit_factor']:.2f}",
         "delta": "> 1.0 = profitable", "delta_value": s["profit_factor"] - 1},
        {"icon": "📉", "label": "Max drawdown", "value": f"{s['max_dd_points']:.2f} pts",
         "delta": f"losing streak {s['worst_streak']} · winning {s['best_streak']}", "delta_value": -1},
    ])
    (st.success if ok else st.warning)(
        "The signal made money after spread in the replay. Past results don't guarantee future ones." if ok else
        "The signal **did not** beat break-even after spread in the replay. Treat the odds as weak evidence "
        "and trade smaller — or not at all — until this turns positive.")
    t = res["trades"].copy()
    eq = go.Figure(go.Scatter(x=local_index(t["exit_time"]), y=t["points"].cumsum().map(money), mode="lines+markers",
                              line=dict(color=GREEN if ok else RED, width=2), name="Equity $"))
    eq.update_yaxes(title_text=f"$ at {lot:.2f} lot")
    st.plotly_chart(style_fig(eq, height=260), width="stretch", key="sc_eq")
    t["time"] = local_index(t["time"]).strftime("%d %b %H:%M")
    t["exit_time"] = local_index(t["exit_time"]).strftime("%H:%M")
    t["success_est"] = (t["success_est"] * 100).round(1)
    t["$ at your lot"] = t["points"].map(money).round(3)
    st.dataframe(t.iloc[::-1].rename(columns={"success_est": "est. success %"}), hide_index=True,
                 width="stretch", height=280)


def best_hours(res, params):
    h = hourly_edge(res["labelled"], params)
    if h.empty:
        return
    off = int(datetime.now(TZ).utcoffset().total_seconds() // 60)
    h.index = [f"{((u * 60 + off) % 1440) // 60:02d}:{((u * 60 + off) % 60):02d}" for u in h.index]
    h = h.sort_index()
    st.caption(f"What a blind BUY or SELL with your TP/SL did, by hour ({tz_name}). Hours with a higher ATR move "
               "enough to reach your target; quiet hours mostly pay spread. Use this to set Trading hours.")
    fig = go.Figure()
    fig.add_bar(x=h.index, y=h["BUY avg pts"], name="BUY avg pts", marker_color=GREEN)
    fig.add_bar(x=h.index, y=h["SELL avg pts"], name="SELL avg pts", marker_color=RED)
    fig.add_scatter(x=h.index, y=h["ATR"], name="ATR", yaxis="y2", line=dict(color=AMBER))
    fig.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", title="ATR", showgrid=False))
    st.plotly_chart(style_fig(fig, height=320), width="stretch", key="sc_hours")
    st.dataframe(h.round(2), width="stretch")


def _apply_opt(over: dict):
    if "hold" in over:
        st.session_state["sc_hold_min"] = int(over["minutes"])
    elif st.session_state.get("sc_mode") == "Points":
        st.session_state["sc_tp"], st.session_state["sc_sl"] = float(over["tp"]), float(over["sl"])
    else:
        st.session_state["sc_tp_atr"], st.session_state["sc_sl_atr"] = float(over["tp"]), float(over["sl"])
