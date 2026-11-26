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
