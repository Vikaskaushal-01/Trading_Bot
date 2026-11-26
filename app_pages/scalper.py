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
