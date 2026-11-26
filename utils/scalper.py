"""Intraday scalping signal engine.

For every candle we know, we ask: "if I had bought (or sold) at this close — exiting at TP / SL, or after a fixed
number of minutes — did I make money after spread?" That labels every candle for both sides. For the current
candle two independent estimators then give the chance of success:

* **Similar setups (k-NN)** — how often a BUY / SELL worked from the most alike past candles.
* **ML model (gradient boosting)** — trained on every labelled candle to learn which indicator combinations
  lead to wins.

The final percentage is their average. Everything is also replayed walk-forward (each decision only uses candles
whose outcome was already known, models retrained on the past only) to show whether the percentages were
accurate and whether the signal made money.

Market-specific details come from the symbol's own trading session (e.g. SPREADEX:SPX trades 18:00–17:00
New York time), so "today", VWAP, the daily high/low and yesterday's levels reset when that market's day starts,
and S&P 500 cash-session context (09:30 New York open, opening range, gap from the previous close) is included.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

NY = "America/New_York"


@dataclass(frozen=True)
class ScalpParams:
    tp: float = 4.0              # take-profit: points, or ATR multiples when tp_mode == "atr" (net of spread)
    sl: float = 3.0              # stop-loss: same unit as tp
    spread: float = 0.6          # broker spread, price points
    horizon: int = 10            # max candles to hold before closing at market (TP/SL mode)
    k: int = 150                 # neighbours used for the similar-setups estimate
    min_edge: float = 0.05       # success rate must beat break-even by this much
    validate_frac: float = 0.35  # share of history replayed walk-forward
    tp_mode: str = "points"      # "points" | "atr"
    trend_filter: bool = False   # BUY only in an up-trend, SELL only in a down-trend (EMA 200 + 15m trend)
    session: tuple | None = None  # your trading hours: (start, end) minutes after 00:00 UTC; None = always
    high_conf: bool = False      # require the 90% lower confidence bound to clear break-even
    exit_mode: str = "tp_sl"     # "tp_sl" = exit at TP / SL (or max hold) | "time" = close after `hold` candles
    hold: int = 3                # candles held in "time" mode (3 × 1m = close after 3 minutes)
    use_ml: bool = True          # blend in the gradient-boosting model
    require_skill: bool = True   # only signal a side whose percentages proved accurate in the replay
    min_auc: float = 0.53        # "proved accurate" = walk-forward ROC-AUC at least this (0.50 = coin flip)

    @property
    def breakeven(self) -> float:
        """Win rate needed not to lose money. Time exits win and lose roughly equal amounts, so ~50%."""
        return 0.5 if self.exit_mode == "time" else self.sl / (self.tp + self.sl)

    @property
    def span(self) -> int:
        """Candles until every trade opened on a candle is finished."""
        return self.hold if self.exit_mode == "time" else self.horizon

    def label_key(self) -> tuple:
        """Everything that changes the trade outcomes / models (not the filters)."""
        return (self.tp, self.sl, self.spread, self.horizon, self.k, self.validate_frac, self.tp_mode,
                self.exit_mode, self.hold, self.use_ml)

    def decision_key(self) -> tuple:
        return (self.min_edge, self.trend_filter, self.session, self.high_conf, self.require_skill, self.min_auc)


@dataclass(frozen=True)
class MarketSession:
    """The symbol's own trading day, from TradingView (e.g. tz America/New_York, session '1800-1700')."""
    tz: str = "UTC"
    start_min: int = 0           # minutes after local midnight when the trading day begins
    us_index: bool = False       # add US cash-session context (09:30–16:00 New York)

    @classmethod
    def from_info(cls, info: dict, symbol: str = "") -> "MarketSession":
        tz = info.get("timezone") or "UTC"
        m = re.match(r"(\d{2})(\d{2})-", str(info.get("session") or ""))
        start = int(m.group(1)) * 60 + int(m.group(2)) if m else 0
        sym = symbol.upper()
        us = any(s in sym for s in ("SPX", "SPY", "ES1", "US500", "NSX", "NAS", "NDX", "US100", "DJI", "US30", "DOW"))
        return cls(tz=tz, start_min=start, us_index=us)


FEATURES = {
    "dist_ema9": "Close vs EMA 9 (ATR units)",
    "ema9_21": "EMA 9 vs EMA 21 — short-term trend",
    "dist_ema50": "Close vs EMA 50",
    "dist_ema200": "Close vs EMA 200",
    "rsi": "RSI 14",
    "rsi7": "RSI 7 (fast)",
    "stoch_k": "Stochastic RSI %K",
    "macd_hist": "MACD histogram",
    "macd_slope": "MACD histogram change",
    "bb_pct": "Bollinger %B",
    "bb_width": "Bollinger band width (squeeze)",
    "adx_dir": "ADX × direction (DI+ vs DI−)",
    "range_pos": "Position in 20-candle high/low range",
    "ret1": "Last candle return",
    "ret5": "5-candle return",
    "ret15": "15-candle return",
    "ret60": "60-candle return",
    "lower_wick": "Lower wick (buying pressure)",
    "upper_wick": "Upper wick (selling pressure)",
    "body": "Candle body (direction × size)",
    "vol_regime": "ATR vs its 100-candle average",
    "vwap_dist": "Close vs session VWAP",
    "vol_z": "Volume z-score",
    "htf5_rsi": "RSI 14 on 5-minute candles",
    "htf15_trend": "Close vs EMA 20 on 15-minute candles",
    "day_hi_dist": "Distance below today's high",
    "day_lo_dist": "Distance above today's low",
    "pdh_dist": "Close vs yesterday's high",
    "pdl_dist": "Close vs yesterday's low",
    "prev_close_dist": "Close vs yesterday's close (gap)",
    "cash_open": "US cash market open (09:30–16:00 New York)",
    "mins_cash": "Minutes since the 09:30 New York open",
    "or_hi_dist": "Close vs opening-range high (09:30–09:45 NY)",
    "or_lo_dist": "Close vs opening-range low (09:30–09:45 NY)",
    "tod_sin": "Time of day in the market's time zone (sin)",
    "tod_cos": "Time of day in the market's time zone (cos)",
}

AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
US_ONLY = ("cash_open", "mins_cash", "or_hi_dist", "or_lo_dist")


def _base_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 3:
        return 1
    return max(1, int(pd.Series(index).diff().dt.total_seconds().median() // 60))


def _htf(df: pd.DataFrame, rule: str, base: int) -> pd.DataFrame:
    """Completed higher-timeframe candles, aligned so each base row only sees bars that had already closed."""
    h = df.resample(rule, label="right", closed="left").agg(AGG).dropna(subset=["Close"])
    h.index = h.index - pd.Timedelta(minutes=base)  # bar ending at L is known at the base row opening at L - base
    return h


def add_scalp_indicators(df: pd.DataFrame, sess: MarketSession | None = None) -> pd.DataFrame:
    sess = sess or MarketSession()
    df = df.copy()
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    atr = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range().replace(0, np.nan)
    df["ATR"] = atr
    for n in (9, 21, 50, 200):
        df[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    df["RSI"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    df["RSI7"] = ta.momentum.RSIIndicator(c, window=7).rsi()
    df["StochK"] = ta.momentum.StochRSIIndicator(c).stochrsi_k() * 100
    df["MACD_hist"] = ta.trend.MACD(c).macd_diff()
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["BB_upper"], df["BB_lower"], df["BB_pct"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_pband()
    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df["ADX"], df["DI_pos"], df["DI_neg"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
    df["HH20"], df["LL20"] = h.rolling(20).max(), l.rolling(20).min()

    # ---- the market's own trading day (e.g. 18:00 New York for SPREADEX:SPX)
    local = df.index.tz_localize("UTC").tz_convert(sess.tz)
    day = (local - pd.Timedelta(minutes=sess.start_min)).tz_localize(None).normalize()
    day = pd.Index(day, name="day")
    tp_ = (h + l + c) / 3
    w = v if (v > 0).any() else pd.Series(1.0, index=df.index)
    df["VWAP"] = (tp_ * w).groupby(day).cumsum().to_numpy() / w.groupby(day).cumsum().to_numpy()
    df["DayHigh"] = h.groupby(day).cummax().to_numpy()
    df["DayLow"] = l.groupby(day).cummin().to_numpy()
    daily = df.groupby(day).agg(PDH=("High", "max"), PDL=("Low", "min"), PDC=("Close", "last")).shift(1)
    for col in ("PDH", "PDL", "PDC"):
        df[col] = daily[col].reindex(day).to_numpy()

    base = _base_minutes(df.index)
    h5, h15 = _htf(df, "5min", base), _htf(df, "15min", base)
    h5["RSI5"] = ta.momentum.RSIIndicator(h5["Close"], window=14).rsi()
    h15["EMA20_15"] = h15["Close"].ewm(span=20, adjust=False).mean()
    df["RSI5"] = h5["RSI5"].reindex(df.index, method="ffill")
    df["EMA20_15"] = h15["EMA20_15"].reindex(df.index, method="ffill")

    # ---- US cash session (09:30–16:00 New York) and its opening range
    ny = df.index.tz_localize("UTC").tz_convert(NY)
    ny_min = pd.Series(ny.hour * 60 + ny.minute, index=df.index)
    weekday = pd.Series(ny.dayofweek < 5, index=df.index)
    df["CashOpen"] = ((ny_min >= 570) & (ny_min < 960) & weekday).astype(float)
    ny_day = pd.Index(ny.tz_localize(None).normalize(), name="nyday")
    in_or = (ny_min >= 570) & (ny_min < 585)
    or_hi = h.where(in_or).groupby(ny_day).transform("max").to_numpy()
    or_lo = l.where(in_or).groupby(ny_day).transform("min").to_numpy()
    formed = (ny_min >= 585).to_numpy()
    df["ORHigh"] = np.where(formed, or_hi, np.nan)
    df["ORLow"] = np.where(formed, or_lo, np.nan)

    f = pd.DataFrame(index=df.index)
    f["dist_ema9"] = (c - df["EMA9"]) / atr
    f["ema9_21"] = (df["EMA9"] - df["EMA21"]) / atr
    f["dist_ema50"] = (c - df["EMA50"]) / atr
    f["dist_ema200"] = (c - df["EMA200"]) / atr
    f["rsi"] = df["RSI"] / 100
    f["rsi7"] = df["RSI7"] / 100
    f["stoch_k"] = df["StochK"] / 100
    f["macd_hist"] = df["MACD_hist"] / atr
    f["macd_slope"] = df["MACD_hist"].diff() / atr
    f["bb_pct"] = df["BB_pct"]
    f["bb_width"] = (df["BB_upper"] - df["BB_lower"]) / atr
    f["adx_dir"] = df["ADX"] / 100 * np.sign(df["DI_pos"] - df["DI_neg"])
    f["range_pos"] = (c - df["LL20"]) / (df["HH20"] - df["LL20"]).replace(0, np.nan)
    f["ret1"] = c.diff() / atr
    f["ret5"] = c.diff(5) / atr
    f["ret15"] = c.diff(15) / atr
    f["ret60"] = c.diff(60) / atr
    body_lo, body_hi = np.minimum(o, c), np.maximum(o, c)
    f["lower_wick"] = (body_lo - l) / atr
    f["upper_wick"] = (h - body_hi) / atr
    f["body"] = (c - o) / atr
    f["vol_regime"] = atr / atr.rolling(100).mean()
    f["vwap_dist"] = (c - df["VWAP"]) / atr
    f["vol_z"] = ((v - v.rolling(50).mean()) / v.rolling(50).std()).fillna(0) if (v > 0).any() else 0.0
    f["htf5_rsi"] = df["RSI5"] / 100
    f["htf15_trend"] = (c - df["EMA20_15"]) / atr
    f["day_hi_dist"] = (c - df["DayHigh"]) / atr
    f["day_lo_dist"] = (c - df["DayLow"]) / atr
    f["pdh_dist"] = (c - df["PDH"]) / atr
    f["pdl_dist"] = (c - df["PDL"]) / atr
    f["prev_close_dist"] = (c - df["PDC"]) / atr
    f["cash_open"] = df["CashOpen"]
    f["mins_cash"] = ((ny_min - 570).clip(-120, 450) / 450).where(weekday, -120 / 450)
    f["or_hi_dist"] = ((c - df["ORHigh"]) / atr).fillna(0)
    f["or_lo_dist"] = ((c - df["ORLow"]) / atr).fillna(0)
    loc_min = local.hour * 60 + local.minute
    f["tod_sin"] = np.sin(2 * np.pi * loc_min / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * loc_min / 1440)
    return df.join(f.clip(-10, 10))


def feature_list(df: pd.DataFrame, sess: MarketSession) -> list[str]:
    feats = [f for f in FEATURES if f in df.columns]
    if not (df["Volume"] > 0).any():
        feats.remove("vol_z")
    if not sess.us_index:
        feats = [f for f in feats if f not in US_ONLY]
    return feats


def trade_levels(data: pd.DataFrame, p: ScalpParams) -> tuple[np.ndarray, np.ndarray]:
    """TP and SL distance in price points for every row (NaN in time-exit mode)."""
    if p.exit_mode == "time":
        return np.full(len(data), np.nan), np.full(len(data), np.nan)
    if p.tp_mode == "atr":
        atr = data["ATR"].to_numpy(float)
        return p.tp * atr, p.sl * atr
    n = len(data)
    return np.full(n, p.tp), np.full(n, p.sl)


def simulate_outcomes(df: pd.DataFrame, p: ScalpParams) -> pd.DataFrame:
    """Spread-adjusted result of a long and a short opened at every close.

    Candle prices are treated as bid. A long buys at ask (close + spread) and exits at bid; a short sells at
    bid and buys back at ask (bid + spread). If TP and SL fall inside the same candle we assume the stop
    was hit first (conservative). Positions still open after `horizon` candles close at market.
    """
    c, h, l = df["Close"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy()
    n = len(df)
    out = {}
    if p.exit_mode == "time":  # buy/sell at this close, close exactly `hold` candles later at market
        idx = np.arange(n) + p.hold
        ok = idx < n
        later = np.full(n, np.nan)
        later[ok] = c[idx[ok]]
        long_pts = later - (c + p.spread)        # buy at ask, sell back at bid
        short_pts = c - (later + p.spread)       # sell at bid, buy back at ask
        exit_i = np.where(ok, idx, -1)
        for side, pts in (("long", long_pts), ("short", short_pts)):
            out[f"{side}_pts"], out[f"{side}_win"], out[f"{side}_exit"] = pts, ok & (pts > 0), exit_i
        return pd.DataFrame(out, index=df.index)
    tp, sl = trade_levels(df, p)
    for side in ("long", "short"):
        pts = np.full(n, np.nan)
        win = np.zeros(n, dtype=bool)
        exit_i = np.full(n, -1)
        open_ = ~np.isnan(tp)
        if side == "long":
            entry = c + p.spread
            tp_lvl, sl_lvl = entry + tp, entry - sl
        else:
            entry = c
            tp_lvl, sl_lvl = entry - tp, entry + sl
        for step in range(1, p.horizon + 1):
            idx = np.arange(n) + step
            ok = idx < n
            hi, lo, cl = (np.full(n, np.nan) for _ in range(3))
            hi[ok], lo[ok], cl[ok] = h[idx[ok]], l[idx[ok]], c[idx[ok]]
            if side == "long":
                hit_sl = open_ & ok & (lo <= sl_lvl)
                hit_tp = open_ & ok & ~hit_sl & (hi >= tp_lvl)
            else:
                hit_sl = open_ & ok & (hi + p.spread >= sl_lvl)
                hit_tp = open_ & ok & ~hit_sl & (lo + p.spread <= tp_lvl)
            pts[hit_sl], exit_i[hit_sl] = -sl[hit_sl], idx[hit_sl]
            pts[hit_tp], win[hit_tp], exit_i[hit_tp] = tp[hit_tp], True, idx[hit_tp]
            open_ &= ~(hit_sl | hit_tp)
            if step == p.horizon:  # time exit at market
                last = open_ & ok
                pts[last] = (cl[last] - entry[last]) if side == "long" else (entry[last] - cl[last] - p.spread)
                exit_i[last] = idx[last]
        known = np.arange(n) + p.horizon < n
        pts[~known], exit_i[~known] = np.nan, -1
        out[f"{side}_pts"], out[f"{side}_win"], out[f"{side}_exit"] = pts, win & known, exit_i
    return pd.DataFrame(out, index=df.index)


# ------------------------------------------------------------------ estimators
def _wilson_low(p: np.ndarray, n: int, z: float = 1.2816) -> np.ndarray:
    """One-sided 90% lower confidence bound of a success rate."""
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    return (centre - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom


def _knn(Zh, yl, ys, pl, ps, Zq, k, prior_l, prior_s, prior_w=20) -> dict:
    k = min(k, len(Zh))
    d = (Zq ** 2).sum(1)[:, None] + (Zh ** 2).sum(1)[None, :] - 2 * Zq @ Zh.T
    nn = np.argpartition(d, k - 1, axis=1)[:, :k]
    raw_l, raw_s = yl[nn].mean(1), ys[nn].mean(1)
    return {
        "knn_long": (raw_l * k + prior_w * prior_l) / (k + prior_w),
        "knn_short": (raw_s * k + prior_w * prior_s) / (k + prior_w),
        "lcb_long": _wilson_low(raw_l, k), "lcb_short": _wilson_low(raw_s, k),
        "exp_long": pl[nn].mean(1), "exp_short": ps[nn].mean(1),
    }


def _fit_ml(X, y):
    if len(np.unique(y)) < 2 or len(y) < 300:
        return None
    m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=60,
                                       l2_regularization=1.0, random_state=0)
    return m.fit(X, y)


def _ml_prob(model, X, fallback: float) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if model is not None else np.full(len(X), fallback)


def _blend(r: dict, ml_l, ml_s, use_ml: bool) -> dict:
    r = dict(r)
    r["ml_long"], r["ml_short"] = ml_l, ml_s
    r["p_long"] = (r["knn_long"] + ml_l) / 2 if use_ml else r["knn_long"]
    r["p_short"] = (r["knn_short"] + ml_s) / 2 if use_ml else r["knn_short"]
    return r


def _auc(won: np.ndarray, prob: np.ndarray) -> float:
    if len(won) < 200 or len(np.unique(won)) < 2:
        return np.nan
    return float(roc_auc_score(won, prob))


P_MIN, P_MAX = 0.05, 0.90  # no 3-minute trade is ever (nearly) certain — never show more extreme odds


class _Platt:
    """Smooth, monotone accuracy correction: logistic fit of real outcomes on the model's log-odds.
    Unlike isotonic regression it cannot jump to 0% / 100% on a handful of unusual candles."""

    def __init__(self, prob: np.ndarray, won: np.ndarray):
        x = np.log(np.clip(prob, 0.01, 0.99) / (1 - np.clip(prob, 0.01, 0.99)))[:, None]
        self.m = LogisticRegression(C=0.5).fit(x, won)
        # A backwards relation (higher % → fewer wins) is treated as "no skill", never flipped into a signal.
        self.flat = float(np.mean(won)) if self.m.coef_[0, 0] <= 0 else None

    def predict(self, prob) -> np.ndarray:
        prob = np.clip(np.asarray(prob, dtype=float), 0.01, 0.99)
        if self.flat is not None:
            return np.clip(np.full(len(prob), self.flat), P_MIN, P_MAX)
        x = np.log(prob / (1 - prob))[:, None]
        return np.clip(self.m.predict_proba(x)[:, 1], P_MIN, P_MAX)


def _iso(prob: np.ndarray, won: np.ndarray):
    if len(prob) < 300 or len(np.unique(won)) < 2:
        return None
    return _Platt(prob, won)


def _calibrate_walk_forward(wf: pd.DataFrame, data: pd.DataFrame, p: ScalpParams) -> pd.DataFrame:
    """Add raw_*, accuracy-corrected p_* and auc_* columns using only earlier replay rows (no peeking)."""
    if wf.empty:
        return wf.assign(raw_long=[], raw_short=[], auc_long=[], auc_short=[])
    wf = wf.copy()
    wf["raw_long"], wf["raw_short"] = wf["p_long"], wf["p_short"]
    pos = data.index.get_indexer(wf.index)
    won = {s_: data[f"{s_}_win"].to_numpy(float)[pos] for s_ in ("long", "short")}
    out = {f"{c_}_{s_}": np.full(len(wf), np.nan) for c_ in ("p", "auc") for s_ in ("long", "short")}
    for chunk in np.array_split(np.arange(len(wf)), max(1, len(wf) // 200)):
        prior = np.where(pos <= pos[chunk[0]] - p.span)[0]
        for s_ in ("long", "short"):
            raw = wf[f"raw_{s_}"].to_numpy(float)
            iso = _iso(raw[prior], won[s_][prior])
            out[f"p_{s_}"][chunk] = iso.predict(raw[chunk]) if iso is not None else np.clip(raw[chunk], P_MIN, P_MAX)
            out[f"auc_{s_}"][chunk] = _auc(won[s_][prior], raw[prior])
    for k_, v in out.items():
        wf[k_] = v
    return wf


def _fit_calibration(wf: pd.DataFrame, data: pd.DataFrame) -> dict:
    """Calibrators and skill from the whole replay, used for the live candle."""
    cal = {}
    for s_ in ("long", "short"):
        if wf.empty:
            cal[s_], cal[f"auc_{s_}"] = None, np.nan
            continue
        raw = wf[f"raw_{s_}"].to_numpy(float)
        won = data.loc[wf.index, f"{s_}_win"].to_numpy(float)
        cal[s_], cal[f"auc_{s_}"] = _iso(raw, won), _auc(won, raw)
    return cal


def _apply_calibration(r: dict, cal: dict) -> dict:
    r = dict(r)
    for s_ in ("long", "short"):
        r[f"raw_{s_}"] = r[f"p_{s_}"]
        if cal.get(s_) is not None:
            r[f"p_{s_}"] = float(cal[s_].predict([r[f"p_{s_}"]])[0])
        else:
            r[f"p_{s_}"] = float(np.clip(r[f"p_{s_}"], P_MIN, P_MAX))
        r[f"auc_{s_}"] = cal.get(f"auc_{s_}", np.nan)
    return r


def _context(data: pd.DataFrame, p: ScalpParams) -> pd.DataFrame:
    """Per-row filter state: in trading hours, trend allows long / short."""
    minutes = data.index.hour * 60 + data.index.minute
    if p.session:
        s, e = p.session
        in_sess = (minutes >= s) & (minutes < e) if s < e else (minutes >= s) | (minutes < e)
    else:
        in_sess = np.ones(len(data), dtype=bool)
    up = (data["Close"] > data["EMA200"]) & (data["htf15_trend"] > 0)
    down = (data["Close"] < data["EMA200"]) & (data["htf15_trend"] < 0)
    return pd.DataFrame({"in_session": np.asarray(in_sess), "trend_up": up.to_numpy(), "trend_down": down.to_numpy()},
                        index=data.index)


def decide(r: dict, ctx: dict, p: ScalpParams) -> tuple[str, str]:
    """Signal and a one-line reason."""
    if not ctx["in_session"]:
        return "WAIT", "Outside your trading hours."
    need = p.breakeven + p.min_edge

    def skilled(side):
        auc = r.get(f"auc_{side}", np.nan)
        return not p.require_skill or (auc is not None and not np.isnan(auc) and auc >= p.min_auc)

    def ok(side):
        confident = r[f"lcb_{side}"] >= p.breakeven or not p.high_conf
        trend = ctx["trend_up"] if side == "long" else ctx["trend_down"]
        return (r[f"p_{side}"] >= need and confident and r[f"exp_{side}"] > 0 and skilled(side)
                and (trend or not p.trend_filter))

    long_ok, short_ok = ok("long"), ok("short")
    if long_ok and (not short_ok or r["exp_long"] >= r["exp_short"]):
        return "BUY", "BUY has a good chance and similar past setups made money on average."
    if short_ok:
        return "SELL", "SELL has a good chance and similar past setups made money on average."
    best = "long" if r["p_long"] >= r["p_short"] else "short"
    name = "BUY" if best == "long" else "SELL"
    if r[f"p_{best}"] >= need and not skilled(best):
        return "WAIT", (f"{name} shows {r[f'p_{best}'] * 100:.0f}%, but the percentages haven't proved accurate "
                        "recently, so no signal.")
    if r[f"p_{best}"] >= need and r[f"exp_{best}"] <= 0:
        return "WAIT", f"{name} wins often enough, but the losses were bigger than the wins on average."
    if p.trend_filter and r[f"p_{best}"] >= need:
        return "WAIT", f"{name} looks good but goes against the trend filter."
    if p.high_conf and r[f"p_{best}"] >= need:
        return "WAIT", "Odds look good but not with high confidence yet."
    return "WAIT", f"Neither side reaches the {need * 100:.0f}% needed — no clear edge right now."


# ------------------------------------------------------------------ heavy part: labels, models, walk-forward
def build(raw: pd.DataFrame, p: ScalpParams, sess: MarketSession | None = None) -> dict:
    """Labels every candle, fits the models and runs the walk-forward replay. Slow (a few seconds); cache it."""
    sess = sess or MarketSession()
    df = add_scalp_indicators(raw, sess)
    feats = feature_list(df, sess)
    data = df.join(simulate_outcomes(df, p))
    n = len(data)
    X = data[feats].to_numpy(dtype=float)
    valid = ~np.isnan(X).any(axis=1)
    if valid.sum() < 400:
        raise ValueError("Not enough candles yet — need at least ~400 with indicators warmed up.")
    mu, sd = X[valid].mean(0), X[valid].std(0)
    sd[sd == 0] = 1
    w = np.ones(len(feats))
    for f_ in ("tod_sin", "tod_cos"):
        w[feats.index(f_)] = 0.7  # time of day matters, but less than the price picture
    Z = (X - mu) / sd * w

    yl, ys = data["long_win"].to_numpy(float), data["short_win"].to_numpy(float)
    pl, ps = data["long_pts"].to_numpy(float), data["short_pts"].to_numpy(float)
    known = valid & ~np.isnan(pl) & ~np.isnan(ps)
    pos = np.arange(n)

    # ---- walk-forward: k-NN refreshed every 200 candles, ML retrained every 750, on the past only
    eligible = np.where(known)[0]
    start = eligible[int(len(eligible) * (1 - p.validate_frac))] if len(eligible) > 600 else n
    wf_idx = eligible[eligible >= start]
    parts = []
    for block in np.array_split(wf_idx, max(1, len(wf_idx) // 1200)) if len(wf_idx) else []:
        past = np.where(known & (pos <= block.min() - p.span))[0]
        if len(past) < max(p.k, 300):
            continue
        ml_l = ml_s = None
        if p.use_ml:
            ml_l, ml_s = _fit_ml(X[past], yl[past]), _fit_ml(X[past], ys[past])
        for chunk in np.array_split(block, max(1, len(block) // 200)):
            past_c = np.where(known & (pos <= chunk.min() - p.span))[0]
            r = _knn(Z[past_c], yl[past_c], ys[past_c], pl[past_c], ps[past_c], Z[chunk], p.k,
                     yl[past_c].mean(), ys[past_c].mean())
            r = _blend(r, _ml_prob(ml_l, X[chunk], yl[past].mean()), _ml_prob(ml_s, X[chunk], ys[past].mean()), p.use_ml)
            parts.append(pd.DataFrame(r, index=data.index[chunk]))
    wf = pd.concat(parts) if parts else pd.DataFrame(
        columns=["knn_long", "knn_short", "lcb_long", "lcb_short", "exp_long", "exp_short", "ml_long", "ml_short",
                 "p_long", "p_short"])

    # ---- accuracy correction, also walk-forward: each chunk is calibrated (and its skill measured) only from
    #      earlier replay candles whose outcome was already known
    wf = _calibrate_walk_forward(wf, data, p)
    cal = _fit_calibration(wf, data)

    # ---- final models on everything known, used to score live candles
    hist = np.where(known)[0]
    models = (_fit_ml(X[hist], yl[hist]), _fit_ml(X[hist], ys[hist])) if p.use_ml else (None, None)
    return {
        "p": p, "sess": sess, "data": data, "feats": feats, "mu": mu, "sd": sd, "w": w,
        "Zh": Z[hist], "yl": yl[hist], "ys": ys[hist], "pl": pl[hist], "ps": ps[hist],
        "base_long": float(yl[hist].mean()), "base_short": float(ys[hist].mean()), "n_hist": int(len(hist)),
        "models": models, "wf": wf, "cal": cal, "built_at": data.index[-1],
    }


# ------------------------------------------------------------------ light part: score the live candle
def score(raw: pd.DataFrame, model: dict, p: ScalpParams, use_last_bar: bool = False) -> dict:
    """Decision for the current candle using a (possibly few minutes old) built model. Fast."""
    sess, feats = model["sess"], model["feats"]
    df = add_scalp_indicators(raw.tail(3000), sess)
    q = len(df) - 1 if use_last_bar else len(df) - 2
    row = df.iloc[q]
    x = row[feats].to_numpy(dtype=float)[None, :]
    if np.isnan(x).any():
        raise ValueError("Indicators are still warming up for the latest candle — give it a minute.")
    z = (x - model["mu"]) / model["sd"] * model["w"]
    r = _knn(model["Zh"], model["yl"], model["ys"], model["pl"], model["ps"], z, p.k,
             model["base_long"], model["base_short"])
    ml_l, ml_s = model["models"]
    r = _blend({k_: float(v[0]) for k_, v in r.items()},
               float(_ml_prob(ml_l, x, model["base_long"])[0]), float(_ml_prob(ml_s, x, model["base_short"])[0]),
               p.use_ml)
    r = _apply_calibration(r, model["cal"])
    ctx = _context(df.iloc[[q]], p).iloc[0].to_dict()
    signal, why = decide(r, ctx, p)

    # walk-forward decisions with the current filters (cheap: probabilities are already in the model)
    data = model["data"]
    wf = model["wf"]
    if len(wf):
        wctx = _context(data.loc[wf.index], p)
        sig = [decide(rw, cw, p)[0] for rw, cw in zip(wf.to_dict("records"), wctx.to_dict("records"))]
        wf = wf.assign(signal=sig)
    else:
        wf = wf.assign(signal=[])
    trades = _replay_trades(data, wf, p)
    tp_arr, sl_arr = trade_levels(df.iloc[[q]], p)
    return {
        "data": df, "features": feats, "q": q, "signal": signal, "why": why, **r, "ctx": ctx,
        "tp_pts": float(tp_arr[0]), "sl_pts": float(sl_arr[0]),
        "base_long": model["base_long"], "base_short": model["base_short"], "n_hist": model["n_hist"],
        "walk_forward": wf, "trades": trades, "stats": trade_stats(trades, p),
        "accuracy": accuracy(data, wf, p), "reasons": explain(row), "built_at": model["built_at"], "labelled": data,
    }


def analyze(raw: pd.DataFrame, p: ScalpParams, use_last_bar: bool = False, sess: MarketSession | None = None) -> dict:
    """Build + score in one go (used by the optimizer and tests)."""
    return score(raw, build(raw, p, sess), p, use_last_bar)
