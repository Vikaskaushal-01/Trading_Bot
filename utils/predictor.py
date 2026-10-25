"""Per-asset ML prediction engine.

* Target: will the close be higher `horizon` trading days from now?
* Validation: purged walk-forward (5 folds, a gap of `horizon` days between
  train and test so overlapping labels never leak). Every metric shown in the
  app comes from these out-of-sample predictions.
* Extras: quantile regression for an expected price range, historical analogs
  (most similar past setups) and threshold-aware strategy evaluation.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.features import (
    DEFAULT_GROUPS, FEATURE_GROUPS, FEATURE_INFO, benchmark_for, compute_features, usable_columns,
)
from utils.indicators import add_indicators, technical_rating
from utils.market import get_currency, get_history

HORIZONS = {"1 day": 1, "1 week": 5, "2 weeks": 10, "1 month": 21}
MODELS = {
    "Ensemble": "Average of Gradient Boosting, Random Forest and Logistic Regression — most robust.",
    "Gradient Boosting": "Sequential shallow trees; captures non-linear interactions.",
    "Random Forest": "Many decorrelated deep-ish trees; stable, resists overfitting.",
    "Extra Trees": "Like Random Forest with random splits; lower variance.",
    "Logistic Regression": "Linear baseline; transparent and hard to overfit.",
}
HISTORY = {"3 years": "3y", "5 years": "5y", "10 years": "10y"}
N_SPLITS = 5

DEFAULT_SETTINGS = {
    "horizon": "1 week", "model": "Ensemble", "history": "5 years",
    "groups": tuple(DEFAULT_GROUPS), "edge": 0.04,
}


def get_settings() -> dict:
    """AI settings chosen in the sidebar (falls back to defaults outside the app)."""
    s = dict(DEFAULT_SETTINGS)
    try:
        for k in s:
            if f"ai_{k}" in st.session_state:
                s[k] = st.session_state[f"ai_{k}"]
    except Exception:
        pass
    s["groups"] = tuple(g for g in FEATURE_GROUPS if g in (s["groups"] or DEFAULT_GROUPS)) or tuple(DEFAULT_GROUPS)
    return s


# ---------------- models ---------------- #

class _Ensemble:
    def __init__(self):
        self.members = [_make("Gradient Boosting"), _make("Random Forest"), _make("Logistic Regression")]

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)


def _make(name: str):
    if name == "Gradient Boosting":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=150,
                                              l2_regularization=1.0, min_samples_leaf=40, random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                      max_features="sqrt", n_jobs=-1, random_state=42)
    if name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=250, max_depth=7, min_samples_leaf=20,
                                    max_features="sqrt", n_jobs=-1, random_state=42)
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    if name == "Ensemble":
        return _Ensemble()
    raise ValueError(name)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def reliability(auc: float) -> tuple[float, str]:
    """Skill weight in [0, 1] from out-of-sample ROC-AUC, plus a label."""
    skill = float(np.clip((auc - 0.5) / 0.06, 0, 1))
    label = ("Strong" if auc >= 0.56 else "Moderate" if auc >= 0.53 else
             "Weak" if auc >= 0.51 else "No edge")
    return skill, label


# ---------------- training (cached, threshold independent) ---------------- #

@st.cache_data(ttl=1800, show_spinner=False, max_entries=200)
def train(symbol: str, horizon: int, model: str, history: str, groups: tuple) -> dict | None:
    period = HISTORY.get(history, history)
    hist = get_history(symbol, period=period)
    if len(hist) < 400:
        return None

    market = vix = None
    if "Market context" in groups:
        market = get_history(benchmark_for(symbol, get_currency(symbol)), period=period)
        vix = get_history("^VIX", period=period)

    feats = compute_features(hist, market, vix)
    cols = usable_columns(feats, list(groups))
    if len(cols) < 3:
        return None

    close = hist["Close"]
    fwd = close.shift(-horizon) / close - 1
    X_all = feats[cols]
    valid = X_all.notna().all(axis=1)
    if not valid.any():
        return None
    latest_date = X_all.index[valid][-1]
    latest = X_all.loc[[latest_date]]

    labeled = valid & fwd.notna()
    X, ret = X_all[labeled], fwd[labeled]
    y = (ret > 0).astype(int)
    if len(X) < 300 or y.nunique() < 2:
        return None

    # Purged walk-forward validation.
    oos = pd.Series(np.nan, index=X.index)
    for tr, te in TimeSeriesSplit(n_splits=N_SPLITS, gap=horizon).split(X):
        if y.iloc[tr].nunique() < 2:
            continue
        oos.iloc[te] = _make(model).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    oos_raw = oos.dropna()
    y_oos, ret_oos = y.loc[oos_raw.index], ret.loc[oos_raw.index]

    # Platt calibration on out-of-sample predictions: makes "60%" mean 60% of such
    # days actually went up, and collapses towards the base rate when there is no skill.
    calibrator = LogisticRegression(C=1.0).fit(_logit(oos_raw.values), y_oos)
    oos = pd.Series(calibrator.predict_proba(_logit(oos_raw.values))[:, 1], index=oos_raw.index)

    final = _make(model).fit(X, y)
    prob_raw = float(final.predict_proba(latest)[0, 1])
    prob_up = float(calibrator.predict_proba(_logit(np.array([prob_raw])))[0, 1])

    # Feature relevance: native importances when available, else a fast reference forest.
    ref = final if hasattr(final, "feature_importances_") else _make("Random Forest").fit(X, y)
    relevance = pd.Series(ref.feature_importances_, index=cols).sort_values(ascending=False)

    # Expected return distribution at the horizon (10th / 50th / 90th percentile).
    quantiles = sorted(
        float(HistGradientBoostingRegressor(loss="quantile", quantile=q, max_depth=3, learning_rate=0.05,
                                            max_iter=120, min_samples_leaf=40, random_state=42)
              .fit(X, ret).predict(latest)[0])
        for q in (0.1, 0.5, 0.9)
    )

    # Historical analogs: nearest past setups on the 12 most relevant features.
    top = list(relevance.index[:12])
    scaler = StandardScaler().fit(X[top])
    dist = np.linalg.norm(scaler.transform(X[top]) - scaler.transform(latest[top]), axis=1)
    nearest = np.argsort(dist)[:30]
    analog_ret = ret.iloc[nearest]

    percentiles = {c: float((X[c] <= latest[c].iloc[0]).mean()) for c in cols}

    return {
        "symbol": symbol, "horizon": horizon, "model": model, "history": history, "groups": groups,
        "features": cols, "n_samples": len(X), "latest_date": latest_date,
        "prob_up": prob_up, "prob_raw": prob_raw,
        "price": float(close.loc[latest_date]),
        "ret_q10": quantiles[0], "ret_q50": quantiles[1], "ret_q90": quantiles[2],
        "oos_proba": oos, "oos_y": y_oos, "oos_ret": ret_oos,
        "close": close, "relevance": relevance,
        "latest_values": latest.iloc[0], "percentiles": percentiles,
        "analog_dates": list(X.index[nearest]), "analog_ret": analog_ret,
        "oos_raw": oos_raw,
        "auc": float(roc_auc_score(y_oos, oos_raw)) if y_oos.nunique() > 1 else 0.5,
        "accuracy": float(((oos > 0.5).astype(int) == y_oos).mean()),
        "baseline": float(max(y_oos.mean(), 1 - y_oos.mean())),
        "brier": float(brier_score_loss(y_oos, oos)),
        "base_rate": float(y_oos.mean()),
    }


# ---------------- evaluation (cheap, depends on thresholds) ---------------- #

def hysteresis_position(proba: pd.Series, buy_level: float, sell_level: float, start: float = 1.0) -> pd.Series:
    """Invested from the start; exit on a SELL (p ≤ sell_level), re-enter on a BUY (p ≥ buy_level)."""
    state = pd.Series(np.nan, index=proba.index)
    state[proba >= buy_level] = 1.0
    state[proba <= sell_level] = 0.0
    return state.ffill().fillna(start)


def levels(res: dict, edge: float) -> tuple[float, float]:
    """BUY/SELL probability levels: the asset's base rate ± the required edge."""
    return res["base_rate"] + edge, res["base_rate"] - edge


def evaluate(res: dict, edge: float, fee: float = 0.001) -> dict:
    buy_th, sell_th = levels(res, edge)
    p, y, fwd = res["oos_proba"], res["oos_y"], res["oos_ret"]
    buys, sells = p >= buy_th, p <= sell_th

    daily = res["close"].pct_change().reindex(p.index).fillna(0.0)
    pos = hysteresis_position(p, buy_th, sell_th).shift(1).fillna(1.0)
    strat = pos * daily - pos.diff().abs().fillna(0.0) * fee
    equity, hold = (1 + strat).cumprod(), (1 + daily).cumprod()
    vol = strat.std() * np.sqrt(252)

    buckets = pd.qcut(p.rank(method="first"), 5, labels=["Very bearish", "Bearish", "Neutral", "Bullish", "Very bullish"])
    by_bucket = pd.DataFrame({"bucket": buckets, "ret": fwd, "up": y, "p": p}).groupby("bucket", observed=True).agg(
        avg_return=("ret", "mean"), hit_rate=("up", "mean"), avg_prob=("p", "mean"), days=("ret", "size"))

    bins = np.clip((p * 10).astype(int), 0, 9)
    calib = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))

    prob = res["prob_up"]
    signal = "BUY" if prob >= buy_th else "SELL" if prob <= sell_th else "HOLD"
    return {
        "signal": signal, "buy_level": buy_th, "sell_level": sell_th,
        "confidence": abs(prob - 0.5) * 2,
        "buy_precision": float(y[buys].mean()) if buys.any() else None,
        "buy_count": int(buys.sum()),
        "buy_avg_ret": float(fwd[buys].mean()) if buys.any() else None,
        "sell_precision": float(1 - y[sells].mean()) if sells.any() else None,
        "sell_count": int(sells.sum()),
        "sell_avg_ret": float(fwd[sells].mean()) if sells.any() else None,
        "avg_ret_all": float(fwd.mean()),
        "equity": equity, "hold_equity": hold,
        "strategy_return": float(equity.iloc[-1] - 1), "hold_return": float(hold.iloc[-1] - 1),
        "sharpe": float(strat.mean() * 252 / vol) if vol > 0 else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "exposure": float(pos.mean()),
        "buckets": by_bucket, "calibration": calib,
    }


def composite(res: dict, tech_score: float) -> dict:
    """Blend AI, technical rating and historical analogs into one -100..+100 score.

    AI and analogs are measured relative to the base rate (how often this asset is
    usually up over the horizon), and the AI's weight shrinks with its reliability.
    """
    base = res["base_rate"]
    ai = float(np.clip((res["prob_up"] - base) / 0.15, -1, 1))
    analog = float(np.clip(((res["analog_ret"] > 0).mean() - base) / 0.25, -1, 1))
    skill, _ = reliability(res["auc"])
    weights = {"ai": 0.5 * skill, "tech": 0.3, "analog": 0.2}
    total = sum(weights.values())
    score = round((weights["ai"] * ai + weights["tech"] * tech_score / 100 + weights["analog"] * analog) / total * 100)
    label = ("Strong Buy" if score >= 40 else "Buy" if score >= 12 else
             "Strong Sell" if score <= -40 else "Sell" if score <= -12 else "Neutral")
    return {"score": score, "label": label, "ai": ai * 100, "analog": analog * 100, "tech": tech_score,
            "weights": {k: v / total for k, v in weights.items()}}


def predict_symbol(symbol: str, settings: dict | None = None) -> dict | None:
    """Train (cached) + evaluate with the current settings; one dict for the UI."""
    s = settings or get_settings()
    res = train(symbol, HORIZONS[s["horizon"]], s["model"], s["history"], tuple(s["groups"]))
    if res is None:
        return None
    out = {**res, **evaluate(res, s["edge"]), "settings": s}
    price = res["price"]
    out.update({
        "target_low": price * (1 + res["ret_q10"]),
        "target_mid": price * (1 + res["ret_q50"]),
        "target_high": price * (1 + res["ret_q90"]),
        "analog_up": float((res["analog_ret"] > 0).mean()),
        "analog_median": float(res["analog_ret"].median()),
    })
    hist = get_history(symbol, period="2y")
    tech = technical_rating(add_indicators(hist))["score"] if len(hist) > 60 else 0
    out["composite"] = composite(res, tech)
    out["skill"], out["reliability"] = reliability(res["auc"])
    return out


def describe(feature: str) -> tuple[str, str]:
    return FEATURE_INFO.get(feature, ("Other", feature))
