"""Tests for the scalper's skill gate and trade simulation."""

import numpy as np
import pandas as pd

from utils.scalper import ScalpParams, _auc_lcb, _wilson_low, decide, simulate_outcomes

RNG = np.random.default_rng(0)
WON = (RNG.random(2000) < 0.45).astype(float)
CTX = {"in_session": True, "trend_up": True, "trend_down": True}


def _row(p_long=0.70, auc=0.60, lcb=0.55):
    """A decision row where BUY clears every check except (optionally) skill."""
    return {"p_long": p_long, "p_short": 0.30, "lcb_long": 0.60, "lcb_short": 0.20,
            "exp_long": 1.0, "exp_short": -1.0, "auc_long": auc, "auc_short": 0.50,
            "auc_lcb_long": lcb, "auc_lcb_short": 0.40}


def _candles(closes, highs=None, lows=None):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range("2026-11-02 14:30", periods=len(closes), freq="3min")
    return pd.DataFrame({"Open": closes, "High": highs if highs is not None else closes,
                         "Low": lows if lows is not None else closes, "Close": closes}, index=idx)


def test_auc_lcb_passes_nan_through():
    assert np.isnan(_auc_lcb(WON, float("nan"), 5))


def test_auc_lcb_is_below_the_auc():
    assert _auc_lcb(WON, 0.60, 1) < 0.60


def test_auc_lcb_tightens_with_more_data():
    assert _auc_lcb(WON, 0.58, 5) > _auc_lcb(WON[:300], 0.58, 5)


def test_auc_lcb_widens_when_trades_overlap():
    assert _auc_lcb(WON, 0.55, 5) < _auc_lcb(WON, 0.55, 1)


def test_small_edge_on_overlapping_trades_is_not_significant():
    # 0.53 on 2,000 candles with 5-candle holds is what chance produces - it must not count as skill
    assert _auc_lcb(WON, 0.53, 5) < 0.5


def test_decide_blocks_auc_below_minimum():
    assert decide(_row(auc=0.51, lcb=0.40), CTX, ScalpParams(exit_mode="time"))[0] == "WAIT"


def test_decide_blocks_auc_that_is_not_significant():
    signal, why = decide(_row(auc=0.56, lcb=0.48), CTX, ScalpParams(exit_mode="time"))
    assert signal == "WAIT" and "proved accurate" in why


def test_decide_allows_significant_auc():
    assert decide(_row(auc=0.60, lcb=0.55), CTX, ScalpParams(exit_mode="time"))[0] == "BUY"


def test_decide_ignores_skill_when_gate_is_off():
    p = ScalpParams(exit_mode="time", require_skill=False)
    assert decide(_row(auc=0.45, lcb=0.30), CTX, p)[0] == "BUY"


def test_decide_respects_trading_hours():
    assert decide(_row(), {**CTX, "in_session": False}, ScalpParams(exit_mode="time"))[0] == "WAIT"


def test_time_exit_pays_the_spread_on_both_sides():
    out = simulate_outcomes(_candles([100, 100, 100, 100]), ScalpParams(exit_mode="time", hold=1, spread=0.5))
    assert out["long_pts"].iloc[0] == -0.5 and out["short_pts"].iloc[0] == -0.5
    assert not out["long_win"].iloc[0] and not out["short_win"].iloc[0]


def test_time_exit_long_profit_after_spread():
    out = simulate_outcomes(_candles([100, 102, 102]), ScalpParams(exit_mode="time", hold=1, spread=0.5))
    assert out["long_pts"].iloc[0] == 1.5 and out["long_win"].iloc[0]


def test_stop_counts_first_when_tp_and_sl_share_a_candle():
    c = _candles([100, 100, 100], highs=[100, 120, 100], lows=[100, 80, 100])
    out = simulate_outcomes(c, ScalpParams(tp=4, sl=3, spread=0.0, horizon=1))
    assert out["long_pts"].iloc[0] == -3 and not out["long_win"].iloc[0]


def test_wilson_lower_bound_is_below_the_rate():
    assert (_wilson_low(np.array([0.6]), 150) < 0.6).all()
