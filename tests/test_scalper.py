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
