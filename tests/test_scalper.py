"""Tests for the scalper's skill gate and trade simulation."""

import numpy as np
import pandas as pd

from utils.scalper import ScalpParams, _auc_lcb, _wilson_low, decide, simulate_outcomes

RNG = np.random.default_rng(0)
WON = (RNG.random(2000) < 0.45).astype(float)

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
