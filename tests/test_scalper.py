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
