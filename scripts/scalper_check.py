"""Out-of-sample health check for the Live Scalper.

Downloads fresh candles from TradingView, replays them with your saved scalper settings and reports whether the
percentages still separate winners from losers. Run it every week or two: settings that worked on old data
often stop working.

    python scripts/scalper_check.py                 # your saved settings
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from utils.scalper import MarketSession, ScalpParams, analyze  # noqa: E402
from utils.tv_feed import PRESETS, TIMEFRAMES, _tv_bars  # noqa: E402

SETTINGS = ROOT / "data" / "scalper_settings.json"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def params_from(s: dict, res_min: int) -> ScalpParams:
    """Same mapping as the Live Scalper page."""
    time_exit = s.get("sc_exit", "Close after X minutes").startswith("Close after")
    atr = s.get("sc_mode", "Points") != "Points"
    return ScalpParams(
        tp=s.get("sc_tp_atr", 1.0) if atr else s.get("sc_tp", 8.0),
        sl=s.get("sc_sl_atr", 1.5) if atr else s.get("sc_sl", 6.0),
        spread=s.get("sc_spread", 0.6), horizon=int(s.get("sc_hz", 20)), k=int(s.get("sc_k", 150)),
        min_edge=s.get("sc_edge", 0.02), tp_mode="atr" if atr else "points",
        trend_filter=s.get("sc_trend", True), high_conf=s.get("sc_hconf", False),
        exit_mode="time" if time_exit else "tp_sl", hold=max(1, -(-int(s.get("sc_hold_min", 15)) // res_min)),
        use_ml=s.get("sc_ml", True), require_skill=s.get("sc_skill", True),
    )


def report(name: str, res: dict):
    s, acc = res["stats"], res["accuracy"]
    print(f"\n== {name}")
    print(f"walk-forward ROC-AUC  BUY {acc['auc_long']:.3f} | SELL {acc['auc_short']:.3f}   (0.500 = coin flip)")
    if not s["trades"]:
        print("no replay trades - the skill gate kept the bot out (WAIT).")
        return
    print(f"{s['trades']} trades on {s['days']} day(s) | win {s['win_rate'] * 100:.0f}% | net {s['net_points']:+.1f} pts "
          f"| profit factor {s['profit_factor']:.2f} | max DD {s['max_dd_points']:.1f} pts")
    tr = res["trades"].assign(day=lambda t: pd.DatetimeIndex(t["time"]).normalize())
    print(tr.groupby("day")["points"].agg(trades="count", points="sum").round(1).to_string())


def calibration(res: dict):
    table = res["accuracy"]["table"]
    if len(table):
        print("\nDoes a higher % really win more often?")
        print(table.round(1).to_string(index=False))




def main():
    s = load_settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", default=PRESETS.get(s.get("sc_preset"), ("SPREADEX:SPX",))[0])
    ap.add_argument("--tf", default=s.get("sc_tf", "3m"), choices=list(TIMEFRAMES))
    ap.add_argument("--bars", type=int, default=5000)
    a = ap.parse_args()

    res_min = int(TIMEFRAMES[a.tf])
    print(f"Downloading {a.bars} x {a.tf} candles of {a.symbol} from TradingView...")
    df, info = _tv_bars(a.symbol, TIMEFRAMES[a.tf], a.bars, timeout=40)
    print(f"{len(df)} candles, {df.index[0]:%Y-%m-%d %H:%M} to {df.index[-1]:%Y-%m-%d %H:%M} UTC")
    sess = MarketSession.from_info(info, a.symbol)
    p = params_from(s, res_min)

    res = analyze(df, p, sess=sess)
    report("Your saved settings", res)
    calibration(res)


if __name__ == "__main__":
    main()
