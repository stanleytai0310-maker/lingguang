#!/usr/bin/env python3
"""Run the full strategy zoo over the cleaned TX data and write result tables.

Outputs (quant-research/results/):
  full_sample.csv      — all strategies, full sample net of costs
  is_oos.csv           — in-sample (…2017-12-31) vs out-of-sample (2018-01-01…)
  subperiods.csv       — per-regime breakdown
  cost_sensitivity.csv — slippage 0 / 1 / 2 pts per side
  equity_curves.csv    — net equity curves (wide) for charting
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import backtest as bt
import strategies as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OOS_START = "2018-01-01"
SUBPERIOD_SPLITS = ["2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01", "2023-01-01"]


def load_merged() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "clean" / "merged.csv", parse_dates=["date"])
    df = df.dropna(subset=["tx_close", "tx_ret"]).reset_index(drop=True)
    return df


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_merged()
    print(f"merged rows={len(df)} range={df['date'].min().date()}..{df['date'].max().date()}")

    full, isoos, subs, costs = [], [], [], []
    curves = {"date": df["date"]}

    # session-hold variants use their own runner (two cost sides per day)
    session_specs = {
        "overnight_long": {"which": "overnight", "direction": 1},
        "intraday_short": {"which": "intraday", "direction": -1},
    }

    def runner(name, slippage_pts=1.0):
        if name in session_specs:
            return bt.run_session(df, slippage_pts=slippage_pts, **session_specs[name])
        fn, kw = st.REGISTRY[name]
        sig, extra_lag = fn(df, **kw)
        return bt.run(df, sig, extra_lag=extra_lag, slippage_pts=slippage_pts)

    def fix_session_stats(name, res, s):
        if name in session_specs:
            active = res[res["pos"] != 0]
            s["n_trades"] = int(len(active))
            s["win_rate_pct"] = round(float((active["net"] > 0).mean() * 100), 1)
            s["avg_trade_pct"] = round(float(active["net"].mean() * 100), 4)
            s["trades_per_yr"] = round(len(active) / (len(res) / bt.TRADING_DAYS), 1)
        return s

    all_names = list(st.REGISTRY) + list(session_specs)
    for name in all_names:
        res = runner(name)
        full.append(fix_session_stats(name, res, bt.stats(res, name)))
        curves[name] = res["equity"].values

        m_is = res[res["date"] < OOS_START].copy()
        m_oos = res[res["date"] >= OOS_START].copy()
        for tag, chunk in [("IS", m_is), ("OOS", m_oos)]:
            if len(chunk) > 60:
                chunk = chunk.copy()
                chunk["equity"] = (1 + chunk["net"]).cumprod()
                s = fix_session_stats(name, chunk, bt.stats(chunk, name))
                s["window"] = tag
                isoos.append(s)

        subs.extend(bt.subperiod_stats(res, name, SUBPERIOD_SPLITS))

        for slip in [0.0, 1.0, 2.0]:
            r2 = runner(name, slippage_pts=slip)
            s2 = fix_session_stats(name, r2, bt.stats(r2, name))
            s2["slippage_pts"] = slip
            costs.append(s2)

    pd.DataFrame(full).to_csv(RESULTS / "full_sample.csv", index=False)
    pd.DataFrame(isoos).to_csv(RESULTS / "is_oos.csv", index=False)
    pd.DataFrame(subs).to_csv(RESULTS / "subperiods.csv", index=False)
    pd.DataFrame(costs).to_csv(RESULTS / "cost_sensitivity.csv", index=False)
    pd.DataFrame(curves).to_csv(RESULTS / "equity_curves.csv", index=False)

    top = (pd.DataFrame(full).sort_values("sharpe", ascending=False)
           [["name", "ann_ret_pct", "ann_vol_pct", "sharpe", "max_dd_pct",
             "t_stat", "n_trades", "win_rate_pct", "profit_factor", "exposure_pct"]])
    print(top.to_string(index=False))
    (RESULTS / "summary.json").write_text(
        json.dumps({"n_strategies": len(full)}, indent=2))


if __name__ == "__main__":
    main()
