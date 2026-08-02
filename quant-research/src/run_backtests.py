#!/usr/bin/env python3
"""Run the full strategy zoo over the cleaned TX data and write result tables.

Execution convention: next-open fills (a close-t signal fills at the open of
t+1) — the only convention executable across the whole sample (no after-hours
session before 2017-05-15). Same-close fills are reported separately in
exec_sensitivity.csv as the optimistic bound.

Flow/PCR signals are published ~15:00 on the signal date, which still
precedes the next morning's 08:45 open, so their publication lag collapses
to zero under next-open execution (it stays one full day under close fills).

Per-strategy stats are computed from each strategy's first VALID date
(inputs exist + indicator warmed up) — quoting a 3-year flow signal on a
28-year frame would dilute every statistic.

Outputs (quant-research/results/):
  full_sample.csv       — all strategies from their valid_from, net of costs
  subsample_split.csv   — pre-2018 vs 2018+ split (NOT true out-of-sample:
                          registry parameters are hard-coded, not IS-chosen)
  subperiods.csv        — per-regime breakdown
  cost_sensitivity.csv  — slippage 0 / 1 / 2 pts + vol-scaled (3pt on >3% days)
  exec_sensitivity.csv  — next-open vs same-close fills
  equity_curves.csv     — net equity curves (full frame) for charting
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import strategies as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPLIT = "2018-01-01"
SUBPERIOD_SPLITS = ["2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01", "2023-01-01"]

# (input column that must exist, indicator warmup in trading days)
WARMUP = {
    "buy_hold": (None, 0),
    "ma_5_20_ls": (None, 20), "ma_20_60_ls": (None, 60), "ma_20_60_lo": (None, 60),
    "ma_50_200_ls": (None, 200), "ma_50_200_lo": (None, 200),
    "donchian_20_10_ls": (None, 20), "donchian_55_20_ls": (None, 55),
    "tsmom_60_ls": (None, 60), "tsmom_120_ls": (None, 120), "tsmom_250_ls": (None, 250),
    "rsi2_lo": (None, 200), "rsi2_ls": (None, 200), "boll_20_2_ls": (None, 200),
    "tom_1_3": (None, 0), "tom_0_3": (None, 0),
    "foreign_z60_ls": ("TXF_外資及陸資", 60),
    "foreign_level_ls": ("TXF_外資及陸資", 0), "foreign_level_lo": ("TXF_外資及陸資", 0),
    "pcr_oi_ls": ("pcr_oi", 0),
    "basis_z120_ls": ("basis", 120), "basis_z120_nodiv_ls": ("basis", 120),
    "mtx_retail_z120_ls": ("MXF_外資及陸資", 120),
    "overnight_long": (None, 0), "intraday_short": (None, 0),
}

SESSION_SPECS = {
    "overnight_long": {"which": "overnight", "direction": 1},
    "intraday_short": {"which": "intraday", "direction": -1},
}


def load_merged() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "clean" / "merged.csv", parse_dates=["date"])
    df = df.dropna(subset=["tx_close", "tx_ret"]).reset_index(drop=True)
    return df


def valid_from(df: pd.DataFrame, name: str) -> pd.Timestamp:
    col, warm = WARMUP.get(name, (None, 0))
    start_pos = 0
    if col is not None and col in df.columns:
        fv = df[col].first_valid_index()
        start_pos = df.index.get_loc(fv) if fv is not None else len(df) - 1
    pos = min(start_pos + warm, len(df) - 1)
    return df["date"].iloc[pos]


def runner(df, name, slippage_pts=1.0, execution="next_open"):
    if name in SESSION_SPECS:
        return bt.run_session(df, slippage_pts=slippage_pts, **SESSION_SPECS[name])
    fn, kw = st.REGISTRY[name]
    sig, lag = fn(df, **kw)
    if execution == "next_open" and lag == 1:
        lag = 0  # 15:00 publication precedes the next 08:45 open
    return bt.run(df, sig, extra_lag=lag, execution=execution,
                  slippage_pts=slippage_pts)


def sliced_stats(res, name, from_date):
    chunk = res[res["date"] >= from_date].copy()
    chunk["equity"] = (1 + chunk["net"]).cumprod()
    s = bt.stats(chunk, name)
    if name in SESSION_SPECS:  # each active day is a round trip
        active = chunk[chunk["pos"] != 0]
        s["n_trades"] = int(len(active))
        s["win_rate_pct"] = round(float((active["net"] > 0).mean() * 100), 1) if len(active) else np.nan
        s["avg_trade_pct"] = round(float(active["net"].mean() * 100), 4) if len(active) else np.nan
        s["trades_per_yr"] = round(len(active) / (len(chunk) / bt.TRADING_DAYS), 1)
    return s


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_merged()
    print(f"merged rows={len(df)} range={df['date'].min().date()}..{df['date'].max().date()}")

    all_names = list(st.REGISTRY) + list(SESSION_SPECS)
    full, split_rows, subs, costs, execs = [], [], [], [], []
    curves = {"date": df["date"]}
    vf_map = {}

    for name in all_names:
        vf = valid_from(df, name)
        vf_map[name] = str(vf.date())
        res = runner(df, name)
        s = sliced_stats(res, name, vf)
        s["valid_from"] = str(vf.date())
        full.append(s)
        curves[name] = res["equity"].values

        for tag, lo_d, hi_d in [("pre2018", vf, pd.Timestamp(SPLIT)),
                                ("post2018", max(vf, pd.Timestamp(SPLIT)),
                                 res["date"].iloc[-1] + pd.Timedelta(days=1))]:
            chunk = res[(res["date"] >= lo_d) & (res["date"] < hi_d)]
            if len(chunk) > 120:
                ss = sliced_stats(chunk, name, lo_d)
                ss["window"] = tag
                split_rows.append(ss)

        res_v = res[res["date"] >= vf].copy()
        res_v["equity"] = (1 + res_v["net"]).cumprod()
        subs.extend(bt.subperiod_stats(res_v, name, SUBPERIOD_SPLITS))

        vol_slip = pd.Series(1.0, index=df.index) + 2.0 * (df["tx_ret"].abs() > 0.03)
        for label, slip in [("0", 0.0), ("1", 1.0), ("2", 2.0), ("vol3", vol_slip)]:
            r2 = runner(df, name, slippage_pts=slip)
            s2 = sliced_stats(r2, name, vf)
            s2["slippage"] = label
            costs.append(s2)

        for mode in ["next_open", "close"]:
            if name in SESSION_SPECS:
                continue
            r3 = runner(df, name, execution=mode)
            s3 = sliced_stats(r3, name, vf)
            execs.append({"name": name, "execution": mode,
                          "cagr_pct": s3["cagr_pct"], "sharpe": s3["sharpe"]})

    pd.DataFrame(full).to_csv(RESULTS / "full_sample.csv", index=False)
    pd.DataFrame(split_rows).to_csv(RESULTS / "subsample_split.csv", index=False)
    pd.DataFrame(subs).to_csv(RESULTS / "subperiods.csv", index=False)
    pd.DataFrame(costs).to_csv(RESULTS / "cost_sensitivity.csv", index=False)
    pd.DataFrame(execs).to_csv(RESULTS / "exec_sensitivity.csv", index=False)
    pd.DataFrame(curves).to_csv(RESULTS / "equity_curves.csv", index=False)
    (RESULTS / "valid_from.json").write_text(json.dumps(vf_map, indent=2, ensure_ascii=False))

    top = (pd.DataFrame(full).sort_values("sharpe", ascending=False)
           [["name", "valid_from", "cagr_pct", "ann_vol_pct", "sharpe", "max_dd_pct",
             "t_stat", "n_trades", "win_rate_pct", "profit_factor", "exposure_pct"]])
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
