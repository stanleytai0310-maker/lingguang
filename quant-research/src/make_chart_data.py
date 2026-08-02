#!/usr/bin/env python3
"""Extract compact JSON chart data for the HTML report artifact."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def weekly(eq: pd.DataFrame, cols: list[str]) -> dict:
    e = eq.set_index("date")[cols].resample("W-FRI").last().dropna(how="all")
    out = {"dates": [d.strftime("%Y-%m-%d") for d in e.index]}
    for c in cols:
        out[c] = [round(float(v), 4) if pd.notna(v) else None for v in e[c]]
    return out


def main():
    eq = pd.read_csv(RESULTS / "equity_curves.csv", parse_dates=["date"])
    rets = eq.set_index("date").pct_change()
    combo = rets[["ma_20_60_lo", "rsi2_lo", "tom_1_3"]].mean(axis=1)
    eq["combo_3"] = (1 + combo.fillna(0)).cumprod().values

    payload = {}
    payload["equity_main"] = weekly(eq, ["buy_hold", "ma_20_60_lo", "rsi2_lo", "combo_3"])
    payload["equity_decay"] = weekly(eq, ["basis_z120_ls", "donchian_55_20_ls", "pcr_oi_ls"])

    m = pd.read_csv(ROOT / "data" / "clean" / "merged.csv", parse_dates=["date"])
    mm = m.dropna(subset=["basis"]).copy()
    mm["month"] = mm["date"].dt.month
    payload["basis_by_month"] = {int(k): round(float(v), 1) for k, v in
                                 mm.groupby("month")["basis"].mean().items()}

    on = (1 + m["tx_ret_overnight"].fillna(0)).cumprod()
    intr = (1 + m["tx_ret_intraday"].fillna(0)).cumprod()
    sess = pd.DataFrame({"date": m["date"], "overnight": on, "intraday": intr})
    payload["session_cum"] = weekly(sess, ["overnight", "intraday"])

    split = pd.read_csv(RESULTS / "subsample_split.csv")
    keep = ["basis_z120_ls", "ma_20_60_lo", "rsi2_lo", "buy_hold", "tom_1_3",
            "pcr_oi_ls", "donchian_55_20_ls", "ma_50_200_lo", "tsmom_60_ls"]
    piv = split[split["name"].isin(keep)].pivot_table(index="name", columns="window",
                                                      values="sharpe")
    piv = piv.dropna(subset=["pre2018", "post2018"])
    payload["prepost_sharpe"] = {n: {"pre": round(float(r["pre2018"]), 2),
                                     "post": round(float(r["post2018"]), 2)}
                                 for n, r in piv.iterrows()}

    (RESULTS / "chart_data.json").write_text(json.dumps(payload))
    sizes = {k: len(json.dumps(v)) for k, v in payload.items()}
    print("chart_data.json written:", sizes, "total", sum(sizes.values()))


if __name__ == "__main__":
    main()
