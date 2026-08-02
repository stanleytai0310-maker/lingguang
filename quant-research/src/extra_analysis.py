#!/usr/bin/env python3
"""Alpha vs buy&hold, defensive combo, and 2026-crash behavior."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ANN = 248


def main():
    eq = pd.read_csv(R / "equity_curves.csv", parse_dates=["date"])
    rets = eq.set_index("date").pct_change()
    out = {}

    bh = rets["buy_hold"]
    for name in ["ma_20_60_lo", "rsi2_lo", "tom_1_3", "basis_z120_ls",
                 "pcr_oi_ls", "donchian_55_20_ls"]:
        r = rets[name]
        for tag, mask in [("full", r.index >= r.index.min()),
                          ("post2018", r.index >= "2018-01-01")]:
            rr, bb = r[mask].dropna(), bh[mask].dropna()
            idx = rr.index.intersection(bb.index)
            rr, bb = rr[idx], bb[idx]
            beta = rr.cov(bb) / bb.var()
            resid = rr - beta * bb
            ir = resid.mean() / resid.std() * np.sqrt(ANN) if resid.std() > 0 else np.nan
            t = resid.mean() / (resid.std() / np.sqrt(len(resid)))
            out.setdefault(name, {})[tag] = {
                "beta": round(float(beta), 2),
                "alpha_ann_pct": round(float(resid.mean() * ANN * 100), 2),
                "IR": round(float(ir), 2), "t_alpha": round(float(t), 2)}

    combo = rets[["ma_20_60_lo", "rsi2_lo", "tom_1_3"]].mean(axis=1).dropna()
    for tag, m in [("full", combo.index >= combo.index.min()),
                   ("post2018", combo.index >= "2018-01-01")]:
        c = combo[m]
        eqc = (1 + c).cumprod()
        out.setdefault("combo_3", {})[tag] = {
            "ann_pct": round(float(eqc.iloc[-1] ** (ANN / len(c)) - 1) * 100, 2),
            "sharpe": round(float(c.mean() / c.std() * np.sqrt(ANN)), 2),
            "max_dd_pct": round(float((eqc / eqc.cummax() - 1).min() * 100), 1)}
    out["combo_corr"] = rets[["ma_20_60_lo", "rsi2_lo", "tom_1_3"]].corr().round(2).to_dict()

    w = rets.loc["2026-05-01":"2026-07-31"]
    for name in ["buy_hold", "ma_20_60_lo", "rsi2_lo", "basis_z120_ls"]:
        out.setdefault("crash_2026", {})[name] = round(
            float((1 + w[name].dropna()).prod() - 1) * 100, 1)
    out["crash_2026"]["combo_3"] = round(
        float((1 + combo.loc["2026-05-01":"2026-07-31"]).prod() - 1) * 100, 1)

    json.dump(out, open(R / "extra_analysis.json", "w"), indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
