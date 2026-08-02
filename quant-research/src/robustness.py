#!/usr/bin/env python3
"""Robustness & validation suite: parameter sensitivity, bootstrap significance,
walk-forward tests, and multiple-testing (deflated Sharpe) adjustment.

Outputs (quant-research/results/):
  param_sensitivity.csv — Sharpe over parameter grids per family
  bootstrap.csv         — stationary-bootstrap p-values for each strategy
  walk_forward.csv      — walk-forward OOS results per family
  deflated_sharpe.csv   — DSR given the number of tried variants
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import backtest as bt
import strategies as st
from run_backtests import load_merged

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RNG = np.random.default_rng(42)

GRIDS = {
    "ma_cross": (st.ma_cross, {"fast": [5, 10, 20, 30, 50], "slow": [20, 60, 120, 200]}),
    "donchian": (st.donchian, {"entry": [10, 20, 40, 55, 80], "exit_": [5, 10, 20]}),
    "tsmom": (st.tsmom, {"lookback": [20, 40, 60, 90, 120, 180, 250]}),
}


def sharpe_of(df, fn, kw):
    sig, lag = fn(df, **kw)
    res = bt.run(df, sig, extra_lag=lag)
    s = bt.stats(res, "")
    return s.get("sharpe", np.nan), s.get("ann_ret_pct", np.nan)


def param_sensitivity(df):
    rows = []
    for fam, (fn, grid) in GRIDS.items():
        keys = list(grid)
        combos = [{}]
        for k in keys:
            combos = [dict(c, **{k: v}) for c in combos for v in grid[k]]
        for kw in combos:
            if fam == "ma_cross" and kw["fast"] >= kw["slow"]:
                continue
            sh, ar = sharpe_of(df, fn, kw)
            rows.append({"family": fam, **kw, "sharpe": sh, "ann_ret_pct": ar})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "param_sensitivity.csv", index=False)
    return out


def stationary_bootstrap(x: np.ndarray, n_boot=2000, mean_block=20):
    """Politis-Romano stationary bootstrap of a daily return series.
    Returns p-value for H0: mean <= 0 (one-sided) and Sharpe distribution."""
    n = len(x)
    p = 1 / mean_block
    sharpes = np.empty(n_boot)
    demeaned = x - x.mean()  # bootstrap under H0
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = RNG.integers(n)
        jump = RNG.random(n) < p
        steps = RNG.integers(0, n, size=n)
        for i in range(1, n):
            idx[i] = steps[i] if jump[i] else (idx[i - 1] + 1) % n
        s = demeaned[idx]
        sd = s.std()
        sharpes[b] = (s.mean() / sd * np.sqrt(bt.TRADING_DAYS)) if sd > 0 else 0.0
    real_sharpe = x.mean() / x.std() * np.sqrt(bt.TRADING_DAYS) if x.std() > 0 else 0.0
    pval = float((sharpes >= real_sharpe).mean())
    return real_sharpe, pval


def bootstrap_all(df=None):
    """Bootstrap every strategy's daily net returns (recovered from the saved
    equity curves, so session-hold variants are covered too)."""
    eq = pd.read_csv(RESULTS / "equity_curves.csv", parse_dates=["date"])
    rows = []
    for name in [c for c in eq.columns if c != "date"]:
        x = eq[name].pct_change().dropna().values
        if (x != 0).sum() < 250:
            continue
        sh, pv = stationary_bootstrap(x)
        rows.append({"name": name, "sharpe": round(sh, 3), "p_value": round(pv, 4),
                     "n_days": len(x)})
    out = pd.DataFrame(rows).sort_values("p_value")
    out.to_csv(RESULTS / "bootstrap.csv", index=False)
    return out


def walk_forward(df, train_years=5, test_years=1):
    """Pick best params per family on rolling train window, apply next year."""
    rows = []
    dates = df["date"]
    y0, y1 = dates.min().year + train_years, dates.max().year
    for fam, (fn, grid) in GRIDS.items():
        keys = list(grid)
        combos = [{}]
        for k in keys:
            combos = [dict(c, **{k: v}) for c in combos for v in grid[k]]
        combos = [c for c in combos
                  if not (fam == "ma_cross" and c.get("fast", 0) >= c.get("slow", 1))]
        oos_parts = []
        for test_start_year in range(y0, y1 + 1, test_years):
            tr_mask = (dates >= f"{test_start_year - train_years}-01-01") & \
                      (dates < f"{test_start_year}-01-01")
            te_mask = (dates >= f"{test_start_year}-01-01") & \
                      (dates < f"{test_start_year + test_years}-01-01")
            if tr_mask.sum() < 500 or te_mask.sum() < 100:
                continue
            tr_df = df[tr_mask].reset_index(drop=True)
            best, best_sh = None, -np.inf
            for kw in combos:
                sh, _ = sharpe_of(tr_df, fn, kw)
                if pd.notna(sh) and sh > best_sh:
                    best, best_sh = kw, sh
            # apply best params on train+test then slice test (warm indicators)
            both = df[tr_mask | te_mask].reset_index(drop=True)
            sig, lag = fn(both, **best)
            res = bt.run(both, sig, extra_lag=lag)
            res_te = res[res["date"] >= f"{test_start_year}-01-01"].copy()
            oos_parts.append(res_te)
            rows.append({"family": fam, "test_year": test_start_year,
                         "chosen": str(best), "train_sharpe": round(best_sh, 2)})
        if oos_parts:
            oos = pd.concat(oos_parts, ignore_index=True)
            oos["equity"] = (1 + oos["net"]).cumprod()
            s = bt.stats(oos, f"WF_{fam}")
            s["family"] = fam
            s["test_year"] = "ALL_OOS"
            rows.append({**{k: s.get(k) for k in
                            ["family", "ann_ret_pct", "sharpe", "max_dd_pct",
                             "t_stat", "n_trades", "win_rate_pct"]},
                         "test_year": "ALL_OOS", "chosen": ""})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "walk_forward.csv", index=False)
    return out


def deflated_sharpe(df):
    """Bailey & López de Prado deflated Sharpe: probability that the observed
    max Sharpe among N tried variants beats the expected max under H0."""
    full = pd.read_csv(RESULTS / "full_sample.csv")
    ps = pd.read_csv(RESULTS / "param_sensitivity.csv")
    n_trials = len(full) + len(ps)
    sh = full.dropna(subset=["sharpe"]).set_index("name")["sharpe"]
    var_sh = float(np.var(pd.concat([sh, ps["sharpe"].dropna()])))
    n_days = int(full["n_days"].max())
    em = 0.5772156649
    sr0 = np.sqrt(var_sh) * ((1 - em) * sps.norm.ppf(1 - 1 / n_trials)
                             + em * sps.norm.ppf(1 - 1 / (n_trials * np.e)))
    rows = []
    for name, s_ann in sh.items():
        sr_d = s_ann / np.sqrt(bt.TRADING_DAYS)     # daily Sharpe
        sr0_d = sr0 / np.sqrt(bt.TRADING_DAYS)
        z = (sr_d - sr0_d) * np.sqrt(n_days - 1) / np.sqrt(1 - 0 + 0.5 * sr_d**2)
        rows.append({"name": name, "sharpe": s_ann, "sr0_annual": round(sr0, 3),
                     "dsr_prob": round(float(sps.norm.cdf(z)), 4),
                     "n_trials": n_trials})
    out = pd.DataFrame(rows).sort_values("dsr_prob", ascending=False)
    out.to_csv(RESULTS / "deflated_sharpe.csv", index=False)
    return out


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_merged()
    print("param sensitivity…")
    ps = param_sensitivity(df)
    print(ps.groupby("family")["sharpe"].describe().round(2).to_string())
    print("bootstrap…")
    print(bootstrap_all(df).to_string(index=False))
    print("walk-forward…")
    wf = walk_forward(df)
    print(wf[wf["test_year"] == "ALL_OOS"].to_string(index=False))
    print("deflated sharpe…")
    print(deflated_sharpe(df).head(10).to_string(index=False))
