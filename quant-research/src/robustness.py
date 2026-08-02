#!/usr/bin/env python3
"""Robustness & validation suite: parameter sensitivity, bootstrap significance,
walk-forward tests, and multiple-testing (deflated Sharpe) adjustment.

All runs use next-open execution, matching run_backtests.py.

Outputs (quant-research/results/):
  param_sensitivity.csv — Sharpe over parameter grids per family
  bootstrap.csv         — stationary-bootstrap p-values (per-strategy series
                          start at its valid_from date)
  walk_forward.csv      — walk-forward OOS results per family (now including
                          the top families: basis, rsi2, pcr)
  deflated_sharpe.csv   — DSR with grid/registry duplicates removed
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


def pcr_width(df, width=0.1, short=True):
    return st.pcr_signal(df, hi=1 + width, lo=1 - width, short=short)


# family: (signal_fn, param grid, warmup = max lookback any combo needs)
GRIDS = {
    "ma_cross": (st.ma_cross, {"fast": [5, 10, 20, 30, 50], "slow": [20, 60, 120, 200]}, 200),
    "donchian": (st.donchian, {"entry": [10, 20, 40, 55, 80], "exit_": [5, 10, 20]}, 80),
    "tsmom": (st.tsmom, {"lookback": [20, 40, 60, 90, 120, 180, 250]}, 250),
    "basis_z": (st.basis_zscore, {"z_win": [60, 120, 250], "th": [0.5, 1.0, 1.5]}, 250),
    "rsi2_lo": (st.rsi2_reversion, {"buy_th": [5, 10, 15], "trend_ma": [100, 200],
                                    "short": [False]}, 200),
    "pcr": (pcr_width, {"width": [0.05, 0.10, 0.15]}, 1),
}


def combos_of(grid: dict) -> list[dict]:
    out = [{}]
    for k, vals in grid.items():
        out = [dict(c, **{k: v}) for c in out for v in vals]
    return [c for c in out if not ("fast" in c and "slow" in c and c["fast"] >= c["slow"])]


def run_one(df, fn, kw, warm_slice=0):
    sig, lag = fn(df, **kw)
    if lag == 1:
        lag = 0  # next-open execution: 15:00 publication precedes next open
    res = bt.run(df, sig, extra_lag=lag, execution="next_open")
    if warm_slice > 0:
        res = res.iloc[warm_slice:].copy()
        res["equity"] = (1 + res["net"]).cumprod()
    return res


def sharpe_of(df, fn, kw, warm_slice=0):
    s = bt.stats(run_one(df, fn, kw, warm_slice), "")
    return s.get("sharpe", np.nan), s.get("cagr_pct", np.nan)


def param_sensitivity(df):
    rows = []
    for fam, (fn, grid, warm) in GRIDS.items():
        for kw in combos_of(grid):
            sh, cagr = sharpe_of(df, fn, kw, warm_slice=warm)
            rows.append({"family": fam, **{k: v for k, v in kw.items() if k != "short"},
                         "sharpe": sh, "cagr_pct": cagr})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "param_sensitivity.csv", index=False)
    return out


def stationary_bootstrap(x: np.ndarray, n_boot=2000, mean_block=20):
    """Politis-Romano stationary bootstrap (vectorized). One-sided p-value
    for H0: true mean <= 0, using the (r+1)/(B+1) estimator."""
    n = len(x)
    p = 1 / mean_block
    demeaned = x - x.mean()
    sharpes = np.empty(n_boot)
    ar = np.arange(n)
    for b in range(n_boot):
        jump = RNG.random(n) < p
        jump[0] = True
        seg_first = np.maximum.accumulate(np.where(jump, ar, 0))
        starts = RNG.integers(0, n, size=n)
        seg_start = starts[seg_first]
        idx = (seg_start + (ar - seg_first)) % n
        s = demeaned[idx]
        sd = s.std()
        sharpes[b] = (s.mean() / sd * np.sqrt(bt.TRADING_DAYS)) if sd > 0 else 0.0
    real_sharpe = x.mean() / x.std() * np.sqrt(bt.TRADING_DAYS) if x.std() > 0 else 0.0
    r = int((sharpes >= real_sharpe).sum())
    pval = (r + 1) / (n_boot + 1)
    return real_sharpe, float(pval)


def bootstrap_all(df=None):
    """Bootstrap every strategy's daily net returns, from its valid_from."""
    eq = pd.read_csv(RESULTS / "equity_curves.csv", parse_dates=["date"])
    full = pd.read_csv(RESULTS / "full_sample.csv")
    vf = dict(zip(full["name"], full.get("valid_from", [None] * len(full))))
    rows = []
    for name in [c for c in eq.columns if c != "date"]:
        series = eq.set_index("date")[name]
        if vf.get(name):
            series = series[series.index >= vf[name]]
        x = series.pct_change().dropna().values
        note = ""
        if (x != 0).sum() < 250:
            note = "insufficient_active_days"
            rows.append({"name": name, "sharpe": np.nan, "p_value": np.nan,
                         "n_days": len(x), "note": note})
            continue
        sh, pv = stationary_bootstrap(x)
        rows.append({"name": name, "sharpe": round(sh, 3), "p_value": round(pv, 4),
                     "n_days": len(x), "note": note})
    out = pd.DataFrame(rows).sort_values("p_value")
    out.to_csv(RESULTS / "bootstrap.csv", index=False)
    return out


def walk_forward(df, train_years=5, test_years=1):
    """Pick best params per family on a rolling train window (scored only on
    the window's warmed-up part), apply to the next year."""
    rows = []
    dates = df["date"]
    y0, y1 = dates.min().year + train_years, dates.max().year
    for fam, (fn, grid, warm) in GRIDS.items():
        combos = combos_of(grid)
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
                sh, _ = sharpe_of(tr_df, fn, kw, warm_slice=warm)
                if pd.notna(sh) and sh > best_sh:
                    best, best_sh = kw, sh
            if best is None:
                continue
            both = df[tr_mask | te_mask].reset_index(drop=True)
            res = run_one(both, fn, best)
            res_te = res[res["date"] >= f"{test_start_year}-01-01"].copy()
            oos_parts.append(res_te)
            rows.append({"family": fam, "test_year": test_start_year,
                         "chosen": str(best), "train_sharpe": round(best_sh, 2)})
        if oos_parts:
            oos = pd.concat(oos_parts, ignore_index=True)
            oos["equity"] = (1 + oos["net"]).cumprod()
            s = bt.stats(oos, f"WF_{fam}")
            rows.append({"family": fam, "test_year": "ALL_OOS", "chosen": "",
                         "train_sharpe": np.nan,
                         **{k: s.get(k) for k in ["cagr_pct", "sharpe", "max_dd_pct",
                                                  "t_stat", "n_trades", "win_rate_pct"]}})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "walk_forward.csv", index=False)
    return out


def deflated_sharpe(df=None):
    """Bailey & López de Prado DSR on ARITHMETIC Sharpe ratios; grid rows
    duplicating registry entries are not double-counted; the true search
    space is larger than counted, so n_trials is a lower bound (making
    dsr_prob an upper bound)."""
    full = pd.read_csv(RESULTS / "full_sample.csv")
    boot = pd.read_csv(RESULTS / "bootstrap.csv")
    ps = pd.read_csv(RESULTS / "param_sensitivity.csv")
    registry_dupes = 8  # ma 20/60, 5/20, 50/200; donchian 20/10, 55/20; tsmom 60/120/250
    n_trials = len(full) + len(ps) - registry_dupes
    sh = boot.dropna(subset=["sharpe"]).set_index("name")["sharpe"]
    var_sh = float(np.var(pd.concat([sh, ps["sharpe"].dropna()])))
    em = 0.5772156649
    sr0 = np.sqrt(var_sh) * ((1 - em) * sps.norm.ppf(1 - 1 / n_trials)
                             + em * sps.norm.ppf(1 - 1 / (n_trials * np.e)))
    n_by_name = dict(zip(boot["name"], boot["n_days"]))
    rows = []
    for name, s_ann in sh.items():
        n_days = int(n_by_name.get(name, 6000))
        sr_d = s_ann / np.sqrt(bt.TRADING_DAYS)
        sr0_d = sr0 / np.sqrt(bt.TRADING_DAYS)
        z = (sr_d - sr0_d) * np.sqrt(n_days - 1) / np.sqrt(1 + 0.5 * sr_d**2)
        rows.append({"name": name, "sharpe_arith": round(float(s_ann), 3),
                     "sr0_annual": round(sr0, 3),
                     "dsr_prob": round(float(sps.norm.cdf(z)), 4),
                     "n_trials": n_trials, "n_days": n_days})
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
    print(deflated_sharpe(df).head(12).to_string(index=False))
