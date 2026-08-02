#!/usr/bin/env python3
"""Vectorized daily backtest engine for TX futures with realistic costs.

Conventions
-----------
- `sig` is the desired position in [-1, 0, 1] decided using information
  available at the close of day t (or earlier).
- Execution: positions take effect at the NEXT close (`extra_lag=0`,
  i.e. contribution_t = sig.shift(1) * ret_t). Data published after the
  day-session close (institutional flows, PCR) should use `extra_lag=1`,
  making the effective shift 2 days — deliberately conservative.
- `ret` must be same-contract close-to-close returns of the continuous
  front-month series (roll jumps excluded by construction).
- Costs are charged per side on every position change:
    cost_frac = (slippage_pts + commission_ntd / point_value) / price + tax_rate
  Defaults: 1.0 pt slippage, NT$60 commission, tax 2e-5, TX point = NT$200.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 240  # Taiwan market ~240-250 sessions/yr


def cost_per_side_frac(price: pd.Series, slippage_pts: float = 1.0,
                       commission_ntd: float = 60.0, tax_rate: float = 2e-5,
                       point_value: float = 200.0) -> pd.Series:
    return (slippage_pts + commission_ntd / point_value) / price + tax_rate


def run(df: pd.DataFrame, sig: pd.Series, extra_lag: int = 0,
        slippage_pts: float = 1.0, commission_ntd: float = 60.0,
        tax_rate: float = 2e-5, point_value: float = 200.0) -> pd.DataFrame:
    """df needs columns: tx_close, tx_ret. Returns per-day frame."""
    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"].values
    pos = sig.reindex(df.index).fillna(0).clip(-1, 1).shift(1 + extra_lag).fillna(0)
    out["pos"] = pos
    out["gross"] = pos * df["tx_ret"].fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    cps = cost_per_side_frac(df["tx_close"], slippage_pts, commission_ntd,
                             tax_rate, point_value)
    out["cost"] = turnover * cps
    out["net"] = out["gross"] - out["cost"]
    out["equity"] = (1 + out["net"]).cumprod()
    return out


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def trades_from(bt: pd.DataFrame) -> pd.DataFrame:
    """Split into trades = maximal runs of constant nonzero position."""
    pos = bt["pos"]
    grp = (pos != pos.shift()).cumsum()
    rows = []
    for _g, chunk in bt.groupby(grp):
        p = chunk["pos"].iloc[0]
        if p == 0:
            continue
        rows.append({
            "start": chunk["date"].iloc[0], "end": chunk["date"].iloc[-1],
            "dir": "L" if p > 0 else "S", "days": len(chunk),
            "ret": float((1 + chunk["net"]).prod() - 1),
        })
    return pd.DataFrame(rows)


def stats(bt: pd.DataFrame, name: str = "") -> dict:
    net = bt["net"]
    n = len(net)
    if n < 10:
        return {"name": name, "n_days": n}
    ann_ret = (1 + net).prod() ** (TRADING_DAYS / n) - 1
    ann_vol = net.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    tstat = net.mean() / (net.std() / np.sqrt(n)) if net.std() > 0 else np.nan
    tr = trades_from(bt)
    years = n / TRADING_DAYS
    return {
        "name": name,
        "start": str(bt["date"].iloc[0].date()) if hasattr(bt["date"].iloc[0], "date") else str(bt["date"].iloc[0]),
        "end": str(bt["date"].iloc[-1].date()) if hasattr(bt["date"].iloc[-1], "date") else str(bt["date"].iloc[-1]),
        "n_days": n,
        "ann_ret_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "max_dd_pct": round(_max_drawdown(bt["equity"]) * 100, 2),
        "t_stat": round(float(tstat), 2),
        "exposure_pct": round(float((bt["pos"] != 0).mean() * 100), 1),
        "n_trades": int(len(tr)),
        "trades_per_yr": round(len(tr) / years, 1) if years > 0 else np.nan,
        "win_rate_pct": round(float((tr["ret"] > 0).mean() * 100), 1) if len(tr) else np.nan,
        "avg_trade_pct": round(float(tr["ret"].mean() * 100), 3) if len(tr) else np.nan,
        "profit_factor": round(float(tr.loc[tr["ret"] > 0, "ret"].sum() /
                                     -tr.loc[tr["ret"] < 0, "ret"].sum()), 2)
        if len(tr) and (tr["ret"] < 0).any() else np.nan,
        "total_cost_pct": round(float(bt["cost"].sum() * 100), 2),
    }


def subperiod_stats(bt: pd.DataFrame, name: str, splits: list[str]) -> list[dict]:
    bounds = [pd.Timestamp(s) for s in splits]
    edges = [bt["date"].iloc[0]] + bounds + [bt["date"].iloc[-1] + pd.Timedelta(days=1)]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        chunk = bt[(bt["date"] >= a) & (bt["date"] < b)].copy()
        if len(chunk) < 60:
            continue
        chunk["equity"] = (1 + chunk["net"]).cumprod()
        s = stats(chunk, f"{name} [{a.date()}..{(b - pd.Timedelta(days=1)).date()}]")
        rows.append(s)
    return rows
