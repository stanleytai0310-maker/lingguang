#!/usr/bin/env python3
"""Vectorized daily backtest engine for TX futures with realistic costs.

Conventions
-----------
- `sig` is the desired position in [-1, 0, 1] decided using information
  available at the close of day t (flows published ~15:00 the same day still
  precede the next morning's open).
- Execution modes:
    execution="next_open" (default): a signal from close t is filled at the
      open of t+1. Day PnL = prev position x overnight return + new position
      x intraday return (same-contract legs). This is executable in every
      era (pre-2017 there was no after-hours session, so nothing can be
      filled AT the close that produced the signal).
    execution="close": fill at the same close that produced the signal
      (optimistic legacy convention; kept for sensitivity comparison).
- extra_lag adds full-day delays on top (signals from data published with a
  longer lag).
- Limit-locked days: when a day trades at a single price (open==high==low),
  no fill is possible; position changes are deferred to the next tradable
  day (the desired position is re-evaluated, not queued).
- `ret` columns must be same-contract returns of the continuous front-month
  series (roll jumps excluded by construction).
- Costs are charged per side on every position change:
    cost_frac = (slippage_pts + commission_ntd / point_value) / price + tax_rate
  Rolling a live position over settlement pays |old|+|new| sides (close the
  old contract, open the new one); a same-day position change is subsumed
  in that (not double-charged).
  Defaults: 1.0 pt slippage, NT$60 commission, tax 2e-5, TX point = NT$200.
  slippage_pts may be a Series (state-dependent slippage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 248  # Taiwan market averages ~248 sessions/yr (1999-2026 data)


def cost_per_side_frac(price: pd.Series, slippage_pts=1.0,
                       commission_ntd: float = 60.0, tax_rate: float = 2e-5,
                       point_value: float = 200.0) -> pd.Series:
    return (slippage_pts + commission_ntd / point_value) / price + tax_rate


def _locked_days(df: pd.DataFrame) -> pd.Series:
    """Full-day limit locks: the day traded at a single price."""
    if {"tx_open", "tx_high", "tx_low"} <= set(df.columns):
        return (df["tx_open"] == df["tx_high"]) & (df["tx_high"] == df["tx_low"])
    return pd.Series(False, index=df.index)


def _effective_pos(desired: pd.Series, locked: pd.Series) -> pd.Series:
    """Defer position changes on locked days to the next tradable day."""
    return desired.mask(locked).ffill().fillna(0)


def _turnover(pos: pd.Series, roll: pd.Series | None) -> pd.Series:
    prev = pos.shift(1).fillna(0)
    t = (pos - prev).abs()
    if roll is not None:
        r = roll.astype(bool)
        t = t.where(~r, prev.abs() + pos.abs())
    return t


def run(df: pd.DataFrame, sig: pd.Series, extra_lag: int = 0,
        execution: str = "next_open", slippage_pts=1.0,
        commission_ntd: float = 60.0, tax_rate: float = 2e-5,
        point_value: float = 200.0) -> pd.DataFrame:
    """df needs: tx_close, tx_ret (+ tx_ret_overnight/tx_ret_intraday and
    tx_open/high/low for next_open mode; tx_roll for roll costs)."""
    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"].values
    locked = _locked_days(df)
    roll = df["tx_roll"] if "tx_roll" in df.columns else None
    base = sig.reindex(df.index).fillna(0).clip(-1, 1)

    if execution == "next_open":
        # signal from close t-1 (already delayed extra_lag days) fills at open t
        desired = base.shift(1 + extra_lag).fillna(0)
        pos_open = _effective_pos(desired, locked)
        prev = pos_open.shift(1).fillna(0)
        on = df.get("tx_ret_overnight", pd.Series(0.0, index=df.index)).fillna(0)
        intra = df.get("tx_ret_intraday", df["tx_ret"]).fillna(0)
        out["pos"] = pos_open
        out["gross"] = prev * on + pos_open * intra
        turnover = _turnover(pos_open, roll)
    elif execution == "close":
        # optimistic legacy convention: the (lagged) signal fills at the very
        # close it is computed from; pos_t is held from close t to close t+1
        desired = base.shift(extra_lag).fillna(0)
        pos = _effective_pos(desired, locked)
        out["pos"] = pos
        out["gross"] = pos.shift(1).fillna(0) * df["tx_ret"].fillna(0)
        turnover = _turnover(pos, roll)
    else:
        raise ValueError(f"unknown execution mode {execution!r}")

    cps = cost_per_side_frac(df["tx_close"], slippage_pts, commission_ntd,
                             tax_rate, point_value)
    out["cost"] = turnover * cps
    out["net"] = out["gross"] - out["cost"]
    out["equity"] = (1 + out["net"]).cumprod()
    return out


def run_session(df: pd.DataFrame, which: str = "overnight", direction: int = 1,
                sig: pd.Series | None = None, extra_lag: int = 0,
                slippage_pts=1.0, commission_ntd: float = 60.0,
                tax_rate: float = 2e-5, point_value: float = 200.0) -> pd.DataFrame:
    """Session-hold backtest (enter and exit within each day).

    which='overnight': long prev close -> today open (tx_ret_overnight).
    which='intraday' : long today open -> today close (tx_ret_intraday).
    Every active day pays TWO sides of costs. Fully locked days are skipped
    (no entry). Results are highly slippage-sensitive; see cost sensitivity.
    """
    col = f"tx_ret_{which}"
    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"].values
    if sig is None:
        sig = pd.Series(1.0, index=df.index)
    locked = _locked_days(df)
    active = sig.reindex(df.index).fillna(0).clip(0, 1).shift(1 + extra_lag).fillna(0)
    active = active.where(~locked, 0.0)
    out["pos"] = active * direction
    out["gross"] = out["pos"] * df[col].fillna(0)
    cps = cost_per_side_frac(df["tx_close"], slippage_pts, commission_ntd,
                             tax_rate, point_value)
    out["cost"] = active * 2 * cps
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
    """Sharpe is the ARITHMETIC daily mean/std annualized by sqrt(248) —
    the estimator the bootstrap and DSR formulas are derived for.
    cagr_pct is the geometric annualized return (what compounding pays)."""
    net = bt["net"]
    n = len(net)
    if n < 10:
        return {"name": name, "n_days": n}
    cagr = (1 + net).prod() ** (TRADING_DAYS / n) - 1
    ann_vol = net.std() * np.sqrt(TRADING_DAYS)
    sharpe = (net.mean() / net.std() * np.sqrt(TRADING_DAYS)) if net.std() > 0 else np.nan
    tstat = net.mean() / (net.std() / np.sqrt(n)) if net.std() > 0 else np.nan
    tr = trades_from(bt)
    years = n / TRADING_DAYS
    return {
        "name": name,
        "start": str(bt["date"].iloc[0].date()) if hasattr(bt["date"].iloc[0], "date") else str(bt["date"].iloc[0]),
        "end": str(bt["date"].iloc[-1].date()) if hasattr(bt["date"].iloc[-1], "date") else str(bt["date"].iloc[-1]),
        "n_days": n,
        "cagr_pct": round(cagr * 100, 2),
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
