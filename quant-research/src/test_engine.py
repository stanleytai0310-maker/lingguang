#!/usr/bin/env python3
"""Sanity tests for the backtest engine and continuous-contract logic.

Run: python3 test_engine.py  (prints PASS/FAIL per check, exits nonzero on fail)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
from prepare_data import build_continuous, third_wednesday

FAILS = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}: {name} {detail}")
    if not cond:
        FAILS.append(name)


def synth(ret, close0=10000.0):
    """Synthetic frame with open == prev close (overnight leg = 0), so
    next-open and close execution coincide unless a test says otherwise."""
    n = len(ret)
    close = close0 * np.cumprod(1 + np.asarray(ret))
    open_ = np.concatenate([[close0], close[:-1]])
    df = pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=n),
        "tx_open": open_, "tx_close": close,
        "tx_high": np.maximum(open_, close) * 1.001,
        "tx_low": np.minimum(open_, close) * 0.999,
        "tx_ret": np.asarray(ret, dtype=float),
    })
    df["tx_ret_overnight"] = df["tx_open"] / np.concatenate([[close0], close[:-1]]) - 1
    df["tx_ret_intraday"] = df["tx_close"] / df["tx_open"] - 1
    return df


# ── third_wednesday ──────────────────────────────────────────────────────
check("3rd-wed 2024-01", third_wednesday(2024, 1) == pd.Timestamp("2024-01-17"))
check("3rd-wed 2024-08", third_wednesday(2024, 8) == pd.Timestamp("2024-08-21"))
check("3rd-wed 2026-08", third_wednesday(2026, 8) == pd.Timestamp("2026-08-19"))

# ── no-lookahead: a same-day-perfect signal must lose its edge once lagged.
n = 200
ret = np.where(np.arange(n) % 2 == 0, 0.01, -0.01)
df = synth(ret)
sig_lookahead = pd.Series(np.sign(ret), index=df.index)
for mode in ["next_open", "close"]:
    res = bt.run(df, sig_lookahead, execution=mode,
                 slippage_pts=0, commission_ntd=0, tax_rate=0)
    check(f"engine lags signals ({mode})", res["net"].mean() < 0,
          f"mean={res['net'].mean():.5f}")

# ── extra_lag shifts one more day (alternating series flips sign back).
res2 = bt.run(df, sig_lookahead, extra_lag=1, execution="close",
              slippage_pts=0, commission_ntd=0, tax_rate=0)
check("extra_lag shifts one more day", res2["net"].mean() > 0,
      f"mean={res2['net'].mean():.5f}")

# ── cost accounting: constant long pays exactly one side (no rolls here).
sig_const = pd.Series(1.0, index=df.index)
res3 = bt.run(df, sig_const, execution="close",
              slippage_pts=1.0, commission_ntd=60, tax_rate=2e-5)
expected_first = (1.0 + 60 / 200) / df["tx_close"].iloc[0] + 2e-5
check("one entry cost only", np.isclose(res3["cost"].sum(), expected_first, rtol=1e-6),
      f"got={res3['cost'].sum():.6f} expect={expected_first:.6f}")

# ── flip long→short pays two sides (entry 1 + flip 2 = 3 total).
sig_flip = pd.Series([1.0] * 100 + [-1.0] * 100, index=df.index)
res4 = bt.run(df, sig_flip, execution="close",
              slippage_pts=1.0, commission_ntd=60, tax_rate=2e-5)
n_sides = res4["pos"].diff().abs().fillna(res4["pos"].abs()).sum()
check("flip = 2 sides + entry = 3 total", np.isclose(n_sides, 3.0), f"sides={n_sides}")

tr = bt.trades_from(res4)
check("two trades extracted", len(tr) == 2, f"n={len(tr)}")

# ── roll-day turnover: constant long over a roll pays |prev|+|new| = 2 sides;
# a flip ON the roll day also pays exactly 2 (close old, open new), not 4.
df_r = synth(np.full(10, 0.001))
df_r["tx_roll"] = [False] * 5 + [True] + [False] * 4
res5 = bt.run(df_r, pd.Series(1.0, index=df_r.index), execution="close",
              slippage_pts=1.0, commission_ntd=0, tax_rate=0)
sides5 = (res5["cost"] * df_r["tx_close"] / 1.0).round(3)
check("roll day charges 2 sides", np.isclose(sides5.iloc[5], 2.0, atol=0.01),
      f"got={sides5.iloc[5]}")
sig_flip_roll = pd.Series([1.0] * 6 + [-1.0] * 4, index=df_r.index)
res6 = bt.run(df_r, sig_flip_roll, execution="close",
              slippage_pts=1.0, commission_ntd=0, tax_rate=0)
# flip executes at index 6? sig from t=5 close (last +1 at t=5): pos changes at t=6
sides6 = (res6["cost"] * df_r["tx_close"] / 1.0).round(3)
check("flip on non-roll day = 2 sides", np.isclose(sides6.iloc[6], 2.0, atol=0.01),
      f"got={sides6.iloc[6]}")

# ── locked-day deferral: full-lock day forbids position changes.
df_l = synth(np.full(10, 0.001))
for c in ["tx_open", "tx_high", "tx_low"]:
    df_l.loc[4, c] = df_l.loc[4, "tx_close"]  # day 4 trades at one price
sig_l = pd.Series([0.0] * 4 + [1.0] * 6, index=df_l.index)
res7 = bt.run(df_l, sig_l, execution="close", slippage_pts=0, commission_ntd=0, tax_rate=0)
check("no fill on locked day", res7["pos"].iloc[4] == 0.0, f"pos={res7['pos'].iloc[4]}")
check("fill resumes next day", res7["pos"].iloc[5] == 1.0, f"pos={res7['pos'].iloc[5]}")

# ── next_open accounting: overnight leg belongs to the previous position.
df_g = synth(np.full(6, 0.0))
df_g.loc[3, "tx_open"] = df_g.loc[2, "tx_close"] * 1.02   # +2% gap on day 3
df_g.loc[3, "tx_close"] = df_g.loc[3, "tx_open"]          # flat intraday
df_g["tx_ret_overnight"] = df_g["tx_open"] / df_g["tx_close"].shift(1) - 1
df_g["tx_ret_intraday"] = df_g["tx_close"] / df_g["tx_open"] - 1
df_g["tx_ret"] = df_g["tx_close"] / df_g["tx_close"].shift(1) - 1
df_g["tx_high"] = df_g[["tx_open", "tx_close"]].max(axis=1) * 1.0001
df_g["tx_low"] = df_g[["tx_open", "tx_close"]].min(axis=1)
# signal turns long at close of day 2 → fills at day-3 open, AFTER the gap
sig_g = pd.Series([0, 0, 1, 1, 1, 1], index=df_g.index, dtype=float)
res8 = bt.run(df_g, sig_g, execution="next_open", slippage_pts=0, commission_ntd=0, tax_rate=0)
check("gap before fill not earned", abs(res8["gross"].iloc[3]) < 1e-12,
      f"gross_day3={res8['gross'].iloc[3]:.6f}")

# ── continuous contract: roll on settlement day; same-contract returns.
rows = []
d = pd.bdate_range("2024-01-10", "2024-01-19")  # settlement 2024-01-17
for day in d:
    for expiry, base in [("202401", 17000.0), ("202402", 16900.0)]:
        px = base + (day.day * 10)
        rows.append({"date": day, "symbol": "TX", "expiry": expiry,
                     "open": px, "high": px + 20, "low": px - 20, "close": px,
                     "volume": 1000, "settle": px, "oi": 50000})
fut = pd.DataFrame(rows)
cont = build_continuous(fut)
jan17 = cont[cont["date"] == "2024-01-17"]
check("rolls to 202402 on settlement day", jan17["expiry"].iloc[0] == "202402")
jan16 = cont[cont["date"] == "2024-01-16"]
check("still 202401 day before settlement", jan16["expiry"].iloc[0] == "202401")
exp_ret = (16900 + 170) / (16900 + 160) - 1
check("roll-day return uses same contract",
      np.isclose(jan17["ret"].iloc[0], exp_ret, rtol=1e-9),
      f"got={jan17['ret'].iloc[0]:.6f} expect={exp_ret:.6f}")
# back-adjusted series: daily changes must equal same-contract returns
adj_ret = cont["adj_close"].pct_change().iloc[1:]
raw_ret = cont["ret"].iloc[1:]
check("adj_close reproduces same-contract returns",
      np.allclose(adj_ret, raw_ret, rtol=1e-9), "")
check("adj anchored at final close",
      np.isclose(cont["adj_close"].iloc[-1], cont["close"].iloc[-1], rtol=1e-12))

# ── stats smoke test.
s = bt.stats(res3, "const_long")
check("stats fields present", all(k in s for k in
      ["cagr_pct", "sharpe", "max_dd_pct", "n_trades", "win_rate_pct"]))

print(f"\n{len(FAILS)} failures")
raise SystemExit(1 if FAILS else 0)
