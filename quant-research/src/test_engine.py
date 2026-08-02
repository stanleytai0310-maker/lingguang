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


# ── third_wednesday ──────────────────────────────────────────────────────
check("3rd-wed 2024-01", third_wednesday(2024, 1) == pd.Timestamp("2024-01-17"))
check("3rd-wed 2024-08", third_wednesday(2024, 8) == pd.Timestamp("2024-08-21"))
check("3rd-wed 2026-08", third_wednesday(2026, 8) == pd.Timestamp("2026-08-19"))

# ── no-lookahead: a signal that "knows" tomorrow's return must not profit
# when properly lagged. Construct alternating +1%/-1% returns; sig equal to
# the sign of the SAME day's return would be lookahead. Engine applies
# shift(1), so a perfect same-day signal becomes a 1-day-late signal, which
# on an alternating series is perfectly wrong.
n = 200
dates = pd.bdate_range("2020-01-01", periods=n)
ret = np.where(np.arange(n) % 2 == 0, 0.01, -0.01)
close = 10000 * np.cumprod(1 + ret)
df = pd.DataFrame({"date": dates, "tx_close": close, "tx_ret": ret})
sig_lookahead = pd.Series(np.sign(ret), index=df.index)
res = bt.run(df, sig_lookahead, slippage_pts=0, commission_ntd=0, tax_rate=0)
check("engine lags signals (lookahead-proof)", res["net"].mean() < 0,
      f"mean={res['net'].mean():.5f}")

# ── with extra_lag=1 the same signal is shifted 2 days → on alternating
# series it becomes exactly right again; verifies extra_lag wiring.
res2 = bt.run(df, sig_lookahead, extra_lag=1, slippage_pts=0, commission_ntd=0, tax_rate=0)
check("extra_lag shifts one more day", res2["net"].mean() > 0,
      f"mean={res2['net'].mean():.5f}")

# ── cost accounting: constant long position should pay exactly one side.
sig_const = pd.Series(1.0, index=df.index)
res3 = bt.run(df, sig_const, slippage_pts=1.0, commission_ntd=60, tax_rate=2e-5)
expected_first = (1.0 + 60 / 200) / df["tx_close"].iloc[1] + 2e-5
check("one entry cost only", np.isclose(res3["cost"].sum(), expected_first, rtol=1e-6),
      f"got={res3['cost'].sum():.6f} expect={expected_first:.6f}")

# ── flip long→short pays two sides.
sig_flip = pd.Series([1.0] * 100 + [-1.0] * 100, index=df.index)
res4 = bt.run(df, sig_flip, slippage_pts=1.0, commission_ntd=60, tax_rate=2e-5)
n_sides = res4["pos"].diff().abs().fillna(res4["pos"].abs()).sum()
check("flip = 2 sides + entry = 3 total", np.isclose(n_sides, 3.0), f"sides={n_sides}")

# ── trade extraction.
tr = bt.trades_from(res4)
check("two trades extracted", len(tr) == 2, f"n={len(tr)}")

# ── continuous contract: two contracts, roll on settlement day; the roll-day
# return must use the NEW contract's own previous close, not the old one's.
rows = []
d = pd.bdate_range("2024-01-10", "2024-01-19")  # settlement 2024-01-17 (3rd Wed)
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
# roll-day return: new contract 16900+170 vs its own prev close 16900+160
exp_ret = (16900 + 170) / (16900 + 160) - 1
check("roll-day return uses same contract",
      np.isclose(jan17["ret"].iloc[0], exp_ret, rtol=1e-9),
      f"got={jan17['ret'].iloc[0]:.6f} expect={exp_ret:.6f}")

# ── stats smoke test.
s = bt.stats(res3, "const_long")
check("stats fields present", all(k in s for k in
      ["ann_ret_pct", "sharpe", "max_dd_pct", "n_trades", "win_rate_pct"]))

print(f"\n{len(FAILS)} failures")
raise SystemExit(1 if FAILS else 0)
