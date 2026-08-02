#!/usr/bin/env python3
"""Signal generators for TX daily strategies.

Every function takes the merged daily frame (see prepare_data.py) and returns
(signal_series, extra_lag). Signals use only information available at the
close of each row's date; extra_lag=1 marks data published after the
day-session close (institutional flows, PCR), pushing execution one more day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn
    return 100 - 100 / (1 + rs)


# ── baselines ────────────────────────────────────────────────────────────

def buy_hold(df):
    return pd.Series(1.0, index=df.index), 0


# ── trend / momentum ─────────────────────────────────────────────────────

def ma_cross(df, fast=20, slow=60, short=True):
    f = df["tx_close"].rolling(fast).mean()
    s = df["tx_close"].rolling(slow).mean()
    sig = pd.Series(np.where(f > s, 1.0, -1.0 if short else 0.0), index=df.index)
    sig[s.isna()] = 0.0
    return sig, 0


def donchian(df, entry=20, exit_=10, short=True):
    hi = df["tx_high"].rolling(entry).max().shift(1)
    lo = df["tx_low"].rolling(entry).min().shift(1)
    xhi = df["tx_high"].rolling(exit_).max().shift(1)
    xlo = df["tx_low"].rolling(exit_).min().shift(1)
    sig = np.zeros(len(df))
    cur = 0.0
    c = df["tx_close"].values
    for i in range(len(df)):
        if np.isnan(hi.iloc[i]) or np.isnan(lo.iloc[i]):
            sig[i] = 0.0
            continue
        if cur == 0:
            if c[i] > hi.iloc[i]:
                cur = 1.0
            elif short and c[i] < lo.iloc[i]:
                cur = -1.0
        elif cur > 0 and c[i] < xlo.iloc[i]:
            cur = -1.0 if (short and c[i] < lo.iloc[i]) else 0.0
        elif cur < 0 and c[i] > xhi.iloc[i]:
            cur = 1.0 if c[i] > hi.iloc[i] else 0.0
        sig[i] = cur
    return pd.Series(sig, index=df.index), 0


def tsmom(df, lookback=120, short=True):
    mom = df["tx_close"].pct_change(lookback)
    sig = pd.Series(np.where(mom > 0, 1.0, -1.0 if short else 0.0), index=df.index)
    sig[mom.isna()] = 0.0
    return sig, 0


# ── mean reversion ───────────────────────────────────────────────────────

def rsi2_reversion(df, buy_th=10, exit_th=60, trend_ma=200, short=True):
    rsi = _rsi(df["tx_close"], 2)
    ma = df["tx_close"].rolling(trend_ma).mean()
    above = df["tx_close"] > ma
    sig = np.zeros(len(df))
    cur = 0.0
    for i in range(len(df)):
        if np.isnan(ma.iloc[i]) or np.isnan(rsi.iloc[i]):
            sig[i] = 0.0
            continue
        if cur == 0:
            if above.iloc[i] and rsi.iloc[i] < buy_th:
                cur = 1.0
            elif short and not above.iloc[i] and rsi.iloc[i] > 100 - buy_th:
                cur = -1.0
        elif cur > 0 and rsi.iloc[i] > exit_th:
            cur = 0.0
        elif cur < 0 and rsi.iloc[i] < 100 - exit_th:
            cur = 0.0
        sig[i] = cur
    return pd.Series(sig, index=df.index), 0


def boll_reversion(df, n=20, k=2.0, trend_ma=200):
    ma = df["tx_close"].rolling(n).mean()
    sd = df["tx_close"].rolling(n).std()
    trend = df["tx_close"].rolling(trend_ma).mean()
    lower = ma - k * sd
    upper = ma + k * sd
    sig = np.zeros(len(df))
    cur = 0.0
    c = df["tx_close"]
    for i in range(len(df)):
        if np.isnan(lower.iloc[i]) or np.isnan(trend.iloc[i]):
            sig[i] = 0.0
            continue
        if cur == 0:
            if c.iloc[i] < lower.iloc[i] and c.iloc[i] > trend.iloc[i]:
                cur = 1.0
            elif c.iloc[i] > upper.iloc[i] and c.iloc[i] < trend.iloc[i]:
                cur = -1.0
        elif cur > 0 and c.iloc[i] > ma.iloc[i]:
            cur = 0.0
        elif cur < 0 and c.iloc[i] < ma.iloc[i]:
            cur = 0.0
        sig[i] = cur
    return pd.Series(sig, index=df.index), 0


# ── seasonality ──────────────────────────────────────────────────────────

def turn_of_month(df, before=1, after=3):
    dates = pd.DatetimeIndex(df["date"])
    month = dates.to_period("M")
    idx_in_month = pd.Series(range(len(dates)), index=df.index)
    sig = np.zeros(len(df))
    pos_in_month = {}
    for i, m in enumerate(month):
        pos_in_month.setdefault(m, []).append(i)
    for m, idxs in pos_in_month.items():
        for j in idxs[:after]:            # first `after` sessions of month
            sig[j] = 1.0
        for j in idxs[-before:]:          # last `before` sessions of month
            sig[j] = 1.0
    return pd.Series(sig, index=df.index), 0


# ── flows (published after day-session close → extra_lag=1) ─────────────

def foreign_oi(df, z_win=60, th=0.0, short=True, col="TXF_外資"):
    cols = [c for c in df.columns if col in c]
    if not cols:
        return pd.Series(0.0, index=df.index), 1
    x = df[cols[0]].astype(float)
    z = (x - x.rolling(z_win).mean()) / x.rolling(z_win).std()
    sig = pd.Series(np.where(z > th, 1.0, np.where(z < -th, -1.0 if short else 0.0, 0.0)),
                    index=df.index)
    sig[z.isna()] = 0.0
    return sig, 1


def foreign_oi_level(df, short=True, col="TXF_外資"):
    cols = [c for c in df.columns if col in c]
    if not cols:
        return pd.Series(0.0, index=df.index), 1
    x = df[cols[0]].astype(float)
    sig = pd.Series(np.where(x > 0, 1.0, -1.0 if short else 0.0), index=df.index)
    sig[x.isna()] = 0.0
    return sig, 1


def pcr_signal(df, hi=1.1, lo=0.9, short=True):
    if "pcr_oi" not in df.columns:
        return pd.Series(0.0, index=df.index), 1
    p = df["pcr_oi"].astype(float) / 100.0  # taifex reports percent
    sig = pd.Series(np.where(p > hi, 1.0, np.where(p < lo, -1.0 if short else 0.0, 0.0)),
                    index=df.index)
    sig[p.isna()] = 0.0
    return sig, 1


# ── basis ────────────────────────────────────────────────────────────────

def basis_zscore(df, z_win=120, th=1.0, short=True):
    b = df["basis"].astype(float)
    z = (b - b.rolling(z_win).mean()) / b.rolling(z_win).std()
    # deep discount vs its own recent norm → contrarian long
    sig = pd.Series(np.where(z < -th, 1.0, np.where(z > th, -1.0 if short else 0.0, 0.0)),
                    index=df.index)
    sig[z.isna()] = 0.0
    return sig, 0


REGISTRY = {
    "buy_hold": (buy_hold, {}),
    "ma_20_60_ls": (ma_cross, {"fast": 20, "slow": 60, "short": True}),
    "ma_20_60_lo": (ma_cross, {"fast": 20, "slow": 60, "short": False}),
    "ma_5_20_ls": (ma_cross, {"fast": 5, "slow": 20, "short": True}),
    "ma_50_200_ls": (ma_cross, {"fast": 50, "slow": 200, "short": True}),
    "ma_50_200_lo": (ma_cross, {"fast": 50, "slow": 200, "short": False}),
    "donchian_20_10_ls": (donchian, {"entry": 20, "exit_": 10, "short": True}),
    "donchian_55_20_ls": (donchian, {"entry": 55, "exit_": 20, "short": True}),
    "tsmom_60_ls": (tsmom, {"lookback": 60, "short": True}),
    "tsmom_120_ls": (tsmom, {"lookback": 120, "short": True}),
    "tsmom_250_ls": (tsmom, {"lookback": 250, "short": True}),
    "rsi2_lo": (rsi2_reversion, {"short": False}),
    "rsi2_ls": (rsi2_reversion, {"short": True}),
    "boll_20_2_ls": (boll_reversion, {}),
    "tom_1_3": (turn_of_month, {}),
    "foreign_z60_ls": (foreign_oi, {"z_win": 60, "short": True}),
    "foreign_level_ls": (foreign_oi_level, {"short": True}),
    "foreign_level_lo": (foreign_oi_level, {"short": False}),
    "pcr_oi_ls": (pcr_signal, {"short": True}),
    "basis_z120_ls": (basis_zscore, {"z_win": 120, "th": 1.0, "short": True}),
}
