#!/usr/bin/env python3
"""Clean raw TAIFEX/TWSE downloads into analysis-ready tables.

Outputs (quant-research/data/clean/):
  tx_continuous.csv   — near-month TX continuous series (roll on settlement day),
                        day-session only, with back-adjusted returns
  taiex.csv           — TAIEX index daily OHLC
  institutional.csv   — foreign investors / dealers / trusts net OI in TXF & MXF
  pcr.csv             — TXO put/call volume & OI ratios
  merged.csv          — everything joined on date
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "clean"


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]
    return df


def _num(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", "").str.replace("-", "", regex=False).replace("", np.nan)
        if s.dtype == object else s,
        errors="coerce",
    )


def third_wednesday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    offset = (2 - d.dayofweek) % 7  # Wednesday = 2
    return d + pd.Timedelta(days=offset + 14)


def load_futures(commodity: str = "TX") -> pd.DataFrame:
    raw = pd.read_csv(DATA / f"futures_daily_{commodity}.csv", low_memory=False)
    raw = _norm_cols(raw)
    colmap = {}
    for c in raw.columns:
        if "交易日期" in c: colmap[c] = "date"
        elif c == "契約": colmap[c] = "symbol"
        elif "到期月份" in c: colmap[c] = "expiry"
        elif c == "開盤價": colmap[c] = "open"
        elif c == "最高價": colmap[c] = "high"
        elif c == "最低價": colmap[c] = "low"
        elif c == "收盤價": colmap[c] = "close"
        elif c == "成交量": colmap[c] = "volume"
        elif c == "結算價": colmap[c] = "settle"
        elif "未沖銷" in c: colmap[c] = "oi"
        elif "交易時段" in c: colmap[c] = "session"
    df = raw.rename(columns=colmap)
    keep = [c for c in ["date", "symbol", "expiry", "open", "high", "low", "close",
                        "volume", "settle", "oi", "session"] if c in df.columns]
    df = df[keep].copy()
    df = df[df["symbol"].astype(str).str.strip() == commodity]
    # keep single-month contracts only (drop spread rows like 202401/202402)
    df["expiry"] = df["expiry"].astype(str).str.strip()
    df = df[df["expiry"].str.fullmatch(r"\d{6}")]
    if "session" in df.columns:
        df["session"] = df["session"].astype(str).str.strip()
        df = df[(df["session"] == "一般") | (df["session"].isin(["nan", "", "-"]))]
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y/%m/%d", errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "settle", "oi"]:
        if c in df.columns:
            df[c] = _num(df[c])
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    return df.sort_values(["date", "expiry"]).reset_index(drop=True)


def build_continuous(fut: pd.DataFrame) -> pd.DataFrame:
    """Near-month continuous: use front contract until the day BEFORE its
    settlement day; from settlement day onward use the next month.
    Returns are computed within the same contract (no roll jump)."""
    fut = fut.copy()
    fut["settlement"] = [third_wednesday(int(e[:4]), int(e[4:6])) for e in fut["expiry"]]
    fut = fut[fut["date"] < fut["settlement"]]  # front contract active before its settlement day
    front = fut.sort_values(["date", "expiry"]).groupby("date").first().reset_index()

    prev_close_same = []
    by_contract = {k: g.set_index("date")["close"] for k, g in fut.groupby("expiry")}
    dates = front["date"].tolist()
    for i, row in front.iterrows():
        if i == 0:
            prev_close_same.append(np.nan)
            continue
        series = by_contract[row["expiry"]]
        prev_dates = series.index[series.index < row["date"]]
        prev_close_same.append(series.loc[prev_dates[-1]] if len(prev_dates) else np.nan)
    front["prev_close_same_contract"] = prev_close_same
    front["ret"] = front["close"] / front["prev_close_same_contract"] - 1
    # log-style back-adjusted level for charts
    front["adj_level"] = front["close"].iloc[-1] * np.exp(
        -(np.log1p(front["ret"].fillna(0))[::-1].cumsum()[::-1] - np.log1p(front["ret"].fillna(0)))
    ) if len(front) else np.nan
    front["roll"] = front["expiry"].ne(front["expiry"].shift(1))
    return front


def load_taiex() -> pd.DataFrame:
    df = pd.read_csv(DATA / "taiex_daily_raw.csv")
    df = _norm_cols(df)
    df.columns = ["date_roc", "open", "high", "low", "close"][: len(df.columns)]

    def roc_to_dt(s: str):
        m = re.match(r"(\d+)/(\d+)/(\d+)", str(s).strip())
        if not m:
            return pd.NaT
        y = int(m.group(1)) + 1911
        return pd.Timestamp(year=y, month=int(m.group(2)), day=int(m.group(3)))

    df["date"] = df["date_roc"].map(roc_to_dt)
    for c in ["open", "high", "low", "close"]:
        df[c] = _num(df[c])
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date")
    return df[["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)


def load_institutional() -> pd.DataFrame:
    frames = []
    for comm in ["TXF", "MXF"]:
        p = DATA / f"institutional_{comm}.csv"
        if not p.exists():
            continue
        raw = _norm_cols(pd.read_csv(p, low_memory=False))
        colmap = {}
        for c in raw.columns:
            if c in ("日期",): colmap[c] = "date"
            elif "身份別" in c or "身分別" in c: colmap[c] = "who"
            elif "多空" in c and "未平倉" in c and "口數" in c: colmap[c] = "net_oi"
        df = raw.rename(columns=colmap)
        if not {"date", "who", "net_oi"} <= set(df.columns):
            continue
        df = df[["date", "who", "net_oi"]].copy()
        df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce")
        df["net_oi"] = _num(df["net_oi"])
        df["who"] = df["who"].astype(str).str.strip()
        df = df.dropna(subset=["date"])
        piv = df.pivot_table(index="date", columns="who", values="net_oi", aggfunc="sum")
        piv.columns = [f"{comm}_{w}" for w in piv.columns]
        frames.append(piv)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).reset_index()


def load_pcr() -> pd.DataFrame:
    p = DATA / "txo_pc_ratio.csv"
    if not p.exists():
        return pd.DataFrame()
    raw = _norm_cols(pd.read_csv(p, low_memory=False))
    colmap = {}
    for c in raw.columns:
        if "日期" in c: colmap[c] = "date"
        elif "成交量比率" in c: colmap[c] = "pcr_vol"
        elif "未平倉量比率" in c: colmap[c] = "pcr_oi"
    df = raw.rename(columns=colmap)
    keep = [c for c in ["date", "pcr_vol", "pcr_oi"] if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce")
    for c in ["pcr_vol", "pcr_oi"]:
        if c in df.columns:
            df[c] = _num(df[c])
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fut = load_futures("TX")
    cont = build_continuous(fut)
    cont.to_csv(OUT / "tx_continuous.csv", index=False)

    taiex = load_taiex()
    taiex.to_csv(OUT / "taiex.csv", index=False)

    inst = load_institutional()
    if len(inst):
        inst.to_csv(OUT / "institutional.csv", index=False)
    pcr = load_pcr()
    if len(pcr):
        pcr.to_csv(OUT / "pcr.csv", index=False)

    merged = cont.rename(columns={c: f"tx_{c}" for c in cont.columns if c != "date"})
    merged = merged.merge(taiex.rename(columns={c: f"ix_{c}" for c in taiex.columns if c != "date"}),
                          on="date", how="left")
    if len(inst):
        merged = merged.merge(inst, on="date", how="left")
    # MTX total market OI (per-date sum over contracts) for the retail ratio
    mtx_path = DATA / "futures_daily_MTX.csv"
    if mtx_path.exists():
        mtx = load_futures("MTX")
        mtx_oi = mtx.groupby("date")["oi"].sum().rename("mtx_total_oi").reset_index()
        merged = merged.merge(mtx_oi, on="date", how="left")
    if len(pcr):
        merged = merged.merge(pcr, on="date", how="left")
    merged["basis"] = merged["tx_close"] - merged["ix_close"]
    merged.to_csv(OUT / "merged.csv", index=False)

    summary = {
        "tx_continuous": {"rows": len(cont),
                          "range": [str(cont['date'].min().date()), str(cont['date'].max().date())],
                          "rolls": int(cont['roll'].sum()),
                          "nan_ret": int(cont['ret'].isna().sum())},
        "taiex": {"rows": len(taiex),
                  "range": [str(taiex['date'].min().date()), str(taiex['date'].max().date())]},
        "institutional_cols": list(inst.columns) if len(inst) else [],
        "pcr_rows": int(len(pcr)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
