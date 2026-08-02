#!/usr/bin/env python3
"""Download official TAIFEX / TWSE historical data for TX (台指期) research.

Datasets:
  1. TAIFEX futures daily OHLC per contract (TX, MTX)   — from 1998-07 / 2001-04
  2. TAIFEX institutional investors by contract (TXF, MXF) — from 2007-07
  3. TAIFEX TXO put/call ratio                           — from 2001-12
  4. TWSE TAIEX index daily OHLC                         — from 1999-01
  5. Yahoo ^TWII daily (cross-check, non-fatal)

Designed to run inside GitHub Actions (unrestricted egress). Each dataset is
fetched in monthly chunks with retries; partial failures are recorded in
failures.json instead of aborting the whole job.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import io
import json
import random
import time
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-data-fetch/1.0"}
TAIFEX = "https://www.taifex.com.tw/cht/3"
FAILURES = []


def month_starts(start: dt.date, end: dt.date):
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield max(cur, start), min(nxt - dt.timedelta(days=1), end)
        cur = nxt


def post_csv(url: str, data: dict, tag: str, tries: int = 4,
             backoff_base: float = 2.0) -> str | None:
    for i in range(tries):
        try:
            r = requests.post(url, data=data, headers=UA, timeout=60)
            if r.status_code == 200 and len(r.content) > 10:
                for enc in ("utf-8-sig", "big5", "cp950"):
                    try:
                        return r.content.decode(enc)
                    except UnicodeDecodeError:
                        continue
                return r.content.decode("utf-8", errors="replace")
            if r.status_code == 429:
                raise RuntimeError(f"http=429 len={len(r.content)}")
            raise RuntimeError(f"http={r.status_code} len={len(r.content)}")
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                FAILURES.append({"tag": tag, "data": data, "error": str(e)})
                return None
            wait = backoff_base ** (i + 1) + random.random() * 2
            if "429" in str(e):
                wait = max(wait, 15.0 + 10 * i)
            time.sleep(wait)
    return None


def parse_csv_text(txt: str) -> pd.DataFrame | None:
    """TAIFEX CSVs end data rows with a trailing comma (one field more than
    the header), which makes pandas silently treat column 1 (the date) as an
    index — and lose it on save. Parse with csv.reader and pad/trim each row
    to the header length instead."""
    import csv as _csv
    lines = [l for l in txt.splitlines() if l.strip() and not l.lstrip().startswith("<")]
    if len(lines) < 2:
        return None
    try:
        parsed = list(_csv.reader(lines))
    except Exception:  # noqa: BLE001
        return None
    header = [h.strip() for h in parsed[0] if h.strip() != ""]
    n = len(header)
    if n < 2:
        return None
    rows = []
    for p in parsed[1:]:
        cells = [c.strip() for c in p]
        if len(cells) > n:
            cells = cells[:n]
        elif len(cells) < n:
            cells += [""] * (n - len(cells))
        rows.append(cells)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=header)


def fetch_monthly(url, base_data, date_keys, start, end, tag, workers=3,
                  pause=0.2, tries=4, backoff_base=2.0):
    chunks = list(month_starts(start, end))

    def one(c):
        s, e = c
        d = dict(base_data)
        d[date_keys[0]] = s.strftime("%Y/%m/%d")
        d[date_keys[1]] = e.strftime("%Y/%m/%d")
        time.sleep(pause + random.random() * pause)
        txt = post_csv(url, d, f"{tag}:{s}", tries=tries, backoff_base=backoff_base)
        return parse_csv_text(txt) if txt else None

    out = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for df in ex.map(one, chunks):
            if df is not None and len(df):
                out.append(df)
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    res = res.drop_duplicates()
    return res


def fetch_futures_daily(out_dir: Path, today: dt.date):
    for commodity, start in [("TX", dt.date(1998, 7, 21)), ("MTX", dt.date(2001, 4, 9))]:
        df = fetch_monthly(
            f"{TAIFEX}/futDataDown",
            {"down_type": "1", "commodity_id": commodity},
            ("queryStartDate", "queryEndDate"),
            start, today, f"fut_{commodity}",
        )
        df.to_csv(out_dir / f"futures_daily_{commodity}.csv", index=False)
        print(f"[futures {commodity}] rows={len(df)}")


def fetch_institutional(out_dir: Path, today: dt.date):
    # futContractsDateDown rate-limits aggressively (429): single worker,
    # long pauses, and patient 429 backoff.
    for commodity in ["TXF", "MXF"]:
        df = fetch_monthly(
            f"{TAIFEX}/futContractsDateDown",
            {"commodityId": commodity},
            ("queryStartDate", "queryEndDate"),
            dt.date(2007, 7, 2), today, f"inst_{commodity}",
            workers=1, pause=2.0, tries=6, backoff_base=3.0,
        )
        df.to_csv(out_dir / f"institutional_{commodity}.csv", index=False)
        print(f"[institutional {commodity}] rows={len(df)}", flush=True)


def fetch_pc_ratio(out_dir: Path, today: dt.date):
    df = fetch_monthly(
        f"{TAIFEX}/pcRatioDown", {},
        ("queryStartDate", "queryEndDate"),
        dt.date(2001, 12, 24), today, "pcr",
    )
    df.to_csv(out_dir / "txo_pc_ratio.csv", index=False)
    print(f"[pc_ratio] rows={len(df)}")


def fetch_taiex(out_dir: Path, today: dt.date):
    rows = []
    for s, _e in month_starts(dt.date(1999, 1, 1), today):
        url = ("https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"
               f"?date={s.strftime('%Y%m01')}&response=json")
        for i in range(4):
            try:
                time.sleep(0.15 + random.random() * 0.2)
                r = requests.get(url, headers=UA, timeout=30)
                j = r.json()
                if j.get("stat") == "OK" and j.get("data"):
                    rows.extend(j["data"])
                break
            except Exception as e:  # noqa: BLE001
                if i == 3:
                    FAILURES.append({"tag": f"taiex:{s}", "error": str(e)})
                time.sleep(2**i)
    df = pd.DataFrame(rows, columns=["date_roc", "open", "high", "low", "close"])
    df.to_csv(out_dir / "taiex_daily_raw.csv", index=False)
    print(f"[taiex] rows={len(df)}")


def fetch_yahoo(out_dir: Path):
    try:
        import yfinance as yf
        df = yf.download("^TWII", start="1997-01-01", progress=False, auto_adjust=False)
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] for c in df.columns]
        df.to_csv(out_dir / "twii_yahoo.csv")
        print(f"[yahoo ^TWII] rows={len(df)}")
    except Exception as e:  # noqa: BLE001
        FAILURES.append({"tag": "yahoo_twii", "error": str(e)})
        print(f"[yahoo ^TWII] FAILED (non-fatal): {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="quant-research/data")
    ap.add_argument("--only", default="",
                    help="comma list: taiex,futures,institutional,pcr,yahoo (default all)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    only = set(args.only.split(",")) if args.only else set()

    def want(name):
        return not only or name in only

    if want("taiex"):
        fetch_taiex(out_dir, today)
    if want("futures"):
        fetch_futures_daily(out_dir, today)
    if want("institutional"):
        fetch_institutional(out_dir, today)
    if want("pcr"):
        fetch_pc_ratio(out_dir, today)
    if want("yahoo"):
        fetch_yahoo(out_dir)

    manifest = {"generated_utc": dt.datetime.utcnow().isoformat(), "files": {}}
    for f in sorted(out_dir.glob("*.csv")):
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        n_lines = sum(1 for _ in f.open(encoding="utf-8", errors="replace")) - 1
        manifest["files"][f.name] = {"rows": n_lines, "sha256_16": h}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "failures.json").write_text(json.dumps(FAILURES, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"failures={len(FAILURES)}")


if __name__ == "__main__":
    main()
