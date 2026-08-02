# 台指期量化研究與驗證（1998–2026）

以期交所 / 證交所官方資料對 24 組 TX 日線策略做含成本回測與多層驗證。
完整報告：`report/index.html`（自包含 HTML，直接開啟即可）。

## 目錄結構

```
quant-research/
├── datafetch/          # GitHub Actions 上執行的官方資料抓取（本地網路受限）
├── data/               # 原始下載（期交所/TWSE CSV）＋ data/clean/ 清理後資料
├── src/
│   ├── prepare_data.py     # 清理、連續近月合約（結算日換月、同合約報酬、回溯調整價）
│   ├── backtest.py         # 回測引擎（次日開盤成交、鎖死日遞延、換月成本、稅+手續費+滑價）
│   ├── strategies.py       # 22 組訊號（趨勢/均值回歸/季節性/籌碼/基差）
│   ├── run_backtests.py    # 全樣本、子樣本切分、成本/執行敏感度
│   ├── robustness.py       # 參數網格、平穩 bootstrap、walk-forward、Deflated Sharpe
│   ├── extra_analysis.py   # α 分解、防禦組合、2026 股災切片
│   ├── test_engine.py      # 20 項引擎正確性測試
│   └── build_report.py     # 由結果表產生 report/index.html
├── results/            # 全部結果表（CSV/JSON）
├── research/           # 市場結構與文獻研究（含來源連結與可信度標注）
└── report/             # 最終報告

## 重跑

```bash
pip install pandas numpy scipy
cd quant-research
python3 src/test_engine.py      # 引擎自測
python3 src/prepare_data.py     # 需 data/ 內原始檔（由 CI workflow 抓取）
python3 src/run_backtests.py
python3 src/robustness.py       # 較耗時（bootstrap + walk-forward）
python3 src/extra_analysis.py && python3 src/make_chart_data.py
python3 src/build_report.py
```

資料更新：修改 `datafetch/` 下任何檔案並推送本分支，`.github/workflows/quant-data-fetch.yml`
會在 GitHub Actions 重抓並把資料 commit 回分支（`--only` 參數控制抓哪些資料集）。

## 主要結論（摘要）

- 通過滾動選參 walk-forward 的家族：RSI2 逢低買（OOS Sharpe 0.57）、基差逆勢（0.48）、PCR（0.45）
- 只做多均線濾網：與買進持有同等報酬、一半回撤（價值在風控不在 α）
- 多空趨勢突破、時段策略（隔夜/日內）、外資方向訊號：不可行或不穩健
- Deflated Sharpe（74 個變體校正）：無策略在 95% 信心下可與資料探勘區分——所有結論以此為前提

本研究為歷史回測，非投資建議。
