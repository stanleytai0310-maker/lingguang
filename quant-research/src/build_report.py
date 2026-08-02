#!/usr/bin/env python3
"""Build the HTML research report from the results tables.
Output: quant-research/report/index.html (self-contained, zh-TW, dual-theme).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report"

NAME_ZH = {
    "buy_hold": "買進持有（基準）",
    "ma_20_60_lo": "均線濾網 20/60（只做多）",
    "ma_50_200_lo": "均線濾網 50/200（只做多）",
    "ma_20_60_ls": "均線交叉 20/60（多空）",
    "ma_5_20_ls": "均線交叉 5/20（多空）",
    "ma_50_200_ls": "均線交叉 50/200（多空）",
    "donchian_20_10_ls": "Donchian 突破 20/10（多空）",
    "donchian_55_20_ls": "Donchian 突破 55/20（多空）",
    "tsmom_60_ls": "時序動能 60 日（多空）",
    "tsmom_120_ls": "時序動能 120 日（多空）",
    "tsmom_250_ls": "時序動能 250 日（多空）",
    "rsi2_lo": "RSI2 逢低買（200MA 之上）",
    "rsi2_ls": "RSI2 均值回歸（多空）",
    "boll_20_2_ls": "布林通道逆勢（多空）",
    "tom_1_3": "月轉換效應（月底1日＋月初3日）",
    "tom_0_3": "月轉換效應（僅月初3日）",
    "foreign_z60_ls": "外資淨未平倉 z60（多空）",
    "foreign_level_ls": "外資淨部位方向（多空）",
    "foreign_level_lo": "外資淨部位方向（只做多）",
    "pcr_oi_ls": "TXO Put/Call 未平倉比（多空）",
    "basis_z120_ls": "期現貨基差 z120 逆勢（多空）",
    "basis_z120_nodiv_ls": "基差 z120（避開除息季）",
    "mtx_retail_z120_ls": "小台散戶反向 z120（多空）",
    "overnight_long": "隔夜持有（收盤買、開盤賣）",
    "intraday_short": "日盤放空（開盤空、收盤回補）",
    "combo_3": "防禦組合（均線濾網＋RSI2＋TOM 等權）",
}


def fmt(v, nd=2):
    if pd.isna(v):
        return "—"
    return f"{v:.{nd}f}"


def table_html(df: pd.DataFrame, cols: list[tuple[str, str, int]], cls="") -> str:
    head = "".join(f"<th>{h}</th>" for _c, h, _n in cols)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c, _h, nd in cols:
            v = r.get(c)
            if c == "name":
                tds.append(f"<td class='tname'>{NAME_ZH.get(v, v)}<span class='tid'>{v}</span></td>")
            elif isinstance(v, str):
                tds.append(f"<td>{v}</td>")
            else:
                tds.append(f"<td class='num'>{fmt(v, nd)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f"<div class='tblwrap'><table class='{cls}'><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def main():
    OUT.mkdir(exist_ok=True)
    full = pd.read_csv(R / "full_sample.csv").sort_values("sharpe", ascending=False)
    split = pd.read_csv(R / "subsample_split.csv")
    boot = pd.read_csv(R / "bootstrap.csv")
    wf = pd.read_csv(R / "walk_forward.csv")
    dsr = pd.read_csv(R / "deflated_sharpe.csv")
    execs = pd.read_csv(R / "exec_sensitivity.csv")
    costs = pd.read_csv(R / "cost_sensitivity.csv")
    chart = json.loads((R / "chart_data.json").read_text())
    extra = json.loads((R / "extra_analysis.json").read_text())
    diag = json.loads((R / "index_diagnostics.json").read_text())

    # ── derived tables ──
    full_cols = [("name", "策略", 0), ("valid_from", "起算日", 0), ("cagr_pct", "年化報酬%", 2),
                 ("sharpe", "Sharpe", 2), ("max_dd_pct", "最大回撤%", 1), ("t_stat", "t 值", 2),
                 ("n_trades", "交易數", 0), ("win_rate_pct", "勝率%", 1), ("exposure_pct", "曝險%", 0)]
    t_full = table_html(full, full_cols)

    piv = split.pivot_table(index="name", columns="window", values="sharpe").reset_index()
    piv = piv.dropna(subset=["pre2018", "post2018"])
    piv["delta"] = piv["post2018"] - piv["pre2018"]
    piv = piv.sort_values("post2018", ascending=False)
    t_split = table_html(piv, [("name", "策略", 0), ("pre2018", "1998–2017 Sharpe", 2),
                               ("post2018", "2018–2026 Sharpe", 2), ("delta", "變化", 2)])

    bt_show = boot.dropna(subset=["p_value"]).head(12)
    t_boot = table_html(bt_show, [("name", "策略", 0), ("sharpe", "Sharpe（算術）", 2),
                                  ("p_value", "bootstrap p 值", 4), ("n_days", "樣本天數", 0)])

    wf_all = wf[wf["test_year"] == "ALL_OOS"].copy()
    t_wf = table_html(wf_all, [("family", "策略家族", 0), ("cagr_pct", "OOS 年化%", 2),
                               ("sharpe", "OOS Sharpe", 2), ("max_dd_pct", "最大回撤%", 1),
                               ("n_trades", "交易數", 0)])

    t_dsr = table_html(dsr.head(10), [("name", "策略", 0), ("sharpe_arith", "Sharpe", 2),
                                      ("sr0_annual", "資料探勘門檻 SR₀", 2),
                                      ("dsr_prob", "DSR 機率", 3), ("n_trials", "試驗數", 0)])

    epiv = execs.pivot_table(index="name", columns="execution", values="sharpe").reset_index()
    epiv["diff"] = epiv["close"] - epiv["next_open"]
    epiv = epiv.reindex(epiv["diff"].abs().sort_values(ascending=False).index).head(10)
    t_exec = table_html(epiv, [("name", "策略", 0), ("next_open", "次日開盤成交", 2),
                               ("close", "同收盤成交（樂觀）", 2), ("diff", "灌水幅度", 2)])

    cpiv = costs.pivot_table(index="name", columns="slippage", values="cagr_pct").reset_index()
    keep = ["buy_hold", "ma_20_60_lo", "rsi2_lo", "basis_z120_ls", "tom_1_3",
            "donchian_55_20_ls", "overnight_long", "intraday_short"]
    cpiv = cpiv[cpiv["name"].isin(keep)]
    t_cost = table_html(cpiv, [("name", "策略", 0), ("0", "滑價 0 點", 2), ("1", "滑價 1 點（基準）", 2),
                               ("2", "滑價 2 點", 2), ("vol3", "波動日 3 點", 2)])

    payload = json.dumps({"chart": chart, "extra": extra, "diag": diag},
                         ensure_ascii=False, separators=(",", ":"))

    html = HTML_TEMPLATE
    for k, v in {
        "T_FULL": t_full, "T_SPLIT": t_split, "T_BOOT": t_boot, "T_WF": t_wf,
        "T_DSR": t_dsr, "T_EXEC": t_exec, "T_COST": t_cost, "PAYLOAD": payload,
    }.items():
        html = html.replace("{{" + k + "}}", v)

    (OUT / "index.html").write_text(html)
    print(f"report written: {OUT/'index.html'} ({len(html)//1024} KB)")


HTML_TEMPLATE = (Path(__file__).parent / "report_template.html").read_text()

if __name__ == "__main__":
    main()
