"""スイング戦略の分析ツール。

1. breakdown: 年別・モジュール別・レジーム別の成績分解
2. sensitivity: 主要パラメータを1つずつ動かした感度分析(頑健性の確認)

使い方:
  python analysis.py breakdown
  python analysis.py sensitivity
"""
import argparse
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backtest import run_backtest
from config import Config
from dataset import build_dataset
from report import OUT_DIR, _stats
from strategies import StrategyContext


def _run(cfg: Config, ds: dict):
    ctx = StrategyContext(ds, cfg)
    return ctx, run_backtest(ctx, cfg, ds["eval_start"])


def _trades_df(result) -> pd.DataFrame:
    return pd.DataFrame([vars(t) for t in result.trades])


def _row(label: str, df: pd.DataFrame) -> str:
    s = _stats(df)
    if s["trades"] == 0:
        return f"{label:<24} トレードなし"
    return (
        f"{label:<24} n={s['trades']:>3}  勝率={s['win_rate']:>5.1%}  "
        f"PF={s['profit_factor']:>5.2f}  損益={s['total_pnl']:>+12,.0f}円"
    )


def cmd_breakdown():
    cfg = Config()
    ds = build_dataset(cfg)
    print("バックテスト実行中...")
    ctx, result = _run(cfg, ds)
    tdf = _trades_df(result)
    eq = result.equity.dropna()

    lines = ["=" * 72, "スイング戦略 成績分解", "=" * 72]

    # ---- 年別 ----
    lines.append("\n■ 年別(資産曲線ベースのリターンとトレード成績)")
    yearly_eq = eq.resample("YE").last()
    prev = cfg.initial_equity
    for ts, v in yearly_eq.items():
        y = ts.year
        ydf = tdf[tdf["exit_date"].dt.year == y] if not tdf.empty else tdf
        lines.append(f"  {y}: 年間リターン {v / prev - 1.0:+7.2%} | " + _row("", ydf).strip())
        prev = v

    # ---- モジュール×年 ----
    lines.append("\n■ モジュール×年の損益(円)")
    if not tdf.empty:
        pv = tdf.pivot_table(
            values="pnl", index="module",
            columns=tdf["exit_date"].dt.year, aggfunc="sum",
        ).fillna(0.0)
        lines.append(pv.round(0).to_string())

    # ---- レジーム別 ----
    lines.append("\n■ エントリー時レジーム別(シグナル日の地合い)")
    if not tdf.empty:
        def regime_at_entry(d):
            p = ctx.date_pos.get(d)
            if p is None or p == 0:
                return "unknown"
            sig_day = ctx.dates[p - 1]
            return "trend" if bool(ctx.trend_regime.get(sig_day, False)) else "range"
        tdf["regime"] = tdf["entry_date"].map(regime_at_entry)
        for reg in ["trend", "range"]:
            lines.append("  " + _row(f"{reg}レジーム", tdf[tdf["regime"] == reg]))

    # ---- 手仕舞い理由別 ----
    lines.append("\n■ 手仕舞い理由別")
    if not tdf.empty:
        for reason in tdf["reason"].unique():
            lines.append("  " + _row(reason, tdf[tdf["reason"] == reason]))

    # ---- 保有日数の分布 ----
    if not tdf.empty:
        lines.append(
            f"\n■ 保有日数: 中央値 {tdf['bars'].median():.0f}日 / "
            f"平均 {tdf['bars'].mean():.1f}日 / 最長 {tdf['bars'].max()}日"
        )

    text = "\n".join(lines)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "analysis_breakdown.txt").write_text(text, encoding="utf-8")
    print(text)
    print("\noutput/analysis_breakdown.txt に保存しました")


def cmd_sensitivity():
    base = Config()
    ds = build_dataset(base)

    variants = [
        ("base(現行設定)", {}),
        ("逆張り乖離 -10%", {"mr_dev25_threshold": -0.10}),
        ("逆張り乖離 -12%", {"mr_dev25_threshold": -0.12}),
        ("逆張り乖離 -18%", {"mr_dev25_threshold": -0.18}),
        ("順張りRS上位5%", {"mom_rs_top_pct": 0.05}),
        ("順張りRS上位15%", {"mom_rs_top_pct": 0.15}),
        ("トレイル5日線", {"mom_trail_sma": 5}),
        ("トレイル20日線", {"mom_trail_sma": 20}),
        ("PEADギャップ+3%", {"pead_gap_min": 0.03}),
        ("PEADギャップ+7%", {"pead_gap_min": 0.07}),
        ("決算またぎ回避なし", {"earnings_avoid_days": 0, "pead_earnings_avoid_days": 0}),
        ("SQ週減額なし", {"sq_size_mult": 1.0}),
        ("信用倍率フィルターなし", {"margin_ratio_max": 999.0}),
    ]

    rows = []
    for i, (label, over) in enumerate(variants):
        cfg = replace(base, **over)
        _, result = _run(cfg, ds)
        eq = result.equity.dropna()
        tdf = _trades_df(result)
        s = _stats(tdf) if not tdf.empty else {"trades": 0}
        total = eq.iloc[-1] / cfg.initial_equity - 1.0 if not eq.empty else 0.0
        dd = (eq / eq.cummax() - 1.0).min() if not eq.empty else 0.0
        rows.append(
            {
                "設定": label,
                "総リターン": f"{total:+.2%}",
                "最大DD": f"{dd:.2%}",
                "トレード数": s.get("trades", 0),
                "勝率": f"{s.get('win_rate', 0):.1%}" if s.get("trades") else "-",
                "PF": f"{s.get('profit_factor', 0):.2f}" if s.get("trades") else "-",
            }
        )
        print(f"[{i + 1}/{len(variants)}] {label}: 総リターン {total:+.2%}")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "analysis_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\n" + df.to_string(index=False))
    print("\noutput/analysis_sensitivity.csv に保存しました")


def main():
    p = argparse.ArgumentParser(description="スイング戦略の分析")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("breakdown", help="年別・レジーム別の成績分解")
    sub.add_parser("sensitivity", help="パラメータ感度分析")
    args = p.parse_args()
    if args.cmd == "breakdown":
        cmd_breakdown()
    else:
        cmd_sensitivity()


if __name__ == "__main__":
    main()
