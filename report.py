"""バックテスト結果の集計とレポート出力。

output/ に以下を保存:
- trades.csv        全トレード明細
- summary.txt       モジュール別・全体の成績サマリー
- equity_curve.png  資産曲線と月次リターン
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BacktestResult

OUT_DIR = Path(__file__).resolve().parent / "output"


def _stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0}
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    gross_win = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    return {
        "trades": len(df),
        "win_rate": len(wins) / len(df),
        "avg_ret": df["ret"].mean(),
        "avg_win": wins["ret"].mean() if len(wins) else 0.0,
        "avg_loss": losses["ret"].mean() if len(losses) else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else np.inf,
        "total_pnl": df["pnl"].sum(),
        "avg_bars": df["bars"].mean(),
    }


def _fmt(name: str, s: dict) -> str:
    if s["trades"] == 0:
        return f"[{name}] トレードなし\n"
    return (
        f"[{name}]\n"
        f"  トレード数     : {s['trades']}\n"
        f"  勝率           : {s['win_rate']:.1%}\n"
        f"  平均リターン   : {s['avg_ret']:+.2%}(勝ち {s['avg_win']:+.2%} / 負け {s['avg_loss']:+.2%})\n"
        f"  プロフィットファクター: {s['profit_factor']:.2f}\n"
        f"  合計損益       : {s['total_pnl']:+,.0f} 円\n"
        f"  平均保有日数   : {s['avg_bars']:.1f}\n"
    )


def make_report(result: BacktestResult, show_plot: bool = False) -> str:
    OUT_DIR.mkdir(exist_ok=True)
    cfg = result.cfg
    eq = result.equity.dropna()

    tdf = pd.DataFrame([vars(t) for t in result.trades])
    if not tdf.empty:
        tdf.to_csv(OUT_DIR / "trades.csv", index=False, encoding="utf-8-sig")

    lines = ["=" * 60, "バックテスト結果", "=" * 60]
    if result.warnings:
        for w in result.warnings:
            lines.append(f"注意: {w}")
        lines.append("-" * 60)

    if eq.empty:
        lines.append("評価期間にデータがありません")
        text = "\n".join(lines)
        (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
        return text

    total_ret = eq.iloc[-1] / cfg.initial_equity - 1.0
    days = (eq.index[-1] - eq.index[0]).days
    cagr = (eq.iloc[-1] / cfg.initial_equity) ** (365.25 / days) - 1.0 if days > 30 else total_ret
    dd = (eq / eq.cummax() - 1.0).min()
    daily_ret = eq.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(245) if daily_ret.std() > 0 else 0.0

    lines += [
        f"期間           : {eq.index[0].date()} 〜 {eq.index[-1].date()}",
        f"初期資金       : {cfg.initial_equity:,.0f} 円",
        f"最終資産       : {eq.iloc[-1]:,.0f} 円",
        f"総リターン     : {total_ret:+.2%}(年率換算 {cagr:+.2%})",
        f"最大ドローダウン: {dd:.2%}",
        f"シャープレシオ : {sharpe:.2f}",
        "-" * 60,
    ]
    if tdf.empty:
        lines.append("トレードが1件も発生しませんでした(条件が厳しすぎる可能性)")
    else:
        lines.append(_fmt("全体", _stats(tdf)))
        for mod in ["momentum", "meanrev", "pead"]:
            label = {"momentum": "② 順張りスイング", "meanrev": "③ 逆張りスイング", "pead": "④ 決算PEAD"}[mod]
            lines.append(_fmt(label, _stats(tdf[tdf["module"] == mod])))
        reason_counts = tdf["reason"].value_counts().to_dict()
        lines.append(f"手仕舞い内訳: {reason_counts}")

    text = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")

    # ---- グラフ ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1])
    axes[0].plot(eq.index, eq.values, lw=1.2)
    axes[0].set_title("Equity curve")
    axes[0].grid(alpha=0.3)
    if result.regime is not None:
        reg = result.regime.reindex(eq.index).fillna(False).astype(bool)
        axes[0].fill_between(
            eq.index, eq.min(), eq.max(), where=reg.values,
            alpha=0.08, color="green", label="trend regime",
        )
        axes[0].legend(loc="upper left")
    monthly = eq.resample("ME").last().pct_change().dropna()
    axes[1].bar(monthly.index, monthly.values, width=20,
                color=["tab:green" if v >= 0 else "tab:red" for v in monthly.values])
    axes[1].set_title("Monthly return")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "equity_curve.png", dpi=120)
    plt.close(fig)

    return text
