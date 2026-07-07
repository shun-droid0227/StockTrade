"""ウォークフォワード検証。

アンカー方式:
- 学習期間は常に 2023-01 から検証期間の直前まで
- 学習期間の成績が最良のパラメータを選び、直後6ヶ月(検証期間)に適用
- 検証期間の成績だけを繋いだ「擬似アウトオブサンプル」成績を、
  固定パラメータ(現行config)と比較する

近似について: 各設定は全期間を1回だけバックテストし、期間別成績は資産曲線の
スライスで評価する(フォールド境界をまたぐポジションの影響は軽微)。

使い方: python walkforward.py
"""
import itertools
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backtest import run_backtest
from config import Config
from dataset import build_dataset
from report import OUT_DIR
from strategies import StrategyContext

# 検証フォールド(検証期間の開始・終了)。学習は2023-01からフォールド直前まで
FOLDS = [
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
    ("2026-01-01", None),
]

# パラメータグリッド(現行configの値を含む16通り)
GRID = {
    "mom_trail_sma": [10, 20],
    "mom_max_hold": [15, 25],
    "mr_dev25_threshold": [-0.12, -0.15],
    "pead_gap_min": [0.03, 0.05],
}


def _slice_metrics(eq: pd.Series, start, end) -> tuple[float, float]:
    s = eq.loc[start:end] if end else eq.loc[start:]
    if len(s) < 2:
        return 0.0, 0.0
    ret = float(s.iloc[-1] / s.iloc[0] - 1.0)
    dd = float((s / s.cummax() - 1.0).min())
    return ret, dd


def _mar(ret: float, dd: float) -> float:
    return ret / max(abs(dd), 0.01)


def main():
    base = Config()
    ds = build_dataset(base)
    print("ベースctx構築中...")
    base_ctx = StrategyContext(ds, base)

    combos = [
        dict(zip(GRID.keys(), vals))
        for vals in itertools.product(*GRID.values())
    ]
    print(f"{len(combos)} 通りを全期間バックテスト中...")
    results = []
    for i, over in enumerate(combos):
        cfg = replace(base, **over)
        ctx = base_ctx.with_config(cfg)
        r = run_backtest(ctx, cfg, ds["eval_start"])
        results.append({"params": over, "equity": r.equity.dropna()})
        print(f"  [{i + 1}/{len(combos)}] {over} 完了")

    base_params = {k: getattr(base, k) for k in GRID}
    base_eq = next(r["equity"] for r in results if r["params"] == base_params)

    lines = ["=" * 78, "ウォークフォワード検証(アンカー方式・6ヶ月フォールド)", "=" * 78]
    wf_rets, base_rets = [], []
    for test_start, test_end in FOLDS:
        train_end = (pd.Timestamp(test_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        best, best_mar = None, -np.inf
        for r in results:
            tr_ret, tr_dd = _slice_metrics(r["equity"], "2023-01-01", train_end)
            m = _mar(tr_ret, tr_dd)
            if m > best_mar:
                best, best_mar = r, m
        te_ret, te_dd = _slice_metrics(best["equity"], test_start, test_end)
        b_ret, _ = _slice_metrics(base_eq, test_start, test_end)
        wf_rets.append(te_ret)
        base_rets.append(b_ret)
        label_end = test_end or "末尾"
        lines.append(
            f"\n■ 検証 {test_start}〜{label_end}"
            f"\n  学習で選ばれた設定: {best['params']}(学習MAR {best_mar:.2f})"
            f"\n  検証期間リターン : WF {te_ret:+.2%}(DD {te_dd:.2%}) / 固定設定 {b_ret:+.2%}"
        )

    wf_total = float(np.prod([1 + r for r in wf_rets]) - 1)
    base_total = float(np.prod([1 + r for r in base_rets]) - 1)
    lines += [
        "",
        "-" * 78,
        f"擬似アウトオブサンプル合計(2024-07〜): WF {wf_total:+.2%} / 固定設定 {base_total:+.2%}",
        "判定の目安: WFが固定設定と同水準以上なら、パラメータ選択手続きが",
        "未来のデータなしでも機能している(=過剰適合していない)ことを示す。",
    ]
    text = "\n".join(lines)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "walkforward.txt").write_text(text, encoding="utf-8")
    print(text)
    print("\noutput/walkforward.txt に保存しました")


if __name__ == "__main__":
    main()
