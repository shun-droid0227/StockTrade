"""エントリーポイント。

使い方:
  python main.py selftest              # APIなしの合成データでエンジン全体を検証
  python main.py run                   # J-Quantsからデータ取得してバックテスト実行
  python main.py run --start 2023-01-01 --end 2026-06-30 --universe 200
"""
import argparse
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import Config


def cmd_selftest():
    from dataset import synthetic_dataset
    from strategies import StrategyContext
    from backtest import run_backtest
    from report import make_report

    cfg = Config(initial_equity=3_000_000)
    print("合成データでセルフテスト実行中(API不要)...")
    ds = synthetic_dataset(cfg)
    ctx = StrategyContext(ds, cfg)
    result = run_backtest(ctx, cfg, ds["eval_start"])
    print(make_report(result))
    n = len(result.trades)
    print(f"\nセルフテスト完了: トレード {n} 件生成、エンジンは正常に動作")
    return 0 if n > 0 else 1


def cmd_run(args):
    from dataset import build_dataset
    from strategies import StrategyContext
    from backtest import run_backtest
    from report import make_report

    cfg = Config()
    if args.start:
        cfg.start = args.start
    if args.end:
        cfg.end = args.end
    if args.universe:
        cfg.universe_size = args.universe
    if args.equity:
        cfg.initial_equity = args.equity

    ds = build_dataset(cfg)
    print("シグナル事前計算中...")
    ctx = StrategyContext(ds, cfg)
    print("バックテスト実行中...")
    result = run_backtest(ctx, cfg, ds["eval_start"])
    print(make_report(result))
    print("\noutput/ に trades.csv, summary.txt, equity_curve.png を保存しました")
    return 0


def main():
    p = argparse.ArgumentParser(description="日本株 短期複合戦略バックテスト")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="合成データでエンジンを検証(API不要)")
    pr = sub.add_parser("run", help="J-Quantsデータでバックテスト")
    pr.add_argument("--start", default="", help="開始日 YYYY-MM-DD")
    pr.add_argument("--end", default="", help="終了日 YYYY-MM-DD(省略時は最新)")
    pr.add_argument("--universe", type=int, default=0, help="ユニバース銘柄数")
    pr.add_argument("--equity", type=float, default=0, help="初期資金(円)")
    args = p.parse_args()
    if args.cmd == "selftest":
        sys.exit(cmd_selftest())
    sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
