"""分足デイトレードバックテスト(DESIGN_INTRADAY.md の Phase 1: ORB)。

流れ:
1. 日足データ(キャッシュ済み)から日ごとのウォッチリストを選定
2. ウォッチリスト銘柄×該当日の1分足を取得(キャッシュ)し、5分足に集約
3. ORB(寄付きレンジブレイクアウト)ルールで日中シミュレーション

執行モデル:
- シグナルは5分足の終値で判定し、次の5分足の始値で執行
- 損切りは足中執行(ギャップ時は始値)。同一足で損切りと利確が両成立なら損切り優先
- 大引け前(2024-11-05以降は15:20、それ以前は14:50)に全決済
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import Trade
from config import Config
from jquants_client import JQuantsClient

SESSION_CHANGE = pd.Timestamp("2024-11-05")  # 大引けが15:00→15:30になった日


def _minutes(t: str) -> int:
    return int(t[:2]) * 60 + int(t[3:5])


def force_close_minute(day: pd.Timestamp) -> int:
    return _minutes("15:20") if day >= SESSION_CHANGE else _minutes("14:50")


# ---------------- ウォッチリスト選定(日足) ----------------
def build_watchlists(ds: dict, cfg: Config, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """日付 -> [(code, gap, score), ...] を返す。判定はすべて寄付き時点で既知の情報のみ。"""
    dates = ds["index"].index

    # 決算の反応日(スイング版と同じロジック)
    reaction: dict[str, set] = {}
    for code, e in ds["earnings"].items():
        s = set()
        for _, row in e.iterrows():
            d, t = row["DisclosedDate"], str(row.get("DisclosedTime") or "15:30")
            i = dates.searchsorted(d)
            if i >= len(dates):
                continue
            if dates[i] == d and t >= "15:00":
                i += 1
            if i < len(dates):
                s.add(dates[i])
        reaction[code] = s

    feats = {}
    for code, p in ds["prices"].items():
        d = p.reindex(dates)
        vol20 = d["Volume"].rolling(20).mean().shift(1)
        feats[code] = pd.DataFrame(
            {
                "gap": d["Open"] / d["Close"].shift(1) - 1.0,
                "rvol": (d["Volume"].shift(1) / vol20).fillna(0.0),
                "turnover20": (d["Close"] * d["Volume"]).rolling(20).mean().shift(1),
                "price_prev": d["Close"].shift(1),
            }
        )

    watch = {}
    for day in dates[(dates >= start) & (dates <= end)]:
        cands = []
        for code, f in feats.items():
            r = f.loc[day]
            if r.isna().any():
                continue
            if r["price_prev"] < cfg.it_min_price or r["turnover20"] < cfg.it_min_turnover:
                continue
            in_play = (
                r["rvol"] >= cfg.it_rvol_threshold
                or abs(r["gap"]) >= cfg.it_gap_threshold
                or day in reaction.get(code, ())
            )
            if not in_play:
                continue
            score = r["rvol"] * (abs(r["gap"]) * 100 + 0.5)
            cands.append((code, float(r["gap"]), float(score)))
        cands.sort(key=lambda x: -x[2])
        if cands:
            watch[day] = cands[: cfg.it_watchlist_size]
    return watch


# ---------------- 分足 → 5分足 ----------------
def to_5min_bars(df: pd.DataFrame) -> pd.DataFrame | None:
    """1分足を5分足に集約し、累積VWAPを付ける。indexは分単位(9:00=540)。"""
    if df.empty:
        return None
    m = df["Time"].map(_minutes)
    # 引けのオークション(単発プリント)はザラ場バーから除外
    body = df[m < _minutes("15:26")].copy()
    if body.empty:
        return None
    mm = body["Time"].map(_minutes)
    body["bucket"] = (mm // 5) * 5
    o = pd.to_numeric(body["O"], errors="coerce")
    h = pd.to_numeric(body["H"], errors="coerce")
    l = pd.to_numeric(body["L"], errors="coerce")
    c = pd.to_numeric(body["C"], errors="coerce")
    vo = pd.to_numeric(body["Vo"], errors="coerce").fillna(0.0)
    va = pd.to_numeric(body["Va"], errors="coerce").fillna(0.0)
    g = pd.DataFrame(
        {"o": o, "h": h, "l": l, "c": c, "vo": vo, "va": va, "bucket": body["bucket"]}
    ).dropna(subset=["c"])
    bars = g.groupby("bucket").agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"),
        vo=("vo", "sum"), va=("va", "sum"),
    )
    cum_vo = bars["vo"].cumsum().replace(0.0, np.nan)
    bars["vwap"] = (bars["va"].cumsum() / cum_vo).ffill()
    return bars


# ---------------- ORBシミュレーション ----------------
@dataclass
class _Pos:
    code: str
    qty: int
    entry: float
    stop: float
    tp: float
    entry_bucket: int
    partial_done: bool = False
    exit_next: str = ""


def run_intraday_backtest(cfg: Config, start: str = "", end: str = "") -> dict:
    from dataset import build_dataset

    ds = build_dataset(cfg)
    dates = ds["index"].index
    start_ts = pd.Timestamp(start or cfg.it_start)
    end_ts = pd.Timestamp(end) if end else dates[-1]

    print("ウォッチリスト選定中...")
    watch = build_watchlists(ds, cfg, start_ts, end_ts)
    n_pairs = sum(len(v) for v in watch.values())
    print(f"  対象 {len(watch)} 日 / 銘柄×日 {n_pairs} 件の分足を取得します")

    client = JQuantsClient()
    equity = cfg.initial_equity
    equity_hist: dict[pd.Timestamp, float] = {}
    trades: list[Trade] = []
    done = 0

    for day, cands in sorted(watch.items()):
        day_str = day.strftime("%Y-%m-%d")
        fc = force_close_minute(day)
        day_start_equity = equity
        day_realized = 0.0
        blocked = False
        entries_today = 0
        traded_codes: set[str] = set()
        positions: list[_Pos] = []
        pendings: dict[str, dict] = {}

        # 分足取得と5分足化
        bars_by_code: dict[str, pd.DataFrame] = {}
        gap_by_code: dict[str, float] = {}
        for code, gap, _ in cands:
            try:
                bars = to_5min_bars(client.minute_bars(code, day_str))
            except Exception:
                continue
            done += 1
            if done % 200 == 0:
                print(f"  分足取得 {done}/{n_pairs}")
            if bars is None or len(bars) < 5:
                continue
            bars_by_code[code] = bars
            gap_by_code[code] = gap

        or_end = _minutes("09:00") + cfg.it_or_minutes
        buckets = sorted({b for bars in bars_by_code.values() for b in bars.index})

        def close_out(pos: _Pos, px: float, bucket: int, reason: str, qty: int | None = None):
            nonlocal equity, day_realized
            q = pos.qty if qty is None else qty
            px = px * (1 - cfg.it_slippage)
            pnl = q * (px - pos.entry) - q * px * cfg.it_commission
            equity += pnl
            day_realized += pnl
            trades.append(
                Trade(
                    code=pos.code, module="orb", entry_date=day, exit_date=day,
                    entry_price=pos.entry, exit_price=px, qty=q, pnl=pnl,
                    ret=px / pos.entry - 1.0,
                    bars=(bucket - pos.entry_bucket) // 5, reason=reason,
                )
            )
            pos.qty -= q

        for b in buckets:
            for code, bars in bars_by_code.items():
                if b not in bars.index:
                    continue
                bar = bars.loc[b]

                # 1) 予約済みエントリーの執行(足の始値)
                if code in pendings and not blocked:
                    sig = pendings.pop(code)
                    if (
                        len(positions) < cfg.it_max_positions
                        and entries_today < cfg.it_max_trades_per_day
                    ):
                        entry = bar["o"] * (1 + cfg.it_slippage)
                        stop = sig["stop"]
                        if stop < entry and (entry - stop) / entry <= cfg.it_max_risk_pct * 1.5:
                            qty = int(cfg.it_risk_per_trade * equity / (entry - stop) / cfg.unit) * cfg.unit
                            while qty > 0 and qty * entry > 0.3 * equity:
                                qty -= cfg.unit
                            if qty > 0:
                                tp = entry + cfg.it_tp_r * (entry - stop)
                                positions.append(
                                    _Pos(code=code, qty=qty, entry=entry, stop=stop,
                                         tp=tp, entry_bucket=b)
                                )
                                entries_today += 1
                                traded_codes.add(code)

                # 2) ポジション管理
                for pos in positions:
                    if pos.code != code or pos.qty <= 0:
                        continue
                    if pos.exit_next:
                        close_out(pos, bar["o"], b, pos.exit_next)
                        continue
                    if b >= fc:
                        close_out(pos, bar["o"], b, "force_close")
                        continue
                    if bar["l"] <= pos.stop:
                        close_out(pos, min(bar["o"], pos.stop), b, "stop")
                        continue
                    if not pos.partial_done and bar["h"] >= pos.tp:
                        half = int(pos.qty / 2 / cfg.unit) * cfg.unit
                        if cfg.unit <= half < pos.qty:
                            close_out(pos, max(bar["o"], pos.tp), b, "tp_half", qty=half)
                        pos.partial_done = True
                    if pos.qty > 0 and bar["c"] < bar["vwap"]:
                        pos.exit_next = "vwap_exit"
                positions = [p for p in positions if p.qty > 0]

                # 3) シグナル判定(足の終値)
                if (
                    blocked
                    or b < or_end
                    or b >= cfg.it_entry_deadline
                    or code in traded_codes
                    or code in pendings
                    or gap_by_code[code] < 0
                ):
                    continue
                or_bars = bars[bars.index < or_end]
                if or_bars.empty:
                    continue
                or_high, or_low = or_bars["h"].max(), or_bars["l"].min()
                or_mid = (or_high + or_low) / 2
                if (or_high - or_low) / or_mid > cfg.it_or_max_width:
                    continue
                prior = bars[(bars.index < b)]
                if prior.empty or bar["c"] <= or_high or bar["c"] <= bar["vwap"]:
                    continue
                if bar["vo"] < cfg.it_vol_mult * prior["vo"].mean():
                    continue
                risk = (bar["c"] - or_mid) / bar["c"]
                if risk <= 0 or risk > cfg.it_max_risk_pct:
                    continue
                pendings[code] = {"stop": or_mid}

            if day_realized <= cfg.it_daily_stop * day_start_equity:
                blocked = True
                pendings.clear()

        # 取り残し(データ末尾までバーが無かった場合)は最終バーの終値で決済
        for pos in positions:
            if pos.qty > 0:
                last_bar = bars_by_code[pos.code].iloc[-1]
                close_out(pos, last_bar["c"], int(bars_by_code[pos.code].index[-1]), "eod")

        equity_hist[day] = equity

    return {"trades": trades, "equity": pd.Series(equity_hist).sort_index(), "cfg": cfg}


def make_intraday_report(result: dict) -> str:
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from report import OUT_DIR, _fmt, _stats

    OUT_DIR.mkdir(exist_ok=True)
    cfg = result["cfg"]
    eq = result["equity"].dropna()
    tdf = pd.DataFrame([vars(t) for t in result["trades"]])
    if not tdf.empty:
        tdf.to_csv(OUT_DIR / "intraday_trades.csv", index=False, encoding="utf-8-sig")

    lines = ["=" * 60, "分足デイトレード(ORB)バックテスト結果", "=" * 60]
    if eq.empty or tdf.empty:
        lines.append("トレードが発生しませんでした")
    else:
        total_ret = eq.iloc[-1] / cfg.initial_equity - 1.0
        days = (eq.index[-1] - eq.index[0]).days
        cagr = (eq.iloc[-1] / cfg.initial_equity) ** (365.25 / days) - 1.0 if days > 30 else total_ret
        dd = (eq / eq.cummax() - 1.0).min()
        lines += [
            f"期間           : {eq.index[0].date()} 〜 {eq.index[-1].date()}",
            f"初期資金       : {cfg.initial_equity:,.0f} 円",
            f"最終資産       : {eq.iloc[-1]:,.0f} 円",
            f"総リターン     : {total_ret:+.2%}(年率換算 {cagr:+.2%})",
            f"最大ドローダウン: {dd:.2%}",
            f"取引日数       : {len(eq)} 日 / トレード発生日はうち {tdf['entry_date'].nunique()} 日",
            "-" * 60,
            _fmt("ORB(平均保有は5分足の本数)", _stats(tdf)),
            f"手仕舞い内訳: {tdf['reason'].value_counts().to_dict()}",
        ]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(eq.index, eq.values, lw=1.2)
        ax.set_title("Intraday ORB equity curve")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "intraday_equity.png", dpi=120)
        plt.close(fig)

    text = "\n".join(lines)
    (OUT_DIR / "intraday_summary.txt").write_text(text, encoding="utf-8")
    return text
