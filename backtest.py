"""イベントループ型バックテストエンジン。

執行モデル:
- シグナルは当日引けで判定し、翌営業日の寄付き成行で執行
- 損切りは場中逆指値(安値が逆指値に達したら約定。ギャップ時は寄付きで約定)
- 利確・トレイル・時間切れは引けで判定し翌日寄付きで執行
- 買いコスト = 寄付き×(1+スリッページ)+手数料
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import Config
from strategies import StrategyContext


@dataclass
class Position:
    code: str
    module: str
    qty: int
    entry_price: float
    entry_date: pd.Timestamp
    stop: float
    tp: float | None = None          # momentum: 半分利確の指値
    target: float | None = None      # meanrev: 25日線への半値戻し水準
    bars_held: int = 0
    partial_done: bool = False
    pending_exit: str = ""           # 空でなければ翌日寄付きで手仕舞い


@dataclass
class Trade:
    code: str
    module: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    ret: float
    bars: int
    reason: str


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series = None
    regime: pd.Series = None
    cfg: Config = None
    warnings: list[str] = field(default_factory=list)


def run_backtest(ctx: StrategyContext, cfg: Config, eval_start: pd.Timestamp) -> BacktestResult:
    dates = ctx.dates
    start_i = int(dates.searchsorted(eval_start))
    cash = cfg.initial_equity
    positions: list[Position] = []
    scheduled: list[dict] = []  # 翌日寄付きで執行するシグナル
    trades: list[Trade] = []
    equity_hist = {}
    month_start_equity = cfg.initial_equity
    entries_blocked = False
    cur_month = None

    def ohlc(code: str, i: int):
        d = ctx.ind[code]
        return (
            float(d["Open"].iloc[i]), float(d["High"].iloc[i]),
            float(d["Low"].iloc[i]), float(d["Close"].iloc[i]),
        )

    def sell(pos: Position, qty: int, price: float, date: pd.Timestamp, reason: str):
        nonlocal cash
        proceeds = qty * price * (1 - cfg.slippage_rate)
        proceeds -= qty * price * cfg.commission_rate
        cash += proceeds
        trades.append(
            Trade(
                code=pos.code, module=pos.module,
                entry_date=pos.entry_date, exit_date=date,
                entry_price=pos.entry_price, exit_price=price, qty=qty,
                pnl=proceeds - qty * pos.entry_price,
                ret=price / pos.entry_price - 1.0,
                bars=pos.bars_held, reason=reason,
            )
        )
        pos.qty -= qty

    for i in range(start_i, len(dates)):
        date = dates[i]

        # ---- 月替わり処理 ----
        if cur_month != (date.year, date.month):
            cur_month = (date.year, date.month)
            month_start_equity = equity_hist[dates[i - 1]] if i > start_i else cash
            entries_blocked = False

        # ---- 1) 寄付き: 予約済み手仕舞い ----
        for pos in positions:
            if pos.pending_exit and pos.qty > 0:
                o = ohlc(pos.code, i)[0]
                if not np.isnan(o):
                    sell(pos, pos.qty, o, date, pos.pending_exit)

        positions = [p for p in positions if p.qty > 0]

        # ---- 2) 寄付き: 新規エントリー ----
        for sig in scheduled:
            if len(positions) >= cfg.max_positions:
                break
            code = sig["code"]
            o = ohlc(code, i)[0]
            if np.isnan(o) or o <= 0:
                continue
            if sig["module"] == "meanrev":
                stop = o * (1 - sig["stop_pct"])
                target = o + 0.5 * (sig["sma25"] - o) if sig["sma25"] > o else None
                tp = None
            else:
                stop = sig["stop"]
                target = None
                tp = None
            if stop >= o:
                continue
            risk_ps = o - stop
            # 損切り幅が想定より大きく開いた場合は見送り
            if sig["module"] == "momentum" and risk_ps / o > cfg.mom_stop_max * 1.5:
                continue
            equity_now = equity_hist.get(dates[i - 1], cash) if i > start_i else cash
            risk_amt = equity_now * cfg.risk_per_trade * sig["size_mult"]
            qty = int(risk_amt / risk_ps / cfg.unit) * cfg.unit
            # 参加率キャップ: 自分の建玉が20日平均出来高のN%を超えないようにする
            vol20 = float(ctx.ind[code]["vol_sma20"].iloc[i - 1]) if i > 0 else np.nan
            if not np.isnan(vol20) and vol20 > 0:
                cap = int(cfg.participation_cap * vol20 / cfg.unit) * cfg.unit
                if cap < cfg.unit:
                    continue
                qty = min(qty, cap)
            max_notional = min(equity_now * cfg.max_position_weight, cash)
            while qty > 0 and qty * o * (1 + cfg.slippage_rate + cfg.commission_rate) > max_notional:
                qty -= cfg.unit
            if qty <= 0:
                continue
            cost = qty * o * (1 + cfg.slippage_rate) + qty * o * cfg.commission_rate
            cash -= cost
            entry_price = cost / qty
            if sig["module"] == "momentum":
                tp = entry_price + sig["tp_r"] * (entry_price - stop)
            positions.append(
                Position(
                    code=code, module=sig["module"], qty=qty,
                    entry_price=entry_price, entry_date=date,
                    stop=stop, tp=tp, target=target,
                )
            )
        scheduled = []

        # ---- 3) 場中: 損切りと半分利確 ----
        for pos in positions:
            if pos.qty <= 0:
                continue
            o, h, l, c = ohlc(pos.code, i)
            if np.isnan(c):
                continue
            if pos.entry_date != date:
                pos.bars_held += 1
            # 損切り優先(同日に両方到達した場合は保守的に損切り扱い)
            if l <= pos.stop:
                px = min(o, pos.stop) if not np.isnan(o) else pos.stop
                sell(pos, pos.qty, px, date, "stop")
                continue
            if pos.module == "momentum" and not pos.partial_done and pos.tp and h >= pos.tp:
                half = int(pos.qty / 2 / cfg.unit) * cfg.unit
                if half >= cfg.unit and half < pos.qty:
                    px = max(o, pos.tp) if not np.isnan(o) else pos.tp
                    sell(pos, half, px, date, "tp_half")
                pos.partial_done = True

        positions = [p for p in positions if p.qty > 0]

        # ---- 3.5) デイトレードモード: 当日引けで全決済(持ち越し禁止) ----
        if cfg.day_trade_only:
            for pos in positions:
                c = ohlc(pos.code, i)[3]
                if np.isnan(c):
                    pos.pending_exit = "day_close"  # 売買停止日は翌日寄付きで決済
                    continue
                sell(pos, pos.qty, c, date, "day_close")
            positions = [p for p in positions if p.qty > 0]

        # ---- 4) 引け: 翌日寄付きでの手仕舞いを予約 ----
        for pos in positions:
            o, h, l, c = ohlc(pos.code, i)
            if np.isnan(c) or pos.pending_exit:
                continue
            d = ctx.ind[pos.code]
            reason = ""
            if pos.module == "momentum":
                # 押し目買い直後はMA割れが常態なので、トレイルは
                # 半分利確後か3日保有後から有効にする
                sma_t = float(d[f"sma{cfg.mom_trail_sma}"].iloc[i])
                trail_active = pos.partial_done or pos.bars_held >= 3
                if trail_active and not np.isnan(sma_t) and c < sma_t:
                    reason = "trail_sma"
                elif pos.bars_held >= cfg.mom_max_hold:
                    reason = "time"
            elif pos.module == "meanrev":
                sma25 = float(d["sma25"].iloc[i])
                if pos.target and c >= pos.target:
                    reason = "target"
                elif not np.isnan(sma25) and c >= sma25:
                    reason = "target"
                elif pos.bars_held >= cfg.mr_time_stop and c < pos.entry_price:
                    reason = "time_stop"
                elif pos.bars_held >= cfg.mr_max_hold:
                    reason = "time"
            elif pos.module == "pead":
                if pos.bars_held >= cfg.pead_max_hold:
                    reason = "time"
            # 決算またぎ回避
            if not reason:
                avoid = cfg.pead_earnings_avoid_days if pos.module == "pead" else cfg.earnings_avoid_days
                nb = ctx.bars_to_next_earnings(pos.code, date)
                if nb is not None and nb <= avoid:
                    reason = "earnings_avoid"
            if reason:
                pos.pending_exit = reason

        # ---- 5) 引け: 評価額と月間DDチェック ----
        pos_value = 0.0
        for pos in positions:
            c = ohlc(pos.code, i)[3]
            d = ctx.ind[pos.code]
            if np.isnan(c):
                c = float(d["Close"].iloc[:i + 1].dropna().iloc[-1])
            pos_value += pos.qty * c
        equity = cash + pos_value
        equity_hist[date] = equity
        if equity / month_start_equity - 1.0 <= cfg.monthly_dd_stop:
            entries_blocked = True

        # ---- 6) 引け: 翌日のシグナル生成 ----
        if entries_blocked or i + 1 >= len(dates):
            continue
        held = {p.code for p in positions}
        slots = cfg.max_positions - len(positions)
        if slots <= 0:
            continue

        def module_risk_mult(module: str) -> float:
            """直近のモジュール成績に応じてリスクを調整(悪化したら縮小→停止)。"""
            if not cfg.adaptive_module_risk:
                return 1.0
            cutoff = date - pd.Timedelta(days=cfg.adaptive_lookback_days)
            recent = [t for t in trades if t.module == module and cutoff <= t.exit_date < date]
            if len(recent) < cfg.adaptive_min_trades:
                return 1.0
            gw = sum(t.pnl for t in recent if t.pnl > 0)
            gl = -sum(t.pnl for t in recent if t.pnl <= 0)
            pf = gw / gl if gl > 0 else 99.0
            if pf < cfg.adaptive_low_pf:
                return 0.0
            if pf < cfg.adaptive_mid_pf:
                return 0.5
            return 1.0
        cands = (
            ctx.momentum_candidates(date)
            + ctx.meanrev_candidates(date)
            + ctx.pead_candidates(date)
        )
        mult = ctx.size_mult(date)
        picked = []
        for sig in sorted(cands, key=lambda s: -s["score"]):
            if len(picked) >= slots:
                break
            if sig["code"] in held or any(p["code"] == sig["code"] for p in picked):
                continue
            if ctx.margin_blocked(sig["code"], date):
                continue
            if ctx.alert_blocked(sig["code"], date):
                continue
            avoid = cfg.pead_earnings_avoid_days if sig["module"] == "pead" else cfg.earnings_avoid_days
            nb = ctx.bars_to_next_earnings(sig["code"], date)
            if nb is not None and nb <= avoid + 3:
                continue
            m_mult = module_risk_mult(sig["module"])
            if m_mult <= 0:
                continue
            sig["size_mult"] = mult * m_mult
            picked.append(sig)
        scheduled = picked

    result = BacktestResult(
        trades=trades,
        equity=pd.Series(equity_hist).sort_index(),
        regime=ctx.trend_regime.reindex(pd.Series(equity_hist).sort_index().index),
        cfg=cfg,
    )
    if ctx.margin is None:
        result.warnings.append("信用需給フィルター無効(margin-interest が未取得)")
    if not ctx.earnings_pos:
        result.warnings.append("PEADと決算またぎ回避が無効(決算データ未取得)")
    if cfg.exclude_margin_alert and ctx.margin_alert is None:
        result.warnings.append("信用規制銘柄フィルター無効(margin-alert が未取得)")
    if not ctx.scale:
        result.warnings.append("規模区分が未取得のため、逆張りは最低価格フィルターのみで稼働")
    return result
