"""シグナル生成ロジック。

3モジュール:
- momentum: 順張りスイング(上昇レジームのみ)
- meanrev:  逆張りスイング(レンジ・下落レジームのみ)
- pead:     決算後ドリフト(レジーム不問)

すべて「当日の引け後に判定 → 翌営業日の寄付きで執行」を前提とする。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
from indicators import add_stock_indicators, adx, sma


class StrategyContext:
    """バックテストで使う事前計算をまとめて保持する。"""

    def __init__(self, dataset: dict, cfg: Config):
        self.cfg = cfg
        self.prices = dataset["prices"]
        self.margin = dataset["margin"]

        # マスターカレンダー = 指数の日付
        self.index_df = dataset["index"]
        self.dates = self.index_df.index
        self.date_pos = {d: i for i, d in enumerate(self.dates)}

        # 指標付き銘柄データ
        self.ind: dict[str, pd.DataFrame] = {}
        for code, df in self.prices.items():
            aligned = df.reindex(self.dates)
            self.ind[code] = add_stock_indicators(aligned, cfg)

        # クロスセクションのレラティブストレングス順位(0〜1)
        rs = pd.DataFrame({c: d["ret_rs"] for c, d in self.ind.items()})
        self.rs_rank = rs.rank(axis=1, pct=True)

        # レジーム判定
        idx = self.index_df
        s = sma(idx["Close"], cfg.regime_sma)
        self.trend_regime = (idx["Close"] > s) & (s.diff(cfg.regime_slope_days) > 0)
        idx_adx = adx(idx["High"], idx["Low"], idx["Close"], cfg.regime_adx_period)
        self.adx_low = idx_adx < cfg.regime_adx_low

        # メジャーSQ週
        self.sq_dates = self._sq_week_dates()

        # 決算: 銘柄ごとの「市場が反応する営業日」のインデックス位置リスト
        self.earnings_pos: dict[str, list[int]] = {}
        for code, e in dataset["earnings"].items():
            pos = sorted({p for p in (self._reaction_pos(r) for _, r in e.iterrows()) if p is not None})
            if pos:
                self.earnings_pos[code] = pos

        # PEADイベントを事前計算 → シグナル日ごとの辞書に展開
        self.pead_signals = self._precompute_pead()

    # ---------- 補助 ----------
    def _next_pos_on_or_after(self, ts: pd.Timestamp) -> int | None:
        i = self.dates.searchsorted(ts)
        return int(i) if i < len(self.dates) else None

    def _reaction_pos(self, row) -> int | None:
        """開示日時から、株価が反応する最初の営業日の位置を返す。"""
        d = row["DisclosedDate"]
        t = str(row.get("DisclosedTime") or "15:30")
        p = self._next_pos_on_or_after(d)
        if p is None:
            return None
        # 場中(15時前)開示なら当日、引け後開示なら翌営業日に反応
        if self.dates[p] == d and t >= "15:00":
            return p + 1 if p + 1 < len(self.dates) else None
        return p

    def _sq_week_dates(self) -> set:
        out = set()
        years = range(self.dates[0].year, self.dates[-1].year + 1)
        for y in years:
            for m in (3, 6, 9, 12):
                fridays = pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D")
                fridays = [d for d in fridays if d.month == m and d.weekday() == 4]
                sq = fridays[1]  # 第2金曜
                monday = sq - pd.Timedelta(days=4)
                for d in pd.date_range(monday, sq):
                    out.add(d.normalize())
        return out

    def size_mult(self, date: pd.Timestamp) -> float:
        m = 1.0
        if date.normalize() in self.sq_dates:
            m *= self.cfg.sq_size_mult
        if bool(self.adx_low.get(date, False)):
            m *= self.cfg.adx_low_size_mult
        return m

    def margin_blocked(self, code: str, date: pd.Timestamp) -> bool:
        """公表ラグを考慮した直近の信用倍率が閾値超なら新規買い禁止。"""
        if self.margin is None or code not in self.margin:
            return False
        p = self.date_pos[date]
        cutoff_p = p - self.cfg.margin_publish_lag
        if cutoff_p < 0:
            return False
        s = self.margin[code]
        s = s[s.index <= self.dates[cutoff_p]]
        if s.empty:
            return False
        return float(s.iloc[-1]) > self.cfg.margin_ratio_max

    def bars_to_next_earnings(self, code: str, date: pd.Timestamp) -> int | None:
        """次の決算反応日までの営業日数。決算データが無ければNone。"""
        pos_list = self.earnings_pos.get(code)
        if not pos_list:
            return None
        p = self.date_pos[date]
        for ep in pos_list:
            if ep > p:
                return ep - p
        return None

    # ---------- ② 順張りスイング ----------
    def momentum_candidates(self, date: pd.Timestamp) -> list[dict]:
        cfg = self.cfg
        if not bool(self.trend_regime.get(date, False)):
            return []
        out = []
        for code, d in self.ind.items():
            r = d.loc[date]
            if r[["Close", "sma10", "sma20", "hh20", "low3"]].isna().any():
                continue
            rank = self.rs_rank.at[date, code] if code in self.rs_rank.columns else np.nan
            if not (rank >= 1.0 - cfg.mom_rs_top_pct):
                continue
            pullback = 1.0 - r["Close"] / r["hh20"]
            if not (cfg.mom_pullback_min <= pullback <= cfg.mom_pullback_max):
                continue
            lo_d, hi_d = cfg.mom_days_since_high
            if not (lo_d <= r["days_since_hh20"] <= hi_d):
                continue
            touched = (r["Low"] <= r["sma10"]) or (r["Low"] <= r["sma20"] * 1.005)
            if not touched or r["Close"] < r["sma20"] * 0.97:
                continue
            prev_close = d["Close"].shift(1).loc[date]
            if not (r["Close"] > r["Open"] and r["Close"] > prev_close):
                continue
            stop = r["low3"] * 0.99
            risk = 1.0 - stop / r["Close"]
            if risk <= 0 or risk > cfg.mom_stop_max:
                continue
            out.append(
                {
                    "code": code, "module": "momentum", "stop": float(stop),
                    "tp_r": cfg.mom_tp_r_multiple, "score": float(rank),
                }
            )
        return out

    # ---------- ③ 逆張りスイング ----------
    def meanrev_candidates(self, date: pd.Timestamp) -> list[dict]:
        cfg = self.cfg
        if bool(self.trend_regime.get(date, False)):
            return []
        out = []
        for code, d in self.ind.items():
            r = d.loc[date]
            if r[["Close", "sma25", "dev25", "rsi2", "vol_sma20"]].isna().any():
                continue
            if r["dev25"] > cfg.mr_dev25_threshold:
                continue
            if r["rsi2"] >= cfg.mr_rsi2_threshold:
                continue
            if r["Volume"] < cfg.mr_volume_mult * r["vol_sma20"]:
                continue
            out.append(
                {
                    "code": code, "module": "meanrev",
                    "stop_pct": cfg.mr_stop, "sma25": float(r["sma25"]),
                    "score": float(-r["dev25"]),
                }
            )
        return out

    # ---------- ④ 決算PEAD ----------
    def _precompute_pead(self) -> dict[pd.Timestamp, list[dict]]:
        cfg = self.cfg
        signals: dict[pd.Timestamp, list[dict]] = {}
        for code, pos_list in self.earnings_pos.items():
            d = self.ind.get(code)
            if d is None:
                continue
            close, open_, high, low = d["Close"], d["Open"], d["High"], d["Low"]
            for rp in pos_list:
                if rp < 1 or rp + cfg.pead_entry_to + 1 >= len(self.dates):
                    continue
                rd = self.dates[rp]
                prev_c = close.iloc[rp - 1]
                o, h, l, c = open_.iloc[rp], high.iloc[rp], low.iloc[rp], close.iloc[rp]
                if np.isnan([prev_c, o, h, l, c]).any() or h <= l:
                    continue
                gap = o / prev_c - 1.0
                if gap < cfg.pead_gap_min:
                    continue
                if (c - l) / (h - l) < cfg.pead_close_range_pct:
                    continue
                gap_mid = (prev_c + o) / 2.0
                # 反応日から2〜5日目に押し目が入ったらシグナル
                for k in range(cfg.pead_entry_from, cfg.pead_entry_to + 1):
                    tp = rp + k
                    if tp >= len(self.dates):
                        break
                    t = self.dates[tp]
                    lo_t, c_t = low.iloc[tp], close.iloc[tp]
                    if np.isnan([lo_t, c_t]).any():
                        continue
                    if lo_t <= c * (1 - cfg.pead_pullback) and c_t >= gap_mid:
                        signals.setdefault(t, []).append(
                            {
                                "code": code, "module": "pead",
                                "stop": float(prev_c), "score": float(gap),
                            }
                        )
                        break  # 1イベント1シグナル
        return signals

    def pead_candidates(self, date: pd.Timestamp) -> list[dict]:
        return list(self.pead_signals.get(date, []))
