"""データセット構築。

J-Quants API から以下を組み立てる:
- prices: dict[code -> OHLCV DataFrame(調整後、DatetimeIndex)]
- index_df: 地合い判定用の指数(TOPIX。取れないプランならユニバース等加重指数で代替)
- earnings: dict[code -> 決算開示日リスト(DisclosedDate, DisclosedTime)]
- margin_ratio: dict[code -> 信用倍率のSeries(週次)] ※Standardプラン以上。無ければNone
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from config import Config
from jquants_client import JQuantsClient, PlanNotAvailableError


def _adjusted_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """株価四本値(V2: /equities/bars/daily)から調整後OHLCVを取り出す。

    V2のフィールド名は O/H/L/C/Vo(未調整)と AdjO/AdjH/AdjL/AdjC/AdjVo(調整後)。
    """
    v2_map = {"Open": "AdjO", "High": "AdjH", "Low": "AdjL", "Close": "AdjC", "Volume": "AdjVo"}
    raw_map = {"Open": "O", "High": "H", "Low": "L", "Close": "C", "Volume": "Vo"}
    cols = {}
    for name in ["Open", "High", "Low", "Close", "Volume"]:
        src = v2_map[name] if v2_map[name] in df.columns else raw_map[name]
        cols[name] = pd.to_numeric(df[src], errors="coerce")
    out = pd.DataFrame(cols)
    out.index = pd.to_datetime(df["Date"])
    out = out.dropna(subset=["Close"])
    return out[out["Volume"] > 0]


def _business_days(cal: pd.DataFrame) -> list[str]:
    """取引カレンダー(V2列: Date, HolDiv)から営業日のみ抽出。HolDiv=0 が非営業日。"""
    return cal[cal["HolDiv"].astype(str) != "0"]["Date"].tolist()


def _recent_business_days(client: JQuantsClient, end: str, n: int) -> list[str]:
    start = (pd.Timestamp(end) - pd.Timedelta(days=int(n * 2.2))).strftime("%Y-%m-%d")
    days = _business_days(client.trading_calendar(start, end))
    return days[-n:]


def resolve_end_date(client: JQuantsClient, cfg: Config) -> str:
    """endが未指定なら、データが実際に存在する最新日を探す(無料プランの12週遅延対応)。"""
    if cfg.end:
        return cfg.end
    today = dt.date.today().strftime("%Y-%m-%d")
    probe_start = (dt.date.today() - dt.timedelta(days=130)).strftime("%Y-%m-%d")
    days = _business_days(client.trading_calendar(probe_start, today))
    for d in reversed(days):
        if not client.daily_quotes_by_date(d).empty:
            return d
    raise RuntimeError("直近130日に株価データが見つかりません。プランと認証情報を確認してください")


def build_universe(client: JQuantsClient, cfg: Config, end: str) -> list[str]:
    """プライム市場のうち、直近の平均売買代金が大きい銘柄コードを返す。

    注意: 「現在の」流動性上位で過去を遡るため、生存バイアスが混入する。
    厳密にやる場合は各時点のユニバースを再構成する必要がある(READMEに記載)。
    """
    listed = client.listed_info()
    prime = set(listed[listed["Mkt"] == "0111"]["Code"])
    days = _recent_business_days(client, end, cfg.liquidity_lookback_days)
    frames = []
    for d in days:
        q = client.daily_quotes_by_date(d)
        if not q.empty:
            frames.append(q[["Code", "Va"]])
    allq = pd.concat(frames)
    allq = allq[allq["Code"].isin(prime)]
    allq["Va"] = pd.to_numeric(allq["Va"], errors="coerce")
    # 平均ではなく中央値を使い、仕手化などの一時的な売買代金スパイクで
    # 低流動性銘柄がランクインするのを防ぐ
    turnover = allq.groupby("Code")["Va"].median().sort_values(ascending=False)
    return turnover.head(cfg.universe_size).index.tolist()


def load_prices(client: JQuantsClient, codes: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    prices = {}
    for i, code in enumerate(codes):
        df = client.daily_quotes_by_code(code, start, end)
        if df.empty:
            continue
        prices[code] = _adjusted_ohlcv(df)
        if (i + 1) % 25 == 0:
            print(f"  株価取得 {i + 1}/{len(codes)}")
    return prices


def load_index(client: JQuantsClient, prices: dict, start: str, end: str) -> pd.DataFrame:
    """TOPIXを試し、ダメならユニバース等加重指数で代替。"""
    try:
        df = client.topix(start, end)
        if not df.empty:
            # 注意: pd.DataFrame(Series辞書, index=...) は新indexへの「整列」に
            # なり全行NaNになるため、構築後に index を付け替える
            out = pd.DataFrame(
                {
                    "Open": pd.to_numeric(df["O"], errors="coerce"),
                    "High": pd.to_numeric(df["H"], errors="coerce"),
                    "Low": pd.to_numeric(df["L"], errors="coerce"),
                    "Close": pd.to_numeric(df["C"], errors="coerce"),
                }
            )
            out.index = pd.to_datetime(df["Date"])
            print("  地合い判定: TOPIX を使用")
            return out
    except PlanNotAvailableError:
        pass
    print("  地合い判定: TOPIX が取得できないため、ユニバース等加重指数で代替")
    rets = pd.DataFrame({c: p["Close"].pct_change() for c, p in prices.items()})
    eq = (1 + rets.mean(axis=1).fillna(0)).cumprod() * 1000
    return pd.DataFrame({"Open": eq, "High": eq, "Low": eq, "Close": eq})


def load_earnings(client: JQuantsClient, codes: list[str]) -> dict[str, pd.DataFrame]:
    """決算開示日(DisclosedDate/DisclosedTime)を銘柄ごとに返す。"""
    earnings = {}
    for i, code in enumerate(codes):
        try:
            df = client.statements(code)
        except PlanNotAvailableError:
            print("  決算データが取得できないプランのため、PEADと決算またぎ回避を無効化")
            return {}
        if df.empty or "DiscDate" not in df.columns:
            continue
        # V2のフィールド名を内部スキーマ名に正規化
        e = df[["DiscDate", "DiscTime"]].rename(
            columns={"DiscDate": "DisclosedDate", "DiscTime": "DisclosedTime"}
        ).copy()
        e["DisclosedDate"] = pd.to_datetime(e["DisclosedDate"])
        earnings[code] = e.dropna(subset=["DisclosedDate"]).sort_values("DisclosedDate")
        if (i + 1) % 50 == 0:
            print(f"  決算取得 {i + 1}/{len(codes)}")
    return earnings


def load_margin_ratio(client: JQuantsClient, codes: list[str]) -> dict[str, pd.Series] | None:
    """信用倍率(買い残/売り残)の週次Series。Standardプラン未満ならNone。"""
    out = {}
    for i, code in enumerate(codes):
        try:
            df = client.weekly_margin_interest(code)
        except PlanNotAvailableError:
            print("  信用残データが取得できないプランのため、信用需給フィルターを無効化")
            return None
        if df.empty:
            continue
        long_v = pd.to_numeric(df["LongVol"], errors="coerce")
        short_v = pd.to_numeric(df["ShrtVol"], errors="coerce").replace(0.0, np.nan)
        ratio = (long_v / short_v).fillna(99.0)
        ratio.index = pd.to_datetime(df["Date"])
        out[code] = ratio.sort_index()
        if (i + 1) % 50 == 0:
            print(f"  信用残取得 {i + 1}/{len(codes)}")
    return out


def load_scale_categories(client: JQuantsClient, codes: list[str]) -> dict[str, str]:
    """TOPIX規模区分(Core30/Large70/Mid400/Small)を銘柄ごとに返す。"""
    listed = client.listed_info()
    if "ScaleCat" not in listed.columns:
        return {}
    m = listed.set_index("Code")["ScaleCat"].to_dict()
    return {c: str(m.get(c, "")) for c in codes}


def load_margin_alert(client: JQuantsClient, codes: list[str]) -> dict[str, list] | None:
    """日々公表(信用規制・注意)銘柄の掲載日リスト。プラン外ならNone。"""
    out = {}
    for i, code in enumerate(codes):
        try:
            df = client.margin_alert(code)
        except PlanNotAvailableError:
            print("  日々公表データが取得できないプランのため、信用規制銘柄フィルターを無効化")
            return None
        if df.empty or "PubDate" not in df.columns:
            continue
        out[code] = sorted(pd.to_datetime(df["PubDate"]).dropna().tolist())
        if (i + 1) % 50 == 0:
            print(f"  日々公表取得 {i + 1}/{len(codes)}")
    return out


def build_dataset(cfg: Config) -> dict:
    client = JQuantsClient()
    end = resolve_end_date(client, cfg)
    fetch_start = (pd.Timestamp(cfg.start) - pd.Timedelta(days=int(cfg.warmup_days * 1.6))).strftime("%Y-%m-%d")
    print(f"期間: {fetch_start} 〜 {end}(指標ウォームアップ込み、評価は {cfg.start} から)")

    print("ユニバース選定中...")
    codes = build_universe(client, cfg, end)
    print(f"  プライム売買代金上位 {len(codes)} 銘柄")

    print("株価取得中...")
    prices = load_prices(client, codes, fetch_start, end)

    print("指数取得中...")
    index_df = load_index(client, prices, fetch_start, end)

    print("決算開示日取得中...")
    earnings = load_earnings(client, list(prices.keys()))

    print("信用残取得中...")
    margin = load_margin_ratio(client, list(prices.keys()))

    print("規模区分・日々公表銘柄取得中...")
    scale = load_scale_categories(client, list(prices.keys()))
    alert = load_margin_alert(client, list(prices.keys())) if cfg.exclude_margin_alert else None

    return {
        "prices": prices,
        "index": index_df,
        "earnings": earnings,
        "margin": margin,
        "scale": scale,
        "margin_alert": alert,
        "eval_start": pd.Timestamp(cfg.start),
    }


# ---------------- セルフテスト用の合成データ ----------------
def synthetic_dataset(cfg: Config, n_codes: int = 40, n_days: int = 500, seed: int = 7) -> dict:
    """APIなしでエンジン全体を検証するための乱数データ。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-04", periods=n_days)
    idx_ret = rng.normal(0.0004, 0.009, n_days)
    trend = np.sin(np.arange(n_days) / 60) * 0.0012
    idx = 2000 * np.exp(np.cumsum(idx_ret + trend))
    index_df = pd.DataFrame({"Open": idx, "High": idx * 1.004, "Low": idx * 0.996, "Close": idx}, index=dates)

    prices, earnings = {}, {}
    for i in range(n_codes):
        code = f"{1301 + i * 7}0"
        beta = rng.uniform(0.5, 1.5)
        alpha = rng.normal(0, 0.0006)
        ret = alpha + beta * (idx_ret + trend) + rng.normal(0, 0.018, n_days)
        volume = rng.integers(100_000, 2_000_000, n_days).astype(float)
        # たまに決算ギャップを混ぜる
        e_days = list(range(int(rng.integers(30, 70)), n_days - 25, 63))
        for d in e_days:
            ret[d] += rng.choice([0.08, -0.06, 0.02], p=[0.4, 0.3, 0.3])
            volume[d] *= 4
        # たまに急落(セリクラ)を混ぜて逆張り条件を作る
        for d in rng.choice(np.arange(100, n_days - 15), size=2, replace=False):
            ret[d:d + 3] -= 0.06
            volume[d:d + 3] *= 5
            ret[d + 3:d + 8] += 0.015
        close = 1500 * np.exp(np.cumsum(ret))
        spread = np.abs(rng.normal(0.008, 0.004, n_days))
        open_ = close * (1 + rng.normal(0, 0.004, n_days))
        body_hi = np.maximum(open_, close)
        body_lo = np.minimum(open_, close)
        df = pd.DataFrame(
            {
                "Open": open_,
                "High": body_hi * (1 + 0.25 * spread),
                "Low": body_lo * (1 - spread),
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )
        prices[code] = df
        earnings[code] = pd.DataFrame(
            {
                "DisclosedDate": [dates[d - 1] for d in e_days],
                "DisclosedTime": ["15:30:00"] * len(e_days),
            }
        )
    return {
        "prices": prices,
        "index": index_df,
        "earnings": earnings,
        "margin": None,
        "scale": {},
        "margin_alert": None,
        "eval_start": dates[cfg.warmup_days],
    }
