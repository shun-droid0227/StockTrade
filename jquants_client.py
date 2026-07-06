"""J-Quants API V2 クライアント。

V2仕様(https://jpx-jquants.com/ja/spec/):
- ベースURL: https://api.jquants.com/v2
- 認証: ダッシュボードから発行した APIキーを x-api-key ヘッダーで送信
  (V1のリフレッシュトークン/IDトークン方式は廃止)
- レスポンス: データ配列は一律 "data" キー、続きは "pagination_key"

このモジュールの方針:
- 全レスポンスを data_cache/ にparquetでキャッシュし、再実行時はAPIを叩かない
- 契約プラン外のエンドポイント(403)は PlanNotAvailableError を投げ、
  呼び出し側で機能を縮退させる
"""
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.jquants.com/v2"
CACHE_DIR = Path(__file__).resolve().parent / "data_cache"


class PlanNotAvailableError(Exception):
    """契約プランで利用できないエンドポイント。"""


class JQuantsClient:
    def __init__(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self._api_key = os.getenv("JQUANTS_API_KEY", "").strip()
        CACHE_DIR.mkdir(exist_ok=True)

    def _headers(self) -> dict:
        if not self._api_key:
            raise RuntimeError(
                ".env に JQUANTS_API_KEY を設定してください"
                "(J-Quantsダッシュボード https://jpx-jquants.com/ で発行。"
                ".env.example を .env にコピーして記入)"
            )
        return {"x-api-key": self._api_key}

    # ---------------- 低レベルGET(ページネーション対応) ----------------
    def _get(self, path: str, params: dict) -> pd.DataFrame:
        params = {k: v for k, v in params.items() if v}
        rows = []
        while True:
            r = None
            for attempt in range(6):
                r = requests.get(
                    f"{BASE}{path}",
                    params=params,
                    headers=self._headers(),
                    timeout=60,
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                break
            if r.status_code == 401:
                raise RuntimeError(
                    "認証エラー(401): JQUANTS_API_KEY が正しいか確認してください"
                )
            if r.status_code == 403:
                raise PlanNotAvailableError(f"{path} は現在の契約プランでは利用できません")
            r.raise_for_status()
            payload = r.json()
            batch = payload.get("data", [])
            if isinstance(batch, list):
                rows.extend(batch)
            pk = payload.get("pagination_key")
            if not pk:
                break
            params["pagination_key"] = pk
        return pd.DataFrame(rows)

    @staticmethod
    def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        """J-Quantsは欠損値を空文字などで返すことがあり、数値と文字列が
        混在したobject列はparquetに書けない。object列は文字列に統一して
        保存し、数値化は読み込み側(pd.to_numeric)で行う。"""
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].map(lambda x: None if pd.isna(x) else str(x))
        return out

    def _cached(self, name: str, path: str, params: dict) -> pd.DataFrame:
        key = hashlib.md5(json.dumps([path, params], sort_keys=True).encode()).hexdigest()[:12]
        f = CACHE_DIR / f"v2_{name}_{key}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        df = self._sanitize_for_parquet(self._get(path, params))
        if not df.empty:
            df.to_parquet(f, index=False)
        return df

    # ---------------- 公開メソッド ----------------
    def trading_calendar(self, from_: str, to: str) -> pd.DataFrame:
        """取引カレンダー。列: Date, HolDiv"""
        return self._cached("calendar", "/markets/calendar", {"from": from_, "to": to})

    def listed_info(self) -> pd.DataFrame:
        """上場銘柄一覧。列: Code, CoName, Mkt, S17, S33 など"""
        return self._cached("master", "/equities/master", {})

    def daily_quotes_by_date(self, date: str) -> pd.DataFrame:
        """全銘柄の株価四本値(1日分)。列: Date, Code, O/H/L/C, Vo, Va, AdjO/AdjH/AdjL/AdjC/AdjVo など"""
        return self._cached(f"bars_d{date}", "/equities/bars/daily", {"date": date})

    def daily_quotes_by_code(self, code: str, from_: str, to: str) -> pd.DataFrame:
        """1銘柄の株価四本値(期間指定)。"""
        return self._cached(
            f"bars_c{code}", "/equities/bars/daily",
            {"code": code, "from": from_, "to": to},
        )

    def topix(self, from_: str, to: str) -> pd.DataFrame:
        """TOPIX四本値。列: Date, O, H, L, C"""
        return self._cached("topix", "/indices/bars/daily/topix", {"from": from_, "to": to})

    def statements(self, code: str) -> pd.DataFrame:
        """財務情報サマリ。列: DiscDate, DiscTime, Code など"""
        return self._cached(f"fin_{code}", "/fins/summary", {"code": code})

    def weekly_margin_interest(self, code: str) -> pd.DataFrame:
        """信用取引週末残高。列: Date, Code, ShrtVol, LongVol など"""
        return self._cached(f"margin_{code}", "/markets/margin-interest", {"code": code})

    def margin_alert(self, code: str) -> pd.DataFrame:
        """日々公表信用取引残高(信用規制・注意銘柄)。列: PubDate, Code, PubReason など"""
        return self._cached(f"alert_{code}", "/markets/margin-alert", {"code": code})

    def earnings_calendar(self) -> pd.DataFrame:
        """決算発表予定日。列: Date, Code, CoName, FY, FQ など"""
        return self._cached("earn_cal", "/equities/earnings-calendar", {})
