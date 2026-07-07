"""テクニカル指標の計算。すべて pandas Series/DataFrame ベース。"""
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 2) -> pd.Series:
    """Wilder方式のRSI。"""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(close.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_ = true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def days_since_rolling_high(high: pd.Series, window: int) -> pd.Series:
    """直近window日の最高値から何日経過したか。当日が高値なら0。"""
    def _f(x):
        return len(x) - 1 - int(np.argmax(x))
    return high.rolling(window).apply(_f, raw=True)


def add_stock_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """個別銘柄のOHLCV DataFrame に戦略で使う指標列を追加する。

    必要列: Open, High, Low, Close, Volume
    """
    out = df.copy()
    c, h, l, v = out["Close"], out["High"], out["Low"], out["Volume"]
    out["sma10"] = sma(c, 10)
    out["sma20"] = sma(c, 20)
    out["sma25"] = sma(c, 25)
    trail_col = f"sma{cfg.mom_trail_sma}"
    if trail_col not in out.columns:
        out[trail_col] = sma(c, cfg.mom_trail_sma)
    out["dev25"] = c / out["sma25"] - 1.0
    out["rsi2"] = rsi(c, 2)
    out["vol_sma20"] = sma(v, 20)
    out["hh20"] = h.rolling(20).max()
    out["days_since_hh20"] = days_since_rolling_high(h, 20)
    out["ret_rs"] = c.pct_change(cfg.mom_rs_lookback)
    out["low3"] = l.rolling(3).min()
    return out
