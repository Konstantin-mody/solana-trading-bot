"""
indicators.py — Technische Indikatoren für den 1d-Bot
Erwartet Tages-OHLCV-Daten von Binance.
"""

import numpy as np
import pandas as pd


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    result = np.full(len(series), np.nan)
    valid = series.dropna()
    if len(valid) < period:
        return pd.Series(result, index=series.index)
    first_idx = series.index.get_loc(valid.index[period - 1])
    result[first_idx] = valid.iloc[:period].mean()
    for i in range(first_idx + 1, len(series)):
        if not np.isnan(result[i - 1]) and not np.isnan(series.iloc[i]):
            result[i] = result[i - 1] * (period - 1) / period + series.iloc[i] / period
    return pd.Series(result, index=series.index)


def _calc_adx(df: pd.DataFrame, period=14):
    high = df["High"]
    low  = df["Low"]
    atr  = df["ATR"]

    up_move   = high.diff()
    down_move = -low.diff()

    dm_plus  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    dm_plus_s  = _wilder_smooth(pd.Series(dm_plus,  index=df.index), period)
    dm_minus_s = _wilder_smooth(pd.Series(dm_minus, index=df.index), period)

    di_plus  = 100 * dm_plus_s  / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus_s / atr.replace(0, np.nan)

    dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = _wilder_smooth(dx, period)

    return di_plus, di_minus, adx


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = df["Close"]
    h  = df["High"]
    lo = df["Low"]

    # ── Moving Averages ──────────────────────────────────────────────────────
    df["MA50"]  = c.rolling(50).mean()
    df["MA200"] = c.rolling(200).mean()
    df["EMA10"] = c.ewm(span=10, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()

    # ── ATR (14, Wilder) ─────────────────────────────────────────────────────
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"]       = _wilder_smooth(tr, 14)
    df["ATR_MA20"]  = df["ATR"].rolling(20).mean()
    df["ATR_spike"] = df["ATR"] / df["ATR_MA20"].replace(0, np.nan)

    # ── RSI (3) ──────────────────────────────────────────────────────────────
    delta  = c.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g3 = _wilder_smooth(gain, 3)
    avg_l3 = _wilder_smooth(loss, 3)
    rs3    = avg_g3 / avg_l3.replace(0, np.nan)
    df["RSI3"] = 100 - 100 / (1 + rs3)

    # ── DI+ / DI- / ADX (14) ────────────────────────────────────────────────
    df["DI_plus"], df["DI_minus"], df["ADX"] = _calc_adx(df, 14)

    # ── OBV (Volumen-Trendbestätigung) ───────────────────────────────────────
    signs         = np.sign(c.diff().fillna(0))
    df["OBV"]     = (signs * df["Volume"]).cumsum()
    df["OBV_MA20"]= df["OBV"].rolling(20).mean()
    df["OBV_bull"]= df["OBV"] > df["OBV_MA20"]

    # ── Volatilität (für Slippage-Modell in backtest.py) ─────────────────────
    df["Vola"] = c.pct_change().mul(100).rolling(20).std()

    # ── Composite-Signale ─────────────────────────────────────────────────────
    df["EMA_bull"]    = (df["EMA10"] > df["EMA21"]) & (df["EMA21"] > df["EMA50"])
    df["EMA_bear"]    = (df["EMA10"] < df["EMA21"]) & (df["EMA21"] < df["EMA50"])
    df["DI_bull"]     = df["DI_plus"] > df["DI_minus"]
    df["DI_bear"]     = df["DI_minus"] > df["DI_plus"]
    df["EMA21_slope"] = df["EMA21"] > df["EMA21"].shift(3)

    # ── Fibonacci-Retracement-Level (Major Swing, 180d) ─────────────────────
    # shift(1) = kein Look-Ahead: Bar t sieht nur Daten bis t-1.
    # 180 Tage ≈ 6 Monate → fängt strukturelle Impulse, keine Mini-Pivots.
    SWING_BARS = 180
    swing_high = h.rolling(SWING_BARS).max().shift(1)
    swing_low  = lo.rolling(SWING_BARS).min().shift(1)

    fib_range = (swing_high - swing_low).clip(lower=0)

    # Mindest-Range: Swing muss ≥15% des Swing-Highs betragen
    valid_range = fib_range >= swing_high * 0.15

    df["Fib_382"]    = np.where(valid_range, swing_high - 0.382 * fib_range, np.nan)
    df["Fib_500"]    = np.where(valid_range, swing_high - 0.500 * fib_range, np.nan)
    df["Fib_618"]    = np.where(valid_range, swing_high - 0.618 * fib_range, np.nan)
    df["Swing_high"] = swing_high
    df["Swing_low"]  = swing_low

    # at_fib_level: Close innerhalb ±3% eines Level UND unterhalb des Swing-Highs
    # (Bedingung "wir sind im Rücksetzer, nicht am neuen ATH")
    TOL = 0.03
    in_retracement = c < swing_high * 0.98
    at_382 = (c - df["Fib_382"]).abs() / df["Fib_382"].replace(0, np.nan) <= TOL
    at_500 = (c - df["Fib_500"]).abs() / df["Fib_500"].replace(0, np.nan) <= TOL
    at_618 = (c - df["Fib_618"]).abs() / df["Fib_618"].replace(0, np.nan) <= TOL
    df["at_fib_level"] = (at_382 | at_500 | at_618) & in_retracement

    return df
