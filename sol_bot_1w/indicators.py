"""
indicators.py — Technische Indikatoren für den 1W-Bot
Erwartet wöchentliche OHLCV-Daten von Binance.
"""

import numpy as np
import pandas as pd

SWING_N = 2   # Wochen auf jeder Seite zur Swing-Bestätigung


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


def detect_swings(df: pd.DataFrame, n: int = SWING_N) -> pd.DataFrame:
    """
    Findet lokale Swing-Hochs und Swing-Tiefs.
    Look-Ahead-Schutz: shift(n) macht Swing bei Bar i erst ab Bar i+n sichtbar.
    d.h. bei Bar T sind Swings aus [0, T-n] sichtbar (vollständig bestätigt).
    """
    highs = df["High"]
    lows  = df["Low"]

    sh_mask = pd.Series(False, index=df.index)
    sl_mask = pd.Series(False, index=df.index)

    for i in range(n, len(df) - n):
        if highs.iloc[i] == highs.iloc[i - n : i + n + 1].max():
            sh_mask.iloc[i] = True
        if lows.iloc[i] == lows.iloc[i - n : i + n + 1].min():
            sl_mask.iloc[i] = True

    # Swing bei Bar i wird nach shift(n) bei Bar i+n sichtbar
    df["Swing_High"] = highs.where(sh_mask).shift(n)
    df["Swing_Low"]  = lows.where(sl_mask).shift(n)
    return df


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

    # ── ATR (14, Wilder) ─────────────────────────────────────────────────────
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = _wilder_smooth(tr, 14)

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    delta  = c.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = _wilder_smooth(gain, 14)
    avg_l  = _wilder_smooth(loss, 14)
    rs     = avg_g / avg_l.replace(0, np.nan)
    df["RSI14"] = 100 - 100 / (1 + rs)

    # ── DI+ / DI- / ADX (14) ────────────────────────────────────────────────
    df["DI_plus"], df["DI_minus"], df["ADX"] = _calc_adx(df, 14)

    # ── Composite-Signale ─────────────────────────────────────────────────────
    df["EMA_bull"] = df["EMA10"] > df["EMA21"]
    df["EMA_bear"] = df["EMA10"] < df["EMA21"]

    # ── Volatilität ───────────────────────────────────────────────────────────
    df["Vola"] = c.pct_change().mul(100).rolling(10).std()

    # ── Vorheriger Close & EMA-Vorwerte (für Crossover-Erkennung) ───────────────
    df["Close_prev"] = c.shift(1)
    df["EMA10_prev"] = df["EMA10"].shift(1)
    df["EMA21_prev"] = df["EMA21"].shift(1)

    # ── Swing Hochs / Tiefs ───────────────────────────────────────────────────
    df = detect_swings(df, n=SWING_N)

    return df
