"""
indicators.py — Technische Indikatoren für den 15min-Bot
Erwartet 15min-OHLCV-Daten von Binance.
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


def compute_daily_macro(df_raw: pd.DataFrame) -> pd.Series:
    """
    EMA21 > EMA50 auf Tagesbasis aus 15min-Daten.
    shift(1) verhindert Look-Ahead.
    """
    close_d = df_raw["Close"].resample("1D").last().dropna()
    ema21   = close_d.ewm(span=21, adjust=False).mean()
    ema50   = close_d.ewm(span=50, adjust=False).mean()
    bull    = (ema21 > ema50).rename("Daily_macro_bull")
    return bull.shift(1).reindex(df_raw.index, method="ffill").fillna(False)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = df["Close"]
    h  = df["High"]
    lo = df["Low"]
    v  = df["Volume"]

    # ── EMAs ────────────────────────────────────────────────────────────────
    df["EMA5"]  = c.ewm(span=5,  adjust=False).mean()
    df["EMA13"] = c.ewm(span=13, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()

    # ── ATR (14 × 15min = 3.5h) ─────────────────────────────────────────────
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"]      = _wilder_smooth(tr, 14)
    df["ATR_MA20"] = df["ATR"].rolling(20).mean()
    df["ATR_spike"] = df["ATR"] / df["ATR_MA20"].replace(0, np.nan)

    # ── RSI (5 und 14) ───────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_g5 = _wilder_smooth(gain, 5)
    avg_l5 = _wilder_smooth(loss, 5)
    rs5    = avg_g5 / avg_l5.replace(0, np.nan)
    df["RSI5"] = 100 - 100 / (1 + rs5)

    avg_g14 = _wilder_smooth(gain, 14)
    avg_l14 = _wilder_smooth(loss, 14)
    rs14    = avg_g14 / avg_l14.replace(0, np.nan)
    df["RSI14"] = 100 - 100 / (1 + rs14)

    # ── DI+ / DI- / ADX (14) ────────────────────────────────────────────────
    df["DI_plus"], df["DI_minus"], df["ADX"] = _calc_adx(df, 14)

    # ── VWAP (rollend 96 Kerzen = 1 Tag bei 15min) ───────────────────────────
    typical      = (h + lo + c) / 3
    df["VWAP96"] = (typical * v).rolling(96).sum() / v.rolling(96).sum()

    # ── Bollinger Bands (20, ±2σ) ─────────────────────────────────────────────
    df["BB_mid"]   = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"].replace(0, np.nan)
    # Reversal-Confirmation: Preis war unter/über Band, schließt jetzt zurück
    df["BB_lower_recovery"] = (c.shift(1) <= df["BB_lower"].shift(1)) & (c > df["BB_lower"])
    df["BB_upper_recovery"] = (c.shift(1) >= df["BB_upper"].shift(1)) & (c < df["BB_upper"])

    # ── Volatilität ──────────────────────────────────────────────────────────
    pct          = c.pct_change() * 100
    df["Vola"]   = pct.rolling(20).std()
    df["Vola_ratio"] = pct.rolling(5).std() / df["Vola"].replace(0, np.nan)

    # ── Daily Macro Bull (EMA21 > EMA50 auf Tagesbasis) ─────────────────────
    df["Daily_macro_bull"] = compute_daily_macro(df)

    # ── Composite-Signale ─────────────────────────────────────────────────────
    df["DI_bull"]    = df["DI_plus"] > df["DI_minus"]
    df["DI_bear"]    = df["DI_minus"] > df["DI_plus"]
    df["EMA_bull"]   = (df["EMA5"] > df["EMA13"]) & (df["EMA13"] > df["EMA21"])
    df["EMA_bear"]   = (df["EMA5"] < df["EMA13"]) & (df["EMA13"] < df["EMA21"])
    df["EMA21_slope"] = df["EMA21"] > df["EMA21"].shift(8)  # 8 × 15min = 2h Aufwärtsslope

    # ── Donchian-Breakout (20 Kerzen, shift(1) verhindert Look-Ahead) ───────
    df["Donch_high"] = df["High"].shift(1).rolling(20).max()
    df["Donch_low"]  = df["Low"].shift(1).rolling(20).min()
    df["Donch_bull_break"] = df["Close"] > df["Donch_high"]
    df["Donch_bear_break"] = df["Close"] < df["Donch_low"]

    return df
