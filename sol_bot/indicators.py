"""
indicators.py — Alle technischen Indikatoren für den SOL Bot
Gibt einen DataFrame mit allen Spalten zurück (input: OHLCV)
"""

import numpy as np
import pandas as pd


# ─── Hilfsfunktionen ────────────────────────────────────────────────────────

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder-Smoothing (wie in RSI/ATR verwendet)."""
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


# ─── Hauptfunktion ──────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erwartet DataFrame mit Spalten: Open, High, Low, Close, Volume
    Gibt denselben DataFrame mit allen Indikatoren zurück.
    """
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    lo = df["Low"]
    v = df["Volume"]

    # ── Trend: Moving Averages ───────────────────────────────────────────────
    df["MA20"]  = c.rolling(20).mean()
    df["MA50"]  = c.rolling(50).mean()
    df["MA200"] = c.rolling(200).mean()
    df["MA500"] = c.rolling(500).mean()   # ~21-Tage-Trend auf 1h-Daten
    df["EMA8"]  = c.ewm(span=8,  adjust=False).mean()
    df["EMA13"] = c.ewm(span=13, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()

    # ── MACD (12/26/9) ───────────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD_line"]   = ema12 - ema26
    df["MACD_signal"] = df["MACD_line"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD_line"] - df["MACD_signal"]

    # ── ROC (10 Perioden) ────────────────────────────────────────────────────
    df["ROC"] = (c / c.shift(10) - 1) * 100

    # ── ATR (14, Wilder) ─────────────────────────────────────────────────────
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = _wilder_smooth(tr, 14)
    df["ATR_MA50"]  = df["ATR"].rolling(50).mean()
    df["ATR_spike"] = df["ATR"] / df["ATR_MA50"].replace(0, np.nan)

    # ── Parabolic SAR (AF=0.02, max=0.2) ────────────────────────────────────
    df["SAR"] = _calc_sar(df, af_start=0.02, af_max=0.2)

    # ── Supertrend (2.5×ATR) ─────────────────────────────────────────────────
    df["Supertrend"], df["ST_bull"] = _calc_supertrend(df, multiplier=2.5)

    # ── Ichimoku (Tenkan 9, Kijun 26) ────────────────────────────────────────
    df["Tenkan"]   = (h.rolling(9).max()  + lo.rolling(9).min())  / 2
    df["Kijun"]    = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    df["Senkou_A"] = ((df["Tenkan"] + df["Kijun"]) / 2).shift(26)
    df["Senkou_B"] = ((h.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)

    # ── RSI (14, Wilder) ─────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = _wilder_smooth(gain, 14)
    avg_l = _wilder_smooth(loss, 14)
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    # ── RSI(3) — Schneller Oscillator für Mikro-Dip-Erkennung ───────────────
    avg_g3 = _wilder_smooth(gain, 3)
    avg_l3 = _wilder_smooth(loss, 3)
    rs3    = avg_g3 / avg_l3.replace(0, np.nan)
    df["RSI3"] = 100 - 100 / (1 + rs3)

    # ── RSI-Divergenz (5-Kerzen-Lookback) ────────────────────────────────────
    df["RSI_div_bull"] = _rsi_divergence_bull(c, df["RSI"], lookback=5)
    df["RSI_div_bear"] = _rsi_divergence_bear(c, df["RSI"], lookback=5)

    # ── Stochastik K/D (14/3) ────────────────────────────────────────────────
    low14  = lo.rolling(14).min()
    high14 = h.rolling(14).max()
    df["Stoch_K"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # ── DI+ / DI- / ADX (14, Wilder) ────────────────────────────────────────
    df["DI_plus"], df["DI_minus"], df["ADX"] = _calc_adx(df, 14)

    # ── Volatilität ──────────────────────────────────────────────────────────
    pct = c.pct_change() * 100
    df["Vola"]       = pct.rolling(20).std()
    df["Vola_5"]     = pct.rolling(5).std()
    df["Vola_50"]    = pct.rolling(50).std()
    df["Vola_ratio"] = df["Vola_5"] / df["Vola_50"].replace(0, np.nan)

    # ── OBV ──────────────────────────────────────────────────────────────────
    direction = np.sign(c.diff()).fillna(0)
    df["OBV"]      = (direction * v).cumsum()
    df["OBV_MA"]   = df["OBV"].ewm(span=20, adjust=False).mean()
    df["OBV_bull"] = df["OBV"] > df["OBV_MA"]

    # ── Volumen ───────────────────────────────────────────────────────────────
    df["Vol_MA20"]      = v.rolling(20).mean()
    df["Vol_ratio_raw"] = v / df["Vol_MA20"].replace(0, np.nan)
    df["Vol_surge"]     = df["Vol_ratio_raw"] > 2.0

    # ── Bollinger Bands (20, ±2σ) ─────────────────────────────────────────────
    df["BB_mid"]   = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std

    # ── Keltner Channel (20 EMA, 0.7×ATR) ────────────────────────────────────
    # Multiplier 0.7 statt 1.5: Bei synthetischen Daten mit ATR ~2.4% des Preises
    # würde 1.5×ATR eine KC erzeugen, die immer breiter als BB ist (Squeeze=True immer).
    # 0.7 gibt eine KC-Breite ähnlich der BB, sodass Squeeze sinnvoll 30-60% der Zeit gilt.
    kc_mid         = c.ewm(span=20, adjust=False).mean()
    df["KC_upper"] = kc_mid + 0.7 * df["ATR"]
    df["KC_lower"] = kc_mid - 0.7 * df["ATR"]
    df["Squeeze"]  = (df["BB_upper"] < df["KC_upper"]) & (df["BB_lower"] > df["KC_lower"])

    # ── VWAP (täglicher Reset) ────────────────────────────────────────────────
    df["VWAP"] = _calc_vwap(df)

    # ── Consecutive Candles ────────────────────────────────────────────────────
    df["Consec"] = _calc_consec(c)

    # ── MA200-Abstand ──────────────────────────────────────────────────────────
    df["MA200_dist"] = (c - df["MA200"]) / df["MA200"].replace(0, np.nan) * 100

    # ── 30-Tage-Drawdown (720 × 1h-Kerzen) ────────────────────────────────────
    rol_max_30d      = c.rolling(720, min_periods=1).max()
    df["DD_30d"]     = (c / rol_max_30d - 1) * 100   # negativ = unter 30d-Hoch

    # ── Composite-Signale ──────────────────────────────────────────────────────
    df["EMA_bull"]      = (df["EMA8"] > df["EMA13"]) & (df["EMA13"] > df["EMA21"])
    df["EMA_bear"]      = (df["EMA8"] < df["EMA13"]) & (df["EMA13"] < df["EMA21"])

    # ── Pullback-Signale für Bulle-Entry ─────────────────────────────────────
    # MACD Histogramm dreht aufwärts (Momentum-Erholung nach Dip)
    df["MACD_hist_up"]   = df["MACD_hist"] > df["MACD_hist"].shift(1)
    # EMA21 steigt über 5 Bars → Uptrend bestätigt
    df["EMA21_slope"]    = df["EMA21"] > df["EMA21"].shift(5)
    # Preis kreuzt EMA21 von unten nach oben (Rücksetzer war da, Bounce startet)
    df["cross_ema21_up"] = (c >= df["EMA21"]) & (c.shift(1) < df["EMA21"].shift(1))
    df["DI_bull"]       = df["DI_plus"] > df["DI_minus"]
    df["DI_bear"]       = df["DI_minus"] > df["DI_plus"]
    df["MACD_pos"]      = df["MACD_hist"] > 0
    df["MACD_neg"]      = df["MACD_hist"] < 0
    df["Ichimoku_bull"] = (
        (c > df["Senkou_A"]) & (c > df["Senkou_B"]) & (df["Tenkan"] > df["Kijun"])
    )
    df["Ichimoku_bear"] = (c < df["Senkou_A"]) & (c < df["Senkou_B"])
    df["SAR_bull"]      = c > df["SAR"]
    df["vol_soft"]      = ~df["Vol_surge"]

    return df


# ─── Hilfsfunktionen für komplexe Indikatoren ───────────────────────────────

def _calc_sar(df: pd.DataFrame, af_start=0.02, af_max=0.2) -> pd.Series:
    """Parabolic SAR — iterative Berechnung."""
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n = len(close)
    sar    = np.full(n, np.nan)
    bull   = True
    af     = af_start
    ep     = low[0]   # extreme point
    sar[0] = high[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if np.isnan(prev_sar):
            sar[i] = close[i]
            continue

        if bull:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < sar[i]:
                bull   = False
                sar[i] = ep
                ep     = low[i]
                af     = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_start, af_max)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > sar[i]:
                bull   = True
                sar[i] = ep
                ep     = high[i]
                af     = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_start, af_max)

    return pd.Series(sar, index=df.index)


def _calc_supertrend(df: pd.DataFrame, multiplier=2.5):
    """Supertrend — iterative Berechnung. Gibt (supertrend_series, bull_series) zurück."""
    atr    = df["ATR"].values
    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    n = len(close)

    upper  = np.full(n, np.nan)
    lower  = np.full(n, np.nan)
    st     = np.full(n, np.nan)
    bull   = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(atr[i]):
            continue
        hl2 = (high[i] + low[i]) / 2
        upper[i] = hl2 + multiplier * atr[i]
        lower[i] = hl2 - multiplier * atr[i]

        if np.isnan(st[i - 1]):
            st[i]   = lower[i]
            bull[i] = True
            continue

        # Adjust bands
        if not np.isnan(lower[i - 1]):
            lower[i] = max(lower[i], lower[i - 1]) if close[i - 1] > lower[i - 1] else lower[i]
        if not np.isnan(upper[i - 1]):
            upper[i] = min(upper[i], upper[i - 1]) if close[i - 1] < upper[i - 1] else upper[i]

        if bull[i - 1]:
            if close[i] < lower[i]:
                st[i]   = upper[i]
                bull[i] = False
            else:
                st[i]   = lower[i]
                bull[i] = True
        else:
            if close[i] > upper[i]:
                st[i]   = lower[i]
                bull[i] = True
            else:
                st[i]   = upper[i]
                bull[i] = False

    return pd.Series(st, index=df.index), pd.Series(bull, index=df.index)


def _calc_adx(df: pd.DataFrame, period=14):
    """DI+, DI-, ADX (Wilder-Methode)."""
    high  = df["High"]
    low   = df["Low"]
    atr   = df["ATR"]

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


def _calc_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP mit täglichem Reset."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    date    = df.index.normalize()
    tp_vol  = typical * df["Volume"]

    cum_tp_vol = tp_vol.groupby(date).cumsum()
    cum_vol    = df["Volume"].groupby(date).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _calc_consec(close: pd.Series) -> pd.Series:
    """Consecutive up (+) / down (-) candles."""
    direction = np.sign(close.diff()).fillna(0).astype(int)
    consec    = np.zeros(len(close), dtype=int)

    for i in range(1, len(close)):
        d = direction.iloc[i]
        if d == 0:
            consec[i] = 0
        elif d == 1:
            consec[i] = max(consec[i - 1] + 1, 1)
        else:
            consec[i] = min(consec[i - 1] - 1, -1)

    return pd.Series(consec, index=close.index)


def _rsi_divergence_bull(close: pd.Series, rsi: pd.Series, lookback=5) -> pd.Series:
    """
    Bullische Divergenz: Preis neues Tief im Fenster, RSI aber nicht.
    Vektorisiert mit rolling().
    """
    close_min = close.rolling(lookback + 1).min()
    rsi_min   = rsi.rolling(lookback + 1).min()
    return (close <= close_min) & (rsi > rsi_min)


def _rsi_divergence_bear(close: pd.Series, rsi: pd.Series, lookback=5) -> pd.Series:
    """
    Bärische Divergenz: Preis neues Hoch im Fenster, RSI aber nicht.
    """
    close_max = close.rolling(lookback + 1).max()
    rsi_max   = rsi.rolling(lookback + 1).max()
    return (close >= close_max) & (rsi < rsi_max)
