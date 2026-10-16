"""
regime.py — 5-Regime-Klassifizierung für den SOL Bot
Priorität: extrem_baer > extrem_bulle > baer > bulle > seitwaerts > neutral
MIN_REGIME = 4 Kerzen Hysterese vor einem Wechsel
"""

import pandas as pd
import numpy as np

MIN_REGIME = 4

REGIMES = ["extrem_baer", "extrem_bulle", "baer", "bulle", "seitwaerts", "neutral"]


def _raw_regime(row) -> str:
    """Klassifiziert eine einzelne Kerze ohne Hysterese."""
    rsi         = row.get("RSI",         50.0)
    atr_sp      = row.get("ATR_spike",   1.0)
    vola_r      = row.get("Vola_ratio",  1.0)
    consec      = row.get("Consec",      0)
    adx         = row.get("ADX",         20.0)
    vola        = row.get("Vola",        1.0)
    di_bull     = row.get("DI_bull",     False)
    di_bear     = row.get("DI_bear",     False)
    ema21_slope = row.get("EMA21_slope", True)
    ema_bull    = row.get("EMA_bull",    False)
    ema_bear    = row.get("EMA_bear",    False)
    ma200d      = row.get("MA200_dist",  0.0)     # nur für Extrem-Regime

    # Nan-Schutz
    if pd.isna(rsi):    rsi    = 50.0
    if pd.isna(atr_sp): atr_sp = 1.0
    if pd.isna(vola_r): vola_r = 1.0
    if pd.isna(consec): consec = 0
    if pd.isna(adx):    adx    = 20.0
    if pd.isna(vola):   vola   = 1.0
    if pd.isna(ma200d): ma200d = 0.0

    # Extrem Bär (unverändert)
    crash_signals = sum([
        rsi    < 25,
        atr_sp > 1.8,
        vola_r > 2.0,
        consec <= -5,
    ])
    if crash_signals >= 2 and ma200d < -5:
        return "extrem_baer"

    # Extrem Bulle (unverändert)
    euph_signals = sum([
        rsi    > 72,
        atr_sp > 1.8,
        vola_r > 1.8,
        consec >= 5,
    ])
    if euph_signals >= 2 and ma200d > 5:
        return "extrem_bulle"

    # Bär: ADX stark + volle bearishe EMA-Alignment + DI- dominiert
    if adx > 22 and di_bear and ema_bear:
        return "baer"

    # Bulle: ADX stark + DI+ dominiert + EMA21 steigt
    if adx > 25 and di_bull and ema21_slope:
        return "bulle"

    # Seitwärts
    if adx < 22 and vola < 1.2:
        return "seitwaerts"

    return "neutral"


def classify_regime(df: pd.DataFrame) -> pd.Series:
    """
    Berechnet das Regime für jeden Zeitpunkt mit MIN_REGIME-Hysterese.
    Gibt eine pd.Series mit Regime-Strings zurück.
    """
    raw = df.apply(_raw_regime, axis=1)

    # Hysterese: Regime wechselt erst nach MIN_REGIME aufeinanderfolgenden
    # Kerzen im neuen Regime
    regimes  = raw.values.tolist()
    smoothed = list(regimes)
    current  = regimes[0]
    count    = 1

    for i in range(1, len(regimes)):
        if regimes[i] == current:
            count += 1
            smoothed[i] = current
        else:
            count += 1
            if count > MIN_REGIME:
                current = regimes[i]
                count   = 1
            smoothed[i] = current

    return pd.Series(smoothed, index=df.index, name="Regime")
