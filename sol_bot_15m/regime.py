"""
regime.py — 3-Regime-Klassifizierung für den 15min-Bot
Regimes: bull, bear, neutral
MIN_REGIME = 8 (8 × 15min = 2h Hysterese) — sweep-ready via Parameter
"""

import pandas as pd
import numpy as np

MIN_REGIME = 8

REGIMES = ["bull", "bear", "neutral"]


def _raw_regime(row) -> str:
    adx       = row.get("ADX",        20.0)
    di_bull   = row.get("DI_bull",    False)
    di_bear   = row.get("DI_bear",    False)
    ema_bull  = row.get("EMA_bull",   False)
    ema_bear  = row.get("EMA_bear",   False)
    ema_slope = row.get("EMA21_slope", True)

    if pd.isna(adx): adx = 20.0

    # Bull: ADX > 18 + DI+ dominant + EMA-Ausrichtung aufwärts
    if adx > 18 and di_bull and (ema_bull or ema_slope):
        return "bull"

    # Bear: ADX > 18 + DI- dominant + EMA-Ausrichtung abwärts
    if adx > 18 and di_bear and ema_bear:
        return "bear"

    return "neutral"


def classify_regime(df: pd.DataFrame, min_regime: int = MIN_REGIME) -> pd.Series:
    raw      = df.apply(_raw_regime, axis=1)
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
            if count > min_regime:
                current = regimes[i]
                count   = 1
            smoothed[i] = current

    return pd.Series(smoothed, index=df.index, name="Regime")
