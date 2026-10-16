"""
regime.py — 2-Regime-Klassifizierung für den 1d-Bot
Regimes: bull, bear (kein Neutral)
Tiebreaker: Preis vs. MA200 wenn weder bull noch bear klar.
MIN_REGIME = 3 (3 Tage Hysterese) — sweep-ready via Parameter
"""

import pandas as pd
import numpy as np

MIN_REGIME = 3

REGIMES = ["bull", "bear"]


def _raw_regime(row) -> str:
    adx       = row.get("ADX",        20.0)
    di_bull   = row.get("DI_bull",    False)
    di_bear   = row.get("DI_bear",    False)
    ema_bull  = row.get("EMA_bull",   False)
    ema_bear  = row.get("EMA_bear",   False)
    ema_slope = row.get("EMA21_slope", True)
    close     = row.get("Close",      0.0)
    ma200     = row.get("MA200",      0.0)

    if pd.isna(adx): adx = 20.0

    # Bull: ADX > 20 + DI+ dominant + EMA10 > EMA21 aufwärts
    if adx > 20 and di_bull and (ema_bull or ema_slope):
        return "bull"

    # Bear: ADX > 20 + DI- dominant + EMA10 < EMA21
    if adx > 20 and di_bear and ema_bear:
        return "bear"

    # Tiebreaker: MA200 (NaN/0 → bull als Default)
    if ma200 > 0 and not pd.isna(ma200):
        return "bull" if close >= ma200 else "bear"
    return "bull"


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
