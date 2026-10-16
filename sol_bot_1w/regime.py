"""
regime.py — 2-Regime-Klassifizierung für den 1W-Bot
Regimes: bull, bear (kein Neutral)

Auf Weekly: MA200 (= ~4 Jahre) als primärer Markt-Schalter.
Kein ADX-Filter — auf Weekly ist Trend fast immer vorhanden.
"""

import pandas as pd
import numpy as np

MIN_REGIME = 2   # 2 Wochen Hysterese

REGIMES = ["bull", "bear"]


def _raw_regime(row) -> str:
    close = row.get("Close", 0.0)
    ma50  = row.get("MA50",  0.0)
    ema21 = row.get("EMA21", 0.0)

    # MA50 weekly ≈ 1 Jahr — klassischer Bull/Bear-Markt-Schalter.
    # Verfügbar ab Woche 50 (ca. Mitte 2021 bei Start 2020-01).
    if ma50 > 0 and not pd.isna(ma50):
        return "bull" if close >= ma50 else "bear"

    # Fallback vor MA50-Verfügbarkeit: EMA21 (≈ 5 Monate)
    if ema21 > 0 and not pd.isna(ema21):
        return "bull" if close >= ema21 else "bear"

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
