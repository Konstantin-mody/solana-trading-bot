"""
strategy.py — EMA-Crossover-Signallogik für den 1W-Bot

Konzept:
  EMA10 kreuzt über EMA21 UND Close > MA50 → Long-Entry
  Exit: nur via SL / TP / Trail (kein Crossover-Exit)
  LONG_ONLY=True: Bear-Crosses werden ignoriert

Signal-Codes:
   0 = Kein Signal
   1 = Long eröffnen (bullish cross + MA50-Filter)
"""
from __future__ import annotations
import pandas as pd


LOOKBACK_K = 52     # Wochen Rückblick (für CLI-Kompatibilität, hier unused)
LONG_ONLY  = True   # Nur Long-Trades (Bear-Crosses ignorieren)

SL_TP = {
    "bull": (1.5, 8.0),
    "bear": (1.5, 4.0),
}

TRAIL_MULT = {
    "bull": 3.0,
    "bear": 3.0,
}

HEBEL = {
    "bull": 2.0,
    "bear": 1.0,
}


def get_signal(row) -> tuple[int, str]:
    """
    EMA10/EMA21 Crossover-Signal mit MA50-Richtungsfilter.

    Long:  EMA10 kreuzt über EMA21 UND (Close > MA50 oder MA50 nicht verfügbar)
    Short: EMA10 kreuzt unter EMA21 UND (Close < MA50 oder MA50 nicht verfügbar)

    Returns:
      (1, "bull")  — bullish cross
      (-2, "bear") — bearish cross
      (0, "")      — kein Signal
    """
    ema10      = row.get("EMA10")
    ema21      = row.get("EMA21")
    ema10_prev = row.get("EMA10_prev")
    ema21_prev = row.get("EMA21_prev")
    close      = row.get("Close")
    ma50       = row.get("MA50")

    if any(x is None or (isinstance(x, float) and pd.isna(x))
           for x in [ema10, ema21, ema10_prev, ema21_prev, close]):
        return 0, ""

    ema10      = float(ema10)
    ema21      = float(ema21)
    ema10_prev = float(ema10_prev)
    ema21_prev = float(ema21_prev)
    close      = float(close)
    ma50_val   = None if (ma50 is None or pd.isna(ma50)) else float(ma50)

    if ema10 > ema21 and ema10_prev < ema21_prev:
        if ma50_val is None or close > ma50_val:
            return 1, "bull"

    if not LONG_ONLY and ema10 < ema21 and ema10_prev > ema21_prev:
        if ma50_val is None or close < ma50_val:
            return -2, "bear"

    return 0, ""
