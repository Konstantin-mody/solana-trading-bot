"""
strategy.py — Signallogik für den 4h-Bot
Strategie: Trend-Following, Swing-Trades über mehrere 4h-Kerzen

Signal-Codes:
  -2 = Short eröffnen
   0 = Kein Signal
   1 = Long eröffnen
"""
from __future__ import annotations
import numpy as np


def _g(row, col, default=False):
    val = row.get(col, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


SL_TP = {
    "bull": (1.0, 8.0),
    "bear": (1.0, 8.0),
}

TRAIL_MULT = {
    "bull": 3.0,
    "bear": None,
}

HEBEL = {
    "bull": 2.0,
    "bear": 1.5,
}


def signal_bull_entry(row) -> int:
    rsi3      = _g(row, "RSI3",      50.0)
    di_bull   = _g(row, "DI_bull",   False)
    ema21     = _g(row, "EMA21",     0.0)
    ema50     = _g(row, "EMA50",     0.0)
    ma50      = _g(row, "MA50",      0.0)
    close     = _g(row, "Close",     0.0)
    atr_spike = _g(row, "ATR_spike", 1.0)
    obv_bull  = _g(row, "OBV_bull",  False)

    if (rsi3 < 50
            and di_bull
            and ema21 > ema50
            and close >= ma50 * 0.92
            and atr_spike < 2.5
            and obv_bull):
        return 1
    return 0


def signal_bull_exit(row) -> bool:
    rsi3    = _g(row, "RSI3",    50.0)
    di_bear = _g(row, "DI_bear", False)
    return rsi3 > 75 or di_bear


def signal_bear_entry(row) -> int:
    rsi3     = _g(row, "RSI3",     50.0)
    di_bear  = _g(row, "DI_bear",  False)
    ema_bear = _g(row, "EMA_bear", False)
    close    = _g(row, "Close",    0.0)
    ma200    = _g(row, "MA200",    0.0)

    if ma200 > 0 and close >= ma200:
        return 0

    if (rsi3 > 65
            and di_bear
            and ema_bear):
        return -2
    return 0


def signal_bear_cover(row) -> bool:
    rsi3    = _g(row, "RSI3",    50.0)
    di_bull = _g(row, "DI_bull", False)
    return rsi3 < 30 or di_bull


def get_signal(row, regime: str) -> int:
    if regime == "bull":
        return signal_bull_entry(row)
    elif regime == "bear":
        return signal_bear_entry(row)
    return 0
