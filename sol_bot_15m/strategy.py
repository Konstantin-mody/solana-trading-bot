"""
strategy.py — 15min-Bot Strategie v4 (Breakout/Momentum)
Ansatz: Donchian-Breakout in bestätigtem Trend
Entry: Close > Donch_high (20 Kerzen) + DI_bull + ADX>25 + EMA21_slope + daily_macro
Exit:  Nur Trail 2×ATR — kein fixes TP, kein Cover
"""
from __future__ import annotations
import numpy as np


def _g(row, col, default=False):
    val = row.get(col, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


# ─── SL/TP-Multiplikatoren ────────────────────────────────────────────────────
# tp_mult=999 = kein fixes TP — Exit ausschließlich via Trail
SL_TP = {
    "bull":    (1.0, 999.0),
    "bear":    (1.0, 999.0),
    "neutral": (1.0, 999.0),
}

TRAIL_MULT = {
    "bull": 2.0,
    "bear": 2.0,
}

HEBEL = {
    "bull": 1.0,
    "bear": 1.0,
}


# ─── Long-Signale ─────────────────────────────────────────────────────────────

def signal_bull_entry(row) -> int:
    """
    Donchian-Breakout Long: Close über 20-Perioden-Hoch + Trend bestätigt.
    ADX>25 filtert Ranging-Phasen, wo Breakouts sofort reversieren.
    """
    donch_break = _g(row, "Donch_bull_break", False)
    di_bull     = _g(row, "DI_bull",          False)
    ema21_slope = _g(row, "EMA21_slope",      False)
    daily_macro = _g(row, "Daily_macro_bull", False)
    adx         = _g(row, "ADX",              0.0)
    atr_spike   = _g(row, "ATR_spike",         1.0)

    if (donch_break
            and di_bull
            and ema21_slope
            and daily_macro
            and adx > 25
            and atr_spike < 2.5):
        return 1
    return 0


def signal_bull_exit(row) -> bool:
    """Kein Cover — Exit ausschließlich via Trail."""
    return False


# ─── Short-Signale ────────────────────────────────────────────────────────────

def signal_bear_entry(row) -> int:
    """
    Donchian-Breakout Short: Close unter 20-Perioden-Tief + Trend bestätigt.
    """
    donch_break = _g(row, "Donch_bear_break", False)
    di_bear     = _g(row, "DI_bear",          False)
    ema21_slope = _g(row, "EMA21_slope",      False)
    daily_macro = _g(row, "Daily_macro_bull", False)
    adx         = _g(row, "ADX",              0.0)
    atr_spike   = _g(row, "ATR_spike",         1.0)

    if (donch_break
            and di_bear
            and not ema21_slope
            and not daily_macro
            and adx > 25
            and atr_spike < 2.5):
        return -2
    return 0


def signal_bear_cover(row) -> bool:
    """Kein Cover — Exit ausschließlich via Trail/SL."""
    return False


# ─── Haupt-Dispatch ───────────────────────────────────────────────────────────

def get_signal(row, regime: str) -> int:
    if regime == "bull":
        return signal_bull_entry(row)
    elif regime == "bear":
        return signal_bear_entry(row)
    return 0
