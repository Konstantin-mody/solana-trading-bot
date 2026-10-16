"""
strategy.py — Signallogik für den 1d-Bot
Strategie: Trend-Following, Swing-Trades über mehrere Tage/Wochen

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


# ─── SL/TP-Multiplikatoren (in ATR-Einheiten) ────────────────────────────────
# Auf 1d ist 1×ATR ≈ 5-8% des Preises — Multiplikatoren entsprechend konservativ.
SL_TP = {
    "bull": (1.0, 8.0),   # SL 1×ATR (~7%), TP 8×ATR — Fate-Analyse: 87% neuer SL-Exits bei 1× sind genuine Verlierer
    "bear": (1.0, 8.0),   # Fate-Analyse: 100% neue SL-Exits bei 1× sind genuine Verlierer (4/4 LOSER, 0 WINNER entgangen)
}

TRAIL_MULT = {
    "bull": 3.0,
    "bear": None,  # kein Trail für Bear-Shorts (feste SL/TP)
}

HEBEL = {
    "bull": 2.0,
    "bear": 1.5,   # Sharpe-Peak bei 1.5× (Funding-Kosten skalieren linear, strukturelles Optimum)
}


# ─── Bull-Signale ─────────────────────────────────────────────────────────────

def signal_bull_entry(row) -> int:
    """
    Bull-Long: Trend-Continuation im bestätigten Aufwärtstrend.
    RSI3<50 auf 1d = Trend-Pause, nicht Dip-Buy. OBV-Bestätigung filtert Fehlausbrüche.
    """
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
            and close >= ma50 * 0.92             # max. 8% unter MA50
            and atr_spike < 2.5                  # kein Crash-Spike
            and obv_bull):                       # Volumen-Trendbestätigung
        return 1
    return 0


def signal_bull_exit(row) -> bool:
    """Cover Long: RSI3 überhitzt oder Trend dreht."""
    rsi3    = _g(row, "RSI3",    50.0)
    di_bear = _g(row, "DI_bear", False)
    return rsi3 > 75 or di_bear


# ─── Bear-Signale ─────────────────────────────────────────────────────────────

def signal_bear_entry(row) -> int:
    """
    Swing-Short: RSI3-Bounce in bestätigtem Abwärtstrend.
    RSI3 > 65 in Bear-Regime = überkaufter Bounce gegen den Trend → Short.
    MA200-Filter: kein Short wenn Preis über MA200 (Recovery-Jahr-Schutz).
    """
    rsi3     = _g(row, "RSI3",     50.0)
    di_bear  = _g(row, "DI_bear",  False)
    ema_bear = _g(row, "EMA_bear", False)
    close    = _g(row, "Close",    0.0)
    ma200    = _g(row, "MA200",    0.0)

    # Kein Short wenn Kurs über MA200 (Recovery-Regime) — NaN/0 = Filter nicht anwenden
    if ma200 > 0 and close >= ma200:
        return 0

    if (rsi3 > 65
            and di_bear
            and ema_bear):
        return -2
    return 0


def signal_bear_cover(row) -> bool:
    """Cover Short: RSI3 erholt sich oder Trend dreht."""
    rsi3   = _g(row, "RSI3",   50.0)
    di_bull = _g(row, "DI_bull", False)
    return rsi3 < 30 or di_bull


# ─── Haupt-Dispatch ───────────────────────────────────────────────────────────

def get_signal(row, regime: str) -> int:
    if regime == "bull":
        return signal_bull_entry(row)
    elif regime == "bear":
        return signal_bear_entry(row)
    return 0
