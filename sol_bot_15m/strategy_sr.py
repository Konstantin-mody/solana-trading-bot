"""
strategy_sr.py — S/R Intraday-Double-Test Signal (Typ 2)

Logik:
  - Support/Resistance mit mind. MIN_TOUCHES Bestätigungen
  - Zweiter Test am gleichen UTC-Tag → Entry-Signal
  - Support hält (Close > level - BREAK) → Long
  - Support bricht (Close < level - BREAK) → Short
  - Resistance hält (Close < level + BREAK) → Short
  - Resistance bricht (Close > level + BREAK) → Long (Breakout)
  - SL direkt hinter dem Level (min 0.3% oder 0.75×ATR)
  - TP: 3×SL-Abstand vom Entry (R/R 3:1)
  - Trail: 2×ATR
"""
from __future__ import annotations
import pandas as pd

from sol_bot_15m.levels import TOUCH_PCT, BREAK_PCT

# ─── Parameter ────────────────────────────────────────────────────────────────
SL_BUFFER    = 0.003   # Mindestabstand SL hinter Level (0.3%)
SL_ATR_MULT  = 0.75    # SL mindestens 0.75×ATR hinter Level
TP_RR        = 3.0     # Risk/Reward 3:1
TRAIL_MULT   = 2.0     # 2×ATR Trailing Stop
HEBEL_SR     = 2.0     # Hebel — enger SL erlaubt mehr Hebelwirkung
MAX_SL_PCT   = 0.025   # Max SL 2.5% vom Entry → kein Trade wenn Level zu weit weg


def get_sr_entry(
    i: int, df: pd.DataFrame, tracker, atr: float
) -> dict | None:
    """
    Gibt Entry-Signal zurück oder None.
    Inkrementell: tracker.process_bar(i, df) muss VOR diesem Aufruf erfolgt sein.

    Rückgabe: {
        "side":   "long" | "short",
        "level":  float,   # Level-Preis (Referenz für SL-Platzierung)
        "sl":     float,   # SL-Preis
        "tp":     float,   # TP-Preis
        "hebel":  float,
        "trail_init": float,  # Initialer Trail-Stop
        "level_type": str,
    }
    """
    ts    = df.index[i]
    today = ts.date()
    close = df["Close"].iloc[i]
    high  = df["High"].iloc[i]
    low   = df["Low"].iloc[i]

    state = tracker.get_level_state_at(i, df)

    for lvl in state:
        p = lvl.price

        # Erster Test heute (vor aktuellem Bar) nötig
        had_test_today = any(
            df.index[tb].date() == today and tb < i
            for tb, k in lvl.touches
            if k in ("touch", "swing")
        )
        if not had_test_today:
            continue

        if lvl.level_type == "support":
            if low > p * (1 + TOUCH_PCT):
                continue   # aktueller Bar berührt Support nicht

            sl_price = min(p * (1 - SL_BUFFER), p - SL_ATR_MULT * atr)

            if close > p * (1 - BREAK_PCT):
                # Support hält → Long
                ep    = close   # Market-Entry am Schlusskurs
                risk  = ep - sl_price
                if risk <= 0 or risk / ep > MAX_SL_PCT:
                    continue
                tp    = ep + TP_RR * risk
                trail = ep - TRAIL_MULT * atr
                return {
                    "side": "long", "level": p, "sl": sl_price, "tp": tp,
                    "hebel": HEBEL_SR, "trail_init": trail, "level_type": "support",
                }

            elif close < p * (1 - BREAK_PCT):
                # Support bricht → Short
                sl_short = max(p * (1 + SL_BUFFER), p + SL_ATR_MULT * atr)
                ep        = close
                risk      = sl_short - ep
                if risk <= 0 or risk / ep > MAX_SL_PCT:
                    continue
                tp = ep - TP_RR * risk
                return {
                    "side": "short", "level": p, "sl": sl_short, "tp": tp,
                    "hebel": HEBEL_SR, "trail_init": None, "level_type": "support_break",
                }

        elif lvl.level_type == "resistance":
            if high < p * (1 - TOUCH_PCT):
                continue

            sl_short = max(p * (1 + SL_BUFFER), p + SL_ATR_MULT * atr)

            if close < p * (1 + BREAK_PCT):
                # Resistance hält → Short
                ep   = close
                risk = sl_short - ep
                if risk <= 0 or risk / ep > MAX_SL_PCT:
                    continue
                tp = ep - TP_RR * risk
                return {
                    "side": "short", "level": p, "sl": sl_short, "tp": tp,
                    "hebel": HEBEL_SR, "trail_init": None, "level_type": "resistance",
                }

            elif close > p * (1 + BREAK_PCT):
                # Resistance bricht → Long (Breakout)
                sl_long = min(p * (1 - SL_BUFFER), p - SL_ATR_MULT * atr)
                ep      = close
                risk    = ep - sl_long
                if risk <= 0 or risk / ep > MAX_SL_PCT:
                    continue
                tp    = ep + TP_RR * risk
                trail = ep - TRAIL_MULT * atr
                return {
                    "side": "long", "level": p, "sl": sl_long, "tp": tp,
                    "hebel": HEBEL_SR, "trail_init": trail, "level_type": "resistance_break",
                }

    return None
