"""
cost_model.py — Geteiltes Kostenmodell für alle Bots (1h, 1d, 15m)
Einmalig definiert, überall importiert.
"""

GEBUEHR  = 0.00075   # 0.075% pro Trade (Maker-Fee Binance)
SLIPPAGE = 0.0003    # 0.03% Slippage
SPREAD   = 0.0002    # 0.02% Spread

# Funding Rate: pro 8h, für Short × Hebel
# Binance charged zu festen UTC-Zeiten (00:00 / 08:00 / 16:00)
FUNDING8 = 0.0001

STARTKAPITAL = 200.0

# ─── Per-Timeframe Funding-Anwendung ─────────────────────────────────────────
#
# 1h:  if i % 8 == 0  → FUNDING8 × hebel × entry_cash
# 1d:  jede Kerze     → FUNDING8 × 3 × hebel × entry_cash  (3 Cycles/Tag)
# 15m: if i % 32 == 0 → FUNDING8 × hebel × entry_cash      (32 × 15min = 8h)
#
# Für 1d ist die Timing-Präzision (UTC 00/08/16 vs. Kerzenende) vernachlässigbar.
FUNDING_1D_PER_CANDLE = 3 * FUNDING8   # Convenience-Konstante für 1d-Engine
FUNDING_CADENCE_1H    = 8
FUNDING_CADENCE_4H    = 2              # alle 2 × 4h-Kerzen = 8h
FUNDING_CADENCE_15M   = 32
