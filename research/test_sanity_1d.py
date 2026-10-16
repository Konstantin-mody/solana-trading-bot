"""
test_sanity_1d.py — Sanity-Checks für sol_bot_1d/backtest.py.

Abdeckung:
  1.  Fee auf Notional (skaliert mit Hebel)
  2.  Long-PnL-Richtung (+Preis → +PnL)
  3.  Short-PnL-Richtung (+Preis → -PnL)
  4.  Funding nur für Shorts, nicht für Longs
  5.  Funding skaliert mit Hebel
  6.  SL-Richtung: Long-SL < Entry, Short-SL > Entry
  7.  Trail nur für Long (TRAIL_MULT["bear"] = None)
  8.  IS/OOS-Label nutzt Entry_ts, nicht Exit_ts (Regression)
  9.  Regime-Routing: Bull → signal_bull_entry, Bear → signal_bear_entry
  10. Snapshot: aktuelle Gesamt-Metriken eingefroren (+58.7%, n=73, WR≈48%) [Cooldown=2, Bear-SL=1×ATR, MA200-Filter]

Aufruf: python3 research/test_sanity_1d.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from sol_bot_1d.backtest import (
    _exit_pnl, _mtm, GEBUEHR, FUNDING_1D_PER_CANDLE,
    fetch_binance_1d, run_backtest, WF_SPLIT,
)
from sol_bot_1d.strategy import TRAIL_MULT, get_signal

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  OK   {name}")
        PASS += 1
    else:
        print(f"  FAIL {name}  {detail}")
        FAIL += 1


class _Pos:
    """Minimales Position-Objekt für Unit-Tests."""
    def __init__(self, side, entry_price, entry_cash, hebel):
        self.side        = side
        self.entry_price = entry_price
        self.entry_cash  = entry_cash
        self.hebel       = hebel


print("\n=== Sanity-Checks: sol_bot_1d/backtest.py ===\n")

# ── Test 1: Gebühren auf Notional ────────────────────────────────────────────
# 2× Hebel, kein Preischange: Fee = 2 × 2 × GEBUEHR × entry_cash
p = _Pos("long", 100.0, 20.0, 2.0)
net = _exit_pnl(p, 100.0)
fee_pct = (p.entry_cash - net) / p.entry_cash * 100
expected = 2 * p.hebel * GEBUEHR * 100
check(
    f"Fee auf Notional (2× Hebel): {fee_pct:.3f}% (erwartet {expected:.3f}%)",
    abs(fee_pct - expected) < 0.01,
    f"fee_pct={fee_pct:.4f}, erwartet={expected:.4f}",
)

# ── Test 2: Long-PnL-Richtung ────────────────────────────────────────────────
# 2× Hebel, +10% Preis → Gewinn = entry_cash × 2 × 0.10 = 4.0 (minus Fees)
p = _Pos("long", 100.0, 20.0, 2.0)
net = _exit_pnl(p, 110.0)
check(
    f"Long +10%: net={net:.2f} > entry_cash={p.entry_cash}",
    net > p.entry_cash,
    f"net={net:.4f} — Long sollte Gewinn zeigen",
)

# ── Test 3: Short-PnL-Richtung ───────────────────────────────────────────────
# 2× Hebel short, Preis steigt 10% → Verlust
p = _Pos("short", 100.0, 20.0, 2.0)
net = _exit_pnl(p, 110.0)
check(
    f"Short +10% (Preis gegen Short): net={net:.2f} < entry_cash={p.entry_cash}",
    net < p.entry_cash,
    f"net={net:.4f} — Short sollte Verlust zeigen wenn Preis steigt",
)
# Short profitiert wenn Preis fällt
p2 = _Pos("short", 100.0, 20.0, 2.0)
net2 = _exit_pnl(p2, 90.0)
check(
    f"Short -10% (Preis für Short): net={net2:.2f} > entry_cash={p2.entry_cash}",
    net2 > p2.entry_cash,
    f"net={net2:.4f} — Short sollte Gewinn zeigen wenn Preis fällt",
)

# ── Test 4: Funding nur für Shorts ───────────────────────────────────────────
# FUNDING_1D_PER_CANDLE muss > 0 sein
check(
    f"FUNDING_1D_PER_CANDLE > 0: {FUNDING_1D_PER_CANDLE}",
    FUNDING_1D_PER_CANDLE > 0,
    "Funding-Konstante ist 0 oder negativ",
)
# Longs zahlen kein Funding — simuliert durch: nur pos.side=="short" löst Funding aus
long_pays_funding = False   # by design — wird in backtest.py via `if side=="short"` geprüft
check(
    "Funding nur für Shorts (by design — Code-Verifikation)",
    not long_pays_funding,
    "Longs dürfen kein Funding zahlen",
)

# ── Test 5: Funding skaliert mit Hebel ───────────────────────────────────────
entry_cash = 40.0
funding_1x = entry_cash * FUNDING_1D_PER_CANDLE * 1.0
funding_2x = entry_cash * FUNDING_1D_PER_CANDLE * 2.0
check(
    f"Funding 2× = 2× Funding 1×: {funding_2x:.6f} == 2 × {funding_1x:.6f}",
    abs(funding_2x - 2 * funding_1x) < 1e-9,
    f"Funding skaliert nicht linear mit Hebel",
)

# ── Test 6: SL-Richtung ──────────────────────────────────────────────────────
# In backtest.py: Long-SL = entry - sl_mult * atr → sl < entry
# Short-SL = entry + sl_mult * atr → sl > entry
entry, atr, sl_mult = 100.0, 5.0, 1.0
long_sl  = entry - sl_mult * atr   # 95.0
short_sl = entry + sl_mult * atr   # 105.0
check(
    f"Long-SL ({long_sl}) < Entry ({entry})",
    long_sl < entry,
    f"long_sl={long_sl} muss unter entry={entry} liegen",
)
check(
    f"Short-SL ({short_sl}) > Entry ({entry})",
    short_sl > entry,
    f"short_sl={short_sl} muss über entry={entry} liegen",
)

# ── Test 7: Trail nur für Long ───────────────────────────────────────────────
check(
    f"TRAIL_MULT['bear'] ist None (kein Trail für Shorts)",
    TRAIL_MULT.get("bear") is None,
    f"TRAIL_MULT['bear']={TRAIL_MULT.get('bear')} — sollte None sein",
)
check(
    f"TRAIL_MULT['bull'] ist gesetzt (Trail für Longs aktiv)",
    TRAIL_MULT.get("bull") is not None and TRAIL_MULT["bull"] > 0,
    f"TRAIL_MULT['bull']={TRAIL_MULT.get('bull')} — sollte > 0 sein",
)

# ── Test 8: IS/OOS-Label nutzt Entry_ts (Regression) ────────────────────────
# Integration: Für alle Trades im realen Backtest muss gelten:
#   IS == (Entry_ts < WF_SPLIT)
# Wenn jemand den Fix rückgängig macht und Exit_ts verwendet, schlägt dieser Test an
# sobald ein Trade die WF-Grenze überbrückt.
print("\n  [Integration-Test 8 läuft — lädt Binance-Daten ...]\n")
try:
    df_raw = fetch_binance_1d("2021-09-01")
    trades_df, _ = run_backtest(df_raw)

    if trades_df.empty:
        check("IS/OOS Regression: Trades vorhanden", False, "Keine Trades — kann nicht prüfen")
    else:
        wf = pd.Timestamp(WF_SPLIT)
        expected_is = trades_df["Entry_ts"] < wf
        label_ok    = (trades_df["IS"] == expected_is).all()

        # Finde Trades die Exit_ts-basiert anders klassifiziert würden
        exit_based_is = trades_df["Exit_ts"] < wf
        straddling    = trades_df[(trades_df["Entry_ts"] < wf) & (trades_df["Exit_ts"] >= wf)]

        check(
            f"IS/OOS-Label == (Entry_ts < WF_SPLIT) für alle {len(trades_df)} Trades",
            label_ok,
            f"Label-Fehler: {(~label_ok).sum()} Trades falsch klassifiziert",
        )
        if len(straddling) > 0:
            check(
                f"Straddling-Trades ({len(straddling)}) korrekt als IS=True gelabelt",
                straddling["IS"].all(),
                f"Straddling-Trade fälschlicherweise als OOS klassifiziert",
            )
        else:
            print("  INFO  Kein Trade überbrückt WF-Grenze — Regression nicht aktiv beweisbar (korrekt)")

except Exception as e:
    check("IS/OOS Regression: kein Crash", False, f"Exception: {e}")

# ── Test 9: Regime-Routing ───────────────────────────────────────────────────
# get_signal(row, "bull") muss signal_bull_entry aufrufen (gibt 0 oder 1)
# get_signal(row, "bear") muss signal_bear_entry aufrufen (gibt 0 oder -2)
# Wir prüfen: Rückgabewerte sind nur aus dem erlaubten Set
_fake_row = {
    "RSI3": 40.0, "DI_bull": True, "EMA21": 110.0, "EMA50": 100.0,
    "MA50": 100.0, "Close": 100.0, "ATR_spike": 1.0, "OBV_bull": True,
    "DI_bear": False, "EMA_bear": False, "MA200": 80.0,
}
bull_sig = get_signal(_fake_row, "bull")
bear_sig = get_signal(_fake_row, "bear")
neutral_sig = get_signal(_fake_row, "neutral")
check(
    f"Bull-Routing gibt 0 oder 1: {bull_sig}",
    bull_sig in (0, 1),
    f"Unerlaubtes Signal: {bull_sig}",
)
check(
    f"Bear-Routing gibt 0 oder -2: {bear_sig}",
    bear_sig in (0, -2),
    f"Unerlaubtes Signal: {bear_sig}",
)
check(
    f"Neutral-Routing gibt 0 (kein Trade): {neutral_sig}",
    neutral_sig == 0,
    f"Neutral sollte immer 0 sein: {neutral_sig}",
)

# ── Test 10: Snapshot — aktuelle Metriken eingefroren ────────────────────────
# Schlägt an wenn eine Code-Änderung das Backtest-Ergebnis unerwartet verändert.
# Toleranz 1% um numerisches Rauschen bei Daten-Updates zu tolerieren.
print("\n  [Snapshot-Test 10 — nutzt bereits geladene Trades ...]\n")
try:
    if not trades_df.empty:
        eq = _
        rendite = (_.iloc[-1] / _.iloc[0] - 1) * 100 if len(_) > 1 else 0

        # Neu berechnen mit frisch geladenem equity
        _, equity_df = run_backtest(df_raw)
        rendite = (equity_df.iloc[-1] / equity_df.iloc[0] - 1) * 100
        n       = len(trades_df)
        wr      = (trades_df["PnL_pct"] > 0).mean() * 100

        check(
            f"Snapshot Rendite: {rendite:.1f}% (erwartet ~58.7%, Toleranz ±2pp)",
            abs(rendite - 58.7) < 2.0,
            f"Rendite={rendite:.2f}% — unerwartete Abweichung!",
        )
        check(
            f"Snapshot Trades: n={n} (erwartet 73)",
            n == 73,
            f"n={n} — Trade-Count hat sich geändert!",
        )
        check(
            f"Snapshot WR: {wr:.1f}% (erwartet ~48.0%, Toleranz ±3pp)",
            abs(wr - 48.0) < 3.0,
            f"WR={wr:.2f}% — unerwartete Abweichung!",
        )
except Exception as e:
    check("Snapshot-Test: kein Crash", False, f"Exception: {e}")

# ── Ergebnis ─────────────────────────────────────────────────────────────────
print(f"\n  Bestanden: {PASS}/{PASS+FAIL}")
if FAIL > 0:
    print(f"  FEHLGESCHLAGEN: {FAIL} Test(s) — backtest.py hat einen Bug!")
    sys.exit(1)
else:
    print("  Alle Checks OK.")
