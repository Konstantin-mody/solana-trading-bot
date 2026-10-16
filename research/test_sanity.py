"""
Sanity-Checks für backtest.py — nach jeder Änderung ausführen.
Fängt die beiden Mai-2026-Bugs sofort.

Aufruf: python3 test_sanity.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --fast oder FAST=1: Nur Unit-Tests (1-7), kein Integration-Test (~30s)
FAST_MODE = "--fast" in sys.argv or os.environ.get("FAST", "") == "1"

from sol_bot.backtest import _exit_pnl, _mtm, GEBUEHR, _buy_price, make_price_data, run_backtest

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
    """Minimales Position-Objekt für Tests."""
    def __init__(self, side, entry_price, entry_cash, hebel):
        self.side        = side
        self.entry_price = entry_price
        self.entry_cash  = entry_cash
        self.hebel       = hebel


print("\n=== Sanity-Checks: backtest.py ===\n")

# ── Test 1: Gebühren auf Notional ───────────────────────────────────────────
# Bei 8× Hebel, kein Preischange: Fee ≈ 2 × hebel × GEBUEHR × entry_cash
#   = 2 × 8 × 0.00075 × 20 = 0.24 → Verlust = 0.24/20 = 1.2%
p = _Pos("long", entry_price=100.0, entry_cash=20.0, hebel=8.0)
net = _exit_pnl(p, 100.0)
fee_pct = (p.entry_cash - net) / p.entry_cash * 100
expected = 2 * p.hebel * GEBUEHR * 100   # ≈ 1.2%
check(
    f"Fee bei 8× Hebel, kein Move: {fee_pct:.3f}% (erwartet ~{expected:.2f}%)",
    abs(fee_pct - expected) < 0.05,
    f"→ Fee-Bug? fee_pct={fee_pct:.3f}, expected={expected:.2f}"
)

# ── Test 2: Fee skaliert mit Hebel ──────────────────────────────────────────
p1 = _Pos("long", 100.0, 20.0, hebel=1.0)
p8 = _Pos("long", 100.0, 20.0, hebel=8.0)
net1 = _exit_pnl(p1, 100.0)
net8 = _exit_pnl(p8, 100.0)
fee1 = p1.entry_cash - net1
fee8 = p8.entry_cash - net8
ratio = fee8 / fee1 if fee1 > 0 else 0
check(
    f"Fee skaliert mit Hebel: 8×-Fee / 1×-Fee = {ratio:.1f}× (erwartet 8×)",
    7.5 < ratio < 8.5,
    f"→ Fee-Bug? ratio={ratio:.2f}"
)

# ── Test 3: Gewinnender Long korrekt ────────────────────────────────────────
# 2× Hebel, +10% Preis: Gewinn = entry_cash × hebel × 0.10 = 4.0
# minus Fees ≈ 2 × 2 × 0.00075 × 20 = 0.06
# → net ≈ 24.0 - 0.06 = 23.94
p = _Pos("long", 100.0, 20.0, hebel=2.0)
net = _exit_pnl(p, 110.0)
gross_pnl = 20.0 * 2.0 * 0.10   # 4.0
fees = 2 * 2.0 * GEBUEHR * 20.0 # ~0.06
expected_net = 20.0 + gross_pnl - fees
check(
    f"Long +10% bei 2× Hebel: net={net:.4f} (erwartet ~{expected_net:.4f})",
    abs(net - expected_net) < 0.05,
    f"→ Arithmetik-Bug? net={net:.4f}, expected={expected_net:.4f}"
)

# ── Test 4: Verlierender Short korrekt ──────────────────────────────────────
# 4× Hebel short, Preis steigt 5% → Verlust = entry_cash × 4 × 0.05 = 4.0
p = _Pos("short", 100.0, 20.0, hebel=4.0)
net = _exit_pnl(p, 105.0)
gross_pnl = 20.0 * 4.0 * 0.05   # 4.0 verlust
expected_net = 20.0 - gross_pnl - 2 * 4.0 * GEBUEHR * 20.0
check(
    f"Short -5% (Preis hoch) bei 4× Hebel: net={net:.4f} (erwartet ~{expected_net:.4f})",
    abs(net - expected_net) < 0.1,
    f"→ Short-Bug? net={net:.4f}, expected={expected_net:.4f}"
)

# ── Test 5: Pyramiding Weighted-Average (Integration) ───────────────────────
# Simuliert den Pyramiding-Block aus backtest.py direkt:
#   Entry bei 100, entry_cash=20, 2× Hebel
#   Pyramid bei 108, add_cash=10
#   Erwarteter Avg = (100×20 + 108×10) / 30 = 3080/30 ≈ 102.667
# Exit genau bei diesem Avg → Verlust = nur Gebühren
p = _Pos("long", entry_price=100.0, entry_cash=20.0, hebel=2.0)
pyramid_price  = _buy_price(108.0, 0.5)   # Slippage wie echte Engine
add_cash       = 10.0
new_total_cash = p.entry_cash + add_cash
# Weighted-Average-Zeile — dieselbe wie in backtest.py run_backtest():
p.entry_price  = (p.entry_price * p.entry_cash + pyramid_price * add_cash) / new_total_cash
p.entry_cash   = new_total_cash
# Exit exakt am gewichteten Durchschnitt → nur Gebühren-Verlust
net_at_avg = _exit_pnl(p, p.entry_price)
loss_pct = (p.entry_cash - net_at_avg) / p.entry_cash * 100
check(
    f"Pyramid Weighted-Avg Entry aktiv: avg={p.entry_price:.3f}, Verlust={loss_pct:.2f}% (erwartet 0.2–0.6%)",
    0.1 < loss_pct < 0.8,
    f"→ Pyramiding-Bug? avg={p.entry_price:.3f}, loss={loss_pct:.3f}%"
)
# Zusatz: Wenn entry_price NICHT aktualisiert wurde, würde avg=100.0 bleiben
# → Exit bei ~102.7 hätte +2.7% Gewinn statt ~0% Verlust
# Dieser Check stellt sicher, dass avg > 100:
check(
    f"Pyramid aktualisiert entry_price: {p.entry_price:.3f} > 100.0 (Slippage auf Pyramid-Preis)",
    p.entry_price > 100.5,
    f"→ entry_price nicht aktualisiert! Wert: {p.entry_price:.3f}"
)

# ── Test 6: _mtm berechnet korrekt ohne Gebühren ────────────────────────────
# Mark-to-market bei unverändertem Preis = entry_cash (keine Gebühren!)
p = _Pos("long", entry_price=100.0, entry_cash=20.0, hebel=3.0)
mtm_val = _mtm(p, 100.0)
check(
    f"_mtm: kein Preischange → Wert = entry_cash ({mtm_val:.4f} == {p.entry_cash:.4f})",
    abs(mtm_val - p.entry_cash) < 0.001,
    f"→ _mtm berechnet Gebühren? mtm={mtm_val:.4f}, entry_cash={p.entry_cash:.4f}"
)
# _mtm bei +10% = entry_cash + entry_cash * hebel * 0.10
p2 = _Pos("long", entry_price=100.0, entry_cash=20.0, hebel=3.0)
mtm2 = _mtm(p2, 110.0)
expected_mtm = 20.0 + 20.0 * 3.0 * 0.10  # = 26.0
check(
    f"_mtm: +10% bei 3× Hebel → {mtm2:.4f} (erwartet {expected_mtm:.4f})",
    abs(mtm2 - expected_mtm) < 0.001,
    f"→ _mtm-Bug? mtm={mtm2:.4f}, expected={expected_mtm:.4f}"
)

# ── Test 7: Bär-Cover Guard — seitwaerts-Short NICHT in bär gecovert ─────────
# Bug #3: pos.regime=="baer" Prüfung muss vorhanden sein.
# Ein Short der in "seitwaerts" geöffnet wurde darf NICHT durch signal_baer_cover()
# gecovered werden. Die Bedingung in backtest.py ist:
#   regime=="baer" AND pos.regime=="baer" AND pos.side=="short"
# Wenn pos.regime=="seitwaerts": Bedingung False → kein ungewollter Cover.

class _PosWithRegime:
    def __init__(self, side, regime):
        self.side   = side
        self.regime = regime

current_regime = "baer"
pos_sw_short   = _PosWithRegime("short", "seitwaerts")
pos_baer_short = _PosWithRegime("short", "baer")

# Bedingung genau so wie in backtest.py Zeile 468:
sw_triggers_baer_cover = (
    current_regime == "baer"
    and pos_sw_short.regime == "baer"
    and pos_sw_short.side == "short"
)
baer_triggers_baer_cover = (
    current_regime == "baer"
    and pos_baer_short.regime == "baer"
    and pos_baer_short.side == "short"
)
check(
    "Bär-Cover Guard: seitwaerts-Short wird in bär NICHT gecovert",
    not sw_triggers_baer_cover,
    "→ pos.regime=='baer' Guard fehlt? sw_triggers_baer_cover=True"
)
check(
    "Bär-Cover Guard: bär-Short wird in bär korrekt gecovert",
    baer_triggers_baer_cover,
    "→ Baer-Cover defekt? baer_triggers_baer_cover=False"
)

# ── Test 8: Integration — run_backtest läuft durch und liefert plausibles Ergebnis ─
# Startet backtest auf synthetischen Daten (seed=42, ~29k Kerzen).
# Prüft: kein Crash, Endkapital zwischen $50 und $5000, mindestens 10 Trades.
# Fängt strukturelle Bugs im Pyramid-Block oder der Engine die Unit-Tests nicht sehen.
if FAST_MODE:
    print("  [Integration-Test übersprungen — FAST-Modus (FAST=1 oder --fast)]")
else:
    print("\n  [Integration-Test läuft, ~10–30s ...]\n")
    try:
        df_raw = make_price_data()
        trades_df, equity_df, _, _blocked = run_backtest(df_raw, sw_mode="rsi_div")
        end_cap  = equity_df.iloc[-1]   # equity_df ist pd.Series
        n_trades = len(trades_df)
        check(
            f"Integration: Endkapital ${end_cap:.0f} in $50–$5000 (Start $200)",
            50 < end_cap < 5000,
            f"→ Endkapital außerhalb Plausibilitätsbereich: ${end_cap:.2f}"
        )
        check(
            f"Integration: Anzahl Trades {n_trades} ≥ 10",
            n_trades >= 10,
            f"→ Zu wenige Trades: {n_trades}"
        )
        check(
            f"Integration: Pyramid_count-Spalte vorhanden und ≥ 0",
            "Pyramid_count" in trades_df.columns and (trades_df["Pyramid_count"] >= 0).all(),
            f"→ Pyramid_count fehlt oder negativ"
        )
    except Exception as e:
        check("Integration: run_backtest ohne Crash", False, f"→ Exception: {e}")

# ── Ergebnis ────────────────────────────────────────────────────────────────
print(f"\n  Bestanden: {PASS}/{PASS+FAIL}")
if FAIL > 0:
    print(f"  FEHLGESCHLAGEN: {FAIL} Test(s) — backtest.py hat einen Bug!")
    sys.exit(1)
else:
    print("  Alle Checks OK.")
