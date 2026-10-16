"""
sl_fate_analysis.py — Fate-Analyse für tightere SL-Multiplier.

Frage: Trades die bei SL=0.5× und SL=1.0× per SL gestoppt werden —
was passiert mit denselben Entries bei SL=2.0× (Baseline)?

Wenn sie bei 2.0× Gewinner sind → Tighter-SL ist slot-rotation artifact.
Wenn sie bei 2.0× auch Verlierer → Tighter-SL rettet echten Edge.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from sol_bot_1d.backtest   import fetch_binance_1d, run_backtest
from sol_bot_1d.indicators import add_indicators
import sol_bot_1d.strategy as strat

print("Lade Daten …")
df_raw = fetch_binance_1d("2021-09-01")
df     = add_indicators(df_raw)

ORIG_BULL_SL_TP = strat.SL_TP["bull"]


def run_at_sl(sl_mult: float) -> pd.DataFrame:
    strat.SL_TP["bull"] = (sl_mult, ORIG_BULL_SL_TP[1])
    strat.SL_TP["bear"] = (sl_mult, strat.SL_TP["bear"][1])
    trades, _ = run_backtest(df_raw)
    strat.SL_TP["bull"] = ORIG_BULL_SL_TP
    strat.SL_TP["bear"] = (ORIG_BULL_SL_TP[0], strat.SL_TP["bear"][1])
    return trades[trades["Regime"] == "bull"].copy() if not trades.empty else pd.DataFrame()


# ─── Baseline: SL=2.0× ────────────────────────────────────────────────────────
print("Laufe SL=2.0× (Baseline) …")
t_base = run_at_sl(2.0)
base_by_entry = t_base.set_index("Entry_ts")

print(f"  Baseline: {len(t_base)} Bull-Trades, "
      f"{(t_base['Exit_reason']=='SL').sum()} SL-Exits")


def fate_report(sl_mult: float):
    print(f"\nLaufe SL={sl_mult}× …")
    t_tight = run_at_sl(sl_mult)
    if t_tight.empty:
        print("  Keine Trades.")
        return

    sl_trades = t_tight[t_tight["Exit_reason"] == "SL"]
    non_sl_base = t_base[t_base["Exit_reason"] != "SL"]

    print(f"\n{'='*72}")
    print(f"  SL={sl_mult}×ATR  |  Bull-Trades gesamt: {len(t_tight)},  SL-Exits: {len(sl_trades)}")
    print(f"{'='*72}")

    # Neue SL-Trades: exits bei sl_mult die NICHT bei 2.0× per SL exitten
    new_sl = sl_trades[~sl_trades["Entry_ts"].isin(
        t_base[t_base["Exit_reason"] == "SL"]["Entry_ts"]
    )]
    print(f"\n  Neu bei SL={sl_mult}× (nicht auch SL bei 2.0×): {len(new_sl)} Trades")

    if new_sl.empty:
        print("  → Keine neuen SL-Exits gegenüber Baseline.")
        return

    print(f"\n  {'Entry_ts':<12}  {'PnL@{:.1f}×'.format(sl_mult):>12}  "
          f"{'Exit@2.0×':>12}  {'PnL@2.0×':>10}  {'Fate'}")
    print("  " + "─" * 65)

    fate_win = 0
    fate_loss = 0
    pnl_rescued = 0.0

    for _, row in new_sl.iterrows():
        entry = row["Entry_ts"]
        pnl_tight = row["PnL_pct"]

        if entry in base_by_entry.index:
            base_row = base_by_entry.loc[entry]
            exit_base = base_row["Exit_reason"]
            pnl_base  = base_row["PnL_pct"]
            fate = "WIN" if pnl_base > 0 else "LOSS"
            if pnl_base > 0:
                fate_win += 1
                pnl_rescued += pnl_base
            else:
                fate_loss += 1
        else:
            exit_base = "N/A (kein Match)"
            pnl_base  = float("nan")
            fate = "?"

        print(f"  {str(entry.date()):<12}  {pnl_tight:>+12.1f}%  "
              f"{exit_base:>12}  {pnl_base:>+10.1f}%  {fate}")

    print(f"\n  Zusammenfassung neue SL-Exits bei {sl_mult}×:")
    print(f"    → Bei 2.0×: {fate_win} WINNER  ({fate_win}/{len(new_sl)*100//100} ≈ "
          f"{fate_win/max(len(new_sl),1)*100:.0f}%)")
    print(f"    → Bei 2.0×: {fate_loss} LOSER")
    print(f"    → Entgangener PnL (rescuable winners): {pnl_rescued:+.1f}% (trade-level Σ)")
    print(f"\n  Interpretation: {'⚠️  ' if fate_win > fate_loss else '✅  '}"
          f"{'Tighter SL stoppt vorwiegend WINNER → Slot-Rotation-Artefakt' if fate_win > fate_loss else 'Tighter SL stoppt vorwiegend LOSER → echter Schutz'}")

    # ─── Vergleich Gesamtverteilung ───────────────────────────────────────────
    print(f"\n  Exit-Mix bei SL={sl_mult}×: {t_tight['Exit_reason'].value_counts().to_dict()}")
    print(f"  Exit-Mix bei SL=2.0×:      {t_base['Exit_reason'].value_counts().to_dict()}")

    # ─── IS/OOS Aufschlüsselung ──────────────────────────────────────────────
    for split_label, mask_tight, mask_base in [
        ("IS",  t_tight["IS"],  t_base["IS"]),
        ("OOS", ~t_tight["IS"], ~t_base["IS"]),
    ]:
        tg = t_tight[mask_tight]
        tb = t_base[mask_base]
        pnl_t = tg["PnL_pct"].sum() if len(tg) else 0
        pnl_b = tb["PnL_pct"].sum() if len(tb) else 0
        wr_t  = (tg["PnL_pct"] > 0).mean() * 100 if len(tg) else 0
        wr_b  = (tb["PnL_pct"] > 0).mean() * 100 if len(tb) else 0
        print(f"  {split_label}: SL={sl_mult}× → n={len(tg)}, WR={wr_t:.0f}%, ΣPnL={pnl_t:+.1f}%  "
              f"| 2.0× → n={len(tb)}, WR={wr_b:.0f}%, ΣPnL={pnl_b:+.1f}%")


fate_report(0.5)
fate_report(1.0)
