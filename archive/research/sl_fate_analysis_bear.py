"""
sl_fate_analysis_bear.py — Fate-Analyse Bear-SL-Multiplikatoren.

Frage: Trades die bei Bear-SL=1.0× oder 1.5×ATR per SL gestoppt werden —
was passiert mit denselben Entries bei SL=2.0× (Baseline)?

Zwei Kategorien:
  (A) Tightened:    SL@tight + SL@baseline  → reine Verlustdeckelung
  (B) Fate-changed: SL@tight + non-SL@base  → genuine LOSER vs. WINNER?

Bear-SL ist close-only (kurs >= sl_price am Tagesschluss).
Bull-SL bleibt fest auf 1.0×ATR — keine Regime-Kontamination.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from sol_bot_1d.backtest import fetch_binance_1d, run_backtest
import sol_bot_1d.strategy as strat

print("Lade Daten …")
df_raw = fetch_binance_1d("2021-09-01")

ORIG_BULL_SL_TP = strat.SL_TP["bull"]   # (1.0, 8.0) — bleibt unverändert
ORIG_BEAR_SL_TP = strat.SL_TP["bear"]   # (2.0, 8.0) — Baseline


def run_at_bear_sl(sl_mult: float) -> pd.DataFrame:
    """Backtest mit geändertem Bear-SL; Bull-SL fest bei 1.0×ATR."""
    strat.SL_TP["bear"] = (sl_mult, ORIG_BEAR_SL_TP[1])
    trades, _ = run_backtest(df_raw)
    strat.SL_TP["bear"] = ORIG_BEAR_SL_TP
    return trades[trades["Regime"] == "bear"].copy() if not trades.empty else pd.DataFrame()


# ─── Baseline: SL=2.0× ────────────────────────────────────────────────────────
print("Laufe Bear-SL=2.0× (Baseline) …")
t_base = run_at_bear_sl(2.0)
base_by_entry = t_base.set_index("Entry_ts")

print(f"  Baseline: {len(t_base)} Bear-Trades, "
      f"{(t_base['Exit_reason'] == 'SL').sum()} SL-Exits, "
      f"ΣPnL={t_base['PnL_pct'].sum():+.1f}%")
print(f"  WR={( t_base['PnL_pct'] > 0).mean()*100:.0f}%  "
      f"IS ΣPnL={t_base[t_base['IS']]['PnL_pct'].sum():+.1f}%  "
      f"OOS ΣPnL={t_base[~t_base['IS']]['PnL_pct'].sum():+.1f}%")


def fate_report(sl_mult: float):
    print(f"\nLaufe Bear-SL={sl_mult}× …")
    t_tight = run_at_bear_sl(sl_mult)
    if t_tight.empty:
        print("  Keine Bear-Trades.")
        return

    sl_trades = t_tight[t_tight["Exit_reason"] == "SL"]
    base_sl_entries = set(t_base[t_base["Exit_reason"] == "SL"]["Entry_ts"])

    print(f"\n{'='*76}")
    print(f"  Bear-SL={sl_mult}×ATR  |  Trades: {len(t_tight)},  SL-Exits: {len(sl_trades)}")
    print(f"{'='*76}")

    # ─── Kategorie A: Tightened (SL auch bei 2.0×) ───────────────────────────
    tightened = sl_trades[sl_trades["Entry_ts"].isin(base_sl_entries)]
    print(f"\n  [A] Tightened (SL@{sl_mult}× UND SL@2.0×): {len(tightened)} Trades")
    delta_a = 0.0
    if not tightened.empty:
        print(f"  {'Entry_ts':<12}  {'PnL@{:.1f}×'.format(sl_mult):>10}  {'PnL@2.0×':>10}  {'Delta':>8}")
        print("  " + "─" * 50)
        for _, row in tightened.iterrows():
            entry  = row["Entry_ts"]
            pnl_t  = row["PnL_pct"]
            pnl_b  = base_by_entry.loc[entry, "PnL_pct"] if entry in base_by_entry.index else float("nan")
            delta  = pnl_t - pnl_b
            delta_a += delta
            print(f"  {str(entry.date()):<12}  {pnl_t:>+10.1f}%  {pnl_b:>+10.1f}%  {delta:>+8.1f}pp")
        print(f"  → ΣDelta A: {delta_a:+.1f}pp  (reine Verlustdeckelung)")

    # ─── Kategorie B: Fate-changed (kein SL bei 2.0×) ────────────────────────
    fate_changed = sl_trades[~sl_trades["Entry_ts"].isin(base_sl_entries)]
    print(f"\n  [B] Fate-changed (SL@{sl_mult}×, nicht SL@2.0×): {len(fate_changed)} Trades")

    fate_win    = 0
    fate_loss   = 0
    pnl_rescued = 0.0
    delta_b     = 0.0

    if fate_changed.empty:
        print("  → Keine neuen SL-Exits gegenüber Baseline.")
    else:
        print(f"\n  {'Entry_ts':<12}  {'PnL@{:.1f}×'.format(sl_mult):>10}  "
              f"{'Exit@2.0×':>12}  {'PnL@2.0×':>10}  Fate")
        print("  " + "─" * 68)
        for _, row in fate_changed.iterrows():
            entry = row["Entry_ts"]
            pnl_t = row["PnL_pct"]
            if entry in base_by_entry.index:
                base_row  = base_by_entry.loc[entry]
                exit_base = base_row["Exit_reason"]
                pnl_base  = base_row["PnL_pct"]
                fate      = "WIN" if pnl_base > 0 else "LOSS"
                delta_b  += pnl_t - pnl_base
                if pnl_base > 0:
                    fate_win    += 1
                    pnl_rescued += pnl_base
                else:
                    fate_loss += 1
            else:
                exit_base = "N/A"
                pnl_base  = float("nan")
                fate      = "?"
            print(f"  {str(entry.date()):<12}  {pnl_t:>+10.1f}%  "
                  f"{exit_base:>12}  {pnl_base:>+10.1f}%  {fate}")

        print(f"\n  Zusammenfassung [B] bei {sl_mult}×:")
        print(f"    → {fate_win} WINNER entgangen (Σ entgangener PnL: {pnl_rescued:+.1f}%)")
        print(f"    → {fate_loss} genuine LOSER gestoppt")
        print(f"    → ΣDelta B: {delta_b:+.1f}pp")
        verdict = "⚠️  Slot-Rotation (WINNER dominieren)" if fate_win > fate_loss else "✅  echter Schutz (LOSER dominieren)"
        print(f"    → {verdict}")

    # ─── Gesamt-Δ ─────────────────────────────────────────────────────────────
    pnl_tight = t_tight["PnL_pct"].sum()
    pnl_base_ = t_base["PnL_pct"].sum()
    print(f"\n  Gesamt Bear ΣPnL: {sl_mult}×={pnl_tight:+.1f}%  vs  2.0×={pnl_base_:+.1f}%  "
          f"(Δ={pnl_tight - pnl_base_:+.1f}pp)")

    # ─── IS/OOS-Aufschlüsselung ───────────────────────────────────────────────
    for label, mask_t, mask_b in [
        ("IS",  t_tight["IS"],  t_base["IS"]),
        ("OOS", ~t_tight["IS"], ~t_base["IS"]),
    ]:
        tg = t_tight[mask_t]
        tb = t_base[mask_b]
        pnl_t_ = tg["PnL_pct"].sum() if len(tg) else 0
        pnl_b_ = tb["PnL_pct"].sum() if len(tb) else 0
        wr_t   = (tg["PnL_pct"] > 0).mean() * 100 if len(tg) else 0
        wr_b   = (tb["PnL_pct"] > 0).mean() * 100 if len(tb) else 0
        print(f"  {label}: {sl_mult}× → n={len(tg)}, WR={wr_t:.0f}%, ΣPnL={pnl_t_:+.1f}%  "
              f"| 2.0× → n={len(tb)}, WR={wr_b:.0f}%, ΣPnL={pnl_b_:+.1f}%")

    # ─── OOS-Stresstest: 2025-03-02 (+46.2% kritischer Winner) ───────────────
    print(f"\n  OOS-Stresstest: 2025-03-02 (kritischer +46.2% OOS-Winner)")
    window = t_tight[(t_tight["Entry_ts"] >= pd.Timestamp("2025-02-26")) &
                     (t_tight["Entry_ts"] <= pd.Timestamp("2025-03-06"))]
    base_w = t_base[(t_base["Entry_ts"] >= pd.Timestamp("2025-02-26")) &
                    (t_base["Entry_ts"] <= pd.Timestamp("2025-03-06"))]
    if base_w.empty:
        print("  → Kein Bear-Trade in diesem Fenster (Datum evtl. Bear-Entry anders)")
    elif window.empty:
        print(f"  → In Baseline: {base_w.iloc[0]['Entry_ts'].date()} PnL={base_w.iloc[0]['PnL_pct']:+.1f}%")
        print(f"  → Bei {sl_mult}×: Trade eliminiert via SL! ⚠️")
    else:
        r = window.iloc[0]
        rb = base_w.iloc[0]
        print(f"  → Baseline: {rb['Entry_ts'].date()} PnL={rb['PnL_pct']:+.1f}%, Exit={rb['Exit_reason']}")
        print(f"  → Bei {sl_mult}×: PnL={r['PnL_pct']:+.1f}%, Exit={r['Exit_reason']} "
              f"{'✅ unverändert' if abs(r['PnL_pct'] - rb['PnL_pct']) < 1 else '⚠️ verändert'}")


fate_report(1.5)
fate_report(1.0)
