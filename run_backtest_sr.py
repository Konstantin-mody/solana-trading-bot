"""
run_backtest_sr.py — S/R-Double-Test-Strategie: Backtest-Runner

Verwendung:
  python3 run_backtest_sr.py
  python3 run_backtest_sr.py --symbol SOLUSDT
  python3 run_backtest_sr.py --symbol BTCUSDT --start 2022-01-01
"""
from __future__ import annotations

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from sol_bot_15m.backtest    import fetch_binance_15m
from sol_bot_15m.backtest_sr import run_backtest_sr, WF_SPLIT
from sol_bot_15m.strategy_sr import MAX_SL_PCT, TP_RR

ABORT_GATE_MIN_TRADES = 60
ABORT_GATE_MIN_AVG    = 0.15   # Ø Netto % > +0.15% nötig


# ─── Statistiken ──────────────────────────────────────────────────────────────

def _stats(t: pd.DataFrame, label: str) -> None:
    if t.empty:
        print(f"  {label}: keine Trades")
        return
    n       = len(t)
    wr      = (t["PnL_pct"] > 0).mean() * 100
    avg     = t["PnL_pct"].mean()
    total   = t["PnL_pct"].sum()
    wins    = t.loc[t["PnL_pct"] > 0, "PnL_pct"]
    losses  = t.loc[t["PnL_pct"] <= 0, "PnL_pct"]
    avg_w   = wins.mean()   if not wins.empty   else 0.0
    avg_l   = losses.mean() if not losses.empty else 0.0
    print(f"  {label}: n={n}  WR={wr:.1f}%  Ø={avg:+.3f}%  ΣPnL={total:+.1f}%  "
          f"Ø-Win={avg_w:+.2f}%  Ø-Loss={avg_l:+.2f}%")


def _level_breakdown(t: pd.DataFrame) -> None:
    for lt in sorted(t["Level_type"].unique()):
        sub = t[t["Level_type"] == lt]
        n   = len(sub)
        wr  = (sub["PnL_pct"] > 0).mean() * 100
        avg = sub["PnL_pct"].mean()
        tot = sub["PnL_pct"].sum()
        print(f"    {lt:<22}  n={n:3d}  WR={wr:4.1f}%  Ø={avg:+.3f}%  ΣPnL={tot:+.1f}%")


def _sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(365 * 96))


def _max_dd(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((equity - peak) / peak).min() * 100)


def _exit_breakdown(t: pd.DataFrame) -> None:
    for reason in ["SL", "TP", "Trail"]:
        sub = t[t["Exit_reason"] == reason]
        if sub.empty:
            continue
        n   = len(sub)
        avg = sub["PnL_pct"].mean()
        tot = sub["PnL_pct"].sum()
        print(f"    {reason:<8}  n={n:3d}  Ø={avg:+.3f}%  ΣPnL={tot:+.1f}%")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Binance-Symbol (default BTCUSDT)")
    parser.add_argument("--start",  default="2021-09-01",
                        help="Start-Datum (default 2021-09-01)")
    args = parser.parse_args()

    sym_label = args.symbol.replace("USDT", "")
    print(f"\n{'='*60}")
    print(f"  S/R-Double-Test-Backtest  |  {sym_label}/USDT 15min")
    print(f"  IS: {args.start} → {WF_SPLIT}  |  OOS: {WF_SPLIT} → heute")
    print(f"{'='*60}\n")

    print(f"Lade {sym_label}/USDT 15min-Daten …")
    df = fetch_binance_15m(args.start, symbol=args.symbol)
    print(f"  {len(df):,} Kerzen geladen\n")

    print("Führe Backtest durch …")
    trades_df, equity_df = run_backtest_sr(df)
    print(f"  Fertig — {len(trades_df)} Trades gesamt\n")

    if trades_df.empty:
        print("  KEIN TRADE — Strategie hat kein Signal produziert.")
        sys.exit(0)

    # ── Split IS / OOS ────────────────────────────────────────────────────
    is_t  = trades_df[trades_df["IS"] == True]
    oos_t = trades_df[trades_df["IS"] == False]

    eq_is  = equity_df[equity_df.index <  pd.Timestamp(WF_SPLIT)]
    eq_oos = equity_df[equity_df.index >= pd.Timestamp(WF_SPLIT)]

    # ── Gesamt-Equity ─────────────────────────────────────────────────────
    start_cap = 200.0
    end_cap   = float(equity_df.iloc[-1])
    rendite   = (end_cap / start_cap - 1) * 100
    sharpe    = _sharpe(equity_df)
    max_dd    = _max_dd(equity_df)

    print(f"{'─'*60}")
    print(f"  Gesamt-Ergebnis")
    print(f"    Startkapital: ${start_cap:.2f}  →  ${end_cap:.2f}")
    print(f"    Rendite:  {rendite:+.1f}%")
    print(f"    Sharpe:   {sharpe:.2f}")
    print(f"    Max DD:   {max_dd:.1f}%")
    print()

    # ── IS ────────────────────────────────────────────────────────────────
    print("  In-Sample (IS):")
    _stats(is_t, "Alle")
    if not is_t.empty:
        _level_breakdown(is_t)
        print("  Exit-Gründe:")
        _exit_breakdown(is_t)
        if not eq_is.empty:
            print(f"  IS Sharpe: {_sharpe(eq_is):.2f}  |  Max DD: {_max_dd(eq_is):.1f}%")
    print()

    # ── OOS ───────────────────────────────────────────────────────────────
    print("  Out-of-Sample (OOS):")
    _stats(oos_t, "Alle")
    if not oos_t.empty:
        _level_breakdown(oos_t)
        print("  Exit-Gründe:")
        _exit_breakdown(oos_t)
        if not eq_oos.empty:
            print(f"  OOS Sharpe: {_sharpe(eq_oos):.2f}  |  Max DD: {_max_dd(eq_oos):.1f}%")
    print()

    # ── Abort-Gate ────────────────────────────────────────────────────────
    print(f"{'─'*60}")
    n_is  = len(is_t)
    avg_is = is_t["PnL_pct"].mean() if not is_t.empty else 0.0
    if n_is >= ABORT_GATE_MIN_TRADES:
        if avg_is < ABORT_GATE_MIN_AVG:
            print(f"  ⛔ ABORT-GATE AUSGELÖST: IS n={n_is} ≥ {ABORT_GATE_MIN_TRADES}"
                  f" und Ø={avg_is:+.3f}% < +{ABORT_GATE_MIN_AVG}%")
            print(f"     → Strategie hat keinen ausreichenden Edge. Projekt einstellen.")
        else:
            print(f"  ✅ Abort-Gate: IS n={n_is}, Ø={avg_is:+.3f}% ≥ +{ABORT_GATE_MIN_AVG}%")
    else:
        print(f"  ⏳ Abort-Gate: IS n={n_is} < {ABORT_GATE_MIN_TRADES} — noch nicht auswertbar")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
