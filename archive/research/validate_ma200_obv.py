"""
validate_ma200_obv.py — Welche Bull-Trades wurden durch MA200+OBV-Filter entfernt?

Vergleicht Bull-Entries vor und nach den neuen Filter-Bedingungen.
Monkey-patcht bt_mod.get_signal (nicht strat.signal_bull_entry),
weil backtest.py `get_signal` direkt importiert.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sol_bot_1d.backtest import run_backtest, fetch_binance_1d
import sol_bot_1d.backtest  as bt_mod
import sol_bot_1d.strategy  as strat


# ─── Entry-Signal OHNE MA200+OBV (Baseline-Logik) ────────────────────────────

def signal_bull_entry_old(row) -> int:
    """Ursprüngliche Logik: kein MA200-Filter, kein OBV-Filter."""
    rsi3      = strat._g(row, "RSI3",      50.0)
    di_bull   = strat._g(row, "DI_bull",   False)
    ema21     = strat._g(row, "EMA21",     0.0)
    ema50     = strat._g(row, "EMA50",     0.0)
    ma50      = strat._g(row, "MA50",      0.0)
    close     = strat._g(row, "Close",     0.0)
    atr_spike = strat._g(row, "ATR_spike", 1.0)

    if (rsi3 < 35
            and di_bull
            and ema21 > ema50
            and close >= ma50 * 0.92
            and atr_spike < 2.5):
        return 1
    return 0


def get_signal_old(row, regime: str) -> int:
    if regime == "bull":
        return signal_bull_entry_old(row)
    elif regime == "bear":
        return strat.signal_bear_entry(row)
    return 0


# ─── Daten ────────────────────────────────────────────────────────────────────
print("Lade Binance 1d SOL/USDT-Daten …")
df_raw = fetch_binance_1d("2021-09-01")
print(f"  {len(df_raw)} Kerzen  |  {df_raw.index[0].date()} → {df_raw.index[-1].date()}\n")


# ─── Alter Backtest (ohne MA200+OBV) ─────────────────────────────────────────
orig_get_signal = bt_mod.get_signal
bt_mod.get_signal = get_signal_old
try:
    old_trades, old_equity = run_backtest(df_raw)
finally:
    bt_mod.get_signal = orig_get_signal

# ─── Neuer Backtest (mit MA200+OBV) ──────────────────────────────────────────
new_trades, new_equity = run_backtest(df_raw)


# ─── Bull-Long isolieren ──────────────────────────────────────────────────────
def bull(df): return df[(df["Regime"] == "bull") & (df["Side"] == "long")].copy()

old_bt = bull(old_trades)
new_bt = bull(new_trades)
old_bt["Entry_ts"] = pd.to_datetime(old_bt["Entry_ts"])
new_bt["Entry_ts"] = pd.to_datetime(new_bt["Entry_ts"])

old_idx = set(old_bt["Entry_ts"])
new_idx = set(new_bt["Entry_ts"])

only_old = sorted(old_idx - new_idx)  # durch Filter entfernt
only_new = sorted(new_idx - old_idx)  # durch veränderte Cooldown neu dazugekommen
common   = sorted(old_idx & new_idx)

print("=" * 68)
print(f"  Bull-Long:  alt={len(old_bt)}  neu={len(new_bt)}  "
      f"gemeinsam={len(common)}  entfernt={len(only_old)}  neu={len(only_new)}")
print("=" * 68)

# ─── Entfernte Trades (nur alt) ───────────────────────────────────────────────
if only_old:
    removed = old_bt[old_bt["Entry_ts"].isin(only_old)].sort_values("Entry_ts")
    print(f"\n  ENTFERNTE TRADES  (n={len(removed)}, durch MA200+OBV-Filter blockiert)")
    print(f"  {'Datum':<12}  {'IS':>4}  {'PnL':>7}  {'Exit':>6}  {'Hold':>6}")
    print("  " + "─" * 45)
    for _, r in removed.iterrows():
        hold = (pd.to_datetime(r["Exit_ts"]) - r["Entry_ts"]).days
        tag  = "IS " if r["IS"] else "OOS"
        print(f"  {str(r['Entry_ts'].date()):<12}  {tag}  "
              f"{r['PnL_pct']:>+6.1f}%  {r['Exit_reason']:<6}  {hold:>4}d")

    sp  = removed["PnL_pct"].sum()
    wr  = (removed["PnL_pct"] > 0).mean() * 100
    n_is  = removed["IS"].sum()
    n_oos = (~removed["IS"]).sum()
    print(f"\n  Σ entfernte Trades: n={len(removed)}  WR={wr:.0f}%  ΣPnL={sp:>+.1f}%  "
          f"(IS={n_is}, OOS={n_oos})")
    sp_is  = removed.loc[removed["IS"],  "PnL_pct"].sum()
    sp_oos = removed.loc[~removed["IS"], "PnL_pct"].sum()
    print(f"  IS ΣPnL={sp_is:>+.1f}%   OOS ΣPnL={sp_oos:>+.1f}%")

# ─── Neu hinzugekommene Trades ────────────────────────────────────────────────
if only_new:
    added = new_bt[new_bt["Entry_ts"].isin(only_new)].sort_values("Entry_ts")
    print(f"\n  NEU HINZUGEKOMMEN  (n={len(added)}, Cooldown-Effekt)")
    print(f"  {'Datum':<12}  {'IS':>4}  {'PnL':>7}  {'Exit':>6}  {'Hold':>6}")
    print("  " + "─" * 45)
    for _, r in added.iterrows():
        hold = (pd.to_datetime(r["Exit_ts"]) - r["Entry_ts"]).days
        tag  = "IS " if r["IS"] else "OOS"
        print(f"  {str(r['Entry_ts'].date()):<12}  {tag}  "
              f"{r['PnL_pct']:>+6.1f}%  {r['Exit_reason']:<6}  {hold:>4}d")
    sp = added["PnL_pct"].sum()
    wr = (added["PnL_pct"] > 0).mean() * 100
    print(f"\n  Σ neue Trades: n={len(added)}  WR={wr:.0f}%  ΣPnL={sp:>+.1f}%")

# ─── Portfolio-Vergleich ──────────────────────────────────────────────────────
from cost_model import STARTKAPITAL

def rendite(eq): return (eq.iloc[-1] - STARTKAPITAL) / STARTKAPITAL * 100
def sharpe(eq):
    r = eq.pct_change().dropna()
    return (r.mean() / r.std()) * np.sqrt(365) if r.std() > 0 else 0.0
def maxdd(eq): return ((eq - eq.cummax()) / eq.cummax()).min() * 100

print("\n" + "=" * 68)
print("  PORTFOLIO-VERGLEICH")
print("=" * 68)
print(f"  {'':26}  {'Rendite':>8}  {'Sharpe':>7}  {'MaxDD':>7}")
print(f"  {'Ohne MA200+OBV':<26}  {rendite(old_equity):>+7.1f}%  "
      f"{sharpe(old_equity):>7.2f}  {maxdd(old_equity):>6.1f}%")
print(f"  {'Mit MA200+OBV':<26}  {rendite(new_equity):>+7.1f}%  "
      f"{sharpe(new_equity):>7.2f}  {maxdd(new_equity):>6.1f}%")
