"""
sweep_1d_bull_exit.py — Sweep über Cover-Exit-Trigger für den 1d-Bot Bull-Long.

Problem: alle 26 Bull-Trades enden via Cover (di_bear ist zu träge → OOS Loser Ø -21%).
Sweep: 3 Exit-Trigger verglichen ohne backtest.py anzufassen (Monkey-Patch).

Tracked: Winner-Median, Loser-Median, ΣPnL, Sharpe  →  IS + OOS + Gesamt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sol_bot_1d.backtest   import run_backtest, fetch_binance_1d, STARTKAPITAL
import sol_bot_1d.backtest  as bt_mod   # Patch-Target: hier liegt die importierte Referenz
import sol_bot_1d.strategy  as strat

# ─── Kandidaten ───────────────────────────────────────────────────────────────

def exit_di_bear(row) -> bool:
    """Baseline: RSI3>75 OR DI-dominant."""
    rsi3    = strat._g(row, "RSI3",    50.0)
    di_bear = strat._g(row, "DI_bear", False)
    return rsi3 > 75 or di_bear


def exit_slope(row) -> bool:
    """Schneller: RSI3>75 OR EMA21-3d-Slope dreht negativ."""
    rsi3  = strat._g(row, "RSI3",       50.0)
    slope = strat._g(row, "EMA21_slope", True)
    return rsi3 > 75 or (not slope)


def exit_not_bull(row) -> bool:
    """Mittel: RSI3>75 OR EMA-Bull-Stack bricht (EMA10 fällt unter EMA21)."""
    rsi3     = strat._g(row, "RSI3",     50.0)
    ema_bull = strat._g(row, "EMA_bull", False)
    return rsi3 > 75 or (not ema_bull)


MODES = [
    ("di_bear (Baseline)",   exit_di_bear),
    ("EMA21_slope==False",   exit_slope),
    ("NOT ema_bull",         exit_not_bull),
]

# ─── Daten ────────────────────────────────────────────────────────────────────
print("Lade Binance 1d SOL/USDT-Daten …")
df_raw = fetch_binance_1d("2021-09-01")
print(f"  {len(df_raw)} Kerzen  |  {df_raw.index[0].date()} → {df_raw.index[-1].date()}\n")


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

def sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    return (rets.mean() / rets.std()) * np.sqrt(365) if rets.std() > 0 else 0.0


def analyze(trades: pd.DataFrame, equity: pd.Series) -> dict:
    if trades.empty:
        return {}
    bt = trades[(trades["Regime"] == "bull") & (trades["Side"] == "long")]
    if bt.empty:
        return {}

    is_  = bt[bt["IS"]]
    oos_ = bt[~bt["IS"]]

    def med(grp, cond): return grp.loc[cond(grp), "PnL_pct"].median() if cond(grp).any() else float("nan")

    return {
        "n_total":    len(bt),
        "n_is":       len(is_),
        "n_oos":      len(oos_),
        "wr_all":     (bt["PnL_pct"] > 0).mean() * 100,
        "sp_all":     bt["PnL_pct"].sum(),
        "w_med_is":   med(is_,  lambda g: g["PnL_pct"] > 0),
        "l_med_is":   med(is_,  lambda g: g["PnL_pct"] <= 0),
        "w_med_oos":  med(oos_, lambda g: g["PnL_pct"] > 0),
        "l_med_oos":  med(oos_, lambda g: g["PnL_pct"] <= 0),
        "sp_is":      is_["PnL_pct"].sum(),
        "sp_oos":     oos_["PnL_pct"].sum(),
        "sharpe":     sharpe(equity),
        "maxdd":      ((equity - equity.cummax()) / equity.cummax()).min() * 100,
        "rendite":    (equity.iloc[-1] - STARTKAPITAL) / STARTKAPITAL * 100,
    }


# ─── Sweep ────────────────────────────────────────────────────────────────────
original   = bt_mod.signal_bull_exit
results    = []
bull_trades = {}   # mode → bull-only DataFrame

for name, fn in MODES:
    bt_mod.signal_bull_exit = fn
    try:
        trades, equity = run_backtest(df_raw)
    finally:
        bt_mod.signal_bull_exit = original

    bt = trades[(trades["Regime"] == "bull") & (trades["Side"] == "long")].copy()
    bt["Entry_ts"] = pd.to_datetime(bt["Entry_ts"])
    bull_trades[name] = bt

    m = analyze(trades, equity)
    m["mode"]   = name
    m["trades"] = trades
    m["equity"] = equity
    results.append(m)
    print(f"  {name} … fertig")

bt_mod.signal_bull_exit = original  # Sicherheit

# ─── Ausgabe ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  PORTFOLIO  (alle Regime, n=61 Trades)")
print("=" * 72)
print(f"  {'Modus':<26}  {'Rendite':>8}  {'Sharpe':>7}  {'MaxDD':>7}")
print("  " + "─" * 55)
for r in results:
    print(f"  {r['mode']:<26}  {r.get('rendite', float('nan')):>+7.1f}%  "
          f"{r.get('sharpe', float('nan')):>7.2f}  {r.get('maxdd', float('nan')):>6.1f}%")

print("\n" + "=" * 72)
print("  BULL-LONG  IS / OOS")
print("=" * 72)
hdr = f"  {'Modus':<26}  {'n':>4}  {'WR':>5}  {'ΣPnL':>7}  {'IS ΣPnL':>8}  {'OOS ΣPnL':>9}"
print(hdr)
print("  " + "─" * 68)
for r in results:
    print(f"  {r['mode']:<26}  {r.get('n_total',0):>4}  "
          f"{r.get('wr_all', float('nan')):>4.0f}%  "
          f"{r.get('sp_all', float('nan')):>+6.1f}%  "
          f"{r.get('sp_is', float('nan')):>+7.1f}%  "
          f"{r.get('sp_oos', float('nan')):>+8.1f}%")

print("\n" + "=" * 72)
print("  PAYOFF  (Winner- vs. Loser-Median, IS / OOS)")
print("=" * 72)
print(f"  {'Modus':<26}  {'IS Win':>7}  {'IS Los':>7}  {'OOS Win':>8}  {'OOS Los':>8}")
print("  " + "─" * 65)
for r in results:
    fmt = lambda v: f"{v:>+7.1f}%" if not (v != v) else f"{'—':>7}"
    print(f"  {r['mode']:<26}  {fmt(r.get('w_med_is',  float('nan')))}  "
          f"{fmt(r.get('l_med_is',  float('nan')))}  "
          f"{fmt(r.get('w_med_oos', float('nan')))}  "
          f"{fmt(r.get('l_med_oos', float('nan')))}")

# ─── Trade-Set-Dekomposition ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  TRADE-SET-DEKOMPOSITION  (vs. Baseline di_bear)")
print("  Frage: Exit-Effekt oder Cooldown-Effekt?")
print("=" * 72)

baseline_key = "di_bear (Baseline)"
base_bt      = bull_trades[baseline_key].set_index("Entry_ts")

for name, alt_bt in bull_trades.items():
    if name == baseline_key:
        continue
    alt_idx = alt_bt.set_index("Entry_ts")

    # Gemeinsame Entries (gleicher Entry_ts)
    common_ts  = base_bt.index.intersection(alt_idx.index)
    only_base  = base_bt.index.difference(alt_idx.index)
    only_alt   = alt_idx.index.difference(base_bt.index)

    common_base = base_bt.loc[common_ts]
    common_alt  = alt_idx.loc[common_ts]

    print(f"\n  [{name}]")
    print(f"    Gemeinsame Entries: {len(common_ts)}  |  Nur Baseline: {len(only_base)}  |  Nur Alt: {len(only_alt)}")

    # Exit-Effekt: gleiche Entries, anderer PnL
    if len(common_ts) > 0:
        delta = common_alt["PnL_pct"] - common_base["PnL_pct"]
        print(f"    PnL-Delta (Alt - Baseline) auf gemeinsamen Trades:")
        print(f"      Median Δ: {delta.median():>+.2f}%  |  Ø Δ: {delta.mean():>+.2f}%  |  "
              f"min: {delta.min():>+.1f}%  |  max: {delta.max():>+.1f}%")
        win_base   = (common_base["PnL_pct"] > 0).sum()
        win_alt    = (common_alt["PnL_pct"]  > 0).sum()
        sum_base   = common_base["PnL_pct"].sum()
        sum_alt    = common_alt["PnL_pct"].sum()
        print(f"      ΣPnL Baseline: {sum_base:>+.1f}%  |  ΣPnL Alt: {sum_alt:>+.1f}%  |  "
              f"Winner Baseline: {win_base}  |  Winner Alt: {win_alt}")

    # Cooldown-Effekt: neue Trades (nur in Alt)
    if len(only_alt) > 0:
        new_tr = alt_idx.loc[only_alt]
        is_new = new_tr[new_tr["IS"]]
        oos_new= new_tr[~new_tr["IS"]]
        sp_new = new_tr["PnL_pct"].sum()
        wr_new = (new_tr["PnL_pct"] > 0).mean() * 100
        print(f"    Neue Trades (Cooldown-Effekt): n={len(only_alt)}  WR={wr_new:.0f}%  ΣPnL={sp_new:>+.1f}%"
              f"  (IS:{len(is_new)}, OOS:{len(oos_new)})")

    # Verlorene Trades (nur in Baseline)
    if len(only_base) > 0:
        lost_tr = base_bt.loc[only_base]
        sp_lost = lost_tr["PnL_pct"].sum()
        wr_lost = (lost_tr["PnL_pct"] > 0).mean() * 100
        print(f"    Entfallene Trades (Baseline only): n={len(only_base)}  WR={wr_lost:.0f}%  ΣPnL={sp_lost:>+.1f}%")
