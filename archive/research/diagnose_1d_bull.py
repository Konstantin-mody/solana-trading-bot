"""
diagnose_1d_bull.py — IS vs. OOS Analyse der 1d-Bot Bull-Trades.

Fragestellung: Warum hat der 1d-Bot im OOS (2025+) schlechtere Bull-Trades?
Vergleicht Entry-Bedingungen, Hold-Zeit, Exit-Gründe, PnL-Verteilung.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sol_bot_1d.backtest   import run_backtest, fetch_binance_1d, WF_SPLIT
from sol_bot_1d.indicators import add_indicators
from sol_bot_1d.regime     import classify_regime

# ─── Daten laden ──────────────────────────────────────────────────────────────
print("Lade Binance 1d SOL/USDT-Daten …")
df_raw = fetch_binance_1d("2021-09-01")
print(f"  {len(df_raw)} Kerzen  |  {df_raw.index[0].date()} → {df_raw.index[-1].date()}\n")

# ─── Indikatoren (für Entry-Snapshot) ─────────────────────────────────────────
df_ind = add_indicators(df_raw)
df_ind["Regime"] = classify_regime(df_ind)

# ─── Backtest ─────────────────────────────────────────────────────────────────
trades, equity = run_backtest(df_raw)
print(f"Trades gesamt: {len(trades)}")
if trades.empty:
    print("Keine Trades.")
    sys.exit()

# ─── Bull-Long Trades isolieren ───────────────────────────────────────────────
bt = trades[(trades["Regime"] == "bull") & (trades["Side"] == "long")].copy()
print(f"Bull-Long Trades:  {len(bt)}  (IS={bt['IS'].sum()}, OOS={(~bt['IS']).sum()})\n")
if len(bt) == 0:
    print("Keine Bull-Trades — Signal zu selektiv.")
    sys.exit()

# ─── Entry-Indikatoren ergänzen ───────────────────────────────────────────────
entry_cols = ["RSI3", "ADX", "ATR", "Close", "EMA21", "EMA50", "MA50",
              "DI_plus", "DI_minus", "EMA_bull", "ATR_spike"]
available  = [c for c in entry_cols if c in df_ind.columns]

bt_idx = pd.to_datetime(bt["Entry_ts"])
snap   = df_ind.reindex(bt_idx.values, method="nearest")[available]
snap.index = bt.index
bt = pd.concat([bt, snap], axis=1)

bt["Hold_days"] = (pd.to_datetime(bt["Exit_ts"]) - pd.to_datetime(bt["Entry_ts"])).dt.days
bt["winner"]    = bt["PnL_pct"] > 0
if "ATR" in bt.columns and "Close" in bt.columns:
    bt["ATR_pct"] = bt["ATR"] / bt["Close"] * 100

# RSI3-Buckets früh setzen, damit alle Splits sie erben
_bins   = [0, 15, 20, 25, 30, 35, 50]
_labels = ["<15", "15-20", "20-25", "25-30", "30-35", ">35"]
if "RSI3" in bt.columns:
    bt["RSI3_bucket"] = pd.cut(bt["RSI3"], bins=_bins, labels=_labels, right=True)


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────

def summary(grp: pd.DataFrame, label: str):
    n   = len(grp)
    wr  = grp["winner"].mean() * 100 if n > 0 else float("nan")
    sp  = grp["PnL_pct"].sum()
    avg = grp["PnL_pct"].mean()
    print(f"  {label:<22}  n={n:>3}  WR={wr:>5.1f}%  ΣPnL={sp:>+7.1f}%  Ø={avg:>+6.2f}%")


def compare(is_grp: pd.DataFrame, oos_grp: pd.DataFrame, col: str, fmt=".2f"):
    if col not in bt.columns:
        return
    is_med  = is_grp[col].median()  if len(is_grp)  > 0 else float("nan")
    oos_med = oos_grp[col].median() if len(oos_grp) > 0 else float("nan")
    diff    = oos_med - is_med
    print(f"  {col:<22}  IS Median={is_med:{fmt}}  |  OOS Median={oos_med:{fmt}}  |  Δ={diff:+{fmt}}")


def exit_breakdown(grp: pd.DataFrame, label: str):
    if "Exit_reason" not in grp.columns or len(grp) == 0:
        return
    print(f"\n  Exit-Gründe ({label}):")
    for reason, g in grp.groupby("Exit_reason"):
        n  = len(g)
        wr = g["winner"].mean() * 100
        sp = g["PnL_pct"].sum()
        print(f"    {reason:<8}  n={n:>3}  WR={wr:>5.1f}%  ΣPnL={sp:>+7.1f}%")


# ─── IS / OOS Split ───────────────────────────────────────────────────────────
is_all   = bt[bt["IS"]]
oos_all  = bt[~bt["IS"]]

is_win   = is_all[is_all["winner"]]
is_loss  = is_all[~is_all["winner"]]
oos_win  = oos_all[oos_all["winner"]]
oos_loss = oos_all[~oos_all["winner"]]

print("=" * 60)
print("  ÜBERBLICK")
print("=" * 60)
summary(is_all,  "IS  gesamt")
summary(is_win,  "IS  Winner")
summary(is_loss, "IS  Loser")
summary(oos_all, "OOS gesamt")
summary(oos_win, "OOS Winner")
summary(oos_loss,"OOS Loser")

# ─── Entry-Bedingungen ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ENTRY-INDIKATOREN  (IS vs. OOS, Median)")
print("=" * 60)
for col in ["RSI3", "ADX", "ATR_pct", "Hold_days"]:
    compare(is_all, oos_all, col)

print("\n  Winner vs. Loser gesamt:")
for col in ["RSI3", "ADX", "Hold_days", "ATR_pct"]:
    compare(bt[bt["winner"]], bt[~bt["winner"]], col)

# ─── Exit-Gründe ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXIT-GRÜNDE")
print("=" * 60)
exit_breakdown(is_all,  "IS")
exit_breakdown(oos_all, "OOS")

# ─── PnL-Verteilung ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PnL-VERTEILUNG")
print("=" * 60)
for label, grp in [("IS  Winner", is_win), ("IS  Loser", is_loss),
                   ("OOS Winner", oos_win), ("OOS Loser", oos_loss)]:
    if len(grp) == 0:
        continue
    p = grp["PnL_pct"]
    print(f"  {label:<14}  min={p.min():>+6.1f}%  p25={p.quantile(.25):>+6.1f}%  "
          f"med={p.median():>+6.1f}%  p75={p.quantile(.75):>+6.1f}%  max={p.max():>+6.1f}%")

# ─── Jahres-Analyse ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  JAHRES-ANALYSE")
print("=" * 60)
bt["Jahr"] = pd.to_datetime(bt["Entry_ts"]).dt.year
for yr, grp in bt.groupby("Jahr"):
    tag = "IS " if grp["IS"].all() else ("OOS" if (~grp["IS"]).all() else "MIX")
    summary(grp, f"{yr}  [{tag}]")

# ─── RSI3-Bucket ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RSI3-BUCKETS  (Entry-RSI3 Verteilung)")
print("=" * 60)
if "RSI3_bucket" in bt.columns:
    print(f"  {'Bucket':<10}  {'IS n':>5}  {'IS WR':>6}  {'OOS n':>6}  {'OOS WR':>7}")
    for bkt in _labels:
        ig = is_all[is_all["RSI3_bucket"] == bkt]
        og = oos_all[oos_all["RSI3_bucket"] == bkt]
        iwr = f"{ig['winner'].mean()*100:.0f}%" if len(ig) > 0 else "—"
        owr = f"{og['winner'].mean()*100:.0f}%" if len(og) > 0 else "—"
        print(f"  {bkt:<10}  {len(ig):>5}  {iwr:>6}  {len(og):>6}  {owr:>7}")

# ─── Hold-Zeit: Winner vs. Loser ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  HOLD-ZEIT ANALYSE")
print("=" * 60)
for label, grp in [("IS  Winner", is_win), ("IS  Loser", is_loss),
                   ("OOS Winner", oos_win), ("OOS Loser", oos_loss)]:
    if len(grp) == 0:
        continue
    h = grp["Hold_days"]
    print(f"  {label:<14}  Ø={h.mean():>5.1f}d  Median={h.median():>4.0f}d  "
          f"min={h.min():>2.0f}d  max={h.max():>3.0f}d")
