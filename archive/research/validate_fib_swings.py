"""
validate_fib_swings.py — Visuelle Prüfung der Swing-Erkennung und Fib-Levels.

Gibt die letzten 15 erkannten Swing-Highs/-Lows aus und zeigt,
an welchen Tagen der Kurs auf einem Fib-Level war.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sol_bot_1d.backtest   import fetch_binance_1d
from sol_bot_1d.indicators import add_indicators

print("Lade Daten …")
df_raw = fetch_binance_1d("2021-09-01")
df     = add_indicators(df_raw)

# ─── Letzte 15 Swing-Highs ────────────────────────────────────────────────────
K = 5
pivot_high_mask = df["High"] == df["High"].rolling(2 * K + 1, center=True).max()
# Pivot liegt bei Bar t, sichtbar ab t+K (nach shift)
raw_pivots_h = df["High"].where(pivot_high_mask).dropna()

print("\n" + "=" * 60)
print("  SWING-HIGHS (letzte 15, mit shift=5 → first-visible-Bar)")
print("=" * 60)
print(f"  {'Pivot-Datum':<14}  {'Sichtbar ab':<14}  {'High':>10}  {'Close':>10}")
print("  " + "─" * 52)
for ts, val in raw_pivots_h.iloc[-15:].items():
    visible_ts = ts + pd.Timedelta(days=K)
    close_at   = df.loc[ts, "Close"]
    print(f"  {str(ts.date()):<14}  {str(visible_ts.date()):<14}  {val:>10.2f}  {close_at:>10.2f}")

# ─── Letzte 15 Swing-Lows ─────────────────────────────────────────────────────
pivot_low_mask = df["Low"] == df["Low"].rolling(2 * K + 1, center=True).min()
raw_pivots_l   = df["Low"].where(pivot_low_mask).dropna()

print("\n" + "=" * 60)
print("  SWING-LOWS (letzte 15)")
print("=" * 60)
print(f"  {'Pivot-Datum':<14}  {'Sichtbar ab':<14}  {'Low':>10}  {'Close':>10}")
print("  " + "─" * 52)
for ts, val in raw_pivots_l.iloc[-15:].items():
    visible_ts = ts + pd.Timedelta(days=K)
    close_at   = df.loc[ts, "Close"]
    print(f"  {str(ts.date()):<14}  {str(visible_ts.date()):<14}  {val:>10.2f}  {close_at:>10.2f}")

# ─── Aktuelle Fib-Levels ──────────────────────────────────────────────────────
last = df.iloc[-1]
print("\n" + "=" * 60)
print(f"  FIB-LEVELS HEUTE ({str(df.index[-1].date())})")
print("=" * 60)
print(f"  Swing-High : {last['Swing_high']:>10.2f}")
print(f"  Swing-Low  : {last['Swing_low']:>10.2f}")
print(f"  Fib 0.382  : {last['Fib_382']:>10.2f}")
print(f"  Fib 0.500  : {last['Fib_500']:>10.2f}")
print(f"  Fib 0.618  : {last['Fib_618']:>10.2f}")
print(f"  Close      : {last['Close']:>10.2f}")
print(f"  at_fib_level: {last['at_fib_level']}")

# ─── Tage am Fib-Level in den letzten 6 Monaten ───────────────────────────────
fib_days = df[df["at_fib_level"]].copy()

# Cluster: zusammenhängende Tage zu einem Setup zusammenfassen
fib_days["gap"] = (fib_days.index.to_series().diff() > pd.Timedelta(days=5)).cumsum()
clusters = fib_days.groupby("gap")

print("\n" + "=" * 68)
print(f"  ALLE FIB-HITS (Gesamt, {df.index[0].date()} → {df.index[-1].date()})")
print(f"  {len(fib_days)} Tage in {len(clusters)} Clustern (Cluster = zusammenhängende Tage)")
print("=" * 68)
print(f"  {'Cluster-Start':<14}  {'n-Tage':>7}  {'Close(1)':>10}  {'Swing-H':>10}  {'Swing-L':>10}  {'Level'}")
print("  " + "─" * 68)
TOL = 0.03
for gid, grp in clusters:
    ts    = grp.index[0]
    cl    = grp.iloc[0]["Close"]
    sh    = grp.iloc[0]["Swing_high"]
    sl    = grp.iloc[0]["Swing_low"]
    levels = []
    if pd.notna(grp.iloc[0]["Fib_382"]) and abs(cl - grp.iloc[0]["Fib_382"]) / grp.iloc[0]["Fib_382"] <= TOL:
        levels.append("0.382")
    if pd.notna(grp.iloc[0]["Fib_500"]) and abs(cl - grp.iloc[0]["Fib_500"]) / grp.iloc[0]["Fib_500"] <= TOL:
        levels.append("0.500")
    if pd.notna(grp.iloc[0]["Fib_618"]) and abs(cl - grp.iloc[0]["Fib_618"]) / grp.iloc[0]["Fib_618"] <= TOL:
        levels.append("0.618")
    print(f"  {str(ts.date()):<14}  {len(grp):>7}  {cl:>10.2f}  {sh:>10.2f}  {sl:>10.2f}  {', '.join(levels) or '—'}")

print(f"\n  Jahres-Verteilung:")
for yr, grp in fib_days.groupby(fib_days.index.year):
    print(f"    {yr}: {len(grp)} Tage")
