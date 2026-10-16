"""
Diskriminator-Analyse: Bulle-Long Entries Winner vs. Loser.
Exportiert Indikatorwerte beim Entry und vergleicht Verteilungen.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sol_bot.backtest import run_backtest, fetch_binance_data, compute_daily_macro

df_raw = fetch_binance_data(start='2021-09-01')
daily_s = compute_daily_macro(df_raw)
extra   = {'Daily_macro_bull': daily_s}

trades, equity, df_full, _ = run_backtest(
    df_raw.copy(),
    sw_mode='rsi_div',
    extra_signals=extra,
    baer_lev_override=6.0,
    bulle_lev_override=1.0,
    extrem_bulle_lev_override=1.0,
)

# Bulle-Long Trades
bt = trades[(trades['Regime'] == 'bulle') & (trades['Side'].str.lower() == 'long')].copy()
print(f"\nBulle-Long Trades gesamt: {len(bt)}")
if len(bt) == 0:
    print("Keine Trades — Signal zu selektiv.")
    sys.exit()

# PnL-Spalte
if 'PnL_pct' not in bt.columns:
    bt['PnL_pct'] = (bt['Exit_price'] / bt['Entry_price'] - 1) * 100

winner = bt[bt['PnL_pct'] > 0]
loser  = bt[bt['PnL_pct'] <= 0]
print(f"Winner: {len(winner)} ({len(winner)/len(bt)*100:.1f}%)")
print(f"Loser:  {len(loser)} ({len(loser)/len(bt)*100:.1f}%)")

# Merge mit df_full auf Entry-Zeitstempel
df_full = df_full.copy()
if 'Entry_ts' in bt.columns:
    ts_col = 'Entry_ts'
else:
    ts_col = bt.columns[0]
    print(f"Verwende Spalte: {ts_col}")

indicators = ['RSI', 'ADX', 'MACD_hist', 'EMA21', 'Close', 'OBV', 'Volume',
              'Stoch_K', 'Stoch_D', 'SAR_bull', 'OBV_bull', 'cross_ema21_up',
              'DI_plus', 'DI_minus', 'MACD_signal']

available = [c for c in indicators if c in df_full.columns]
print(f"\nVerfügbare Indikatoren: {available}\n")

# Entry-Zeitstempel → df_full Row
bt_merged = bt.copy()
bt_merged.index = pd.to_datetime(bt_merged[ts_col])
df_full.index   = pd.to_datetime(df_full.index)

# Close-zu-EMA21-Distanz berechnen
if 'Close' in df_full.columns and 'EMA21' in df_full.columns:
    df_full['close_ema21_pct'] = (df_full['Close'] / df_full['EMA21'] - 1) * 100
    available.append('close_ema21_pct')

entry_data = df_full.reindex(bt_merged.index, method='nearest')[available]
bt_merged[available] = entry_data.values

bt_merged['winner'] = bt_merged['PnL_pct'] > 0

print("=" * 60)
print("Indikator-Verteilung: Winner vs Loser bei Entry")
print("=" * 60)

for col in available:
    if col in bt_merged.columns:
        w_vals = bt_merged.loc[bt_merged['winner'], col].dropna()
        l_vals = bt_merged.loc[~bt_merged['winner'], col].dropna()
        if len(w_vals) == 0 and len(l_vals) == 0:
            continue
        if bt_merged[col].dtype == bool or set(bt_merged[col].dropna().unique()).issubset({True, False, 0, 1}):
            # Boolean
            w_true = w_vals.mean() if len(w_vals) > 0 else float('nan')
            l_true = l_vals.mean() if len(l_vals) > 0 else float('nan')
            print(f"  {col:<25} Winner: {w_true:.1%}  |  Loser: {l_true:.1%}")
        else:
            w_med = w_vals.median() if len(w_vals) > 0 else float('nan')
            l_med = l_vals.median() if len(l_vals) > 0 else float('nan')
            w_p25 = w_vals.quantile(0.25) if len(w_vals) > 0 else float('nan')
            l_p75 = l_vals.quantile(0.75) if len(l_vals) > 0 else float('nan')
            diff  = w_med - l_med if not (np.isnan(w_med) or np.isnan(l_med)) else float('nan')
            print(f"  {col:<25} Winner Median: {w_med:>8.2f}  |  Loser Median: {l_med:>8.2f}  |  Diff: {diff:>+.2f}")

# Jahres-Analyse
print("\n" + "=" * 60)
print("Jahres-Analyse")
print("=" * 60)
bt_merged['Jahr'] = bt_merged.index.year
for yr, grp in bt_merged.groupby('Jahr'):
    n  = len(grp)
    wr = grp['winner'].mean() * 100
    sp = grp['PnL_pct'].sum()
    print(f"  {yr}: n={n:>3}  WR={wr:>5.1f}%  ΣPnL={sp:>+7.1f}%")

# Exit-Reason
if 'Exit_reason' in bt_merged.columns:
    print("\nExit-Gründe:")
    for reason, grp in bt_merged.groupby('Exit_reason'):
        n  = len(grp)
        wr = grp['winner'].mean() * 100
        sp = grp['PnL_pct'].sum()
        print(f"  {reason:<15}: n={n:>3}  WR={wr:>5.1f}%  ΣPnL={sp:>+7.1f}%")
