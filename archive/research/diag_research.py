"""
Forschungs-Diagnose: EB Regime-Alter-Diskriminator + Neutral-Long Test
Läuft als eigenständiges Skript (kein Inline-Python-Problem).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_backtest import fetch_binance_data
from sol_bot.backtest import run_backtest, compute_daily_macro
import pandas as pd
import numpy as np

print("Lade Daten...")
df_raw = fetch_binance_data(start="2021-09-01")
extra  = {"Daily_macro_bull": compute_daily_macro(df_raw)}

PROD = dict(
    extra_signals=extra,
    baer_lev_override=6.0, bulle_lev_override=1.0,
    extrem_bulle_lev_override=2.0, extrem_bulle_tp_mult=10.0,
    bulle_mode="rsi3", bulle_tp_mult=10.0, bulle_trail_mult=4.0,
    bulle_lev_high_override=2.0, bulle_adx_thresh_override=40.0,
)

def metrics(trades, equity, label=""):
    df_t = trades
    eq   = pd.Series(equity.values)
    rendite  = (eq.iloc[-1] / 200 - 1) * 100
    ret      = eq.pct_change().dropna()
    sharpe   = ret.mean() / ret.std() * (24*365)**0.5 if ret.std() > 0 else 0
    roll_max = eq.expanding().max()
    maxdd    = ((eq - roll_max) / roll_max).min() * 100
    return {"label": label, "end_eq": eq.iloc[-1],
            "rendite": rendite, "sharpe": sharpe, "maxdd": maxdd,
            "trades": df_t}

# ─── Schritt 1: Produktion (Baseline) ────────────────────────────────────────
print("\nLaufe Produktion...")
t_prod, eq_prod, df_full, _ = run_backtest(df_raw.copy(), sw_mode="neutral_sw_short", **PROD)
base = metrics(t_prod, eq_prod, "Produktion (neutral_sw_short)")

# ─── Schritt 2: EB Regime-Alter Diskriminator ────────────────────────────────
print("\n=== EB REGIME-ALTER DISKRIMINATOR ===")

# Für jeden EB-Trade: berechne regime_age bei Entry aus df_full
df_full_copy = df_full.copy()
df_full_copy["eb_age"] = 0
eb_age_counter = 0
prev_r = None
for ts, row in df_full_copy.iterrows():
    r = row["Regime"]
    if r == "extrem_bulle":
        if prev_r != "extrem_bulle":
            eb_age_counter = 0
        eb_age_counter += 1
        df_full_copy.at[ts, "eb_age"] = eb_age_counter
    else:
        eb_age_counter = 0
    prev_r = r

eb_trades = t_prod[(t_prod["Regime"] == "extrem_bulle") & (t_prod["Side"] == "long")].copy()
eb_trades["pnl_dollar"] = eb_trades["Entry_cash"] * eb_trades["PnL_pct"] / 100.0

# eb_age aus df_full_copy je Trade zuordnen
def get_age(entry_ts):
    ts = pd.Timestamp(entry_ts)
    if ts in df_full_copy.index:
        return df_full_copy.at[ts, "eb_age"]
    # nächster Index
    idx = df_full_copy.index.get_indexer([ts], method='nearest')[0]
    return df_full_copy["eb_age"].iloc[idx]

eb_trades["eb_age_at_entry"] = eb_trades["Entry_ts"].apply(get_age)

print(f"\nEB Trades gesamt: n={len(eb_trades)}")
print("\nAlter bei Entry vs Exit-Reason:")
print(eb_trades.groupby("Exit_reason")["eb_age_at_entry"].describe().round(1))

print("\nEB Age-Buckets (ΣPnL$, WR, n):")
bins   = [0, 12, 24, 48, 96, 9999]
labels = ["≤12h", "13-24h", "25-48h", "49-96h", ">96h"]
eb_trades["age_bucket"] = pd.cut(eb_trades["eb_age_at_entry"], bins=bins, labels=labels)
for bkt in labels:
    sub = eb_trades[eb_trades["age_bucket"] == bkt]
    if len(sub) == 0:
        continue
    wr  = (sub["PnL_pct"] > 0).mean() * 100
    pnl = sub["pnl_dollar"].sum()
    tp_n = (sub["Exit_reason"] == "TP").sum()
    trail_n = (sub["Exit_reason"] == "Trail").sum()
    print(f"  {bkt:8}: n={len(sub):3}, WR={wr:.0f}%, ΣPnL$={pnl:+.0f}, TP={tp_n}, Trail={trail_n}")

# ─── Schritt 3: neutral_sw_full Test ─────────────────────────────────────────
print("\n\nLaufe neutral_sw_full...")
t_nsf, eq_nsf, _, _ = run_backtest(df_raw.copy(), sw_mode="neutral_sw_full", **PROD)
nsf = metrics(t_nsf, eq_nsf, "neutral_sw_full")

# Neutral-Long Detail
nl = t_nsf[(t_nsf["Regime"] == "neutral") & (t_nsf["Side"] == "long")].copy()
nl["year"] = pd.to_datetime(nl["Entry_ts"]).dt.year
nl["pnl_dollar"] = nl["Entry_cash"] * nl["PnL_pct"] / 100.0
print(f"\nneutral_sw_full: Neutral-Long n={len(nl)}, ΣPnL$={nl['pnl_dollar'].sum():+.0f}")
print("Nach Jahr:")
for yr in sorted(nl["year"].unique()):
    sub = nl[nl["year"]==yr]
    wr  = (sub["PnL_pct"]>0).mean()*100
    print(f"  {yr}: n={len(sub)}, WR={wr:.0f}%, ΣPnL$={sub['pnl_dollar'].sum():+.0f}")

# ─── Zusammenfassung ─────────────────────────────────────────────────────────
print("\n\n" + "="*60)
print("ZUSAMMENFASSUNG")
print("="*60)
fmt = "{:45} | {:>10} | {:>8} | {:>7} | {:>8}"
print(fmt.format("Variante", "Endkapital", "Rendite", "Sharpe", "MaxDD"))
print("-"*85)
for r in [base, nsf]:
    print(fmt.format(
        r["label"][:45], f"${r['end_eq']:.0f}",
        f"{r['rendite']:+.1f}%", f"{r['sharpe']:.2f}", f"{r['maxdd']:+.1f}%"
    ))
