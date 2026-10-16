"""
diagnose_fib_setups.py — Profitabilitäts-Check der Fib-Setups.

Simuliert Fib-Signaltage (RSI3<35 + at_fib + Close>MA200) als Trades
durch die existierende Exit-Logik — ohne den Signal-Path zu ändern.

Frage: Sind die Setups in Bull UND Neutral gewinnträchtig?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sol_bot_1d.backtest   import fetch_binance_1d, WF_SPLIT, _buy_price, _sell_price, STARTKAPITAL
from sol_bot_1d.indicators import add_indicators
from sol_bot_1d.regime     import classify_regime
from sol_bot_1d.strategy   import signal_bull_exit, SL_TP, TRAIL_MULT, _g
from cost_model            import FUNDING_1D_PER_CANDLE

HEBEL       = 2.0
MAX_HOLD    = 30    # max. 30 Tage halten, dann Zwangs-Cover

print("Lade Daten …")
df_raw = fetch_binance_1d("2021-09-01")
df     = add_indicators(df_raw)
df["Regime"] = classify_regime(df)


def simulate_entries(signal_mask: pd.Series, label: str) -> pd.DataFrame:
    """
    Simuliert alle Signal-Tage als eigenständige Trades (keine Cooldown-Interaktion).
    Exit: signal_bull_exit OR SL (2×ATR) OR TP (8×ATR) OR max. 30 Tage.
    """
    trades = []
    entry_days = df[signal_mask].index

    for entry_ts in entry_days:
        if entry_ts not in df.index:
            continue
        entry_row = df.loc[entry_ts]
        atr_e     = entry_row["ATR"] if not pd.isna(entry_row["ATR"]) else entry_row["Close"] * 0.02
        vola_e    = entry_row["Vola"] if not pd.isna(entry_row["Vola"]) else 2.0
        ep        = _buy_price(entry_row["Close"], vola_e)

        sl_mult, tp_mult = SL_TP.get("bull", (2.0, 8.0))
        sl    = ep - sl_mult * atr_e
        tp    = ep + tp_mult * atr_e
        trail = ep - (TRAIL_MULT.get("bull", 3.0) or 3.0) * atr_e

        exit_ts     = None
        exit_price  = None
        exit_reason = None
        cash_sim    = 100.0   # normiert auf 100$
        entry_cash  = cash_sim

        # Durch folgende Kerzen iterieren
        future = df.loc[entry_ts:].iloc[1: MAX_HOLD + 1]
        for fts, frow in future.iterrows():
            kurs = frow["Close"]
            atr  = frow["ATR"] if not pd.isna(frow["ATR"]) else kurs * 0.02
            vola = frow["Vola"] if not pd.isna(frow["Vola"]) else 2.0

            if kurs <= sl:
                exit_reason = "SL"
                exit_price  = _sell_price(sl, vola)
                exit_ts     = fts
                break
            elif kurs >= tp:
                exit_reason = "TP"
                exit_price  = _sell_price(tp, vola)
                exit_ts     = fts
                break
            elif signal_bull_exit(frow):
                exit_reason = "Cover"
                exit_price  = _sell_price(kurs, vola)
                exit_ts     = fts
                break

            # Trail-Update
            trail_mult = TRAIL_MULT.get("bull", 3.0) or 3.0
            new_trail  = kurs - trail_mult * atr
            trail      = max(trail, new_trail)
            if kurs <= trail and trail > sl:
                exit_reason = "Trail"
                exit_price  = _sell_price(trail, vola)
                exit_ts     = fts
                break
        else:
            # Max-Hold erreicht
            last = future.iloc[-1] if len(future) > 0 else entry_row
            exit_reason = "MaxHold"
            exit_price  = _sell_price(last["Close"], last["Vola"] if not pd.isna(last["Vola"]) else 2.0)
            exit_ts     = future.index[-1] if len(future) > 0 else entry_ts

        if exit_price is None:
            continue

        # PnL berechnen (vereinfacht: keine Funding da Long)
        from sol_bot_1d.backtest import _exit_pnl, Position
        pos = Position("long", ep, entry_cash, HEBEL, sl, tp, trail, "bull", 0)
        proceeds = _exit_pnl(pos, exit_price)
        pnl_pct  = (proceeds - entry_cash) / entry_cash * 100

        trades.append({
            "Entry_ts":    entry_ts,
            "Exit_ts":     exit_ts,
            "Exit_reason": exit_reason,
            "PnL_pct":     pnl_pct,
            "IS":          exit_ts < pd.Timestamp(WF_SPLIT) if exit_ts else True,
            "Regime":      df.loc[entry_ts, "Regime"],
            "RSI3":        df.loc[entry_ts, "RSI3"],
        })

    return pd.DataFrame(trades)


def report(df_t: pd.DataFrame, label: str):
    if df_t.empty:
        print(f"  {label}: keine Trades")
        return
    is_  = df_t[df_t["IS"]]
    oos_ = df_t[~df_t["IS"]]

    def sp(g):  return g["PnL_pct"].sum() if len(g) > 0 else float("nan")
    def wr(g):  return (g["PnL_pct"] > 0).mean() * 100 if len(g) > 0 else float("nan")
    def med_w(g): return g.loc[g["PnL_pct"]>0, "PnL_pct"].median() if (g["PnL_pct"]>0).any() else float("nan")
    def med_l(g): return g.loc[g["PnL_pct"]<=0, "PnL_pct"].median() if (g["PnL_pct"]<=0).any() else float("nan")

    print(f"\n  [{label}]  n={len(df_t)}  (IS={len(is_)}, OOS={len(oos_)})")
    print(f"    Gesamt:  WR={wr(df_t):>5.1f}%  ΣPnL={sp(df_t):>+7.1f}%  "
          f"Win-Med={med_w(df_t):>+6.1f}%  Los-Med={med_l(df_t):>+6.1f}%")
    print(f"    IS:      WR={wr(is_):>5.1f}%  ΣPnL={sp(is_):>+7.1f}%  "
          f"Win-Med={med_w(is_):>+6.1f}%  Los-Med={med_l(is_):>+6.1f}%")
    print(f"    OOS:     WR={wr(oos_):>5.1f}%  ΣPnL={sp(oos_):>+7.1f}%  "
          f"Win-Med={med_w(oos_):>+6.1f}%  Los-Med={med_l(oos_):>+6.1f}%")

    print(f"    Exit-Mix: {df_t['Exit_reason'].value_counts().to_dict()}")


# ─── Masken definieren ────────────────────────────────────────────────────────
fib  = df["at_fib_level"].fillna(False)
r35  = df["RSI3"] < 35
ma2  = df["MA200"].isna() | (df["MA200"] == 0) | (df["Close"] > df["MA200"])
bull_reg    = df["Regime"] == "bull"
neutral_reg = df["Regime"] == "neutral"
bear_reg    = df["Regime"] == "bear"

core = fib & r35 & ma2

print("\n" + "=" * 68)
print("  FIB-SETUP PROFITABILITÄTS-DIAGNOSE")
print("  (jeder Signal-Tag = eigenständiger Trade, Max-Hold 30 Tage)")
print("=" * 68)

t_bull    = simulate_entries(core & bull_reg,    "Bull-Regime")
t_neutral = simulate_entries(core & neutral_reg, "Neutral-Regime")
t_bear    = simulate_entries(core & bear_reg,    "Bear-Regime (Referenz)")
t_all     = simulate_entries(core,               "Alle Regime")

report(t_bull,    "Bull-Regime   ")
report(t_neutral, "Neutral-Regime")
report(t_bear,    "Bear-Regime   ")
report(t_all,     "Alle Regime   ")
