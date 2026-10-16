"""
explore_cooldown_1d.py — Cooldown-Sweep für den 1d-Bot

Hypothese: 3-Tage-Cooldown nach Exit verhindert schnelle Re-Entries in laufende
Trends. Im Stitch-Muster (z.B. Jan-18 → Jan-24) liegt zwischen Exit und Re-Entry
3 Tage Zwangspause. Mit 1 Tag würden wir dieselben Trends früher re-entrieren.

Getestete Varianten: Cooldown ∈ {0, 1, 2, 3}

Pre-committed Gates:
  - MaxDD < -15%
  - OOS Bull ΣPnL > +30.7% (aktuell)
  - Gesamt-Rendite > 62%
  - OOS-Sharpe ≥ 0.74
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sol_bot_1d.backtest import (
    fetch_binance_1d, WF_SPLIT,
    Position, _buy_price, _sell_price, _exit_pnl, _mtm
)
from sol_bot_1d.indicators import add_indicators
from sol_bot_1d.regime     import classify_regime
from sol_bot_1d.strategy   import (
    get_signal, signal_bull_exit, signal_bear_cover,
    SL_TP, TRAIL_MULT, HEBEL
)
from cost_model import FUNDING_1D_PER_CANDLE, STARTKAPITAL


def run_backtest_cooldown(df_raw, cooldown=3, max_pos_size=0.20, min_regime=3):
    df = add_indicators(df_raw)
    df["Regime"] = classify_regime(df, min_regime=min_regime)

    cash          = STARTKAPITAL
    pos: Position | None = None
    equity        = []
    trades        = []
    equity_window: list[float] = []
    cooldown_left = 0
    DD_RESET_BARS = 90

    for i, (ts, row) in enumerate(df.iterrows()):
        kurs   = row["Close"]
        atr    = row["ATR"]  if not pd.isna(row["ATR"])  else 0.02 * kurs
        vola   = row["Vola"] if not pd.isna(row["Vola"]) else 2.0
        regime = row["Regime"]

        if pos is not None and pos.side == "short":
            cash -= pos.entry_cash * FUNDING_1D_PER_CANDLE * pos.hebel

        port_value = cash if pos is None else (cash + _mtm(pos, kurs))
        equity_window.append(port_value)
        if len(equity_window) > DD_RESET_BARS:
            equity_window.pop(0)
        peak = max(equity_window)
        dd   = (port_value - peak) / peak
        equity.append({"ts": ts, "equity": port_value})

        if pos is not None:
            exit_reason = None
            exit_price  = kurs

            if pos.side == "long":
                if kurs <= pos.sl_price:
                    exit_reason, exit_price = "SL", _sell_price(pos.sl_price, vola)
                elif pos.tp_price is not None and kurs >= pos.tp_price:
                    exit_reason, exit_price = "TP", _sell_price(pos.tp_price, vola)
                elif signal_bull_exit(row):
                    exit_reason, exit_price = "Cover", _sell_price(kurs, vola)

                if pos.trail_price is not None and atr > 0 and exit_reason is None:
                    trail_mult  = TRAIL_MULT.get(pos.regime, 3.0) or 3.0
                    new_trail   = kurs - trail_mult * atr
                    pos.trail_price = max(pos.trail_price, new_trail)
                    if kurs <= pos.trail_price:
                        exit_reason, exit_price = "Trail", _sell_price(pos.trail_price, vola)
            else:
                if kurs >= pos.sl_price:
                    exit_reason, exit_price = "SL", _buy_price(pos.sl_price, vola)
                elif pos.tp_price is not None and kurs <= pos.tp_price:
                    exit_reason, exit_price = "TP", _buy_price(pos.tp_price, vola)
                elif signal_bear_cover(row):
                    exit_reason, exit_price = "Cover", _buy_price(kurs, vola)

            if exit_reason:
                proceeds = _exit_pnl(pos, exit_price)
                pnl_pct  = (proceeds - pos.entry_cash) / pos.entry_cash * 100
                trades.append({
                    "Entry_ts":    df.index[pos.candle_idx],
                    "Exit_ts":     ts,
                    "Side":        pos.side,
                    "Hebel":       pos.hebel,
                    "PnL_pct":     pnl_pct,
                    "Regime":      pos.regime,
                    "Exit_reason": exit_reason,
                    "IS":          df.index[pos.candle_idx] < pd.Timestamp(WF_SPLIT),
                })
                cash         += proceeds
                pos           = None
                cooldown_left = cooldown

        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if pos is None:
            sig = get_signal(row, regime)
            if sig != 0:
                sl_mult, tp_mult = SL_TP.get(regime, (2.0, 8.0))
                hebel            = HEBEL.get(regime, 2.0)
                if dd < -0.20:
                    hebel = 1.0
                entry_cash = cash * max_pos_size

                if sig == 1:
                    ep    = _buy_price(kurs, vola)
                    sl    = ep - sl_mult * atr
                    tp    = ep + tp_mult * atr
                    trail = ep - (TRAIL_MULT.get(regime, 3.0) or 3.0) * atr
                    pos   = Position("long", ep, entry_cash, hebel, sl, tp, trail, regime, i)
                elif sig == -2:
                    ep  = _sell_price(kurs, vola)
                    sl  = ep + sl_mult * atr
                    tp  = ep - tp_mult * atr
                    pos = Position("short", ep, entry_cash, hebel, sl, tp, None, regime, i)

                cash -= entry_cash

    equity_df = pd.DataFrame(equity).set_index("ts")["equity"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, equity_df


def report(trades_df, equity_df, label):
    wf       = pd.Timestamp(WF_SPLIT)
    rendite  = (equity_df.iloc[-1] / equity_df.iloc[0] - 1) * 100
    rets     = equity_df.pct_change().dropna()
    sharpe   = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0
    maxdd    = ((equity_df - equity_df.expanding().max()) / equity_df.expanding().max()).min() * 100

    oos_df   = trades_df[trades_df["IS"] == False]
    bull_oos = oos_df[oos_df["Regime"] == "bull"]
    oos_eq   = equity_df[equity_df.index >= wf]
    oos_sh   = 0.0
    if len(oos_eq) > 1:
        r = oos_eq.pct_change().dropna()
        oos_sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0

    is_df    = trades_df[trades_df["IS"] == True]
    bull_is  = is_df[is_df["Regime"] == "bull"]
    bear_is  = is_df[is_df["Regime"] == "bear"]
    bear_oos = oos_df[oos_df["Regime"] == "bear"]

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    n = len(trades_df)
    wr = (trades_df['PnL_pct']>0).mean()*100 if n else 0
    print(f"  Gesamt:   n={n}, WR={wr:.1f}%, Rendite={rendite:.1f}%, Sharpe={sharpe:.2f}, MaxDD={maxdd:.1f}%")
    print(f"  OOS:      n={len(oos_df)}, Sharpe={oos_sh:.2f}")

    for grp, name in [(bull_is,'IS Bull'), (bear_is,'IS Bear'), (bull_oos,'OOS Bull'), (bear_oos,'OOS Bear')]:
        if len(grp):
            wr_g = (grp['PnL_pct']>0).mean()*100
            sp   = grp['PnL_pct'].sum()
            print(f"  {name}: n={len(grp)}, WR={wr_g:.1f}%, ΣPnL={sp:.1f}%")

    print()
    g1 = maxdd >= -15.0
    g2 = len(bull_oos) > 0 and bull_oos['PnL_pct'].sum() > 30.7
    g3 = rendite > 62.0
    g4 = oos_sh >= 0.74
    print(f"  Gates: MaxDD<-15% {'✅' if g1 else '❌'}  OOS-Bull>30.7% {'✅' if g2 else '❌'}  Rendite>62% {'✅' if g3 else '❌'}  OOS-Sh≥0.74 {'✅' if g4 else '❌'}")
    all_pass = g1 and g2 and g3 and g4
    print(f"  → {'✅ ALLE GATES' if all_pass else '❌ Gates nicht erfüllt'}")


if __name__ == "__main__":
    print("Lade Daten ...")
    df_raw = fetch_binance_1d("2021-09-01")

    for cd in [3, 2, 1, 0]:
        label = f"Cooldown={cd} Tage" + (" (Baseline)" if cd == 3 else "")
        print(f"\nLaufe {label} ...")
        t, e = run_backtest_cooldown(df_raw, cooldown=cd)
        report(t, e, label)

    print("\n" + "="*60)
    print("Fertig.")
