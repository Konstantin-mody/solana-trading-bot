"""
explore_lever_size_1d.py — Hebel × Position-Size Sweep für den 1d-Bot

Diagnose vorab (2026-05-07):
  - 0 TP-Exits in 62 Trades → TP-Weitung ist tot, RSI3-Cover feuert immer zuerst
  - 0 Trail-Exits → Trail toter Code
  - MaxDD aktuell ~-8% mit 20% PosSize → Puffer vorhanden

Sweep Teil 1 — Bull-Hebel:    {2.0, 2.5, 3.0, 3.5}× (Bear fix 1.5×)
Sweep Teil 2 — Position-Size: {20%, 22%, 25%, 27%, 30%}  (Bull-Hebel fix 2.0×)

Pre-committed Gates:
  - MaxDD < -15%
  - OOS Bull ΣPnL ≥ 58.7% (Baseline CD=2)
  - Gesamt-Rendite ≥ 63.1% (Baseline)
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

COOLDOWN = 2


def run_backtest_custom(
    df_raw: pd.DataFrame,
    bull_hebel: float = 2.0,
    max_pos_size: float = 0.20,
    min_regime: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                cooldown_left = COOLDOWN

        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if pos is None:
            sig = get_signal(row, regime)
            if sig != 0:
                sl_mult, tp_mult = SL_TP.get(regime, (2.0, 8.0))
                hebel = bull_hebel if regime == "bull" else HEBEL.get(regime, 2.0)
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
                    ep    = _sell_price(kurs, vola)
                    sl    = ep + sl_mult * atr
                    tp    = ep - tp_mult * atr
                    pos   = Position("short", ep, entry_cash, hebel, sl, tp, None, regime, i)

                cash -= entry_cash

    equity_df = pd.DataFrame(equity).set_index("ts")["equity"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, equity_df


def report(trades_df, equity_df, label):
    wf      = pd.Timestamp(WF_SPLIT)
    rendite = (equity_df.iloc[-1] / equity_df.iloc[0] - 1) * 100
    rets    = equity_df.pct_change().dropna()
    sharpe  = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0
    maxdd   = ((equity_df - equity_df.expanding().max()) / equity_df.expanding().max()).min() * 100

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

    g1 = maxdd >= -15.0
    g2 = len(bull_oos) > 0 and bull_oos["PnL_pct"].sum() >= 58.7
    g3 = rendite >= 63.1
    g4 = oos_sh >= 0.74
    all_pass = g1 and g2 and g3 and g4

    flag = "✅" if all_pass else "❌"
    print(f"\n{flag} {label}")
    print(f"   Rendite={rendite:.1f}%  Sharpe={sharpe:.2f}  MaxDD={maxdd:.1f}%  n={len(trades_df)}")
    print(f"   OOS-Sharpe={oos_sh:.2f}  OOS-Bull-ΣPnL={bull_oos['PnL_pct'].sum():.1f}% (n={len(bull_oos)})")
    print(f"   Gates: MaxDD<-15% {'✅' if g1 else '❌'}  OOS-Bull≥58.7% {'✅' if g2 else '❌'}  Rendite≥63.1% {'✅' if g3 else '❌'}  OOS-Sh≥0.74 {'✅' if g4 else '❌'}")

    return {
        "label": label, "rendite": rendite, "sharpe": sharpe,
        "maxdd": maxdd, "oos_sh": oos_sh,
        "oos_bull_sum": bull_oos["PnL_pct"].sum() if len(bull_oos) else 0,
        "all_pass": all_pass,
    }


if __name__ == "__main__":
    print("Lade Daten ...")
    df_raw = fetch_binance_1d("2021-09-01")

    results = []

    # ── Teil 1: Bull-Hebel-Sweep (PosSize fix 20%) ───────────────────────────
    print("\n" + "="*60)
    print("TEIL 1 — Bull-Hebel-Sweep (PosSize=20%, Bear=1.5×)")
    print("="*60)
    for hebel in [2.0, 2.5, 3.0, 3.5]:
        label = f"Bull-Hebel={hebel:.1f}×" + (" (Baseline)" if hebel == 2.0 else "")
        t, e = run_backtest_custom(df_raw, bull_hebel=hebel, max_pos_size=0.20)
        r = report(t, e, label)
        results.append(r)

    # ── Teil 2: Position-Size-Sweep (Bull-Hebel fix 2.0×) ───────────────────
    print("\n" + "="*60)
    print("TEIL 2 — Position-Size-Sweep (Bull-Hebel=2.0×)")
    print("="*60)
    for size in [0.20, 0.22, 0.25, 0.27, 0.30]:
        label = f"PosSize={size*100:.0f}%" + (" (Baseline)" if size == 0.20 else "")
        t, e = run_backtest_custom(df_raw, bull_hebel=2.0, max_pos_size=size)
        r = report(t, e, label)
        results.append(r)

    # ── Zusammenfassung ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG — alle Gates-bestehenden Varianten:")
    print("="*60)
    passing = [r for r in results if r["all_pass"]]
    if passing:
        for r in sorted(passing, key=lambda x: x["rendite"], reverse=True):
            print(f"  ✅ {r['label']:35s}  Rendite={r['rendite']:.1f}%  MaxDD={r['maxdd']:.1f}%  OOS-Sh={r['oos_sh']:.2f}")
    else:
        print("  ❌ Keine Variante besteht alle Gates.")
    print()
