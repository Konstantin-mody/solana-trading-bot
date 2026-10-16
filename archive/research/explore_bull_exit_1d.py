"""
explore_bull_exit_1d.py — Bull-Exit-Signal Sweep für den 1d-Bot

Untersucht ob weniger aggressive RSI3-Exit-Schwellen die Winner länger laufen lassen.

Hintergrund:
  - 16/21 Cover-Exits sind RSI3-only (DI_bear nicht aktiv), Ø PnL +23.8%
  -  5/21 Cover-Exits sind DI_bear-only (Verlust-Begrenzung), Ø PnL -8.1%
  - RSI3>75 stoppt auch Trades mit RSI3=76-85 (nicht extremer Überhitzung)
  - Trail ist aktuell toter Code (RSI3>75 feuert immer zuerst)

Getestete Varianten:
  A: Baseline (RSI3>75 OR DI_bear) — Status quo
  B: RSI3>85 OR DI_bear — nur extreme Überhitzung exiten
  C: DI_bear only — RSI3-Exit komplett entfernt, Trail + DI_bear

Pre-committed Acceptance Gates:
  - MaxDD < -15%
  - OOS Bull ΣPnL > +30.7% (aktuell)
  - Gesamt-Rendite > 62% (aktuell)
  - OOS-Sharpe ≥ 0.74 (aktuell)
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
    get_signal, signal_bear_cover,
    SL_TP, TRAIL_MULT, HEBEL, _g
)
from cost_model import FUNDING_1D_PER_CANDLE, STARTKAPITAL


def make_bull_exit(rsi3_thresh):
    """Erstellt eine Bull-Exit-Funktion mit konfigurierbarer RSI3-Schwelle."""
    def signal_bull_exit_custom(row) -> bool:
        rsi3    = _g(row, "RSI3",    50.0)
        di_bear = _g(row, "DI_bear", False)
        if rsi3_thresh is None:
            return di_bear
        return rsi3 > rsi3_thresh or di_bear
    return signal_bull_exit_custom


def run_backtest_custom_exit(
    df_raw: pd.DataFrame,
    bull_exit_fn,
    max_pos_size: float = 0.20,
    min_regime: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_indicators(df_raw)
    df["Regime"] = classify_regime(df, min_regime=min_regime)

    cash          = STARTKAPITAL
    pos: Position | None = None
    equity        = []
    trades        = []
    peak          = STARTKAPITAL
    DD_RESET_BARS = 90
    equity_window: list[float] = []
    cooldown_left = 0
    COOLDOWN      = 3

    for i, (ts, row) in enumerate(df.iterrows()):
        kurs   = row["Close"]
        atr    = row["ATR"]  if not pd.isna(row["ATR"])  else 0.02 * kurs
        vola   = row["Vola"] if not pd.isna(row["Vola"]) else 2.0
        regime = row["Regime"]

        if pos is not None and pos.side == "short":
            funding_cost = pos.entry_cash * FUNDING_1D_PER_CANDLE * pos.hebel
            cash -= funding_cost

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
                    exit_reason = "SL"
                    exit_price  = _sell_price(pos.sl_price, vola)
                elif pos.tp_price is not None and kurs >= pos.tp_price:
                    exit_reason = "TP"
                    exit_price  = _sell_price(pos.tp_price, vola)
                elif bull_exit_fn(row) and exit_reason is None:
                    exit_reason = "Cover"
                    exit_price  = _sell_price(kurs, vola)

                if pos.trail_price is not None and atr > 0 and exit_reason is None:
                    trail_mult  = TRAIL_MULT.get(pos.regime, 3.0) or 3.0
                    new_trail   = kurs - trail_mult * atr
                    pos.trail_price = max(pos.trail_price, new_trail)
                    if kurs <= pos.trail_price:
                        exit_reason = "Trail"
                        exit_price  = _sell_price(pos.trail_price, vola)

            else:
                if kurs >= pos.sl_price:
                    exit_reason = "SL"
                    exit_price  = _buy_price(pos.sl_price, vola)
                elif pos.tp_price is not None and kurs <= pos.tp_price:
                    exit_reason = "TP"
                    exit_price  = _buy_price(pos.tp_price, vola)
                elif signal_bear_cover(row) and exit_reason is None:
                    exit_reason = "Cover"
                    exit_price  = _buy_price(kurs, vola)

            if exit_reason:
                proceeds    = _exit_pnl(pos, exit_price)
                pnl_pct     = (proceeds - pos.entry_cash) / pos.entry_cash * 100
                trades.append({
                    "Entry_ts":    df.index[pos.candle_idx],
                    "Exit_ts":     ts,
                    "Side":        pos.side,
                    "Entry_price": pos.entry_price,
                    "Exit_price":  exit_price,
                    "Hebel":       pos.hebel,
                    "PnL_pct":     pnl_pct,
                    "Entry_cash":  pos.entry_cash,
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
                hebel            = HEBEL.get(regime, 2.0)

                if dd < -0.20:
                    hebel = 1.0

                entry_cash = cash * max_pos_size

                if sig == 1:
                    ep    = _buy_price(kurs, vola)
                    sl    = ep - sl_mult * atr
                    tp    = ep + tp_mult * atr
                    trail = ep - (TRAIL_MULT.get(regime, 3.0) or 3.0) * atr
                    pos   = Position("long", ep, entry_cash, hebel,
                                     sl, tp, trail, regime, i)

                elif sig == -2:
                    ep    = _sell_price(kurs, vola)
                    sl    = ep + sl_mult * atr
                    tp    = ep - tp_mult * atr
                    pos   = Position("short", ep, entry_cash, hebel,
                                     sl, tp, None, regime, i)

                cash -= entry_cash

    equity_df = pd.DataFrame(equity).set_index("ts")["equity"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, equity_df


def report(trades_df, equity_df, label):
    wf = pd.Timestamp(WF_SPLIT)
    rendite  = (equity_df.iloc[-1] / equity_df.iloc[0] - 1) * 100
    rets     = equity_df.pct_change().dropna()
    sharpe   = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0
    roll_max = equity_df.expanding().max()
    dd_ser   = (equity_df - roll_max) / roll_max
    maxdd    = dd_ser.min() * 100

    is_df    = trades_df[trades_df["IS"] == True]
    oos_df   = trades_df[trades_df["IS"] == False]
    bull_is  = is_df[is_df["Regime"]  == "bull"]
    bull_oos = oos_df[oos_df["Regime"] == "bull"]

    oos_eq   = equity_df[equity_df.index >= wf]
    oos_r    = oos_eq.pct_change().dropna()
    oos_sh   = oos_r.mean() / oos_r.std() * np.sqrt(365) if oos_r.std() > 0 else 0

    print(f"\n{'='*60}")
    print(f"Variante: {label}")
    print(f"{'='*60}")
    print(f"  Gesamt:  n={len(trades_df)}, Rendite={rendite:.1f}%, Sharpe={sharpe:.2f}, MaxDD={maxdd:.1f}%")
    print(f"  OOS:     n={len(oos_df)}, Sharpe={oos_sh:.2f}")

    if len(bull_is):
        bi_wr  = (bull_is['PnL_pct']>0).mean()*100
        bi_sum = bull_is['PnL_pct'].sum()
        bi_dur = (bull_is['Exit_ts']-bull_is['Entry_ts']).dt.days.mean()
        exits  = bull_is['Exit_reason'].value_counts().to_dict()
        print(f"  IS Bull: n={len(bull_is)}, WR={bi_wr:.1f}%, ΣPnL={bi_sum:.1f}%, Ø_dur={bi_dur:.1f}d  {exits}")
    if len(bull_oos):
        bo_wr  = (bull_oos['PnL_pct']>0).mean()*100
        bo_sum = bull_oos['PnL_pct'].sum()
        bo_dur = (bull_oos['Exit_ts']-bull_oos['Entry_ts']).dt.days.mean()
        exits  = bull_oos['Exit_reason'].value_counts().to_dict()
        print(f"  OOS Bull: n={len(bull_oos)}, WR={bo_wr:.1f}%, ΣPnL={bo_sum:.1f}%, Ø_dur={bo_dur:.1f}d  {exits}")

    # Gates
    print()
    g1 = maxdd >= -15.0
    g2 = len(bull_oos) > 0 and bull_oos['PnL_pct'].sum() > 30.7
    g3 = rendite > 62.0
    g4 = oos_sh >= 0.74
    print(f"  Gates: MaxDD<-15% {'✅' if g1 else '❌'}  OOS-Bull>30.7% {'✅' if g2 else '❌'}  Rendite>62% {'✅' if g3 else '❌'}  OOS-Sh≥0.74 {'✅' if g4 else '❌'}")


if __name__ == "__main__":
    print("Lade Daten ...")
    df_raw = fetch_binance_1d("2021-09-01")

    variants = [
        ("A: Baseline (RSI3>75)",   make_bull_exit(75)),
        ("B: RSI3>85",              make_bull_exit(85)),
        ("C: DI_bear only (kein RSI3)", make_bull_exit(None)),
    ]

    results = []
    for label, exit_fn in variants:
        print(f"\nLaufe {label} ...")
        t, e = run_backtest_custom_exit(df_raw, exit_fn)
        report(t, e, label)
        results.append((label, t, e))

    print("\n" + "="*60)
    print("Fertig. Gates oben.")
