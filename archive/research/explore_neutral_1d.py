"""
explore_neutral_1d.py — Neutral-Regime-Signal Exploration für den 1d-Bot

Testet Long-Only-Signale im neutralen Regime (ADX≤20 / gemischte Richtung).
Keine Änderungen an Produktionscode — alles in-process.

Ansätze:
  A: RSI3 < 25 (einfach, mehr Trades)
  B: RSI3 < 25 + at_fib_level (selektiver, weniger Rauschen)
  C: RSI3 < 25 + OBV_bull (Volumen-Bestätigung)

Pre-committed Acceptance Gates (vor dem Lauf):
  - MaxDD < -15%
  - OOS Neutral ΣPnL > 0%
  - OOS Neutral n ≥ 10
  - OOS-Sharpe ≥ 0.74 (kein Rückschritt)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sol_bot_1d.backtest import (
    fetch_binance_1d, run_backtest, WF_SPLIT,
    Position, _buy_price, _sell_price, _exit_pnl, _mtm, GEBUEHR
)
from sol_bot_1d.indicators import add_indicators
from sol_bot_1d.regime     import classify_regime
from sol_bot_1d.strategy   import (
    get_signal, signal_bull_exit, signal_bear_cover,
    SL_TP, TRAIL_MULT, HEBEL, _g
)
from cost_model import FUNDING_1D_PER_CANDLE, SLIPPAGE, SPREAD, STARTKAPITAL

# ── Neutral-Signal-Definitionen ───────────────────────────────────────────────

def signal_neutral_A(row) -> int:
    """RSI3 < 25 — einfach, mehr Trades."""
    rsi3      = _g(row, "RSI3",      50.0)
    atr_spike = _g(row, "ATR_spike", 1.0)
    if rsi3 < 25 and atr_spike < 2.0:
        return 1
    return 0


def signal_neutral_B(row) -> int:
    """RSI3 < 30 + Fibonacci-Level — selektiver."""
    rsi3      = _g(row, "RSI3",         50.0)
    at_fib    = _g(row, "at_fib_level", False)
    atr_spike = _g(row, "ATR_spike",    1.0)
    if rsi3 < 30 and at_fib and atr_spike < 2.0:
        return 1
    return 0


def signal_neutral_C(row) -> int:
    """RSI3 < 25 + OBV_bull — Volumen-Bestätigung."""
    rsi3      = _g(row, "RSI3",      50.0)
    obv_bull  = _g(row, "OBV_bull",  False)
    atr_spike = _g(row, "ATR_spike", 1.0)
    if rsi3 < 25 and obv_bull and atr_spike < 2.0:
        return 1
    return 0


def signal_neutral_exit(row) -> bool:
    """Exit Neutral-Long: RSI3 erholt sich stark oder Trend dreht bärisch."""
    rsi3    = _g(row, "RSI3",    50.0)
    di_bear = _g(row, "DI_bear", False)
    return rsi3 > 65 or di_bear


# ── Backtest mit patchbarem Neutral-Signal ────────────────────────────────────

def run_backtest_with_neutral(
    df_raw: pd.DataFrame,
    neutral_signal_fn,
    max_pos_size: float = 0.20,
    min_regime: int = 3,
    neutral_sl_mult: float = 1.5,
    neutral_tp_mult: float = 5.0,
    neutral_hebel:   float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vollständiger Backtest mit optionalem Neutral-Long-Signal.
    Nur Long im neutralen Regime; Exit via signal_neutral_exit.
    """
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

        # ── Funding (Short × 3 Cycles/Tag) ──────────────────────────────
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

        # ── SL / TP / Trail / Cover ──────────────────────────────────────
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
                else:
                    # Exit-Signal je nach Regime des Trades
                    if pos.regime == "neutral":
                        if signal_neutral_exit(row):
                            exit_reason = "Cover"
                            exit_price  = _sell_price(kurs, vola)
                    else:
                        if signal_bull_exit(row) and exit_reason is None:
                            exit_reason = "Cover"
                            exit_price  = _sell_price(kurs, vola)

                # Trail-Update (nur für Bull-Regime-Trades)
                if pos.trail_price is not None and atr > 0 and exit_reason is None:
                    trail_mult  = TRAIL_MULT.get(pos.regime, 3.0) or 3.0
                    new_trail   = kurs - trail_mult * atr
                    pos.trail_price = max(pos.trail_price, new_trail)
                    if kurs <= pos.trail_price:
                        exit_reason = "Trail"
                        exit_price  = _sell_price(pos.trail_price, vola)

            else:  # short
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

        # ── Cooldown ─────────────────────────────────────────────────────
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        # ── Entry ────────────────────────────────────────────────────────
        if pos is None:
            sig = get_signal(row, regime)

            # Neutral-Signal (Long only)
            if sig == 0 and regime == "neutral":
                sig = neutral_signal_fn(row)

            if sig != 0:
                if regime == "neutral" and sig == 1:
                    sl_mult = neutral_sl_mult
                    tp_mult = neutral_tp_mult
                    hebel   = neutral_hebel
                    trail_p = None  # Kein Trail für Neutral (Bounce-Trade)
                else:
                    sl_mult, tp_mult = SL_TP.get(regime, (2.0, 8.0))
                    hebel            = HEBEL.get(regime, 2.0)

                if dd < -0.20:
                    hebel = 1.0

                entry_cash = cash * max_pos_size

                if sig == 1:  # long
                    ep    = _buy_price(kurs, vola)
                    sl    = ep - sl_mult * atr
                    tp    = ep + tp_mult * atr
                    if regime != "neutral":
                        trail_p = ep - (TRAIL_MULT.get(regime, 3.0) or 3.0) * atr
                    pos   = Position("long", ep, entry_cash, hebel,
                                     sl, tp, trail_p, regime, i)

                elif sig == -2:  # short
                    ep    = _sell_price(kurs, vola)
                    sl    = ep + sl_mult * atr
                    tp    = ep - tp_mult * atr
                    pos   = Position("short", ep, entry_cash, hebel,
                                     sl, tp, None, regime, i)

                cash -= entry_cash

    equity_df = pd.DataFrame(equity).set_index("ts")["equity"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, equity_df


# ── Metriken ─────────────────────────────────────────────────────────────────

def metrics(trades_df, equity_df, label=""):
    wf = pd.Timestamp(WF_SPLIT)

    rendite  = (equity_df.iloc[-1] / equity_df.iloc[0] - 1) * 100
    rets     = equity_df.pct_change().dropna()
    sharpe   = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0
    roll_max = equity_df.expanding().max()
    dd_ser   = (equity_df - roll_max) / roll_max
    maxdd    = dd_ser.min() * 100

    n_total  = len(trades_df)
    wr_total = (trades_df["PnL_pct"] > 0).mean() * 100 if n_total else 0

    is_df    = trades_df[trades_df["IS"] == True]
    oos_df   = trades_df[trades_df["IS"] == False]

    is_neu   = is_df[is_df["Regime"]  == "neutral"]
    oos_neu  = oos_df[oos_df["Regime"] == "neutral"]

    print(f"\n{'='*60}")
    print(f"Signal: {label}")
    print(f"{'='*60}")
    print(f"  Gesamt:   n={n_total}, WR={wr_total:.1f}%, Rendite={rendite:.1f}%, Sharpe={sharpe:.2f}, MaxDD={maxdd:.1f}%")
    print()

    if len(is_df):
        is_rendite = is_df["PnL_pct"].sum()
        is_wr      = (is_df["PnL_pct"] > 0).mean() * 100
        print(f"  IS:       n={len(is_df)}, WR={is_wr:.1f}%, ΣPnL={is_rendite:.1f}%")
    if len(oos_df):
        oos_rendite = oos_df["PnL_pct"].sum()
        oos_wr      = (oos_df["PnL_pct"] > 0).mean() * 100
        print(f"  OOS:      n={len(oos_df)}, WR={oos_wr:.1f}%, ΣPnL={oos_rendite:.1f}%")
    print()

    if len(is_neu):
        in_wr   = (is_neu["PnL_pct"] > 0).mean() * 100
        in_sum  = is_neu["PnL_pct"].sum()
        in_mean = is_neu["PnL_pct"].mean()
        print(f"  IS  Neutral: n={len(is_neu)}, WR={in_wr:.1f}%, ΣPnL={in_sum:.1f}%, Ø={in_mean:.2f}%")
    if len(oos_neu):
        on_wr   = (oos_neu["PnL_pct"] > 0).mean() * 100
        on_sum  = oos_neu["PnL_pct"].sum()
        on_mean = oos_neu["PnL_pct"].mean()
        print(f"  OOS Neutral: n={len(oos_neu)}, WR={on_wr:.1f}%, ΣPnL={on_sum:.1f}%, Ø={on_mean:.2f}%")
    else:
        print(f"  OOS Neutral: n=0 — ❌ Gate (n≥10) nicht erfüllt")

    # Gate-Check
    print()
    print("  Gates:")
    g1 = maxdd >= -15.0
    print(f"    MaxDD < -15%:          {'✅' if g1 else '❌'} ({maxdd:.1f}%)")
    g2 = len(oos_neu) >= 10
    print(f"    OOS Neutral n ≥ 10:    {'✅' if g2 else '❌'} (n={len(oos_neu)})")
    if len(oos_neu):
        oos_neu_pnl = oos_neu["PnL_pct"].sum()
        g3 = oos_neu_pnl > 0
        print(f"    OOS Neutral ΣPnL > 0:  {'✅' if g3 else '❌'} ({oos_neu_pnl:.1f}%)")
    sharpe_oos = None
    if len(oos_df):
        oos_eq = equity_df[equity_df.index >= wf]
        oos_r  = oos_eq.pct_change().dropna()
        sharpe_oos = oos_r.mean() / oos_r.std() * np.sqrt(365) if oos_r.std() > 0 else 0
        g4 = sharpe_oos >= 0.74
        print(f"    OOS-Sharpe ≥ 0.74:     {'✅' if g4 else '❌'} ({sharpe_oos:.2f})")

    return trades_df, equity_df


# ── Hauptlauf ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Lade Daten ...")
    df_raw = fetch_binance_1d("2021-09-01")

    # Baseline (kein Neutral-Signal)
    print("\nBaseline läuft ...")
    t_base, e_base = run_backtest(df_raw)
    metrics(t_base, e_base, "Baseline (kein Neutral)")

    # Signal A: RSI3 < 25
    print("\nSignal A läuft ...")
    t_a, e_a = run_backtest_with_neutral(df_raw, signal_neutral_A)
    metrics(t_a, e_a, "A: RSI3<25")

    # Signal B: RSI3 < 30 + Fibonacci-Level
    print("\nSignal B läuft ...")
    t_b, e_b = run_backtest_with_neutral(df_raw, signal_neutral_B)
    metrics(t_b, e_b, "B: RSI3<30 + Fib-Level")

    # Signal C: RSI3 < 25 + OBV_bull
    print("\nSignal C läuft ...")
    t_c, e_c = run_backtest_with_neutral(df_raw, signal_neutral_C)
    metrics(t_c, e_c, "C: RSI3<25 + OBV_bull")

    print("\n" + "="*60)
    print("Fertig. Gates-Zusammenfassung oben.")
