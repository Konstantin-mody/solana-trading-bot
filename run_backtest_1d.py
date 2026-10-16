"""
run_backtest_1d.py — Einstiegspunkt für den 1d-Bot-Backtest

Verwendung:
    python3 run_backtest_1d.py                  # Standard-Lauf
    python3 run_backtest_1d.py --min_regime 2   # MIN_REGIME-Sweep
    python3 run_backtest_1d.py --min_regime 4
    python3 run_backtest_1d.py --wf             # Walk-Forward-Bericht (IS vs. OOS)
"""

import argparse
import numpy as np
import pandas as pd

from sol_bot_1d.backtest import run_backtest, fetch_binance_1d, WF_SPLIT, STARTKAPITAL
from sol_bot_1w.backtest import fetch_binance_1w
from sol_bot_1w.indicators import add_indicators as add_indicators_1w
from sol_bot_1w.regime import classify_regime as classify_regime_1w


def compute_macro_regime(target_index: pd.DatetimeIndex) -> pd.Series:
    """1W-MA50-Regime täglich resampelt (kein Look-Ahead: .shift(1))."""
    df_1w = fetch_binance_1w("2020-01-01")
    df_1w = add_indicators_1w(df_1w)
    weekly_regime = classify_regime_1w(df_1w)
    # shift(1): Regime der abgeschlossenen Woche, nicht der laufenden
    weekly_regime = weekly_regime.shift(1)
    return weekly_regime.reindex(target_index, method="ffill")


def sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    return (rets.mean() / rets.std()) * np.sqrt(365)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd   = (equity - peak) / peak
    return dd.min() * 100


def print_report(label: str, trades_df: pd.DataFrame, equity: pd.Series, df_raw: pd.DataFrame,
                 start_capital: float = None):
    final        = equity.iloc[-1]
    start_cap    = start_capital if start_capital is not None else equity.iloc[0]
    rendite      = (final - start_cap) / start_cap * 100
    bah_ret      = (df_raw["Close"].iloc[-1] / df_raw["Close"].iloc[0] - 1) * 100
    sr      = sharpe(equity)
    dd      = max_drawdown(equity)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Endkapital:    ${final:.2f}")
    print(f"  Rendite:       {rendite:+.1f}%  (BaH: {bah_ret:+.1f}%)")
    print(f"  Sharpe:        {sr:.2f}")
    print(f"  Max Drawdown:  {dd:.1f}%")

    if trades_df.empty:
        print("  Keine Trades.")
        return

    n   = len(trades_df)
    wr  = (trades_df["PnL_pct"] > 0).mean() * 100
    avg = trades_df["PnL_pct"].mean()
    print(f"  Trades:        {n}  |  WR: {wr:.1f}%  |  Ø PnL: {avg:+.2f}%")

    if "Regime" in trades_df.columns:
        print()
        for regime, grp in trades_df.groupby("Regime"):
            r_wr  = (grp["PnL_pct"] > 0).mean() * 100
            r_sum = grp["PnL_pct"].sum()
            print(f"    {regime:<10}  n={len(grp):>3}  WR={r_wr:.0f}%  ΣPnL={r_sum:+.1f}%")

    if "Exit_reason" in trades_df.columns:
        print()
        for reason, grp in trades_df.groupby("Exit_reason"):
            r_wr = (grp["PnL_pct"] > 0).mean() * 100
            print(f"    {reason:<8}  n={len(grp):>3}  WR={r_wr:.0f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_regime", type=int, default=3,
                        help="MIN_REGIME Hysterese-Sweep (default 3, try 2 / 4)")
    parser.add_argument("--wf", action="store_true",
                        help="Walk-Forward-Bericht: IS vs. OOS separat")
    parser.add_argument("--regime_sweep", action="store_true",
                        help="Sweep MIN_REGIME {2, 3, 4} auf einmal")
    parser.add_argument("--macro_filter", action="store_true",
                        help="1W-MA50-Regime als Richtungsfilter (Vergleich mit/ohne)")
    parser.add_argument("--bear_trail_sweep", action="store_true",
                        help="Bear-Trail-Multiplikator Sweep: None, 1.5, 2.0, 2.5, 3.0, 4.0")
    args = parser.parse_args()

    print("Lade Binance 1d SOL/USDT-Daten …")
    df_raw = fetch_binance_1d("2021-09-01")
    print(f"  {len(df_raw)} Tageskerzen | {df_raw.index[0].date()} → {df_raw.index[-1].date()}\n")

    if args.regime_sweep:
        print(f"{'MIN_REGIME':<12} {'Trades':>7} {'WR%':>6} {'Rendite%':>9} {'Sharpe':>7} {'MaxDD%':>7}")
        print("─" * 55)
        for mr in [2, 3, 4]:
            t, eq = run_backtest(df_raw, min_regime=mr)
            if t.empty:
                print(f"  mr={mr}  keine Trades")
                continue
            wr = (t["PnL_pct"] > 0).mean() * 100
            r  = (eq.iloc[-1] - STARTKAPITAL) / STARTKAPITAL * 100
            sr = sharpe(eq)
            dd = max_drawdown(eq)
            print(f"  {mr:<10} {len(t):>7} {wr:>5.1f}% {r:>+8.1f}% {sr:>7.2f} {dd:>6.1f}%")
        return

    if args.macro_filter:
        print("Lade 1W-Macro-Regime …")
        macro = compute_macro_regime(df_raw.index)
        print(f"  1W-Regime-Verteilung: bull={( macro=='bull').sum()}d, bear={(macro=='bear').sum()}d, NaN={(macro.isna()).sum()}d\n")
        print("─── OHNE Macro-Filter ───────────────────────────────────")
        t0, eq0 = run_backtest(df_raw, min_regime=args.min_regime)
        print_report("1d-Bot (ohne Filter)", t0, eq0, df_raw)
        print("\n─── MIT Macro-Filter (1W-MA50) ──────────────────────────")
        t1, eq1 = run_backtest(df_raw, min_regime=args.min_regime, macro_regime=macro)
        print_report("1d-Bot (mit Macro-Filter)", t1, eq1, df_raw)
        return

    if args.bear_trail_sweep:
        sweep_vals = [None, 1.5, 2.0, 2.5, 3.0, 4.0]
        print(f"\n{'Trail':>6}  {'Rendite':>8}  {'Sharpe':>7}  {'MaxDD':>6}  "
              f"{'Bear IS n':>9}  {'Bear IS ΣPnL':>12}  {'Bear OOS n':>10}  {'Bear OOS ΣPnL':>13}  "
              f"{'Trail n':>7}  {'Cover n':>7}  {'SL n':>5}")
        print("─" * 115)
        for tm in sweep_vals:
            t, eq = run_backtest(df_raw, min_regime=args.min_regime, bear_trail_mult=tm)
            if t.empty:
                label = f"{'off':>6}" if tm is None else f"{tm:>6.1f}"
                print(f"{label}  {'no trades':>8}")
                continue
            r  = (eq.iloc[-1] - STARTKAPITAL) / STARTKAPITAL * 100
            sr = sharpe(eq)
            dd = max_drawdown(eq)

            bear = t[t["Regime"] == "bear"]
            bear_is  = bear[bear["IS"]]
            bear_oos = bear[~bear["IS"]]
            bear_is_sum  = bear_is["PnL_pct"].sum()
            bear_oos_sum = bear_oos["PnL_pct"].sum()

            trail_n  = (t["Exit_reason"] == "Trail").sum()
            cover_n  = (t["Exit_reason"] == "Cover").sum()
            sl_n     = (t["Exit_reason"] == "SL").sum()

            label = "off" if tm is None else f"{tm:.1f}"
            print(f"{label:>6}  {r:>+7.1f}%  {sr:>7.2f}  {dd:>5.1f}%  "
                  f"{len(bear_is):>9}  {bear_is_sum:>+11.1f}%  {len(bear_oos):>10}  {bear_oos_sum:>+12.1f}%  "
                  f"{trail_n:>7}  {cover_n:>7}  {sl_n:>5}")
        return

    trades_df, equity = run_backtest(df_raw, min_regime=args.min_regime)
    print_report(f"1d-Bot  (MIN_REGIME={args.min_regime})", trades_df, equity, df_raw)

    if args.wf and not trades_df.empty and "IS" in trades_df.columns:
        split = pd.Timestamp(WF_SPLIT)
        print(f"\n{'─'*55}")
        print(f"  Walk-Forward  (Split: {WF_SPLIT})")

        is_trades = trades_df[trades_df["IS"]]
        is_eq     = equity[equity.index < split]
        if not is_trades.empty and len(is_eq) > 1:
            print_report("  In-Sample", is_trades, is_eq, df_raw[df_raw.index < split])

        oos_trades = trades_df[~trades_df["IS"]]
        oos_eq     = equity[equity.index >= split]
        if not oos_trades.empty and len(oos_eq) > 1:
            print_report("  Out-of-Sample", oos_trades, oos_eq, df_raw[df_raw.index >= split])


if __name__ == "__main__":
    main()
