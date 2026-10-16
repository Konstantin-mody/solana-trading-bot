"""
run_backtest_4h.py — Einstiegspunkt für den 4h-Bot-Backtest

Verwendung:
    python3 run_backtest_4h.py          # Standard-Lauf
    python3 run_backtest_4h.py --wf     # Walk-Forward (IS vs. OOS)
"""

import argparse
import numpy as np
import pandas as pd

from sol_bot_4h.backtest import run_backtest, fetch_binance_4h, WF_SPLIT, STARTKAPITAL
from sol_bot_1w.backtest import fetch_binance_1w
from sol_bot_1w.indicators import add_indicators as add_indicators_1w
from sol_bot_1w.regime import classify_regime as classify_regime_1w


def compute_macro_regime(target_index: pd.DatetimeIndex) -> pd.Series:
    df_1w = fetch_binance_1w("2020-01-01")
    df_1w = add_indicators_1w(df_1w)
    weekly_regime = classify_regime_1w(df_1w)
    weekly_regime = weekly_regime.shift(1)
    return weekly_regime.reindex(target_index, method="ffill")

CANDLES_PER_DAY = 6   # 24h / 4h


def sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    return (rets.mean() / rets.std()) * np.sqrt(365 * CANDLES_PER_DAY)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd   = (equity - peak) / peak
    return dd.min() * 100


def print_report(label: str, trades_df: pd.DataFrame, equity: pd.Series,
                 df_raw: pd.DataFrame, start_capital: float = None):
    final     = equity.iloc[-1]
    start_cap = start_capital if start_capital is not None else equity.iloc[0]
    rendite   = (final - start_cap) / start_cap * 100
    bah_ret   = (df_raw["Close"].iloc[-1] / df_raw["Close"].iloc[0] - 1) * 100
    sr        = sharpe(equity)
    dd        = max_drawdown(equity)

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
    parser.add_argument("--wf", action="store_true",
                        help="Walk-Forward-Bericht: IS vs. OOS separat")
    parser.add_argument("--min_regime", type=int, default=6)
    parser.add_argument("--macro_filter", action="store_true",
                        help="1W-MA50-Regime als Richtungsfilter (Vergleich mit/ohne)")
    args = parser.parse_args()

    print("Lade Binance 4h SOL/USDT-Daten …")
    df_raw = fetch_binance_4h("2021-09-01")
    print(f"  {len(df_raw)} 4h-Kerzen | {df_raw.index[0].date()} → {df_raw.index[-1].date()}\n")

    if args.macro_filter:
        print("Lade 1W-Macro-Regime …")
        macro = compute_macro_regime(df_raw.index)
        print(f"  1W-Regime-Verteilung: bull={(macro=='bull').sum()}h, bear={(macro=='bear').sum()}h, NaN={(macro.isna()).sum()}h\n")
        print("─── OHNE Macro-Filter ───────────────────────────────────")
        t0, eq0 = run_backtest(df_raw, min_regime=args.min_regime)
        print_report("4h-Bot (ohne Filter)", t0, eq0, df_raw, start_capital=STARTKAPITAL)
        print("\n─── MIT Macro-Filter (1W-MA50) ──────────────────────────")
        t1, eq1 = run_backtest(df_raw, min_regime=args.min_regime, macro_regime=macro)
        print_report("4h-Bot (mit Macro-Filter)", t1, eq1, df_raw, start_capital=STARTKAPITAL)
        return

    trades_df, equity = run_backtest(df_raw, min_regime=args.min_regime)
    print_report(f"4h-Bot  (MIN_REGIME={args.min_regime})", trades_df, equity, df_raw,
                 start_capital=STARTKAPITAL)

    if args.wf and not trades_df.empty and "IS" in trades_df.columns:
        split = pd.Timestamp(WF_SPLIT)
        print(f"\n{'─'*55}")
        print(f"  Walk-Forward  (Split: {WF_SPLIT})")

        is_trades = trades_df[trades_df["IS"]]
        is_eq     = equity[equity.index < split]
        if not is_trades.empty and len(is_eq) > 1:
            print_report("  In-Sample", is_trades, is_eq, df_raw[df_raw.index < split],
                         start_capital=STARTKAPITAL)

        oos_trades = trades_df[~trades_df["IS"]]
        oos_eq     = equity[equity.index >= split]
        if not oos_trades.empty and len(oos_eq) > 1:
            oos_start = is_eq.iloc[-1] if len(is_eq) > 0 else STARTKAPITAL
            print_report("  Out-of-Sample", oos_trades, oos_eq,
                         df_raw[df_raw.index >= split], start_capital=oos_start)


if __name__ == "__main__":
    main()
