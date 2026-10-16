"""
run_backtest_1w.py — Einstiegspunkt für den 1W-Bot-Backtest

Verwendung:
    python3 run_backtest_1w.py          # Standard-Lauf
    python3 run_backtest_1w.py --wf     # Walk-Forward (IS vs. OOS)
"""

import argparse
import numpy as np
import pandas as pd

from sol_bot_1w.backtest  import run_backtest, fetch_binance_1w, WF_SPLIT, STARTKAPITAL


def sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    return (rets.mean() / rets.std()) * np.sqrt(52)


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

    if not trades_df.empty:
        print()
        print("  Alle Trades:")
        for _, t in trades_df.iterrows():
            tag = "IS" if t.get("IS", True) else "OOS"
            print(f"    [{tag}] {t['Entry_ts'].date()} → {t['Exit_ts'].date()}"
                  f"  {t['Side']:>5}  {t['PnL_pct']:>+7.1f}%  ({t['Exit_reason']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wf",   action="store_true",
                        help="Walk-Forward-Bericht: IS vs. OOS separat")
    args = parser.parse_args()

    print("Lade Binance 1W SOL/USDT-Daten …")
    df_raw = fetch_binance_1w("2020-01-01")
    print(f"  {len(df_raw)} Wochenkerzen | {df_raw.index[0].date()} → {df_raw.index[-1].date()}\n")

    trades_df, equity = run_backtest(df_raw)
    print_report("1W-Bot  (EMA10/EMA21-Crossover)", trades_df, equity, df_raw,
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
