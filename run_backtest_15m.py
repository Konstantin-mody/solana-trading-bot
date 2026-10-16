"""
run_backtest_15m.py — Einstiegspunkt für den 15min-Bot-Backtest

Verwendung:
    python3 run_backtest_15m.py                     # Standard-Lauf
    python3 run_backtest_15m.py --min_regime 4      # MIN_REGIME-Sweep
    python3 run_backtest_15m.py --min_regime 16
    python3 run_backtest_15m.py --wf                # Walk-Forward-Bericht
    python3 run_backtest_15m.py --regime_sweep      # Sweep {4, 8, 16}
    python3 run_backtest_15m.py --daily_limit 3     # Max Trades/Tag anpassen

HARTES GATE vor Live-Einsatz:
  - Netto-Edge ≥ 0.25%/Trade
  - n ≥ 100 in-sample
  - OOS positiv
"""

import argparse
import numpy as np
import pandas as pd

from sol_bot_15m.backtest import (run_backtest, fetch_binance_15m,
                                   WF_SPLIT, STARTKAPITAL, MAX_DAILY_TRADES)

ANNUALIZE = np.sqrt(365 * 96)   # 35.040 Kerzen/Jahr


def sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    return (rets.mean() / rets.std()) * ANNUALIZE


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd   = (equity - peak) / peak
    return dd.min() * 100


def print_report(label: str, trades_df: pd.DataFrame, equity: pd.Series,
                 df_raw: pd.DataFrame, check_gate: bool = False,
                 start_capital: float = None):
    final     = equity.iloc[-1]
    start_cap = start_capital if start_capital is not None else equity.iloc[0]
    rendite   = (final - start_cap) / start_cap * 100
    bah_ret   = (df_raw["Close"].iloc[-1] / df_raw["Close"].iloc[0] - 1) * 100
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

    n           = len(trades_df)
    wr          = (trades_df["PnL_pct"] > 0).mean() * 100
    avg_net_pct = trades_df["PnL_pct"].mean()
    days        = (df_raw.index[-1] - df_raw.index[0]).days
    trades_day  = n / max(days, 1)

    print(f"  Trades:        {n}  ({trades_day:.1f}/Tag)  |  WR: {wr:.1f}%  |  Ø Netto: {avg_net_pct:+.3f}%")

    if check_gate:
        print()
        gate_net  = avg_net_pct >= 0.25
        gate_n    = n >= 100
        print(f"  Gate Netto ≥ 0.25%/Trade: {'✅' if gate_net else '❌'}  ({avg_net_pct:+.3f}%)")
        print(f"  Gate n ≥ 100 in-sample:   {'✅' if gate_n  else '❌'}  (n={n})")

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
    parser.add_argument("--min_regime", type=int, default=8,
                        help="MIN_REGIME Sweep (default 8, try 4 / 16)")
    parser.add_argument("--daily_limit", type=int, default=MAX_DAILY_TRADES,
                        help=f"Max Trades/Tag (default {MAX_DAILY_TRADES})")
    parser.add_argument("--wf", action="store_true",
                        help="Walk-Forward-Bericht: IS vs. OOS separat")
    parser.add_argument("--regime_sweep", action="store_true",
                        help="Sweep MIN_REGIME {4, 8, 16} auf einmal")
    parser.add_argument("--symbol", type=str, default="SOLUSDT",
                        help="Binance-Symbol (default SOLUSDT, z.B. BTCUSDT)")
    args = parser.parse_args()

    sym_label = args.symbol.replace("USDT", "")
    print(f"Lade Binance 15min {sym_label}/USDT-Daten …")
    df_raw = fetch_binance_15m("2021-09-01", symbol=args.symbol)
    print(f"  {len(df_raw):,} Kerzen | {df_raw.index[0].date()} → {df_raw.index[-1].date()}\n")

    if args.regime_sweep:
        print(f"{'MIN_REGIME':<12} {'Trades':>7} {'/Tag':>5} {'WR%':>6} {'Ø Netto%':>9} {'Sharpe':>7} {'MaxDD%':>7}")
        print("─" * 65)
        days = (df_raw.index[-1] - df_raw.index[0]).days
        for mr in [4, 8, 16]:
            t, eq = run_backtest(df_raw, min_regime=mr,
                                 max_daily_trades=args.daily_limit)
            if t.empty:
                print(f"  mr={mr}  keine Trades")
                continue
            wr       = (t["PnL_pct"] > 0).mean() * 100
            avg_net  = t["PnL_pct"].mean()
            r        = (eq.iloc[-1] - STARTKAPITAL) / STARTKAPITAL * 100
            sr       = sharpe(eq)
            dd       = max_drawdown(eq)
            tpd      = len(t) / max(days, 1)
            print(f"  {mr:<10} {len(t):>7} {tpd:>4.1f} {wr:>5.1f}% {avg_net:>+8.3f}% {sr:>7.2f} {dd:>6.1f}%")
        return

    trades_df, equity = run_backtest(df_raw, min_regime=args.min_regime,
                                      max_daily_trades=args.daily_limit)
    print_report(f"15min-Bot {sym_label}  (MIN_REGIME={args.min_regime}, limit={args.daily_limit}/Tag)",
                 trades_df, equity, df_raw, check_gate=True)

    if args.wf and not trades_df.empty and "IS" in trades_df.columns:
        split = pd.Timestamp(WF_SPLIT)
        print(f"\n{'─'*55}")
        print(f"  Walk-Forward  (Split: {WF_SPLIT})")

        is_trades = trades_df[trades_df["IS"]]
        is_eq     = equity[equity.index < split]
        is_raw    = df_raw[df_raw.index < split]
        if not is_trades.empty and len(is_eq) > 1:
            print_report("  In-Sample (bis 2024)", is_trades, is_eq, is_raw, check_gate=True)

        oos_trades = trades_df[~trades_df["IS"]]
        oos_eq     = equity[equity.index >= split]
        oos_raw    = df_raw[df_raw.index >= split]
        if not oos_trades.empty and len(oos_eq) > 1:
            oos_gate = (oos_trades["PnL_pct"] > 0).any()
            print_report("  Out-of-Sample (2025+)", oos_trades, oos_eq, oos_raw)
            print(f"  Gate OOS positiv: {'✅' if oos_eq.iloc[-1] > oos_eq.iloc[0] else '❌'}")


if __name__ == "__main__":
    main()
