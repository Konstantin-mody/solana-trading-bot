"""
diag_4h_bull.py — Trade-Qualitäts-Sweep für 4h Bull-Entry-Filter-Konfigurationen

Vergleicht IS-Bull-Trades (WR, ΣPnL, n) für verschiedene Filter-Kombinationen,
um zu entscheiden ob und wie signal_bull_entry verbessert werden kann.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import sol_bot_4h.strategy as strat
from sol_bot_4h.backtest import run_backtest, fetch_binance_4h, WF_SPLIT


def _fmt(grp, label_w: int = 40) -> str:
    if grp is None or grp.empty:
        return f"  {'':>{label_w}}  n=  0  WR=  ---  ΣPnL=     ---  Ø=    ---"
    n    = len(grp)
    wr   = (grp["PnL_pct"] > 0).mean() * 100
    spnl = grp["PnL_pct"].sum()
    avg  = grp["PnL_pct"].mean()
    return f"n={n:>3}  WR={wr:>5.1f}%  ΣPnL={spnl:>+8.1f}%  Ø={avg:>+6.2f}%"


def bull_stats(trades_df: pd.DataFrame, label: str) -> None:
    if trades_df.empty:
        print(f"  {label:<40}  IS: n=  0  ---    OOS: n=  0  ---")
        return
    is_bull  = trades_df[ trades_df["IS"]  & (trades_df["Regime"] == "bull")]
    oos_bull = trades_df[~trades_df["IS"]  & (trades_df["Regime"] == "bull")]
    is_str   = _fmt(is_bull  if not is_bull.empty  else None)
    oos_str  = _fmt(oos_bull if not oos_bull.empty else None)
    print(f"  {label:<40}  IS: {is_str}   OOS: {oos_str}")


def make_bull(rsi_thresh: float = 50.0, use_di: bool = True, use_obv: bool = True,
              ma50_factor: float = 0.92):
    """Gibt eine signal_bull_entry-Funktion mit angepassten Parametern zurück."""
    from sol_bot_4h.strategy import _g

    def fn(row) -> int:
        rsi3      = _g(row, "RSI3",      50.0)
        di_bull   = _g(row, "DI_bull",   False)
        ema21     = _g(row, "EMA21",     0.0)
        ema50     = _g(row, "EMA50",     0.0)
        ma50      = _g(row, "MA50",      0.0)
        close     = _g(row, "Close",     0.0)
        atr_spike = _g(row, "ATR_spike", 1.0)
        obv_bull  = _g(row, "OBV_bull",  False)

        if (rsi3 < rsi_thresh
                and (not use_di  or di_bull)
                and ema21 > ema50
                and close >= ma50 * ma50_factor
                and atr_spike < 2.5
                and (not use_obv or obv_bull)):
            return 1
        return 0

    return fn


def run_config(df_raw: pd.DataFrame, label: str, fn) -> None:
    orig = strat.signal_bull_entry
    strat.signal_bull_entry = fn
    try:
        trades_df, _ = run_backtest(df_raw)
        bull_stats(trades_df, label)
    finally:
        strat.signal_bull_entry = orig


def main() -> None:
    print("Lade Binance 4h SOL/USDT-Daten …")
    df_raw = fetch_binance_4h("2021-09-01")
    print(f"  {len(df_raw)} Kerzen | {df_raw.index[0].date()} → {df_raw.index[-1].date()}\n")

    print(f"  {'Konfiguration':<40}  IS: n / WR / ΣPnL / Ø                    OOS: n / WR / ΣPnL / Ø")
    print(f"  {'─'*120}")

    configs = [
        # ── Baseline & Filter-Isolation ──────────────────────────────────────
        ("Baseline (RSI3<50, di+obv)",        make_bull(50, True,  True )),
        ("Ohne di_bull  (RSI3<50, obv)",       make_bull(50, False, True )),
        ("Ohne obv_bull (RSI3<50, di)",        make_bull(50, True,  False)),
        ("Ohne di+obv   (RSI3<50)",            make_bull(50, False, False)),

        # ── RSI3-Schwellenwert-Sweep (alle Filter) ───────────────────────────
        ("RSI3<30  (di+obv)",                  make_bull(30, True,  True )),
        ("RSI3<35  (di+obv)",                  make_bull(35, True,  True )),
        ("RSI3<40  (di+obv)",                  make_bull(40, True,  True )),
        ("RSI3<45  (di+obv)",                  make_bull(45, True,  True )),

        # ── RSI3-Sweep ohne di_bull ───────────────────────────────────────────
        ("RSI3<30  (obv, kein di)",             make_bull(30, False, True )),
        ("RSI3<35  (obv, kein di)",             make_bull(35, False, True )),
        ("RSI3<40  (obv, kein di)",             make_bull(40, False, True )),

        # ── MA50-Proximity-Sweep (di+obv, RSI3<50) ───────────────────────────
        ("MA50-Filter  0.95 (di+obv)",         make_bull(50, True,  True,  0.95)),
        ("MA50-Filter  1.00 (di+obv)",         make_bull(50, True,  True,  1.00)),
        ("MA50-Filter  0.85 (di+obv)",         make_bull(50, True,  True,  0.85)),
    ]

    for label, fn in configs:
        run_config(df_raw, label, fn)

    print()


if __name__ == "__main__":
    main()
