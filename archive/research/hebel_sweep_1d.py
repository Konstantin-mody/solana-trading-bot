"""
hebel_sweep_1d.py — Unabhängige Hebel-Sweeps für Bull und Bear (1d-Bot).

Schritt 1: Bull-Sweep  (Bear bleibt fix bei 2×)
Schritt 2: Bear-Sweep  (Bull bleibt fix am Bull-Optimum aus Schritt 1)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import sol_bot_1d.strategy as strat
from sol_bot_1d.backtest import fetch_binance_1d, run_backtest

print("Lade Daten …")
df_raw = fetch_binance_1d("2021-09-01")

ORIG_BULL = strat.HEBEL["bull"]
ORIG_BEAR = strat.HEBEL["bear"]
WF_SPLIT  = pd.Timestamp("2025-01-01")

BULL_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
BEAR_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def _sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    return (r.mean() / r.std() * (365 ** 0.5)) if r.std() > 0 else 0


def _oos_sharpe(equity: pd.Series) -> float:
    oos_eq = equity[equity.index >= WF_SPLIT]
    return _sharpe(oos_eq) if len(oos_eq) > 5 else float("nan")


def summary(label: str, trades: pd.DataFrame, equity: pd.Series):
    if trades.empty:
        print(f"  {label}: keine Trades")
        return
    is_t  = trades[trades["IS"]]
    oos_t = trades[~trades["IS"]]
    bull  = trades[trades["Regime"] == "bull"]
    bear  = trades[trades["Regime"] == "bear"]

    rend   = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    sharpe = _sharpe(equity)
    oos_sh = _oos_sharpe(equity)
    peak   = equity.expanding().max()
    dd     = ((equity - peak) / peak).min() * 100

    print(f"  {label:35s}  "
          f"Rend={rend:+6.1f}%  Sh={sharpe:.2f}  OOS-Sh={oos_sh:.2f}  MaxDD={dd:+5.1f}%  "
          f"n={len(trades)}  "
          f"Bull ΣPnL={bull['PnL_pct'].sum():+6.1f}%  "
          f"Bear ΣPnL={bear['PnL_pct'].sum():+6.1f}%  "
          f"IS={is_t['PnL_pct'].sum():+6.1f}%  OOS={oos_t['PnL_pct'].sum():+6.1f}%")


def run_sweep(label: str, mults: list[float], regime: str, fixed_regime: str, fixed_mult: float):
    print(f"\n{'='*100}")
    print(f"  {label}  (fix: {fixed_regime}={fixed_mult}×)")
    print(f"{'='*100}")

    strat.HEBEL[fixed_regime] = fixed_mult
    best_sharpe = -999
    best_mult   = None

    for m in mults:
        strat.HEBEL[regime] = m
        trades, equity = run_backtest(df_raw)
        strat.HEBEL[regime] = ORIG_BULL if regime == "bull" else ORIG_BEAR

        if trades.empty:
            continue
        ret_series = equity.pct_change().dropna()
        sh = (ret_series.mean() / ret_series.std() * (365 ** 0.5)) if ret_series.std() > 0 else 0
        dd = ((equity - equity.expanding().max()) / equity.expanding().max()).min() * 100

        summary(f"{regime}={m}×", trades, equity)

        # MaxDD-Gate < 20%
        if dd > -20 and sh > best_sharpe:
            best_sharpe = sh
            best_mult   = m

    strat.HEBEL[fixed_regime] = ORIG_BEAR if fixed_regime == "bear" else ORIG_BULL
    strat.HEBEL[regime]       = ORIG_BULL if regime == "bull" else ORIG_BEAR
    print(f"\n  → Optimum (Sharpe, MaxDD<20%): {regime}={best_mult}×  (Sharpe={best_sharpe:.2f})")
    return best_mult


# ─── Schritt 1: Bull-Sweep ────────────────────────────────────────────────────
bull_opt = run_sweep("Schritt 1: Bull-Hebel-Sweep", BULL_MULTS,
                     regime="bull", fixed_regime="bear", fixed_mult=ORIG_BEAR)

# ─── Schritt 2: Bear-Sweep ────────────────────────────────────────────────────
bear_opt = run_sweep("Schritt 2: Bear-Hebel-Sweep", BEAR_MULTS,
                     regime="bear", fixed_regime="bull", fixed_mult=bull_opt or ORIG_BULL)

print(f"\n{'='*100}")
print(f"  ERGEBNIS: Bull={bull_opt}×  Bear={bear_opt}×")
print(f"{'='*100}")

# ─── Finale Kombination ───────────────────────────────────────────────────────
strat.HEBEL["bull"] = bull_opt or ORIG_BULL
strat.HEBEL["bear"] = bear_opt or ORIG_BEAR
trades_final, equity_final = run_backtest(df_raw)
strat.HEBEL["bull"] = ORIG_BULL
strat.HEBEL["bear"] = ORIG_BEAR

print()
summary("Finale Kombination", trades_final, equity_final)
