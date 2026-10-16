"""
explore_neutral_breakout_1d.py — Volatility-Compression-Breakout

Diagnose v1: Signal `Regime=="neutral" AND ADX>25 AND DI_bull` = 0 Treffer.
Ursache: Regime-Engine flippt sofort auf "bull" wenn ADX>20+DI_bull (nach count>MIN_REGIME
 Persistenz-Tagen in neutral) — mutual exclusion.

Korrektur v2: Signal feuert im bull-Regime, aber nur am ERSTEN bull-Tag nach Kompression.
  compression_active[t] = max(ADX[t-K:t-1]) < ADX_LOW     (shift(1) → kein Look-Ahead)
  Signal: regime=="bull" AND compression_active AND prev_regime!="bull"
          AND ADX > ADX_HIGH AND DI_bull
  → Momentum-Entry am Regime-Flip-Tag, ergänzt das normale RSI3-Dip-Entry.

Pre-committed Gates (ALLE müssen bestehen):
  1. MaxDD < -15%
  2. Rendite ≥ 63.1%
  3. OOS-Sharpe ≥ 0.74
  4. OOS-Bull ΣPnL ≥ 58.7%   (Cooldown-Interferenz-Check)
  5. OOS-Bear ΣPnL ≥ 30.7%   (Cooldown-Interferenz-Check)
  6. Neue OOS Breakout-n ≥ 3  (Statistical floor)
  7. Robustheit: Monotonie über 3×3-Sweep ADX_LOW {12,15,18} × ADX_HIGH {22,25,28}

Skopus: Long-only. SL=1.0×ATR, TP=8×ATR, Hebel=2.0×, Trail=3.0×ATR, PosSize=20%.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sol_bot_1d.backtest import (
    fetch_binance_1d, WF_SPLIT,
    _buy_price, _sell_price, GEBUEHR,
)
from sol_bot_1d.indicators import add_indicators
from sol_bot_1d.regime     import classify_regime
from sol_bot_1d.strategy   import (
    signal_bull_entry, signal_bull_exit,
    signal_bear_entry, signal_bear_cover,
    SL_TP, TRAIL_MULT, HEBEL
)
from cost_model import FUNDING_1D_PER_CANDLE, STARTKAPITAL

COOLDOWN      = 2
COMPRESSION_K = 5   # Rollfenster für Kompression (Tage)


# ── Lokales Position-Objekt ───────────────────────────────────────────────────
class Pos:
    __slots__ = ["side", "entry_price", "entry_cash", "hebel",
                 "sl_price", "tp_price", "trail_price", "regime", "candle_idx",
                 "is_breakout"]

    def __init__(self, side, entry_price, entry_cash, hebel,
                 sl_price, tp_price, trail_price, regime, candle_idx,
                 is_breakout: bool = False):
        self.side        = side
        self.entry_price = entry_price
        self.entry_cash  = entry_cash
        self.hebel       = hebel
        self.sl_price    = sl_price
        self.tp_price    = tp_price
        self.trail_price = trail_price
        self.regime      = regime
        self.candle_idx  = candle_idx
        self.is_breakout = is_breakout


def _exit_pnl_local(pos: Pos, exit_price: float) -> float:
    if pos.side == "long":
        gross = pos.entry_cash * pos.hebel * (exit_price / pos.entry_price - 1)
    else:
        gross = pos.entry_cash * pos.hebel * (1 - exit_price / pos.entry_price)
    fee = GEBUEHR * 2 * pos.hebel * pos.entry_cash
    return pos.entry_cash + gross - fee


def _mtm_local(pos: Pos, kurs: float) -> float:
    if pos.side == "long":
        gross = pos.entry_cash * pos.hebel * (kurs / pos.entry_price - 1)
    else:
        gross = pos.entry_cash * pos.hebel * (1 - kurs / pos.entry_price)
    fee = GEBUEHR * 2 * pos.hebel * pos.entry_cash
    return pos.entry_cash + gross - fee


def _compression_breakout_signal(row, adx_low: float, adx_high: float) -> bool:
    """
    True wenn der aktuelle (bull-Regime-)Tag ein Breakout aus Kompression ist.
    Erfordert: compression_active UND prev_regime != "bull" UND ADX > adx_high.
    Beide Werte sind aus vorberechneten df-Spalten (kein Look-Ahead).
    """
    compression_active = row.get("compression_active", False)
    prev_regime        = row.get("prev_regime", "neutral")
    adx                = row.get("ADX", 0.0)

    if pd.isna(compression_active) or pd.isna(adx):
        return False
    return bool(compression_active and prev_regime != "bull" and adx > adx_high)


# ── Backtest-Engine ───────────────────────────────────────────────────────────

def run_backtest_breakout(
    df_raw: pd.DataFrame,
    adx_low:    float = 15.0,
    adx_high:   float = 25.0,
    min_regime: int   = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    df = add_indicators(df_raw)
    df["Regime"]             = classify_regime(df, min_regime=min_regime)
    df["compression_active"] = df["ADX"].rolling(COMPRESSION_K).max().shift(1) < adx_low
    df["prev_regime"]        = df["Regime"].shift(1).fillna("neutral")

    cash          = STARTKAPITAL
    pos: Pos | None = None
    equity        = []
    trades        = []
    breakout_details: list[dict] = []
    equity_window: list[float]   = []
    cooldown_left = 0
    DD_RESET_BARS = 90

    for i, (ts, row) in enumerate(df.iterrows()):
        kurs   = row["Close"]
        atr    = row["ATR"]  if not pd.isna(row["ATR"])  else 0.02 * kurs
        vola   = row["Vola"] if not pd.isna(row["Vola"]) else 2.0
        regime = row["Regime"]

        if pos is not None and pos.side == "short":
            cash -= pos.entry_cash * FUNDING_1D_PER_CANDLE * pos.hebel

        port_value = cash if pos is None else (cash + _mtm_local(pos, kurs))
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
                    trail_mult  = TRAIL_MULT.get("bull", 3.0) or 3.0
                    new_trail   = kurs - trail_mult * atr
                    pos.trail_price = max(pos.trail_price, new_trail)
                    if kurs <= pos.trail_price:
                        exit_reason, exit_price = "Trail", _sell_price(pos.trail_price, vola)
            else:  # short
                if kurs >= pos.sl_price:
                    exit_reason, exit_price = "SL", _buy_price(pos.sl_price, vola)
                elif pos.tp_price is not None and kurs <= pos.tp_price:
                    exit_reason, exit_price = "TP", _buy_price(pos.tp_price, vola)
                elif signal_bear_cover(row):
                    exit_reason, exit_price = "Cover", _buy_price(kurs, vola)

            if exit_reason:
                proceeds = _exit_pnl_local(pos, exit_price)
                pnl_pct  = (proceeds - pos.entry_cash) / pos.entry_cash * 100
                is_flag  = df.index[pos.candle_idx] < pd.Timestamp(WF_SPLIT)
                trades.append({
                    "Entry_ts":    df.index[pos.candle_idx],
                    "Exit_ts":     ts,
                    "Side":        pos.side,
                    "Hebel":       pos.hebel,
                    "PnL_pct":     pnl_pct,
                    "Regime":      pos.regime,
                    "Exit_reason": exit_reason,
                    "IS":          is_flag,
                    "is_breakout": pos.is_breakout,
                })
                if pos.is_breakout:
                    breakout_details.append({
                        "Entry_ts":    df.index[pos.candle_idx],
                        "Exit_ts":     ts,
                        "PnL_pct":     pnl_pct,
                        "Exit_reason": exit_reason,
                        "IS":          is_flag,
                        "ADX_high":    adx_high,
                    })
                cash         += proceeds
                pos           = None
                cooldown_left = COOLDOWN

        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if pos is None:
            if regime == "bull":
                sl_mult, tp_mult = SL_TP.get("bull", (1.0, 8.0))
                hebel = HEBEL.get("bull", 2.0)
                if dd < -0.20:
                    hebel = 1.0

                # Primär: RSI3-Dip (Standard-Signal)
                sig        = signal_bull_entry(row)
                is_breakout = False

                # Sekundär: Kompression-Breakout (wenn kein Dip-Signal)
                if sig == 0 and _compression_breakout_signal(row, adx_low, adx_high):
                    sig        = 1
                    is_breakout = True

                if sig == 1:
                    ep    = _buy_price(kurs, vola)
                    sl    = ep - sl_mult * atr
                    tp    = ep + tp_mult * atr
                    trail = ep - (TRAIL_MULT.get("bull", 3.0) or 3.0) * atr
                    pos   = Pos("long", ep, cash * 0.20, hebel, sl, tp, trail,
                                "bull", i, is_breakout)
                    cash -= pos.entry_cash

            elif regime == "bear":
                sl_mult, tp_mult = SL_TP.get("bear", (1.0, 8.0))
                hebel = HEBEL.get("bear", 1.5)
                if dd < -0.20:
                    hebel = 1.0
                sig = signal_bear_entry(row)
                if sig == -2:
                    ep  = _sell_price(kurs, vola)
                    sl  = ep + sl_mult * atr
                    tp  = ep - tp_mult * atr
                    pos = Pos("short", ep, cash * 0.20, hebel, sl, tp, None,
                              "bear", i, False)
                    cash -= pos.entry_cash

    equity_df = pd.DataFrame(equity).set_index("ts")["equity"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, equity_df, breakout_details


# ── Gate-Prüfung & Report ─────────────────────────────────────────────────────

GATES = {
    "maxdd":      -15.0,
    "rendite":     63.1,
    "oos_sh":       0.74,
    "oos_bull":    58.7,
    "oos_bear":    30.7,
    "breakout_n":   3,
}


def report(trades_df, equity_df, breakout_details, label):
    wf      = pd.Timestamp(WF_SPLIT)
    rendite = (equity_df.iloc[-1] / equity_df.iloc[0] - 1) * 100
    rets    = equity_df.pct_change().dropna()
    sharpe  = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0
    maxdd   = ((equity_df - equity_df.expanding().max()) / equity_df.expanding().max()).min() * 100

    oos_eq = equity_df[equity_df.index >= wf]
    oos_sh = 0.0
    if len(oos_eq) > 1:
        r = oos_eq.pct_change().dropna()
        oos_sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0

    oos_bull_sum, oos_bear_sum = 0.0, 0.0
    if not trades_df.empty:
        oos_df   = trades_df[trades_df["IS"] == False]
        bull_oos = oos_df[oos_df["Regime"] == "bull"]
        bear_oos = oos_df[oos_df["Regime"] == "bear"]
        oos_bull_sum = bull_oos["PnL_pct"].sum() if len(bull_oos) else 0.0
        oos_bear_sum = bear_oos["PnL_pct"].sum() if len(bear_oos) else 0.0

    oos_breakouts = [b for b in breakout_details if not b["IS"]]
    breakout_n    = len(oos_breakouts)

    g1 = maxdd >= GATES["maxdd"]
    g2 = rendite >= GATES["rendite"]
    g3 = oos_sh >= GATES["oos_sh"]
    g4 = oos_bull_sum >= GATES["oos_bull"]
    g5 = oos_bear_sum >= GATES["oos_bear"]
    g6 = breakout_n >= GATES["breakout_n"]
    all_pass = g1 and g2 and g3 and g4 and g5 and g6

    flag = "✅" if all_pass else "❌"
    print(f"\n{flag} {label}")
    print(f"   Rendite={rendite:.1f}%  Sharpe={sharpe:.2f}  MaxDD={maxdd:.1f}%  n={len(trades_df)}")
    print(f"   OOS-Sharpe={oos_sh:.2f}  OOS-Bull={oos_bull_sum:.1f}%  OOS-Bear={oos_bear_sum:.1f}%")
    print(f"   Breakout-Trades (OOS): {breakout_n}")
    print(f"   Gates: MaxDD {'✅' if g1 else '❌'}  Rendite {'✅' if g2 else '❌'}  "
          f"OOS-Sh {'✅' if g3 else '❌'}  OOS-Bull {'✅' if g4 else '❌'}  "
          f"OOS-Bear {'✅' if g5 else '❌'}  Break-n≥3 {'✅' if g6 else '❌'}")

    return {
        "label":       label,
        "rendite":     rendite,
        "sharpe":      sharpe,
        "maxdd":       maxdd,
        "oos_sh":      oos_sh,
        "oos_bull":    oos_bull_sum,
        "oos_bear":    oos_bear_sum,
        "breakout_n":  breakout_n,
        "all_pass":    all_pass,
    }


def print_breakout_details(breakout_details, label=""):
    if not breakout_details:
        print("  (keine Breakout-Trades)")
        return
    print(f"\n  Breakout-Trades{' — ' + label if label else ''}:")
    for b in breakout_details:
        is_str = "IS " if b["IS"] else "OOS"
        print(f"    {is_str}  {b['Entry_ts'].date()} → {b['Exit_ts'].date()}  "
              f"PnL={b['PnL_pct']:+.1f}%  Exit={b['Exit_reason']}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Lade Daten ...")
    df_raw = fetch_binance_1d("2021-09-01")

    # ── Feasibility-Check ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FEASIBILITY-CHECK v2 — Breakout im bull-Regime nach Kompression")
    print("  Signal: regime=='bull' AND prev_regime!='bull'")
    print("          AND compression_active AND ADX>25")
    print("="*60)
    df_check = add_indicators(df_raw)
    df_check["Regime"]             = classify_regime(df_check, min_regime=3)
    df_check["compression_active"] = df_check["ADX"].rolling(COMPRESSION_K).max().shift(1) < 15.0
    df_check["prev_regime"]        = df_check["Regime"].shift(1).fillna("neutral")

    breakout_mask = (
        (df_check["Regime"] == "bull") &
        (df_check["prev_regime"] != "bull") &
        df_check["compression_active"] &
        (df_check["ADX"] > 25.0)
    )
    wf = pd.Timestamp(WF_SPLIT)
    n_total = breakout_mask.sum()
    n_oos   = breakout_mask[breakout_mask.index >= wf].sum()
    n_is    = breakout_mask[breakout_mask.index < wf].sum()
    print(f"  Potenzielle Breakout-Signaltage: {n_total} gesamt  (IS: {n_is}, OOS: {n_oos})")
    print(f"  Alle bull-Tage: {(df_check['Regime']=='bull').sum()}")
    print(f"  Regime-Flips neutral→bull: "
          f"{((df_check['Regime']=='bull') & (df_check['prev_regime']=='neutral')).sum()}")
    if n_oos < 3:
        print(f"  ⚠️  OOS < 3 — Gate 6 strukturell nicht erfüllbar. Ausgabe trotzdem vollständig.")
    else:
        print(f"  ✅ OOS ≥ 3 Signaltage — Backtest sinnvoll.")

    print("\n  Breakout-Signaltage (alle, kein Cooldown):")
    for dt in df_check.index[breakout_mask]:
        oos_str = "OOS" if dt >= wf else "IS "
        row = df_check.loc[dt]
        print(f"    {oos_str}  {dt.date()}  ADX={row['ADX']:.1f}  "
              f"prev_regime={row['prev_regime']}  "
              f"compression={row['compression_active']}")

    # ── 3×3 Sweep ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("3×3 ROBUSTHEITS-SWEEP — ADX_LOW × ADX_HIGH")
    print("="*60)

    adx_lows  = [12.0, 15.0, 18.0]
    adx_highs = [22.0, 25.0, 28.0]
    sweep_results: dict = {}

    for adx_low in adx_lows:
        for adx_high in adx_highs:
            label = f"ADX_LOW={adx_low:.0f}  ADX_HIGH={adx_high:.0f}"
            t, e, bd = run_backtest_breakout(df_raw, adx_low=adx_low, adx_high=adx_high)
            r = report(t, e, bd, label)
            sweep_results[(adx_low, adx_high)] = r

    # ── Nominale Detail-Ausgabe ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("DETAIL — Breakout-Trades bei Nominal (ADX_LOW=15, ADX_HIGH=25)")
    print("="*60)
    t_nom, e_nom, bd_nom = run_backtest_breakout(df_raw, adx_low=15.0, adx_high=25.0)
    print_breakout_details(bd_nom)

    # ── Monotonie-Tabelle ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MONOTONIE — Rendite und Break-n(OOS) über Sweep")
    print("="*60)
    print("\n  Rendite (ADX_HIGH →):")
    print("  ADX_LOW  " + "  ".join(f"AH={ah:.0f}" for ah in adx_highs))
    for adx_low in adx_lows:
        vals = [f"{sweep_results[(adx_low, ah)]['rendite']:6.1f}%" for ah in adx_highs]
        print(f"  AL={adx_low:.0f}     " + "  ".join(vals))

    print("\n  Break-n OOS:")
    for adx_low in adx_lows:
        vals = [f"n={sweep_results[(adx_low, ah)]['breakout_n']}" for ah in adx_highs]
        print(f"  AL={adx_low:.0f}     " + "  ".join(vals))

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)
    passing = [(k, v) for k, v in sweep_results.items() if v["all_pass"]]
    if passing:
        for (al, ah), v in sorted(passing, key=lambda x: x[1]["rendite"], reverse=True):
            print(f"  ✅ ADX_LOW={al:.0f}  ADX_HIGH={ah:.0f}  "
                  f"Rendite={v['rendite']:.1f}%  MaxDD={v['maxdd']:.1f}%  "
                  f"OOS-Sh={v['oos_sh']:.2f}  Break-n(OOS)={v['breakout_n']}")
    else:
        print("  ❌ Keine Variante besteht alle Gates.")

    renditen = [v["rendite"] for v in sweep_results.values()]
    swing    = max(renditen) - min(renditen)
    print(f"\n  Rendite-Swing: {swing:.1f}pp — "
          f"{'⚠️  Non-Monotonie-Risiko prüfen!' if swing > 20 else '✅ akzeptabel'}")
    print()
