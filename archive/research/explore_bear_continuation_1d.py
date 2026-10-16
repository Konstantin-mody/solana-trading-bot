"""
explore_bear_continuation_1d.py — Bear-Trend-Continuation-Signal für den 1d-Bot

Kontext (2026-05-07):
  - Baseline: Rendite=+63.1%, OOS-Sharpe=0.74, MaxDD=-8%, OOS-Bear-n=9, OOS-Bear-ΣPnL=+30.7%
  - Bear-Regime: 519 Tage total
  - Aktuell: RSI3>65 + DI_bear + EMA_bear + Close<MA200 = ~53 tradeable Tage
  - Hypothese-Material: RSI3∈[40,65) + ADX>25 + Close<MA200 = 109 zusätzliche Tage

Sektion 0 — Diagnostik: Cover-Quelle der OOS-Bear-Trades (RSI3<30 vs DI_bull)
Sektion 1 — Hypothese A: RSI3-Threshold-Sweep {65, 60, 55, 50}, Cover unverändert
Sektion 2 — Hypothese B: Zwei-Tier-Entry (bestehend + TC), TC-Cover = DI_bull only
             ADX-Sweep {25, 28, 32} für den TC-Threshold

Pre-committed Gates (gesetzt VOR Sweep):
  - MaxDD < -15%
  - OOS Bear ΣPnL ≥ 30.7% (Baseline)
  - Rendite ≥ 70% (Baseline war 63.1%)
  - OOS-Sharpe ≥ 0.74 (Baseline)
  - OOS-Bear-n ≥ 14 (Baseline 9, Δ≥+5)

Hinweis: selbst "Pass" = strukturell weit von Gate 8 (Rendite≥500%) entfernt.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sol_bot_1d.backtest import (
    fetch_binance_1d, WF_SPLIT,
    _buy_price, _sell_price, _exit_pnl, _mtm,
    GEBUEHR,
)
from sol_bot_1d.indicators import add_indicators
from sol_bot_1d.regime     import classify_regime
from sol_bot_1d.strategy   import (
    get_signal, signal_bull_entry, signal_bull_exit,
    SL_TP, TRAIL_MULT, HEBEL,
    _g,
)
from cost_model import FUNDING_1D_PER_CANDLE, STARTKAPITAL

COOLDOWN  = 2
WF        = pd.Timestamp(WF_SPLIT)


# ─── Lokale Position-Klasse (cover_strict-Flag) ───────────────────────────────

class Pos:
    __slots__ = [
        "side", "entry_price", "entry_cash", "hebel",
        "sl_price", "tp_price", "trail_price", "regime",
        "candle_idx", "cover_strict",
    ]

    def __init__(self, side, entry_price, entry_cash, hebel,
                 sl_price, tp_price, trail_price, regime, candle_idx,
                 cover_strict: bool = False):
        self.side         = side
        self.entry_price  = entry_price
        self.entry_cash   = entry_cash
        self.hebel        = hebel
        self.sl_price     = sl_price
        self.tp_price     = tp_price
        self.trail_price  = trail_price
        self.regime       = regime
        self.candle_idx   = candle_idx
        self.cover_strict = cover_strict  # True → Cover nur auf DI_bull (kein RSI3-Floor)


# ─── Signale (parametrisierbar) ───────────────────────────────────────────────

def bear_entry_a(row, rsi3_thresh: float) -> int:
    """Hypothese A: klassisches Bounce-Entry, RSI3-Threshold variabel."""
    rsi3     = _g(row, "RSI3",     50.0)
    di_bear  = _g(row, "DI_bear",  False)
    ema_bear = _g(row, "EMA_bear", False)
    close    = _g(row, "Close",    0.0)
    ma200    = _g(row, "MA200",    0.0)
    if ma200 > 0 and close >= ma200:
        return 0
    if rsi3 > rsi3_thresh and di_bear and ema_bear:
        return -2
    return 0


def bear_entry_tc(row, adx_thresh: float) -> int:
    """
    Hypothese B — Tier 2: Trend-Continuation-Entry.
    RSI3∈[40,65) + ADX>adx_thresh + DI_bear + EMA_bear + Close<MA200
    → kein RSI3-Bounce-Prämisse nötig, kein überkaufter Zustand
    """
    rsi3     = _g(row, "RSI3",     50.0)
    adx      = _g(row, "ADX",      0.0)
    di_bear  = _g(row, "DI_bear",  False)
    ema_bear = _g(row, "EMA_bear", False)
    close    = _g(row, "Close",    0.0)
    ma200    = _g(row, "MA200",    0.0)
    if ma200 > 0 and close >= ma200:
        return 0
    if (40.0 <= rsi3 < 65.0
            and adx > adx_thresh
            and di_bear
            and ema_bear):
        return -2
    return 0


def bear_cover_standard(row) -> bool:
    """Bestehendes Cover: RSI3<30 OR DI_bull."""
    return _g(row, "RSI3", 50.0) < 30 or _g(row, "DI_bull", False)


def bear_cover_strict(row) -> bool:
    """TC-Cover: nur DI_bull (kein RSI3-Floor)."""
    return bool(_g(row, "DI_bull", False))


# ─── Backtest-Engine ──────────────────────────────────────────────────────────

def _run(
    df_raw: pd.DataFrame,
    rsi3_thresh_a: float | None = 65.0,  # None → Hypothese B
    adx_thresh_b:  float | None = None,  # None → nur Hypothese A
    min_regime: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    rsi3_thresh_a: Float → Hypothese A (nur ein Tier, RSI3>thresh)
    adx_thresh_b:  Float → Hypothese B aktiviert (Tier-2 TC-Entry mit ADX>thresh)
    Beide können kombiniert werden (A-Tier1=65 + B-Tier2=adx_thresh_b).
    """
    df = add_indicators(df_raw)
    df["Regime"] = classify_regime(df, min_regime=min_regime)

    cash          = STARTKAPITAL
    pos: Pos | None = None
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
                    exit_reason, exit_price = "SL",    _sell_price(pos.sl_price, vola)
                elif pos.tp_price is not None and kurs >= pos.tp_price:
                    exit_reason, exit_price = "TP",    _sell_price(pos.tp_price, vola)
                elif signal_bull_exit(row):
                    exit_reason, exit_price = "Cover", _sell_price(kurs, vola)

                if pos.trail_price is not None and atr > 0 and exit_reason is None:
                    trail_mult = TRAIL_MULT.get(pos.regime, 3.0) or 3.0
                    new_trail  = kurs - trail_mult * atr
                    pos.trail_price = max(pos.trail_price, new_trail)
                    if kurs <= pos.trail_price:
                        exit_reason, exit_price = "Trail", _sell_price(pos.trail_price, vola)

            else:  # short
                if kurs >= pos.sl_price:
                    exit_reason, exit_price = "SL", _buy_price(pos.sl_price, vola)
                elif pos.tp_price is not None and kurs <= pos.tp_price:
                    exit_reason, exit_price = "TP", _buy_price(pos.tp_price, vola)
                else:
                    cover_fn = bear_cover_strict if pos.cover_strict else bear_cover_standard
                    if cover_fn(row):
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
                    "IS":          df.index[pos.candle_idx] < WF,
                    "TC":          pos.cover_strict,
                })
                cash         += proceeds
                pos           = None
                cooldown_left = COOLDOWN

        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if pos is None:
            sig = get_signal(row, regime)
            cover_strict_flag = False

            # Bear-Regime: eigene Entry-Logik überschreiben
            if regime == "bear":
                sig = 0

                # Tier 1 (Baseline-Bounce, RSI3>thresh)
                if rsi3_thresh_a is not None:
                    sig = bear_entry_a(row, rsi3_thresh_a)
                    cover_strict_flag = False

                # Tier 2 (TC-Signal), nur wenn kein Tier-1-Entry
                if sig == 0 and adx_thresh_b is not None:
                    sig = bear_entry_tc(row, adx_thresh_b)
                    cover_strict_flag = True

            if sig != 0:
                sl_mult, tp_mult = SL_TP.get(regime, (2.0, 8.0))
                hebel = HEBEL.get(regime, 2.0)
                if dd < -0.20:
                    hebel = 1.0
                entry_cash = cash * 0.20

                if sig == 1:
                    ep    = _buy_price(kurs, vola)
                    sl    = ep - sl_mult * atr
                    tp    = ep + tp_mult * atr
                    trail = ep - (TRAIL_MULT.get(regime, 3.0) or 3.0) * atr
                    pos   = Pos("long", ep, entry_cash, hebel,
                                sl, tp, trail, regime, i, False)
                elif sig == -2:
                    ep  = _sell_price(kurs, vola)
                    sl  = ep + sl_mult * atr
                    tp  = ep - tp_mult * atr
                    pos = Pos("short", ep, entry_cash, hebel,
                              sl, tp, None, regime, i, cover_strict_flag)

                cash -= entry_cash

    equity_df = pd.DataFrame(equity).set_index("ts")["equity"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, equity_df


# ─── Reporting ────────────────────────────────────────────────────────────────

GATES = {
    "maxdd":    -15.0,
    "oos_bear": 30.7,
    "rendite":  70.0,
    "oos_sh":   0.74,
    "bear_n":   14,
}


def report(trades_df, equity_df, label, show_exit_breakdown=False):
    rendite = (equity_df.iloc[-1] / equity_df.iloc[0] - 1) * 100
    rets    = equity_df.pct_change().dropna()
    sharpe  = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0
    maxdd   = ((equity_df - equity_df.expanding().max()) / equity_df.expanding().max()).min() * 100

    oos_df   = trades_df[trades_df["IS"] == False] if not trades_df.empty else pd.DataFrame()
    bear_oos = oos_df[oos_df["Regime"] == "bear"] if not oos_df.empty else pd.DataFrame()
    oos_eq   = equity_df[equity_df.index >= WF]
    oos_sh   = 0.0
    if len(oos_eq) > 1:
        r = oos_eq.pct_change().dropna()
        oos_sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0

    bear_sum = bear_oos["PnL_pct"].sum() if not bear_oos.empty else 0
    bear_n   = len(bear_oos)
    n_total  = len(trades_df)

    g1 = maxdd   >= GATES["maxdd"]
    g2 = bear_sum >= GATES["oos_bear"]
    g3 = rendite >= GATES["rendite"]
    g4 = oos_sh  >= GATES["oos_sh"]
    g5 = bear_n  >= GATES["bear_n"]
    all_pass = g1 and g2 and g3 and g4 and g5

    flag = "✅" if all_pass else "❌"
    print(f"\n{flag} {label}")
    print(f"   Rendite={rendite:.1f}%  Sharpe={sharpe:.2f}  MaxDD={maxdd:.1f}%  n={n_total}")
    print(f"   OOS-Sharpe={oos_sh:.2f}  OOS-Bear-ΣPnL={bear_sum:.1f}% (n={bear_n})")
    print(f"   Gates: MaxDD {'✅' if g1 else '❌'}"
          f"  OOS-Bear≥30.7% {'✅' if g2 else '❌'}"
          f"  Rendite≥70% {'✅' if g3 else '❌'}"
          f"  OOS-Sh≥0.74 {'✅' if g4 else '❌'}"
          f"  Bear-n≥14 {'✅' if g5 else '❌'}")

    if show_exit_breakdown and not bear_oos.empty:
        exits = bear_oos["Exit_reason"].value_counts()
        print(f"   OOS-Bear-Exits: {dict(exits)}")

    return {
        "label": label, "rendite": rendite, "sharpe": sharpe, "maxdd": maxdd,
        "oos_sh": oos_sh, "bear_sum": bear_sum, "bear_n": bear_n,
        "all_pass": all_pass,
    }


# ─── Sektion 0: Cover-Quelle Diagnostik ──────────────────────────────────────

def diag_cover_source(df_raw: pd.DataFrame):
    print("\n" + "="*60)
    print("SEKTION 0 — Cover-Quelle Diagnostik (OOS Bear Cover-Exits)")
    print("="*60)
    trades_df, _ = _run(df_raw, rsi3_thresh_a=65.0, adx_thresh_b=None)

    if trades_df.empty:
        print("  Keine Trades.")
        return

    df_ind = add_indicators(df_raw)

    oos_df   = trades_df[trades_df["IS"] == False]
    bear_cov = oos_df[(oos_df["Regime"] == "bear") & (oos_df["Exit_reason"] == "Cover")]

    print(f"\n  OOS-Bear-Trades total: {len(oos_df[oos_df['Regime']=='bear'])}")
    print(f"  davon Cover-Exits:     {len(bear_cov)}")

    rsi3_trigger  = 0
    di_trigger    = 0
    both_trigger  = 0

    for _, tr in bear_cov.iterrows():
        exit_ts = tr["Exit_ts"]
        if exit_ts not in df_ind.index:
            continue
        row   = df_ind.loc[exit_ts]
        rsi3  = _g(row, "RSI3",   50.0)
        di_bull = _g(row, "DI_bull", False)
        r30   = rsi3 < 30
        di    = bool(di_bull)
        if r30 and di:
            both_trigger += 1
        elif r30:
            rsi3_trigger += 1
        elif di:
            di_trigger += 1

        print(f"    {exit_ts.date()}  RSI3={rsi3:.1f}  DI_bull={di_bull}  "
              f"→ {'RSI3<30' if r30 else ''}{'&DI_bull' if r30 and di else 'DI_bull' if di else ''}")

    print(f"\n  Cover-Quelle Breakdown:")
    print(f"    RSI3<30 allein:  {rsi3_trigger}")
    print(f"    DI_bull allein:  {di_trigger}")
    print(f"    Beide:           {both_trigger}")
    if rsi3_trigger + di_trigger + both_trigger == 0:
        print("    (alle Cover über SL oder kein Cover)")

    pct_di = (di_trigger + both_trigger) / max(len(bear_cov), 1) * 100
    print(f"\n  DI_bull-Anteil (inkl. Both): {pct_di:.0f}%")
    if pct_di >= 60:
        print("  → RSI3<30-Floor ist für TC-Cover irrelevant (DI_bull dominiert).")
    else:
        print("  → RSI3<30-Floor ist aktiv — TC-Cover-Removal kann Hold-Zeit verlängern.")


# ─── Hauptprogramm ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Lade Daten ...")
    df_raw = fetch_binance_1d("2021-09-01")

    # ── Sektion 0: Diagnostik ────────────────────────────────────────────────
    diag_cover_source(df_raw)

    results = []

    # ── Sektion 1: Hypothese A — RSI3-Threshold-Sweep ───────────────────────
    print("\n" + "="*60)
    print("SEKTION 1 — Hypothese A: RSI3-Threshold-Sweep (Cover unverändert)")
    print("(Baseline: RSI3>65, Cover=RSI3<30 OR DI_bull)")
    print("="*60)
    for thresh in [65, 60, 55, 50]:
        label = f"A: RSI3>{thresh}" + (" (Baseline)" if thresh == 65 else "")
        t, e = _run(df_raw, rsi3_thresh_a=float(thresh), adx_thresh_b=None)
        r = report(t, e, label, show_exit_breakdown=True)
        results.append(r)

    # ── Sektion 2: Hypothese B — Zwei-Tier + ADX-Sweep ─────────────────────
    print("\n" + "="*60)
    print("SEKTION 2 — Hypothese B: Baseline + TC-Tier (Cover=DI_bull only)")
    print("(Tier-1: RSI3>65 bestehend, Tier-2: RSI3∈[40,65) + ADX>X)")
    print("="*60)
    for adx_t in [25, 28, 32]:
        label = f"B: Tier1(RSI3>65) + Tier2(ADX>{adx_t})"
        t, e = _run(df_raw, rsi3_thresh_a=65.0, adx_thresh_b=float(adx_t))
        r = report(t, e, label, show_exit_breakdown=True)
        results.append(r)

    # ── Zusammenfassung ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG — alle Gates-bestehenden Varianten:")
    print("="*60)
    passing = [r for r in results if r["all_pass"]]
    if passing:
        for r in sorted(passing, key=lambda x: x["rendite"], reverse=True):
            print(f"  ✅ {r['label']:45s}  Rendite={r['rendite']:.1f}%  "
                  f"MaxDD={r['maxdd']:.1f}%  OOS-Sh={r['oos_sh']:.2f}  "
                  f"OOS-Bear-n={r['bear_n']}")
    else:
        print("  ❌ Keine Variante besteht alle Gates.")
    print()
    print("  Erinnerung: selbst Gates-Pass ≈ +70% Rendite — Gate 8 (≥500%) bleibt strukturelle Lücke.")
    print()
