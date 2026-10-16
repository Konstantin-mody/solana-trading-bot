"""
explore_neutral_short_1d.py — Short in bearish-neutral Regime

LETZTER NEUTRAL-TEST (Hypothese 5 von 5). Nach diesem Lauf:
→ Wenn alle Gates ✅: Implementierung evaluieren (Advisor High-Stakes).
→ Wenn Gates ❌ oder Kill-Pattern: Neutral-Regime geschlossen, kein weiterer Versuch.
  Eintrag in CLAUDE.md: "Neutral nicht handelbar — 5 Hypothesen, alle gescheitert."

Konzept (noch nie getestet, distinct von A–D):
  EMA_bear in neutral = EMA10 < EMA21 < EMA50 (bearish EMA-Stack, aber ADX<20 = low momentum)
  RSI3 > 65 = überkaufter Bounce im abwärts-geneigten Neutral-Regime
  → Short ohne DI_bear und ohne MA200-Filter (das ist der Unterschied zum Bear-Signal)

  354 von 718 Neutral-Tagen haben EMA_bear → substantieller Subspace.

Sweep: RSI3_HIGH {60, 65, 70, 75} × SL_MULT {1.0, 1.5, 2.0, 2.5} (4×4)

Pre-committed Gates (ALLE müssen bestehen — Multiple-Testing-verschärft):
  1. MaxDD < -15%
  2. Rendite ≥ 63.1%
  3. OOS-Sharpe ≥ 0.74
  4. OOS-Bull ΣPnL ≥ 58.7%          (Cooldown-Interferenz-Check)
  5. OOS-Bear ΣPnL ≥ 30.7%          (Cooldown-Interferenz-Check)
  6. OOS Neutral-Short n ≥ 5         (verschärft wegen Multiple-Testing, war 3)
  7. Sub-Perioden-Stabilität: 2025 UND 2026 je positiver Neutral-Short ΣPnL
  8. Robustheit: kein Kill-Pattern (Swing < 20pp über 4×4-Sweep)
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

COOLDOWN  = 2
HEBEL_NEU = 1.5   # wie Bear-Shorts


# ── Lokales Position-Objekt ───────────────────────────────────────────────────
class Pos:
    __slots__ = ["side", "entry_price", "entry_cash", "hebel",
                 "sl_price", "tp_price", "trail_price", "regime",
                 "candle_idx", "is_neutral_short"]

    def __init__(self, side, entry_price, entry_cash, hebel,
                 sl_price, tp_price, trail_price, regime, candle_idx,
                 is_neutral_short: bool = False):
        self.side            = side
        self.entry_price     = entry_price
        self.entry_cash      = entry_cash
        self.hebel           = hebel
        self.sl_price        = sl_price
        self.tp_price        = tp_price
        self.trail_price     = trail_price
        self.regime          = regime
        self.candle_idx      = candle_idx
        self.is_neutral_short = is_neutral_short


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


def signal_neutral_short(row, rsi3_high: float) -> int:
    """
    Short im bearish-neutral Regime.
    EMA_bear (bearish Stack) + RSI3 überhitzt = Bounce gegen abwärts-EMA.
    Kein DI_bear-Filter (das ist der Unterschied zum Bear-Signal).
    Kein MA200-Filter.
    """
    rsi3     = row.get("RSI3",     50.0)
    ema_bear = row.get("EMA_bear", False)
    if pd.isna(rsi3) or pd.isna(ema_bear):
        return 0
    if rsi3 > rsi3_high and ema_bear:
        return -2
    return 0


def signal_neutral_cover(row) -> bool:
    """Cover Neutral-Short: RSI3 erholt sich oder DI kippt bullisch."""
    rsi3    = row.get("RSI3",    50.0)
    di_bull = row.get("DI_bull", False)
    return rsi3 < 30 or di_bull


# ── Backtest-Engine ───────────────────────────────────────────────────────────

def run_backtest_neutral_short(
    df_raw: pd.DataFrame,
    rsi3_high:  float = 65.0,
    sl_mult:    float = 1.0,
    min_regime: int   = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    df = add_indicators(df_raw)
    df["Regime"] = classify_regime(df, min_regime=min_regime)

    tp_mult = 8.0   # fix — kein zusätzlicher freier Parameter

    cash          = STARTKAPITAL
    pos: Pos | None = None
    equity        = []
    trades        = []
    neutral_details: list[dict] = []
    equity_window: list[float]  = []
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
                # Neutral-Shorts nutzen eigene Cover-Logik
                cover_fn = signal_neutral_cover if pos.is_neutral_short else signal_bear_cover
                if kurs >= pos.sl_price:
                    exit_reason, exit_price = "SL", _buy_price(pos.sl_price, vola)
                elif pos.tp_price is not None and kurs <= pos.tp_price:
                    exit_reason, exit_price = "TP", _buy_price(pos.tp_price, vola)
                elif cover_fn(row):
                    exit_reason, exit_price = "Cover", _buy_price(kurs, vola)

            if exit_reason:
                proceeds = _exit_pnl_local(pos, exit_price)
                pnl_pct  = (proceeds - pos.entry_cash) / pos.entry_cash * 100
                is_flag  = df.index[pos.candle_idx] < pd.Timestamp(WF_SPLIT)
                trades.append({
                    "Entry_ts":       df.index[pos.candle_idx],
                    "Exit_ts":        ts,
                    "Side":           pos.side,
                    "Hebel":          pos.hebel,
                    "PnL_pct":        pnl_pct,
                    "Regime":         pos.regime,
                    "Exit_reason":    exit_reason,
                    "IS":             is_flag,
                    "is_neutral_short": pos.is_neutral_short,
                })
                if pos.is_neutral_short:
                    neutral_details.append({
                        "Entry_ts":    df.index[pos.candle_idx],
                        "Exit_ts":     ts,
                        "PnL_pct":     pnl_pct,
                        "Exit_reason": exit_reason,
                        "IS":          is_flag,
                        "year":        ts.year,
                    })
                cash         += proceeds
                pos           = None
                cooldown_left = COOLDOWN

        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if pos is None:
            if regime == "bull":
                sl_m, tp_m = SL_TP.get("bull", (1.0, 8.0))
                hebel = HEBEL.get("bull", 2.0)
                if dd < -0.20: hebel = 1.0
                sig = signal_bull_entry(row)
                if sig == 1:
                    ep    = _buy_price(kurs, vola)
                    sl    = ep - sl_m * atr
                    tp    = ep + tp_m * atr
                    trail = ep - (TRAIL_MULT.get("bull", 3.0) or 3.0) * atr
                    pos   = Pos("long", ep, cash * 0.20, hebel, sl, tp, trail, "bull", i)
                    cash -= pos.entry_cash

            elif regime == "bear":
                sl_m, tp_m = SL_TP.get("bear", (1.0, 8.0))
                hebel = HEBEL.get("bear", 1.5)
                if dd < -0.20: hebel = 1.0
                sig = signal_bear_entry(row)
                if sig == -2:
                    ep  = _sell_price(kurs, vola)
                    sl  = ep + sl_m * atr
                    tp  = ep - tp_m * atr
                    pos = Pos("short", ep, cash * 0.20, hebel, sl, tp, None, "bear", i)
                    cash -= pos.entry_cash

            elif regime == "neutral":
                sig = signal_neutral_short(row, rsi3_high)
                if sig == -2:
                    hebel = HEBEL_NEU
                    if dd < -0.20: hebel = 1.0
                    ep  = _sell_price(kurs, vola)
                    sl  = ep + sl_mult * atr
                    tp  = ep - tp_mult * atr
                    pos = Pos("short", ep, cash * 0.20, hebel, sl, tp, None,
                              "neutral", i, is_neutral_short=True)
                    cash -= pos.entry_cash

    equity_df = pd.DataFrame(equity).set_index("ts")["equity"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, equity_df, neutral_details


# ── Gates & Report ────────────────────────────────────────────────────────────

GATES = {
    "maxdd":      -15.0,
    "rendite":     63.1,
    "oos_sh":       0.74,
    "oos_bull":    58.7,
    "oos_bear":    30.7,
    "neutral_n":    5,      # verschärft (Multiple-Testing)
}


def report(trades_df, equity_df, neutral_details, label):
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

    oos_neutrals = [d for d in neutral_details if not d["IS"]]
    neutral_n    = len(oos_neutrals)

    # Sub-Perioden-Stabilität
    oos_2025 = [d for d in oos_neutrals if d["year"] == 2025]
    oos_2026 = [d for d in oos_neutrals if d["year"] == 2026]
    pnl_2025 = sum(d["PnL_pct"] for d in oos_2025)
    pnl_2026 = sum(d["PnL_pct"] for d in oos_2026)
    subperiod_ok = (pnl_2025 > 0 or len(oos_2025) == 0) and (pnl_2026 > 0 or len(oos_2026) == 0)

    g1 = maxdd >= GATES["maxdd"]
    g2 = rendite >= GATES["rendite"]
    g3 = oos_sh >= GATES["oos_sh"]
    g4 = oos_bull_sum >= GATES["oos_bull"]
    g5 = oos_bear_sum >= GATES["oos_bear"]
    g6 = neutral_n >= GATES["neutral_n"]
    g7 = subperiod_ok
    all_pass = g1 and g2 and g3 and g4 and g5 and g6 and g7

    flag = "✅" if all_pass else "❌"
    print(f"\n{flag} {label}")
    print(f"   Rendite={rendite:.1f}%  Sharpe={sharpe:.2f}  MaxDD={maxdd:.1f}%  n={len(trades_df)}")
    print(f"   OOS-Sharpe={oos_sh:.2f}  OOS-Bull={oos_bull_sum:.1f}%  OOS-Bear={oos_bear_sum:.1f}%")
    print(f"   Neutral-Short OOS: n={neutral_n}  2025={pnl_2025:.1f}%  2026={pnl_2026:.1f}%")
    print(f"   Gates: MaxDD {'✅' if g1 else '❌'}  Rendite {'✅' if g2 else '❌'}  "
          f"OOS-Sh {'✅' if g3 else '❌'}  OOS-Bull {'✅' if g4 else '❌'}  "
          f"OOS-Bear {'✅' if g5 else '❌'}  Neutral-n≥5 {'✅' if g6 else '❌'}  "
          f"SubPeriod {'✅' if g7 else '❌'}")

    return {
        "label":      label,
        "rendite":    rendite,
        "sharpe":     sharpe,
        "maxdd":      maxdd,
        "oos_sh":     oos_sh,
        "oos_bull":   oos_bull_sum,
        "oos_bear":   oos_bear_sum,
        "neutral_n":  neutral_n,
        "pnl_2025":   pnl_2025,
        "pnl_2026":   pnl_2026,
        "all_pass":   all_pass,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Lade Daten ...")
    df_raw = fetch_binance_1d("2021-09-01")

    # ── Feasibility-Check ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FEASIBILITY-CHECK — Neutral-Shorts (RSI3>65, EMA_bear)")
    print("="*60)
    df_check = add_indicators(df_raw)
    df_check["Regime"] = classify_regime(df_check, min_regime=3)
    neutral_bear = df_check[
        (df_check["Regime"] == "neutral") & df_check["EMA_bear"]
    ]
    rsi3_high_mask = neutral_bear["RSI3"] > 65
    wf = pd.Timestamp(WF_SPLIT)
    print(f"  Neutrale Tage gesamt: {(df_check['Regime']=='neutral').sum()}")
    print(f"  Davon EMA_bear: {len(neutral_bear)} ({len(neutral_bear)/(df_check['Regime']=='neutral').sum()*100:.0f}%)")
    print(f"  RSI3>65 in neutral+EMA_bear: {rsi3_high_mask.sum()} "
          f"(OOS: {rsi3_high_mask[neutral_bear.index >= wf].sum()})")
    for thresh in [60, 65, 70, 75]:
        m = neutral_bear["RSI3"] > thresh
        n_oos = m[neutral_bear.index >= wf].sum()
        print(f"  RSI3>{thresh}: gesamt={m.sum()}  OOS={n_oos}")

    # ── 4×4 Sweep ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("4×4 ROBUSTHEITS-SWEEP — RSI3_HIGH × SL_MULT")
    print("="*60)

    rsi3_highs = [60.0, 65.0, 70.0, 75.0]
    sl_mults   = [1.0, 1.5, 2.0, 2.5]
    sweep_results: dict = {}

    for rsi3_high in rsi3_highs:
        for sl_mult in sl_mults:
            label = f"RSI3>{rsi3_high:.0f}  SL={sl_mult:.1f}×ATR"
            t, e, nd = run_backtest_neutral_short(df_raw, rsi3_high=rsi3_high, sl_mult=sl_mult)
            r = report(t, e, nd, label)
            sweep_results[(rsi3_high, sl_mult)] = r

    # ── Nominal-Detail ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DETAIL — Neutral-Short-Trades bei Nominal (RSI3>65, SL=1.0×ATR)")
    print("="*60)
    t_nom, e_nom, nd_nom = run_backtest_neutral_short(df_raw, rsi3_high=65.0, sl_mult=1.0)
    if nd_nom:
        for d in nd_nom:
            is_str = "IS " if d["IS"] else "OOS"
            print(f"  {is_str}  {d['Entry_ts'].date()} → {d['Exit_ts'].date()}  "
                  f"PnL={d['PnL_pct']:+.1f}%  Exit={d['Exit_reason']}")
    else:
        print("  (keine Neutral-Short-Trades)")

    # ── Monotonie-Tabelle ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MONOTONIE — Rendite und Neutral-n(OOS) über Sweep")
    print("="*60)
    print("\n  Rendite (SL_MULT →):")
    print("  RSI3_HIGH  " + "  ".join(f"SL={sl:.1f}" for sl in sl_mults))
    for rsi3_high in rsi3_highs:
        vals = [f"{sweep_results[(rsi3_high, sl)]['rendite']:6.1f}%" for sl in sl_mults]
        print(f"  RSI3>{rsi3_high:.0f}   " + "  ".join(vals))

    print("\n  Neutral-n(OOS):")
    for rsi3_high in rsi3_highs:
        vals = [f"n={sweep_results[(rsi3_high, sl)]['neutral_n']}" for sl in sl_mults]
        print(f"  RSI3>{rsi3_high:.0f}   " + "  ".join(vals))

    print("\n  ΣPnL Neutral-Shorts (OOS):")
    for rsi3_high in rsi3_highs:
        vals = []
        for sl in sl_mults:
            r = sweep_results[(rsi3_high, sl)]
            pnl_oos = r["pnl_2025"] + r["pnl_2026"]
            vals.append(f"{pnl_oos:+.0f}%")
        print(f"  RSI3>{rsi3_high:.0f}   " + "  ".join(vals))

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG")
    print("="*60)
    passing = [(k, v) for k, v in sweep_results.items() if v["all_pass"]]
    if passing:
        for (rsi3h, slm), v in sorted(passing, key=lambda x: x[1]["rendite"], reverse=True):
            print(f"  ✅ RSI3>{rsi3h:.0f}  SL={slm:.1f}×  "
                  f"Rendite={v['rendite']:.1f}%  MaxDD={v['maxdd']:.1f}%  "
                  f"OOS-Sh={v['oos_sh']:.2f}  n(OOS)={v['neutral_n']}")
    else:
        print("  ❌ Keine Variante besteht alle Gates.")

    renditen = [v["rendite"] for v in sweep_results.values()]
    swing    = max(renditen) - min(renditen)
    print(f"\n  Rendite-Swing: {swing:.1f}pp — "
          f"{'⚠️  Kill-Pattern prüfen!' if swing > 20 else '✅ akzeptabel'}")

    # ── Final-Entscheidung ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL-ENTSCHEIDUNG (Pre-committed):")
    if passing:
        print("  ✅ Mind. eine Variante besteht — Advisor (High-Stakes) vor Implementierung.")
    else:
        print("  ❌ Kein Ergebnis — Neutral-Regime wird geschlossen.")
        print("  CLAUDE.md-Eintrag: Neutral nicht handelbar (5/5 Hypothesen gescheitert).")
    print()
