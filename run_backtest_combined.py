"""
run_backtest_combined.py — Kombinierter Backtest: 1d + 4h + 1w aus $200-Pool

Konzept:
  Drei Engines teilen sich einen gemeinsamen Cash-Pool ($200 Startkapital).
  Positionsgröße = % des aktuell freien Cash (schrumpft wenn andere Bots in Trade).
  SL/TP/Trail für alle Bots auf 4h-Granularität geprüft (realistischer als Daily).
  1d-Signale nur am Day-Open (00:00 UTC) — basierend auf Vortags-Close.
  1w-Signale nur am Week-Open (Montag 00:00 UTC) — basierend auf Vorwochs-Close.
  DD-Schutz pro Bot separat (eigene virtuelle Equity).

Look-Ahead-Schutz:
  1d-Indikatoren: +24h shift → am Day-Open verfügbar (Vortags-Close bereits abgeschlossen).
  1w-Indikatoren: +168h shift → am Week-Open verfügbar (Vorwochs-Close abgeschlossen).
"""

import sys, os, argparse
from collections import deque
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cost_model import GEBUEHR, SLIPPAGE, SPREAD, FUNDING8, FUNDING_CADENCE_4H

from sol_bot_4h.indicators import add_indicators as add_ind_4h
from sol_bot_4h.regime     import classify_regime as classify_regime_4h
from sol_bot_4h.strategy   import (
    get_signal as get_signal_4h,
    signal_bull_exit as bull_exit_4h,
    signal_bear_cover as bear_cover_4h,
    SL_TP as SL_TP_4h, TRAIL_MULT as TRAIL_MULT_4h, HEBEL as HEBEL_4h,
)
from sol_bot_1d.indicators import add_indicators as add_ind_1d
from sol_bot_1d.regime     import classify_regime as classify_regime_1d
from sol_bot_1d.strategy   import (
    get_signal as get_signal_1d,
    signal_bull_exit as bull_exit_1d,
    signal_bear_cover as bear_cover_1d,
    SL_TP as SL_TP_1d, TRAIL_MULT as TRAIL_MULT_1d, HEBEL as HEBEL_1d,
)
from sol_bot_1w.indicators import add_indicators as add_ind_1w
from sol_bot_1w.strategy   import (
    get_signal as get_signal_1w,
    SL_TP as SL_TP_1w, TRAIL_MULT as TRAIL_MULT_1w, HEBEL as HEBEL_1w,
)
from sol_bot_4h.backtest import fetch_binance_4h
from sol_bot_1d.backtest import fetch_binance_1d
from sol_bot_1w.backtest import fetch_binance_1w

WF_SPLIT      = "2025-01-01"
TOTAL_CAPITAL = 200.0

POS_SIZE_4H = 0.20
POS_SIZE_1D = 0.20
POS_SIZE_1W = 1 / 3

COOLDOWN_4H_BARS   = 12   # 4h-Bars (= 2 Tage)
COOLDOWN_1D_OPENS  = 2    # Day-Open-Events (= 2 Tage)
COOLDOWN_1W_OPENS  = 4    # Week-Open-Events (= 4 Wochen)

MIN_HOLD_1D_BARS   = 6    # 4h-Bars (= 24h) bevor 1d-SL/TP/Trail/Cover feuern darf

DD_WINDOW_4H = 90 * 6    # 4h-Bars ≈ 90 Tage
DD_WINDOW_1D = 90         # Day-Open-Events ≈ 90 Tage
DD_WINDOW_1W = 52         # Week-Open-Events ≈ 1 Jahr


# ─── Preismodell ──────────────────────────────────────────────────────────────

def _buy_price(kurs: float, vola: float) -> float:
    slip = min(SLIPPAGE + vola * 0.00003, 0.001)
    return kurs * (1 + slip + SPREAD / 2)


def _sell_price(kurs: float, vola: float) -> float:
    slip = min(SLIPPAGE + vola * 0.00003, 0.001)
    return kurs * (1 - slip - SPREAD / 2)


def _exit_pnl(entry_cash: float, hebel: float,
               entry_price: float, exit_price: float, side: str) -> float:
    if side == "long":
        raw = (exit_price - entry_price) / entry_price
    else:
        raw = (entry_price - exit_price) / entry_price
    notional    = entry_cash * hebel
    gross_pnl   = notional * raw
    fee_entry   = notional * GEBUEHR
    fee_exit    = abs(notional * exit_price / entry_price) * GEBUEHR
    return max(entry_cash + gross_pnl - fee_entry - fee_exit, 0.01)


def _mtm(entry_cash: float, hebel: float,
          entry_price: float, kurs: float, side: str) -> float:
    if side == "long":
        raw = (kurs - entry_price) / entry_price
    else:
        raw = (entry_price - kurs) / entry_price
    return max(entry_cash + entry_cash * hebel * raw, 0.01)


def _nan_default(val, default):
    if val is None:
        return default
    try:
        if np.isnan(float(val)):
            return default
    except (TypeError, ValueError):
        return default
    return float(val)


# ─── Position ─────────────────────────────────────────────────────────────────

class Position:
    __slots__ = ["side", "entry_price", "entry_cash", "hebel",
                 "sl_price", "tp_price", "trail_price", "regime", "candle_idx"]

    def __init__(self, side, entry_price, entry_cash, hebel,
                 sl_price, tp_price, trail_price, regime, candle_idx):
        self.side        = side
        self.entry_price = entry_price
        self.entry_cash  = entry_cash
        self.hebel       = hebel
        self.sl_price    = sl_price
        self.tp_price    = tp_price
        self.trail_price = trail_price
        self.regime      = regime
        self.candle_idx  = candle_idx


# ─── Hauptsimulation ──────────────────────────────────────────────────────────

def run_backtest_combined(
    df_4h_raw: pd.DataFrame,
    df_1d_raw: pd.DataFrame,
    df_1w_raw: pd.DataFrame,
    pos_size_4h: float = POS_SIZE_4H,
    pos_size_1d: float = POS_SIZE_1D,
    pos_size_1w: float = POS_SIZE_1W,
    min_regime_4h: int = 6,
    min_regime_1d: int = 3,
    global_dd_brake: float | None = None,   # z.B. 0.22 → Entries pausieren wenn Portfolio-DD < -22%
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Gemeinsamer $200-Pool-Backtest für 1d + 4h + 1w Bots.
    Gibt (trades_df, equity_series) zurück.
    """

    # ── Indikatoren berechnen ──────────────────────────────────────────────────
    df_4h = add_ind_4h(df_4h_raw.copy())
    df_4h["Regime"] = classify_regime_4h(df_4h, min_regime=min_regime_4h)

    df_1d = add_ind_1d(df_1d_raw.copy())
    df_1d["Regime"] = classify_regime_1d(df_1d, min_regime=min_regime_1d)

    df_1w = add_ind_1w(df_1w_raw.copy())

    # ── Zeitachsen-Alignment (Look-Ahead-sicher) ───────────────────────────────
    # 1d-Bar (Index = Open-Zeitpunkt) enthält Close bis Index + 24h.
    # Signal feuert am nächsten Day-Open (00:00 UTC) = Index + 24h.
    df_1d_shifted        = df_1d.copy()
    df_1d_shifted.index  = df_1d_shifted.index + pd.Timedelta(hours=24)
    df_1d_on_4h          = df_1d_shifted.reindex(df_4h.index, method="ffill")

    # 1w-Bar (Index = Monday 00:00) enthält Close bis Index + 168h (nächster Mo).
    # Signal feuert am nächsten Week-Open = Index + 168h.
    df_1w_shifted        = df_1w.copy()
    df_1w_shifted.index  = df_1w_shifted.index + pd.Timedelta(hours=168)
    df_1w_on_4h          = df_1w_shifted.reindex(df_4h.index, method="ffill")

    # ── Startzustand ──────────────────────────────────────────────────────────
    cash   = TOTAL_CAPITAL
    pos_4h: Position | None = None
    pos_1d: Position | None = None
    pos_1w: Position | None = None

    # Cooldowns
    cd_4h   = 0   # 4h-Bars
    cd_1d   = 0   # Day-Open-Events
    cd_1w   = 0   # Week-Open-Events

    # Virtuelle Equity pro Bot (für DD-Schutz, je 1/3 Startkapital)
    INIT_BOT = TOTAL_CAPITAL / 3
    virt_4h  = INIT_BOT
    virt_1d  = INIT_BOT
    virt_1w  = INIT_BOT
    win_4h: deque[float] = deque([INIT_BOT], maxlen=DD_WINDOW_4H)
    win_1d: deque[float] = deque([INIT_BOT], maxlen=DD_WINDOW_1D)
    win_1w: deque[float] = deque([INIT_BOT], maxlen=DD_WINDOW_1W)

    all_trades: list[dict] = []
    equity:     list[dict] = []
    global_peak = TOTAL_CAPITAL   # Portfolio-Peak für globale DD-Bremse

    for i, (ts, row_4h) in enumerate(df_4h.iterrows()):
        kurs   = float(row_4h["Close"])
        vola   = _nan_default(row_4h.get("Vola"), 5.0)
        atr_4h = _nan_default(row_4h.get("ATR"),  0.10 * kurs)

        row_1d = df_1d_on_4h.iloc[i]
        row_1w = df_1w_on_4h.iloc[i]

        is_day_open  = (ts.hour == 0)
        is_week_open = (ts.hour == 0 and ts.weekday() == 0)

        # ── Funding Short alle 2 × 4h-Bars (= 8h) ───────────────────────────
        if i % FUNDING_CADENCE_4H == 0:
            if pos_4h is not None and pos_4h.side == "short":
                cash -= pos_4h.entry_cash * FUNDING8 * pos_4h.hebel
            if pos_1d is not None and pos_1d.side == "short":
                cash -= pos_1d.entry_cash * FUNDING8 * pos_1d.hebel
            # 1w ist LONG_ONLY → kein Funding

        # ── Portfolio-Wert ────────────────────────────────────────────────────
        pv = cash
        if pos_4h: pv += _mtm(pos_4h.entry_cash, pos_4h.hebel, pos_4h.entry_price, kurs, pos_4h.side)
        if pos_1d: pv += _mtm(pos_1d.entry_cash, pos_1d.hebel, pos_1d.entry_price, kurs, pos_1d.side)
        if pos_1w: pv += _mtm(pos_1w.entry_cash, pos_1w.hebel, pos_1w.entry_price, kurs, pos_1w.side)
        equity.append({"ts": ts, "equity": pv})
        global_peak = max(global_peak, pv)
        global_brake = (global_dd_brake is not None
                        and (pv - global_peak) / global_peak < -global_dd_brake)

        # ── Cooldown-Countdown (vor Exit, damit Exit-Iteration korrekt zählt) ──
        cd_4h = max(0, cd_4h - 1)
        if is_day_open  and cd_1d > 0: cd_1d -= 1
        if is_week_open and cd_1w > 0: cd_1w -= 1

        # ── 4h-Bot: Exit ──────────────────────────────────────────────────────
        if pos_4h is not None:
            ex_r = None
            ex_p = kurs

            if pos_4h.side == "long":
                if kurs <= pos_4h.sl_price:
                    ex_r, ex_p = "SL", _sell_price(pos_4h.sl_price, vola)
                elif pos_4h.tp_price and kurs >= pos_4h.tp_price:
                    ex_r, ex_p = "TP", _sell_price(pos_4h.tp_price, vola)
                elif bull_exit_4h(row_4h):
                    ex_r, ex_p = "Cover", _sell_price(kurs, vola)
                if ex_r is None and pos_4h.trail_price is not None and atr_4h > 0:
                    tm = TRAIL_MULT_4h.get(pos_4h.regime, 3.0) or 3.0
                    pos_4h.trail_price = max(pos_4h.trail_price, kurs - tm * atr_4h)
                    if kurs <= pos_4h.trail_price:
                        ex_r, ex_p = "Trail", _sell_price(pos_4h.trail_price, vola)
            else:
                if kurs >= pos_4h.sl_price:
                    ex_r, ex_p = "SL", _buy_price(pos_4h.sl_price, vola)
                elif pos_4h.tp_price and kurs <= pos_4h.tp_price:
                    ex_r, ex_p = "TP", _buy_price(pos_4h.tp_price, vola)
                elif bear_cover_4h(row_4h):
                    ex_r, ex_p = "Cover", _buy_price(kurs, vola)

            if ex_r:
                proceeds = _exit_pnl(pos_4h.entry_cash, pos_4h.hebel,
                                     pos_4h.entry_price, ex_p, pos_4h.side)
                pnl      = proceeds - pos_4h.entry_cash
                all_trades.append({
                    "bot": "4h",
                    "Entry_ts":    df_4h.index[pos_4h.candle_idx],
                    "Exit_ts":     ts,
                    "Side":        pos_4h.side,
                    "Entry_price": pos_4h.entry_price,
                    "Exit_price":  ex_p,
                    "Hebel":       pos_4h.hebel,
                    "PnL_pct":     pnl / pos_4h.entry_cash * 100,
                    "Entry_cash":  pos_4h.entry_cash,
                    "Regime":      pos_4h.regime,
                    "Exit_reason": ex_r,
                    "IS":          df_4h.index[pos_4h.candle_idx] < pd.Timestamp(WF_SPLIT),
                })
                virt_4h += pnl
                cash    += proceeds
                pos_4h   = None
                cd_4h    = COOLDOWN_4H_BARS

        # ── 1d-Bot: Exit (nur am Day-Open, wie Standalone) ────────────────────
        if pos_1d is not None and is_day_open:
            atr_1d = _nan_default(row_1d.get("ATR"), 0.02 * kurs)
            ex_r   = None
            ex_p   = kurs

            if pos_1d.side == "long":
                if kurs <= pos_1d.sl_price:
                    ex_r, ex_p = "SL", _sell_price(pos_1d.sl_price, vola)
                elif pos_1d.tp_price and kurs >= pos_1d.tp_price:
                    ex_r, ex_p = "TP", _sell_price(pos_1d.tp_price, vola)
                elif bull_exit_1d(row_1d):
                    ex_r, ex_p = "Cover", _sell_price(kurs, vola)
                if ex_r is None and pos_1d.trail_price is not None and atr_1d > 0:
                    tm = TRAIL_MULT_1d.get(pos_1d.regime, 3.0) or 3.0
                    pos_1d.trail_price = max(pos_1d.trail_price, kurs - tm * atr_1d)
                    if kurs <= pos_1d.trail_price:
                        ex_r, ex_p = "Trail", _sell_price(pos_1d.trail_price, vola)
            else:
                if kurs >= pos_1d.sl_price:
                    ex_r, ex_p = "SL", _buy_price(pos_1d.sl_price, vola)
                elif pos_1d.tp_price and kurs <= pos_1d.tp_price:
                    ex_r, ex_p = "TP", _buy_price(pos_1d.tp_price, vola)
                elif bear_cover_1d(row_1d):
                    ex_r, ex_p = "Cover", _buy_price(kurs, vola)

            if ex_r:
                proceeds = _exit_pnl(pos_1d.entry_cash, pos_1d.hebel,
                                     pos_1d.entry_price, ex_p, pos_1d.side)
                pnl      = proceeds - pos_1d.entry_cash
                all_trades.append({
                    "bot": "1d",
                    "Entry_ts":    df_4h.index[pos_1d.candle_idx],
                    "Exit_ts":     ts,
                    "Side":        pos_1d.side,
                    "Entry_price": pos_1d.entry_price,
                    "Exit_price":  ex_p,
                    "Hebel":       pos_1d.hebel,
                    "PnL_pct":     pnl / pos_1d.entry_cash * 100,
                    "Entry_cash":  pos_1d.entry_cash,
                    "Regime":      pos_1d.regime,
                    "Exit_reason": ex_r,
                    "IS":          df_4h.index[pos_1d.candle_idx] < pd.Timestamp(WF_SPLIT),
                })
                virt_1d += pnl
                cash    += proceeds
                pos_1d   = None
                cd_1d    = COOLDOWN_1D_OPENS

        # ── 1w-Bot: Exit (LONG_ONLY) ──────────────────────────────────────────
        if pos_1w is not None:
            atr_1w = _nan_default(row_1w.get("ATR"), 0.10 * kurs)
            ex_r   = None
            ex_p   = kurs

            if kurs <= pos_1w.sl_price:
                ex_r, ex_p = "SL", _sell_price(pos_1w.sl_price, vola)
            elif pos_1w.tp_price and kurs >= pos_1w.tp_price:
                ex_r, ex_p = "TP", _sell_price(pos_1w.tp_price, vola)
            elif ex_r is None and pos_1w.trail_price is not None and atr_1w > 0:
                tm = TRAIL_MULT_1w.get(pos_1w.regime, 3.0) or 3.0
                pos_1w.trail_price = max(pos_1w.trail_price, kurs - tm * atr_1w)
                if kurs <= pos_1w.trail_price:
                    ex_r, ex_p = "Trail", _sell_price(pos_1w.trail_price, vola)

            if ex_r:
                proceeds = _exit_pnl(pos_1w.entry_cash, pos_1w.hebel,
                                     pos_1w.entry_price, ex_p, pos_1w.side)
                pnl      = proceeds - pos_1w.entry_cash
                all_trades.append({
                    "bot": "1w",
                    "Entry_ts":    df_4h.index[pos_1w.candle_idx],
                    "Exit_ts":     ts,
                    "Side":        pos_1w.side,
                    "Entry_price": pos_1w.entry_price,
                    "Exit_price":  ex_p,
                    "Hebel":       pos_1w.hebel,
                    "PnL_pct":     pnl / pos_1w.entry_cash * 100,
                    "Entry_cash":  pos_1w.entry_cash,
                    "Regime":      pos_1w.regime,
                    "Exit_reason": ex_r,
                    "IS":          df_4h.index[pos_1w.candle_idx] < pd.Timestamp(WF_SPLIT),
                })
                virt_1w += pnl
                cash    += proceeds
                pos_1w   = None
                cd_1w    = COOLDOWN_1W_OPENS

        # ── Virtuelle Equity für DD-Tracking aktualisieren ────────────────────
        cur_4h = virt_4h + ((_mtm(pos_4h.entry_cash, pos_4h.hebel,
                                   pos_4h.entry_price, kurs, pos_4h.side) - pos_4h.entry_cash)
                             if pos_4h else 0)
        cur_1d = virt_1d + ((_mtm(pos_1d.entry_cash, pos_1d.hebel,
                                   pos_1d.entry_price, kurs, pos_1d.side) - pos_1d.entry_cash)
                             if pos_1d else 0)
        cur_1w = virt_1w + ((_mtm(pos_1w.entry_cash, pos_1w.hebel,
                                   pos_1w.entry_price, kurs, pos_1w.side) - pos_1w.entry_cash)
                             if pos_1w else 0)
        win_4h.append(cur_4h)
        if is_day_open:  win_1d.append(cur_1d)
        if is_week_open: win_1w.append(cur_1w)

        # ── 4h-Bot: Entry ─────────────────────────────────────────────────────
        if pos_4h is None and cd_4h == 0 and cash >= 1.0 and pos_size_4h > 0 and not global_brake:
            regime_4h = row_4h.get("Regime", "")
            if regime_4h in ("bull", "bear"):
                sig = get_signal_4h(row_4h, regime_4h)
                if sig != 0:
                    sl_m, tp_m = SL_TP_4h.get(regime_4h, (2.0, 8.0))
                    hebel      = HEBEL_4h.get(regime_4h, 2.0)
                    peak_4h    = max(win_4h) if win_4h else INIT_BOT
                    if peak_4h > 0 and (cur_4h - peak_4h) / peak_4h < -0.20:
                        hebel = 1.0
                    ec = cash * pos_size_4h
                    if sig == 1:
                        ep    = _buy_price(kurs, vola)
                        trail = TRAIL_MULT_4h.get(regime_4h, 3.0) or 3.0
                        pos_4h = Position("long", ep, ec, hebel,
                                          ep - sl_m * atr_4h, ep + tp_m * atr_4h,
                                          ep - trail * atr_4h, regime_4h, i)
                    else:
                        ep    = _sell_price(kurs, vola)
                        pos_4h = Position("short", ep, ec, hebel,
                                          ep + sl_m * atr_4h, ep - tp_m * atr_4h,
                                          None, regime_4h, i)
                    cash -= ec

        # ── 1d-Bot: Entry (nur am Day-Open) ───────────────────────────────────
        if is_day_open and pos_1d is None and cd_1d == 0 and cash >= 1.0 and pos_size_1d > 0 and not global_brake:
            regime_1d = row_1d.get("Regime", "")
            if pd.notna(regime_1d) and regime_1d in ("bull", "bear"):
                atr_1d = _nan_default(row_1d.get("ATR"), 0.02 * kurs)
                sig    = get_signal_1d(row_1d, regime_1d)
                if sig != 0:
                    sl_m, tp_m = SL_TP_1d.get(regime_1d, (1.0, 8.0))
                    hebel      = HEBEL_1d.get(regime_1d, 2.0)
                    peak_1d    = max(win_1d) if win_1d else INIT_BOT
                    if peak_1d > 0 and (cur_1d - peak_1d) / peak_1d < -0.20:
                        hebel = 1.0
                    ec = cash * pos_size_1d
                    if sig == 1:
                        ep    = _buy_price(kurs, vola)
                        trail = TRAIL_MULT_1d.get(regime_1d, 3.0) or 3.0
                        pos_1d = Position("long", ep, ec, hebel,
                                          ep - sl_m * atr_1d, ep + tp_m * atr_1d,
                                          ep - trail * atr_1d, regime_1d, i)
                    else:
                        ep    = _sell_price(kurs, vola)
                        pos_1d = Position("short", ep, ec, hebel,
                                          ep + sl_m * atr_1d, ep - tp_m * atr_1d,
                                          None, regime_1d, i)
                    cash -= ec

        # ── 1w-Bot: Entry (nur am Week-Open) ──────────────────────────────────
        if is_week_open and pos_1w is None and cd_1w == 0 and cash >= 1.0 and pos_size_1w > 0 and not global_brake:
            atr_1w  = _nan_default(row_1w.get("ATR"), 0.10 * kurs)
            sig_1w, regime_1w = get_signal_1w(row_1w)
            if sig_1w == 1:
                sl_m, tp_m = SL_TP_1w.get(regime_1w, (1.5, 8.0))
                hebel      = HEBEL_1w.get(regime_1w, 2.0)
                peak_1w    = max(win_1w) if win_1w else INIT_BOT
                if peak_1w > 0 and (cur_1w - peak_1w) / peak_1w < -0.20:
                    hebel = 1.0
                ec    = cash * pos_size_1w
                ep    = _buy_price(kurs, vola)
                trail = TRAIL_MULT_1w.get(regime_1w, 3.0) or 3.0
                pos_1w = Position("long", ep, ec, hebel,
                                  ep - sl_m * atr_1w, ep + tp_m * atr_1w,
                                  ep - trail * atr_1w, regime_1w, i)
                cash -= ec

    equity_s   = pd.DataFrame(equity).set_index("ts")["equity"]
    trades_df  = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    return trades_df, equity_s


# ─── Kennzahlen ───────────────────────────────────────────────────────────────

def _sharpe(eq: pd.Series, periods_per_year: float) -> float:
    r = eq.pct_change().dropna()
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def _max_dd(eq: pd.Series) -> float:
    roll_max = eq.cummax()
    dd       = (eq - roll_max) / roll_max
    return float(dd.min())


def _print_bot_stats(label: str, df: pd.DataFrame, equity: pd.Series) -> None:
    if df.empty:
        print(f"\n  {label}: keine Trades")
        return

    def section(mask, tag):
        sub = df[mask]
        if sub.empty:
            return
        wr    = (sub["PnL_pct"] > 0).mean() * 100
        sigma = sub["PnL_pct"].std()
        print(f"    {tag:4s}  n={len(sub):3d}  WR={wr:5.1f}%  "
              f"ΣPnL={sub['PnL_pct'].sum():+8.1f}%  "
              f"Ø={sub['PnL_pct'].mean():+6.2f}%/Trade  σ={sigma:.1f}%")

    print(f"\n  ── {label} ({'IS+OOS' if 'IS' in df.columns else 'gesamt'}) ──")
    section(df["IS"] == True,  "IS ")
    section(df["IS"] == False, "OOS")
    section(pd.Series([True] * len(df), index=df.index), "ALL")


def _print_stats(trades: pd.DataFrame, equity: pd.Series, label: str = "Combined") -> None:
    wf = pd.Timestamp(WF_SPLIT)
    eq_is  = equity[equity.index <  wf]
    eq_oos = equity[equity.index >= wf]

    start_cap = equity.iloc[0]
    end_cap   = equity.iloc[-1]
    ret_total = (end_cap / start_cap - 1) * 100

    sharpe_4h = _sharpe(equity, 365 * 6)
    max_dd    = _max_dd(equity)

    print(f"\n{'='*60}")
    print(f"  {label} — Gesamtübersicht")
    print(f"{'='*60}")
    print(f"  Startkapital:  ${start_cap:>8.2f}")
    print(f"  Endkapital:    ${end_cap:>8.2f}")
    print(f"  Rendite:       {ret_total:+.1f}%")
    print(f"  Sharpe (4h):   {sharpe_4h:.2f}")
    print(f"  Max Drawdown:  {max_dd:.1%}")

    if not eq_is.empty and not eq_oos.empty:
        ret_is  = (eq_is.iloc[-1]  / eq_is.iloc[0]  - 1) * 100
        ret_oos = (eq_oos.iloc[-1] / eq_oos.iloc[0] - 1) * 100
        print(f"  IS-Rendite:    {ret_is:+.1f}%  ({eq_is.index[0].date()} – {eq_is.index[-1].date()})")
        print(f"  OOS-Rendite:   {ret_oos:+.1f}%  ({eq_oos.index[0].date()} – {eq_oos.index[-1].date()})")

    if trades.empty:
        print("  Keine Trades.")
        return

    print(f"\n  Trades gesamt: {len(trades)}")
    for bot in ["4h", "1d", "1w"]:
        sub = trades[trades["bot"] == bot] if "bot" in trades.columns else pd.DataFrame()
        _print_bot_stats(bot, sub, equity)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kombinierter 1d+4h+1w Backtest")
    parser.add_argument("--sanity_4h", action="store_true",
                        help="Nur 4h-Bot aktiv (pos_size_1d=0, pos_size_1w=0) — Sanity-Check")
    parser.add_argument("--sanity_1d", action="store_true",
                        help="Nur 1d-Bot aktiv — Sanity-Check")
    parser.add_argument("--sanity_1w", action="store_true",
                        help="Nur 1w-Bot aktiv — Sanity-Check")
    args = parser.parse_args()

    print("Lade Daten …")
    df_4h_raw = fetch_binance_4h("2021-09-01")
    df_1d_raw = fetch_binance_1d("2021-09-01")
    df_1w_raw = fetch_binance_1w("2020-01-01")

    # Wochentag-Check: Binance 1w sollte immer Montag (0) sein
    weekdays = df_1w_raw.index.weekday.unique().tolist()
    if weekdays != [0]:
        print(f"  WARNUNG: 1w-Daten Wochentage = {weekdays} (erwartet [0] = Montag)")
    else:
        print("  1w-Wochentag OK: Montag ✓")

    ps4h = 0.0 if (args.sanity_1d or args.sanity_1w) else POS_SIZE_4H
    ps1d = 0.0 if (args.sanity_4h or args.sanity_1w) else POS_SIZE_1D
    ps1w = 0.0 if (args.sanity_4h or args.sanity_1d) else POS_SIZE_1W

    label = "Combined (1d+4h+1w)"
    if args.sanity_4h: label = "Sanity: nur 4h"
    if args.sanity_1d: label = "Sanity: nur 1d"
    if args.sanity_1w: label = "Sanity: nur 1w"

    print(f"Simuliere: {label} …")
    trades, equity = run_backtest_combined(
        df_4h_raw, df_1d_raw, df_1w_raw,
        pos_size_4h=ps4h, pos_size_1d=ps1d, pos_size_1w=ps1w,
    )

    _print_stats(trades, equity, label)


if __name__ == "__main__":
    main()
