"""
backtest.py — Backtest-Engine für den 1W-Bot (EMA-Crossover)

Strategie:
  EMA10 kreuzt EMA21 nach oben → Long
  EMA10 kreuzt EMA21 nach unten → Short
  Exit: SL / TP / Trail (kein Crossover-Exit)

Parameter vs. 1d-Engine:
- Funding: 21× FUNDING8 pro Kerze (7 Tage × 3 Cycles)
- Startkapital: $60 (30% des 1d-Systems)
- Sharpe annualisiert auf √52
- Walk-Forward-Split: IS bis 2024-12-31, OOS ab 2025-01-01
- Cooldown: 4 Kerzen (= 4 Wochen)
"""

import time
import requests
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cost_model import GEBUEHR, SLIPPAGE, SPREAD, FUNDING8

from sol_bot_1w.indicators import add_indicators
from sol_bot_1w.strategy   import get_signal, SL_TP, TRAIL_MULT, HEBEL, LOOKBACK_K

WF_SPLIT              = "2025-01-01"
STARTKAPITAL          = 60.0
COOLDOWN              = 4       # 4 Wochen
FUNDING_1W_PER_CANDLE = 21 * FUNDING8   # 7 Tage × 3 Cycles


# ─── Datenabruf ───────────────────────────────────────────────────────────────

def fetch_binance_1w(start: str = "2020-01-01") -> pd.DataFrame:
    url      = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int(pd.Timestamp.now().timestamp() * 1000)
    rows     = []

    ts = start_ms
    while ts < end_ms:
        r = requests.get(url, params={
            "symbol": "SOLUSDT", "interval": "1w",
            "startTime": ts, "limit": 1000
        }, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        ts = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.05)

    df = pd.DataFrame(rows, columns=[
        "OpenTime", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVol", "Trades", "TakerBase", "TakerQuote", "Ignore"
    ])
    df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
    df = df.set_index("OpenTime")
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.index.name = None
    return df


# ─── Kostenmodell ─────────────────────────────────────────────────────────────

def _buy_price(kurs: float, vola: float) -> float:
    slip = min(SLIPPAGE + vola * 0.00003, 0.001)
    return kurs * (1 + slip + SPREAD / 2)


def _sell_price(kurs: float, vola: float) -> float:
    slip = min(SLIPPAGE + vola * 0.00003, 0.001)
    return kurs * (1 - slip - SPREAD / 2)


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


def _exit_pnl(pos: Position, exit_price: float) -> float:
    if pos.side == "long":
        raw = (exit_price - pos.entry_price) / pos.entry_price
    else:
        raw = (pos.entry_price - exit_price) / pos.entry_price

    notional_entry = pos.entry_cash * pos.hebel
    gross_pnl      = notional_entry * raw
    fee_entry      = notional_entry * GEBUEHR
    exit_notional  = abs(notional_entry * exit_price / pos.entry_price)
    fee_exit       = exit_notional * GEBUEHR
    net            = pos.entry_cash + gross_pnl - fee_entry - fee_exit
    return max(net, 0.01)


def _mtm(pos: Position, kurs: float) -> float:
    if pos.side == "long":
        raw = (kurs - pos.entry_price) / pos.entry_price
    else:
        raw = (pos.entry_price - kurs) / pos.entry_price
    return max(pos.entry_cash + pos.entry_cash * pos.hebel * raw, 0.01)


# ─── Backtest-Engine ──────────────────────────────────────────────────────────

def run_backtest(df_raw: pd.DataFrame,
                 max_pos_size: float = 0.30,
                 lookback_k:   int   = LOOKBACK_K,
                 sl_mult:      float | None = None,
                 hebel_bull:   float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_indicators(df_raw)

    cash          = STARTKAPITAL
    pos: Position | None = None
    equity        = []
    trades        = []
    peak          = STARTKAPITAL
    DD_RESET_BARS = 52
    equity_window: list[float] = []
    cooldown_left = 0

    for i, (ts, row) in enumerate(df.iterrows()):
        kurs   = row["Close"]
        atr    = row["ATR"]  if not pd.isna(row["ATR"])  else 0.10 * kurs
        vola   = row["Vola"] if not pd.isna(row["Vola"]) else 5.0

        # ── Funding Short: 21 Cycles pro Weekly-Kerze ───────────────────────
        if pos is not None and pos.side == "short":
            cash -= pos.entry_cash * FUNDING_1W_PER_CANDLE * pos.hebel

        port_value = cash if pos is None else (cash + _mtm(pos, kurs))
        equity_window.append(port_value)
        if len(equity_window) > DD_RESET_BARS:
            equity_window.pop(0)
        peak = max(equity_window)
        dd   = (port_value - peak) / peak

        equity.append({"ts": ts, "equity": port_value})

        # ── SL / TP / Trail ──────────────────────────────────────────────────
        if pos is not None:
            exit_reason = None
            exit_price  = kurs

            if pos.side == "long":
                if kurs <= pos.sl_price:
                    exit_reason = "SL"
                    exit_price  = _sell_price(pos.sl_price, vola)
                elif pos.tp_price is not None and kurs >= pos.tp_price:
                    exit_reason = "TP"
                    exit_price  = _sell_price(pos.tp_price, vola)

                if pos.trail_price is not None and atr > 0 and exit_reason is None:
                    trail_mult  = TRAIL_MULT.get(pos.regime, 3.0) or 3.0
                    new_trail   = kurs - trail_mult * atr
                    pos.trail_price = max(pos.trail_price, new_trail)
                    if kurs <= pos.trail_price:
                        exit_reason = "Trail"
                        exit_price  = _sell_price(pos.trail_price, vola)

            else:  # short
                if kurs >= pos.sl_price:
                    exit_reason = "SL"
                    exit_price  = _buy_price(pos.sl_price, vola)
                elif pos.tp_price is not None and kurs <= pos.tp_price:
                    exit_reason = "TP"
                    exit_price  = _buy_price(pos.tp_price, vola)

                if pos.trail_price is not None and atr > 0 and exit_reason is None:
                    trail_mult  = TRAIL_MULT.get(pos.regime, 3.0) or 3.0
                    new_trail   = kurs + trail_mult * atr
                    pos.trail_price = min(pos.trail_price, new_trail)
                    if kurs >= pos.trail_price:
                        exit_reason = "Trail"
                        exit_price  = _buy_price(pos.trail_price, vola)

            if exit_reason:
                proceeds    = _exit_pnl(pos, exit_price)
                pnl_pct     = (proceeds - pos.entry_cash) / pos.entry_cash * 100
                trades.append({
                    "Entry_ts":    df.index[pos.candle_idx],
                    "Exit_ts":     ts,
                    "Side":        pos.side,
                    "Entry_price": pos.entry_price,
                    "Exit_price":  exit_price,
                    "Hebel":       pos.hebel,
                    "PnL_pct":     pnl_pct,
                    "Entry_cash":  pos.entry_cash,
                    "Regime":      pos.regime,
                    "Exit_reason": exit_reason,
                    "IS":          df.index[pos.candle_idx] < pd.Timestamp(WF_SPLIT),
                })
                cash         += proceeds
                pos           = None
                cooldown_left = COOLDOWN

        # ── Cooldown ──────────────────────────────────────────────────────────
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        # ── Signal ────────────────────────────────────────────────────────────
        if pos is None:
            sig, regime = get_signal(row)

            if sig == 0:
                continue

            _sl, tp_mult = SL_TP.get(regime, (2.0, 6.0))
            sl_mult_eff  = sl_mult if sl_mult is not None else _sl
            hebel        = HEBEL.get(regime, 1.0)
            if regime == "bull" and hebel_bull is not None:
                hebel = hebel_bull

            if dd < -0.20:
                hebel = 1.0

            entry_cash = cash * max_pos_size

            if sig == 1:
                ep    = _buy_price(kurs, vola)
                sl    = ep - sl_mult_eff * atr
                tp    = ep + tp_mult * atr
                trail = ep - (TRAIL_MULT.get(regime, 3.0) or 3.0) * atr
                pos   = Position("long", ep, entry_cash, hebel,
                                 sl, tp, trail, regime, i)
                cash -= entry_cash

    equity_df = pd.DataFrame(equity).set_index("ts")["equity"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, equity_df
