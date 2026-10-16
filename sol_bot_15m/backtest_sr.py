"""
backtest_sr.py — Backtest-Engine für die S/R-Double-Test-Strategie (Typ 2)

Wichtige Unterschiede zur standard backtest.py:
- Inkrementeller LevelTracker: process_bar(i) → get_sr_entry(i) — kein Look-Ahead
- SL/TP aus strategy_sr (level-basiert, nicht ATR×Mult)
- Trail für Longs wenn trail_init gesetzt
- Kein Regime-Filter
- MAX_DAILY_TRADES = 2
- COOLDOWN = 8 Bars (2h) nach Exit
- Walk-Forward: IS bis 2024-12-31, OOS ab 2025-01-01
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from cost_model import (GEBUEHR, SLIPPAGE, SPREAD, FUNDING8,
                        FUNDING_CADENCE_15M, STARTKAPITAL)
from sol_bot_15m.indicators import add_indicators
from sol_bot_15m.levels     import LevelTracker
from sol_bot_15m.strategy_sr import get_sr_entry, TRAIL_MULT

WF_SPLIT         = "2025-01-01"
MAX_DAILY_TRADES = 2
COOLDOWN         = 8   # 8 × 15min = 2h
POS_SIZE         = 0.15


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
                 "sl_price", "tp_price", "trail_price",
                 "level_type", "candle_idx"]

    def __init__(self, side, entry_price, entry_cash, hebel,
                 sl_price, tp_price, trail_price, level_type, candle_idx):
        self.side        = side
        self.entry_price = entry_price
        self.entry_cash  = entry_cash
        self.hebel       = hebel
        self.sl_price    = sl_price
        self.tp_price    = tp_price
        self.trail_price = trail_price
        self.level_type  = level_type
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

def run_backtest_sr(
    df_raw: pd.DataFrame,
    pos_size: float = POS_SIZE,
    max_daily_trades: int = MAX_DAILY_TRADES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inkrementeller S/R-Backtest (Typ 2: Intraday-Double-Test).
    Rückgabe: (trades_df, equity_df)
    """
    df = add_indicators(df_raw)

    tracker      = LevelTracker()
    cash         = STARTKAPITAL
    pos: Position | None = None
    equity       = []
    trades       = []

    cooldown_left = 0
    trades_today  = 0
    current_day   = None

    for i, (ts, row) in enumerate(df.iterrows()):
        kurs = row["Close"]
        atr  = row["ATR"]  if not pd.isna(row["ATR"])  else 0.003 * kurs
        vola = row["Vola"] if not pd.isna(row["Vola"]) else 1.0

        # ── Level inkrementell aktualisieren ─────────────────────────────
        tracker.process_bar(i, df)

        # Tages-Counter zurücksetzen
        day = ts.date()
        if day != current_day:
            current_day  = day
            trades_today = 0

        port_value = cash if pos is None else (cash + _mtm(pos, kurs))
        equity.append({"ts": ts, "equity": port_value})

        # ── Funding (Short, alle 32 Kerzen = 8h) ────────────────────────
        if pos is not None and pos.side == "short" and i % FUNDING_CADENCE_15M == 0:
            cash -= pos.entry_cash * FUNDING8 * pos.hebel

        # ── SL / TP / Trail ──────────────────────────────────────────────
        if pos is not None:
            exit_reason = None
            exit_price  = kurs

            if pos.side == "long":
                if kurs <= pos.sl_price:
                    exit_reason = "SL"
                    exit_price  = _sell_price(pos.sl_price, vola)
                elif kurs >= pos.tp_price:
                    exit_reason = "TP"
                    exit_price  = _sell_price(pos.tp_price, vola)
                elif pos.trail_price is not None and atr > 0:
                    new_trail       = kurs - TRAIL_MULT * atr
                    pos.trail_price = max(pos.trail_price, new_trail)
                    if kurs <= pos.trail_price:
                        exit_reason = "Trail"
                        exit_price  = _sell_price(pos.trail_price, vola)

            else:  # short
                if kurs >= pos.sl_price:
                    exit_reason = "SL"
                    exit_price  = _buy_price(pos.sl_price, vola)
                elif kurs <= pos.tp_price:
                    exit_reason = "TP"
                    exit_price  = _buy_price(pos.tp_price, vola)

            if exit_reason:
                proceeds = _exit_pnl(pos, exit_price)
                pnl_pct  = (proceeds - pos.entry_cash) / pos.entry_cash * 100
                trades.append({
                    "Entry_ts":    df.index[pos.candle_idx],
                    "Exit_ts":     ts,
                    "Side":        pos.side,
                    "Entry_price": pos.entry_price,
                    "Exit_price":  exit_price,
                    "Hebel":       pos.hebel,
                    "PnL_pct":     pnl_pct,
                    "Entry_cash":  pos.entry_cash,
                    "Level_type":  pos.level_type,
                    "Exit_reason": exit_reason,
                    "IS":          df.index[pos.candle_idx] < pd.Timestamp(WF_SPLIT),
                })
                cash         += proceeds
                pos           = None
                cooldown_left = COOLDOWN
                trades_today += 1

        # ── Cooldown / Daily-Limit ────────────────────────────────────────
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if trades_today >= max_daily_trades:
            continue

        # ── Entry ─────────────────────────────────────────────────────────
        if pos is None:
            sig = get_sr_entry(i, df, tracker, atr)
            if sig is not None:
                entry_cash = cash * pos_size
                if sig["side"] == "long":
                    ep = _buy_price(kurs, vola)
                else:
                    ep = _sell_price(kurs, vola)

                pos = Position(
                    side        = sig["side"],
                    entry_price = ep,
                    entry_cash  = entry_cash,
                    hebel       = sig["hebel"],
                    sl_price    = sig["sl"],
                    tp_price    = sig["tp"],
                    trail_price = sig["trail_init"],
                    level_type  = sig["level_type"],
                    candle_idx  = i,
                )
                cash -= entry_cash

    equity_df = pd.DataFrame(equity).set_index("ts")["equity"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, equity_df
