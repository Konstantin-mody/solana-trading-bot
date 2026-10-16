"""
paper_bot_combined.py — Paper Trading Bot: Kombiniertes 1d+4h+1w System

Spiegelt run_backtest_combined.py exakt.
Läuft alle 4h (00:05, 04:05, …). Virtuelle Positionen, echte Binance-Preise.

Start: python3 paper_bot_combined.py
Stop:  Ctrl+C  (State wird atomar gespeichert)

Konfigurationsübersicht (beim Live-Start verifizieren):
  TOTAL_CAPITAL = 200.0
  POS_SIZE_4H   = 0.20  (20% des freien Cash)
  POS_SIZE_1D   = 0.20
  POS_SIZE_1W   = 1/3
  COOLDOWN_4H_BARS  = 12  (= 2 Tage)
  COOLDOWN_1D_OPENS = 2   (= 2 Tage)
  COOLDOWN_1W_OPENS = 4   (= 4 Wochen)
  global_dd_brake   = None (deaktiviert — Analyse ergab kontraproduktiv)
"""

import json, sqlite3, time, os, sys, logging
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cost_model import GEBUEHR, SLIPPAGE, SPREAD, FUNDING8

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

# ── Konfiguration ──────────────────────────────────────────────────────────────

STATE_FILE  = "paper_state_combined.json"
DB_FILE     = "paper_trades_combined.db"
LOG_FILE    = "paper_bot_combined.log"

TOTAL_CAPITAL     = 200.0
POS_SIZE_4H       = 0.20
POS_SIZE_1D       = 0.20
POS_SIZE_1W       = 1 / 3
COOLDOWN_4H_BARS  = 12
COOLDOWN_1D_OPENS = 2
COOLDOWN_1W_OPENS = 4
DD_WINDOW_4H      = 90 * 6
DD_WINDOW_1D      = 90
DD_WINDOW_1W      = 52

LOOKBACK_4H = 600   # 4h-Kerzen (~100 Tage; deckt MA200=800h ab)
LOOKBACK_1D = 300   # 1d-Kerzen (~300 Tage; deckt MA200=200d ab)
LOOKBACK_1W = 80    # 1w-Kerzen (~80 Wochen; deckt EMA21 ab)

FUNDING_HOURS = {0, 8, 16}   # Binance Perpetual Funding UTC

# ── Logging ────────────────────────────────────────────────────────────────────

log = logging.getLogger("paper_combined")


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(LOG_FILE)],
    )


# ── Preismodell (identisch zu run_backtest_combined.py) ────────────────────────

def _buy_price(kurs: float, vola: float) -> float:
    slip = min(SLIPPAGE + vola * 0.00003, 0.001)
    return kurs * (1 + slip + SPREAD / 2)


def _sell_price(kurs: float, vola: float) -> float:
    slip = min(SLIPPAGE + vola * 0.00003, 0.001)
    return kurs * (1 - slip - SPREAD / 2)


def _exit_pnl(entry_cash: float, hebel: float,
               entry_price: float, exit_price: float, side: str) -> float:
    raw = (exit_price - entry_price) / entry_price if side == "long" \
          else (entry_price - exit_price) / entry_price
    notional  = entry_cash * hebel
    gross_pnl = notional * raw
    fee_entry = notional * GEBUEHR
    fee_exit  = abs(notional * exit_price / entry_price) * GEBUEHR
    return max(entry_cash + gross_pnl - fee_entry - fee_exit, 0.01)


def _mtm(entry_cash: float, hebel: float,
          entry_price: float, kurs: float, side: str) -> float:
    raw = (kurs - entry_price) / entry_price if side == "long" \
          else (entry_price - kurs) / entry_price
    return max(entry_cash + entry_cash * hebel * raw, 0.01)


def _nan(val, default):
    if val is None:
        return default
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


# ── Position ───────────────────────────────────────────────────────────────────

class Position:
    __slots__ = ["side", "entry_price", "entry_cash", "hebel",
                 "sl_price", "tp_price", "trail_price", "regime", "entry_ts"]

    def __init__(self, side, entry_price, entry_cash, hebel,
                 sl_price, tp_price, trail_price, regime, entry_ts):
        self.side        = side
        self.entry_price = entry_price
        self.entry_cash  = entry_cash
        self.hebel       = hebel
        self.sl_price    = sl_price
        self.tp_price    = tp_price
        self.trail_price = trail_price
        self.regime      = regime
        self.entry_ts    = str(entry_ts)


def _pos_to_dict(pos) -> dict | None:
    if pos is None:
        return None
    return {s: getattr(pos, s) for s in Position.__slots__}


def _pos_from_dict(d: dict | None) -> Position | None:
    if d is None:
        return None
    p = object.__new__(Position)
    for s in Position.__slots__:
        setattr(p, s, d.get(s))
    return p


# ── SQLite ─────────────────────────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            bot         TEXT,
            Entry_ts    TEXT,
            Exit_ts     TEXT,
            Side        TEXT,
            Entry_price REAL,
            Exit_price  REAL,
            Hebel       REAL,
            PnL_pct     REAL,
            Entry_cash  REAL,
            Regime      TEXT,
            Exit_reason TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS equity_log (
            ts        TEXT PRIMARY KEY,
            portfolio REAL,
            cash      REAL,
            pos_4h    TEXT,
            pos_1d    TEXT,
            pos_1w    TEXT
        )
    """)
    con.commit()
    con.close()


def _log_trade(bot: str, entry_ts, exit_ts, side, ep, xp, hebel, pnl, ec, regime, reason):
    con = sqlite3.connect(DB_FILE)
    con.execute(
        "INSERT INTO trades (bot,Entry_ts,Exit_ts,Side,Entry_price,Exit_price,"
        "Hebel,PnL_pct,Entry_cash,Regime,Exit_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (bot, str(entry_ts), str(exit_ts), side, ep, xp, hebel, pnl, ec, regime, reason),
    )
    con.commit()
    con.close()


def _log_equity(ts, portfolio, cash, pos_4h, pos_1d, pos_1w):
    def _ps(p):
        return f"{p.side}@{p.entry_price:.4f}" if p else None
    con = sqlite3.connect(DB_FILE)
    con.execute(
        "INSERT OR REPLACE INTO equity_log (ts,portfolio,cash,pos_4h,pos_1d,pos_1w) "
        "VALUES (?,?,?,?,?,?)",
        (str(ts), portfolio, cash, _ps(pos_4h), _ps(pos_1d), _ps(pos_1w)),
    )
    con.commit()
    con.close()


# ── State ──────────────────────────────────────────────────────────────────────

INIT_BOT = TOTAL_CAPITAL / 3


def _default_state() -> dict:
    return {
        "cash":       TOTAL_CAPITAL,
        "pos_4h":     None,
        "pos_1d":     None,
        "pos_1w":     None,
        "cd_4h":      0,
        "cd_1d":      0,
        "cd_1w":      0,
        "virt_4h":    INIT_BOT,
        "virt_1d":    INIT_BOT,
        "virt_1w":    INIT_BOT,
        "win_4h":     [INIT_BOT],
        "win_1d":     [INIT_BOT],
        "win_1w":     [INIT_BOT],
        "last_ts":    None,
        "live_since": None,
    }


def load_state() -> dict:
    if not Path(STATE_FILE).exists():
        return _default_state()
    with open(STATE_FILE) as f:
        raw = json.load(f)
    raw["pos_4h"] = _pos_from_dict(raw.get("pos_4h"))
    raw["pos_1d"] = _pos_from_dict(raw.get("pos_1d"))
    raw["pos_1w"] = _pos_from_dict(raw.get("pos_1w"))
    return raw


def save_state(state: dict):
    tmp = STATE_FILE + ".tmp"
    s = {**state,
         "pos_4h": _pos_to_dict(state["pos_4h"]),
         "pos_1d": _pos_to_dict(state["pos_1d"]),
         "pos_1w": _pos_to_dict(state["pos_1w"])}
    with open(tmp, "w") as f:
        json.dump(s, f, default=str)
    os.replace(tmp, STATE_FILE)


# ── Daten-Fetch ────────────────────────────────────────────────────────────────

def _fetch_with_retry(fn, start: str, label: str) -> pd.DataFrame | None:
    for attempt in range(3):
        try:
            df = fn(start)
            if len(df) < 10:
                raise ValueError(f"Zu wenige Kerzen: {len(df)}")
            return df
        except Exception as e:
            wait = [5, 30, 120][attempt]
            log.warning("Fetch %s Fehler (Versuch %d/3): %s — warte %ds",
                        label, attempt + 1, e, wait)
            if attempt < 2:
                time.sleep(wait)
    log.error("Fetch %s nach 3 Versuchen fehlgeschlagen.", label)
    return None


def fetch_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    now = pd.Timestamp.now("UTC").tz_localize(None)
    s4h = (now - pd.Timedelta(hours=LOOKBACK_4H)).strftime("%Y-%m-%d")
    s1d = (now - pd.Timedelta(days=LOOKBACK_1D)).strftime("%Y-%m-%d")
    s1w = (now - pd.Timedelta(weeks=LOOKBACK_1W)).strftime("%Y-%m-%d")

    df4 = _fetch_with_retry(fetch_binance_4h, s4h, "4h")
    df1 = _fetch_with_retry(fetch_binance_1d, s1d, "1d")
    dfw = _fetch_with_retry(fetch_binance_1w, s1w, "1w")
    if df4 is None or df1 is None or dfw is None:
        return None
    return df4, df1, dfw


# ── Indikatoren & Alignment (identisch zu run_backtest_combined.py) ────────────

def build_aligned_frames(df_4h_raw, df_1d_raw, df_1w_raw):
    df_4h = add_ind_4h(df_4h_raw.copy())
    df_4h["Regime"] = classify_regime_4h(df_4h, min_regime=6)

    df_1d = add_ind_1d(df_1d_raw.copy())
    df_1d["Regime"] = classify_regime_1d(df_1d, min_regime=3)

    df_1w = add_ind_1w(df_1w_raw.copy())

    # +24h / +168h Shift (Look-Ahead-Schutz)
    df_1d_sh       = df_1d.copy()
    df_1d_sh.index = df_1d_sh.index + pd.Timedelta(hours=24)
    df_1d_on_4h    = df_1d_sh.reindex(df_4h.index, method="ffill")

    df_1w_sh       = df_1w.copy()
    df_1w_sh.index = df_1w_sh.index + pd.Timedelta(hours=168)
    df_1w_on_4h    = df_1w_sh.reindex(df_4h.index, method="ffill")

    return df_4h, df_1d_on_4h, df_1w_on_4h


# ── Bar-Logik (spiegelt run_backtest_combined.py exakt) ────────────────────────

def process_bar(ts, row_4h, row_1d, row_1w, state: dict) -> list[dict]:
    """
    Verarbeitet eine 4h-Kerze. Mutiert state in-place.
    Gibt Liste der geschlossenen Trades zurück (0–3 Elemente).
    """
    kurs   = float(row_4h["Close"])
    vola   = _nan(row_4h.get("Vola"), 5.0)
    atr_4h = _nan(row_4h.get("ATR"),  0.10 * kurs)

    is_day_open  = (ts.hour == 0)
    is_week_open = (ts.hour == 0 and ts.weekday() == 0)

    cash   = state["cash"]
    pos_4h = state["pos_4h"]
    pos_1d = state["pos_1d"]
    pos_1w = state["pos_1w"]
    cd_4h  = state["cd_4h"]
    cd_1d  = state["cd_1d"]
    cd_1w  = state["cd_1w"]
    virt_4h = state["virt_4h"]
    virt_1d = state["virt_1d"]
    virt_1w = state["virt_1w"]
    win_4h  = deque(state["win_4h"], maxlen=DD_WINDOW_4H)
    win_1d  = deque(state["win_1d"], maxlen=DD_WINDOW_1D)
    win_1w  = deque(state["win_1w"], maxlen=DD_WINDOW_1W)

    closed_trades: list[dict] = []

    # ── Funding Short (Wall-Clock, alle 8h) ──────────────────────────────────
    if ts.hour in FUNDING_HOURS:
        if pos_4h is not None and pos_4h.side == "short":
            cash -= pos_4h.entry_cash * FUNDING8 * pos_4h.hebel
        if pos_1d is not None and pos_1d.side == "short":
            cash -= pos_1d.entry_cash * FUNDING8 * pos_1d.hebel

    # ── Cooldown-Countdown ────────────────────────────────────────────────────
    cd_4h = max(0, cd_4h - 1)
    if is_day_open  and cd_1d > 0: cd_1d -= 1
    if is_week_open and cd_1w > 0: cd_1w -= 1

    # ── 4h-Bot: Exit ──────────────────────────────────────────────────────────
    if pos_4h is not None:
        ex_r = ex_p = None
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
            pnl = proceeds - pos_4h.entry_cash
            closed_trades.append({"bot": "4h", "entry_ts": pos_4h.entry_ts,
                "exit_ts": ts, "side": pos_4h.side, "ep": pos_4h.entry_price,
                "xp": ex_p, "hebel": pos_4h.hebel,
                "pnl_pct": pnl / pos_4h.entry_cash * 100,
                "entry_cash": pos_4h.entry_cash, "regime": pos_4h.regime,
                "reason": ex_r})
            virt_4h += pnl
            cash    += proceeds
            pos_4h   = None
            cd_4h    = COOLDOWN_4H_BARS

    # ── 1d-Bot: Exit (nur am Day-Open) ────────────────────────────────────────
    if pos_1d is not None and is_day_open:
        atr_1d = _nan(row_1d.get("ATR"), 0.02 * kurs)
        ex_r = ex_p = None
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
            pnl = proceeds - pos_1d.entry_cash
            closed_trades.append({"bot": "1d", "entry_ts": pos_1d.entry_ts,
                "exit_ts": ts, "side": pos_1d.side, "ep": pos_1d.entry_price,
                "xp": ex_p, "hebel": pos_1d.hebel,
                "pnl_pct": pnl / pos_1d.entry_cash * 100,
                "entry_cash": pos_1d.entry_cash, "regime": pos_1d.regime,
                "reason": ex_r})
            virt_1d += pnl
            cash    += proceeds
            pos_1d   = None
            cd_1d    = COOLDOWN_1D_OPENS

    # ── 1w-Bot: Exit ──────────────────────────────────────────────────────────
    if pos_1w is not None:
        atr_1w = _nan(row_1w.get("ATR"), 0.10 * kurs)
        ex_r = ex_p = None
        if kurs <= pos_1w.sl_price:
            ex_r, ex_p = "SL", _sell_price(pos_1w.sl_price, vola)
        elif pos_1w.tp_price and kurs >= pos_1w.tp_price:
            ex_r, ex_p = "TP", _sell_price(pos_1w.tp_price, vola)
        elif pos_1w.trail_price is not None and atr_1w > 0:
            tm = TRAIL_MULT_1w.get(pos_1w.regime, 3.0) or 3.0
            pos_1w.trail_price = max(pos_1w.trail_price, kurs - tm * atr_1w)
            if kurs <= pos_1w.trail_price:
                ex_r, ex_p = "Trail", _sell_price(pos_1w.trail_price, vola)
        if ex_r:
            proceeds = _exit_pnl(pos_1w.entry_cash, pos_1w.hebel,
                                  pos_1w.entry_price, ex_p, pos_1w.side)
            pnl = proceeds - pos_1w.entry_cash
            closed_trades.append({"bot": "1w", "entry_ts": pos_1w.entry_ts,
                "exit_ts": ts, "side": pos_1w.side, "ep": pos_1w.entry_price,
                "xp": ex_p, "hebel": pos_1w.hebel,
                "pnl_pct": pnl / pos_1w.entry_cash * 100,
                "entry_cash": pos_1w.entry_cash, "regime": pos_1w.regime,
                "reason": ex_r})
            virt_1w += pnl
            cash    += proceeds
            pos_1w   = None
            cd_1w    = COOLDOWN_1W_OPENS

    # ── Virtuelle Equity für DD-Tracking ─────────────────────────────────────
    def _cur(virt, pos):
        if pos is None:
            return virt
        return virt + (_mtm(pos.entry_cash, pos.hebel,
                            pos.entry_price, kurs, pos.side) - pos.entry_cash)

    cur_4h = _cur(virt_4h, pos_4h)
    cur_1d = _cur(virt_1d, pos_1d)
    cur_1w = _cur(virt_1w, pos_1w)
    win_4h.append(cur_4h)
    if is_day_open:  win_1d.append(cur_1d)
    if is_week_open: win_1w.append(cur_1w)

    # ── 4h-Bot: Entry ─────────────────────────────────────────────────────────
    if pos_4h is None and cd_4h == 0 and cash >= 1.0:
        regime_4h = row_4h.get("Regime", "")
        if regime_4h in ("bull", "bear"):
            sig = get_signal_4h(row_4h, regime_4h)
            if sig != 0:
                sl_m, tp_m = SL_TP_4h.get(regime_4h, (2.0, 8.0))
                hebel      = HEBEL_4h.get(regime_4h, 2.0)
                peak_4h    = max(win_4h) if win_4h else INIT_BOT
                if peak_4h > 0 and (cur_4h - peak_4h) / peak_4h < -0.20:
                    hebel = 1.0
                ec = cash * POS_SIZE_4H
                if sig == 1:
                    ep    = _buy_price(kurs, vola)
                    trail = (TRAIL_MULT_4h.get(regime_4h, 3.0) or 3.0) * atr_4h
                    pos_4h = Position("long", ep, ec, hebel,
                                      ep - sl_m * atr_4h, ep + tp_m * atr_4h,
                                      ep - trail, regime_4h, ts)
                else:
                    ep    = _sell_price(kurs, vola)
                    pos_4h = Position("short", ep, ec, hebel,
                                      ep + sl_m * atr_4h, ep - tp_m * atr_4h,
                                      None, regime_4h, ts)
                cash -= ec

    # ── 1d-Bot: Entry (nur am Day-Open) ───────────────────────────────────────
    if is_day_open and pos_1d is None and cd_1d == 0 and cash >= 1.0:
        regime_1d = row_1d.get("Regime", "")
        if pd.notna(regime_1d) and regime_1d in ("bull", "bear"):
            atr_1d = _nan(row_1d.get("ATR"), 0.02 * kurs)
            sig    = get_signal_1d(row_1d, regime_1d)
            if sig != 0:
                sl_m, tp_m = SL_TP_1d.get(regime_1d, (1.0, 8.0))
                hebel      = HEBEL_1d.get(regime_1d, 2.0)
                peak_1d    = max(win_1d) if win_1d else INIT_BOT
                if peak_1d > 0 and (cur_1d - peak_1d) / peak_1d < -0.20:
                    hebel = 1.0
                ec = cash * POS_SIZE_1D
                if sig == 1:
                    ep    = _buy_price(kurs, vola)
                    trail = (TRAIL_MULT_1d.get(regime_1d, 3.0) or 3.0) * atr_1d
                    pos_1d = Position("long", ep, ec, hebel,
                                      ep - sl_m * atr_1d, ep + tp_m * atr_1d,
                                      ep - trail, regime_1d, ts)
                else:
                    ep    = _sell_price(kurs, vola)
                    pos_1d = Position("short", ep, ec, hebel,
                                      ep + sl_m * atr_1d, ep - tp_m * atr_1d,
                                      None, regime_1d, ts)
                cash -= ec

    # ── 1w-Bot: Entry (nur am Week-Open) ──────────────────────────────────────
    if is_week_open and pos_1w is None and cd_1w == 0 and cash >= 1.0:
        atr_1w    = _nan(row_1w.get("ATR"), 0.10 * kurs)
        sig_1w, regime_1w = get_signal_1w(row_1w)
        if sig_1w == 1:
            sl_m, tp_m = SL_TP_1w.get(regime_1w, (1.5, 8.0))
            hebel      = HEBEL_1w.get(regime_1w, 2.0)
            peak_1w    = max(win_1w) if win_1w else INIT_BOT
            if peak_1w > 0 and (cur_1w - peak_1w) / peak_1w < -0.20:
                hebel = 1.0
            ec    = cash * POS_SIZE_1W
            ep    = _buy_price(kurs, vola)
            trail = (TRAIL_MULT_1w.get(regime_1w, 3.0) or 3.0) * atr_1w
            pos_1w = Position("long", ep, ec, hebel,
                              ep - sl_m * atr_1w, ep + tp_m * atr_1w,
                              ep - trail, regime_1w, ts)
            cash -= ec

    # ── State zurückschreiben ─────────────────────────────────────────────────
    state["cash"]    = cash
    state["pos_4h"]  = pos_4h
    state["pos_1d"]  = pos_1d
    state["pos_1w"]  = pos_1w
    state["cd_4h"]   = cd_4h
    state["cd_1d"]   = cd_1d
    state["cd_1w"]   = cd_1w
    state["virt_4h"] = virt_4h
    state["virt_1d"] = virt_1d
    state["virt_1w"] = virt_1w
    state["win_4h"]  = list(win_4h)
    state["win_1d"]  = list(win_1d)
    state["win_1w"]  = list(win_1w)
    state["last_ts"] = str(ts)

    return closed_trades


# ── Sleep bis nächster 4h-Bar ──────────────────────────────────────────────────

def _sleep_until_next_bar():
    now    = pd.Timestamp.now("UTC").tz_localize(None)
    next_4h = now.floor("4h") + pd.Timedelta(hours=4)
    target  = next_4h + pd.Timedelta(minutes=1, seconds=5)
    wait    = (target - now).total_seconds()
    log.info("Schlafe %.0fs bis %s UTC", wait, target.strftime("%Y-%m-%d %H:%M:%S"))
    time.sleep(max(wait, 1))


# ── Hauptloop ──────────────────────────────────────────────────────────────────

def run():
    _setup_logging()
    log.info("=== Paper Bot Combined (1d+4h+1w) gestartet ===")
    init_db()
    state = load_state()

    if state.get("live_since") is None:
        today = str(pd.Timestamp.now("UTC").tz_localize(None).normalize())
        state["live_since"] = today
        save_state(state)
        log.info("Live-Start gesetzt: %s", today)

    log.info("State: cash=$%.2f  last_ts=%s  live_since=%s",
             state["cash"], state.get("last_ts"), state.get("live_since"))

    while True:
        _sleep_until_next_bar()
        log.info("--- 4h Bar-Check ---")

        data = fetch_all()
        if data is None:
            log.error("Datenabruf fehlgeschlagen — überspringe Bar.")
            continue

        df_4h_raw, df_1d_raw, df_1w_raw = data
        df_4h, df_1d_on_4h, df_1w_on_4h = build_aligned_frames(
            df_4h_raw, df_1d_raw, df_1w_raw
        )

        now_utc  = pd.Timestamp.now("UTC").tz_localize(None)
        df_done  = df_4h[df_4h.index <= now_utc - pd.Timedelta(minutes=5)]
        last_str = state.get("last_ts")
        if last_str:
            df_done = df_done[df_done.index > pd.Timestamp(last_str)]

        if df_done.empty:
            log.info("Keine neuen Kerzen.")
            continue

        live_since_ts = pd.Timestamp(state["live_since"])

        for ts, row_4h in df_done.iterrows():
            if ts < live_since_ts:
                state["last_ts"] = str(ts)
                continue

            row_1d = df_1d_on_4h.loc[ts] if ts in df_1d_on_4h.index else pd.Series(dtype=float)
            row_1w = df_1w_on_4h.loc[ts] if ts in df_1w_on_4h.index else pd.Series(dtype=float)

            pos_before = (state["pos_4h"], state["pos_1d"], state["pos_1w"])
            trades     = process_bar(ts, row_4h, row_1d, row_1w, state)
            pos_after  = (state["pos_4h"], state["pos_1d"], state["pos_1w"])

            pv = state["cash"]
            for pos in pos_after:
                if pos:
                    pv += _mtm(pos.entry_cash, pos.hebel,
                               pos.entry_price, float(row_4h["Close"]), pos.side)
            _log_equity(ts, pv, state["cash"], *pos_after)

            for t in trades:
                _log_trade(t["bot"], t["entry_ts"], t["exit_ts"], t["side"],
                           t["ep"], t["xp"], t["hebel"], t["pnl_pct"],
                           t["entry_cash"], t["regime"], t["reason"])
                log.info("EXIT  bot=%-3s | %-8s | %s | PnL: %+.2f%% | Cash: $%.2f",
                          t["bot"], t["reason"], t["regime"],
                          t["pnl_pct"], state["cash"])

            for bot_key, pos_b, pos_a in zip(
                    ("4h", "1d", "1w"), pos_before, pos_after):
                if pos_b is None and pos_a is not None:
                    log.info("ENTRY bot=%-3s | %s | %s | ep=%.4f | sl=%.4f | tp=%s | hebel=%.1f×",
                              bot_key, pos_a.regime, pos_a.side, pos_a.entry_price,
                              pos_a.sl_price,
                              f"{pos_a.tp_price:.4f}" if pos_a.tp_price else "None",
                              pos_a.hebel)

            log.info("Bar %s | cash=$%.2f | pv=$%.2f | 4h=%s 1d=%s 1w=%s",
                      ts, state["cash"], pv,
                      f"{pos_after[0].side}" if pos_after[0] else "—",
                      f"{pos_after[1].side}" if pos_after[1] else "—",
                      f"{pos_after[2].side}" if pos_after[2] else "—")

        save_state(state)
        log.info("State gespeichert.")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        log.info("Bot gestoppt (Ctrl+C).")
