"""
paper_bot.py — Paper Trading Bot (lokale Simulation mit echten Binance-Preisen)

Start: python3 paper_bot.py
Stop:  Ctrl+C  (State wird gespeichert)

Verarbeitet stündlich die letzte vollständige Kerze. Virtuelle Position, echte Preise.
"""
import json, sqlite3, time, os, sys, logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from sol_bot.backtest import (
    GEBUEHR, SLIPPAGE, SPREAD, FUNDING8, STARTKAPITAL,
    _buy_price, _sell_price, _exit_pnl, _mtm,
    fetch_binance_data, compute_daily_macro, Position,
)
from sol_bot.indicators import add_indicators
from sol_bot.regime import classify_regime
from sol_bot.strategy import (
    get_signal, calc_hebel, calc_position_size,
    signal_bulle_exit, signal_bulle_pyramid, signal_extrem_bulle_pyramid,
    signal_seitwaerts_cover, signal_baer_cover, signal_extrem_baer_cover,
    SL_TP, _g, signal_bulle_diag, signal_baer_diag,
)

# ── Konfiguration ─────────────────────────────────────────────────────────────

STATE_FILE = "paper_state.json"
DB_FILE    = "paper_trades.db"
LOG_FILE   = "paper_bot.log"
LOOKBACK   = 4000    # ~167 Tage; EMA50_daily braucht ~3600 Bars für Warmup

PROD_PARAMS = dict(
    sw_mode                  = "neutral_sw_short",
    bulle_mode               = "rsi3",
    max_pos_size             = 0.10,
    baer_lev_override        = 6.0,
    bulle_lev_override       = 1.0,
    extrem_bulle_lev_override= 2.0,
    extrem_bulle_tp_mult     = 10.0,
    bulle_tp_mult            = 10.0,
    bulle_trail_mult         = 4.0,
    bulle_lev_high_override  = 2.0,
    bulle_adx_thresh_override= 40.0,
    sw_sl_mult               = None,
    sw_tp_mult               = None,
    sw_lev_override          = None,
    sw_short_lev_override    = None,
    extrem_bulle_sl_mult     = None,
    extrem_bulle_trail_mult  = None,
    bulle_sl_mult            = None,
)

COOLDOWN      = 5
DD_RESET_BARS = 90 * 24
FUNDING_HOURS = {0, 8, 16}   # Binance Perpetual Funding UTC-Stunden

# ── Logging ───────────────────────────────────────────────────────────────────

log = logging.getLogger("paper_bot")


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE),
        ],
    )


# ── SQLite ────────────────────────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            Entry_ts          TEXT,
            Exit_ts           TEXT,
            Side              TEXT,
            Entry_price       REAL,
            Exit_price        REAL,
            Hebel             REAL,
            PnL_pct           REAL,
            Entry_cash        REAL,
            Regime            TEXT,
            Exit_reason       TEXT,
            Trigger_id        TEXT,
            RSI_entry         REAL,
            Dist_ema21_entry  REAL,
            Regime_age_entry  REAL,
            ADX_entry         REAL,
            Vola_ratio_entry  REAL,
            Pyramid_count     INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS equity_log (
            ts           TEXT PRIMARY KEY,
            portfolio    REAL,
            cash         REAL,
            regime       TEXT,
            has_position INTEGER
        )
    """)
    con.commit()
    con.close()


def _log_trade(trade: dict):
    con = sqlite3.connect(DB_FILE)
    con.execute("""
        INSERT INTO trades (
            Entry_ts, Exit_ts, Side, Entry_price, Exit_price, Hebel, PnL_pct,
            Entry_cash, Regime, Exit_reason, Trigger_id, RSI_entry,
            Dist_ema21_entry, Regime_age_entry, ADX_entry, Vola_ratio_entry, Pyramid_count
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        str(trade.get("Entry_ts")), str(trade.get("Exit_ts")),
        trade.get("Side"), trade.get("Entry_price"), trade.get("Exit_price"),
        trade.get("Hebel"), trade.get("PnL_pct"), trade.get("Entry_cash"),
        trade.get("Regime"), trade.get("Exit_reason"), trade.get("Trigger_id"),
        trade.get("RSI_entry"), trade.get("Dist_ema21_entry"),
        trade.get("Regime_age_entry"), trade.get("ADX_entry"),
        trade.get("Vola_ratio_entry"), trade.get("Pyramid_count"),
    ))
    con.commit()
    con.close()


def _log_equity(ts, portfolio, cash, regime, has_position):
    con = sqlite3.connect(DB_FILE)
    con.execute(
        "INSERT OR REPLACE INTO equity_log (ts, portfolio, cash, regime, has_position) VALUES (?,?,?,?,?)",
        (str(ts), portfolio, cash, regime, int(has_position)),
    )
    con.commit()
    con.close()


# ── State-Serialisierung ──────────────────────────────────────────────────────

def _pos_to_dict(pos) -> dict | None:
    if pos is None:
        return None
    return {s: getattr(pos, s) for s in Position.__slots__}


def _pos_from_dict(d: dict | None):
    if d is None:
        return None
    p = object.__new__(Position)
    for s in Position.__slots__:
        setattr(p, s, d.get(s))
    return p


def _default_state() -> dict:
    return {
        "cash":          STARTKAPITAL,
        "pos":           None,
        "peak":          STARTKAPITAL,
        "cooldown_left": 0,
        "equity_window": [],
        "regime_age":    0,
        "prev_regime":   None,
        "atl_peak":      STARTKAPITAL,
        "last_ts":       None,
        "live_since":    None,
    }


def load_state() -> dict:
    if not Path(STATE_FILE).exists():
        return _default_state()
    with open(STATE_FILE, "r") as f:
        raw = json.load(f)
    raw["pos"] = _pos_from_dict(raw.get("pos"))
    return raw


def save_state(state: dict):
    tmp = STATE_FILE + ".tmp"
    serializable = {**state, "pos": _pos_to_dict(state["pos"])}
    with open(tmp, "w") as f:
        json.dump(serializable, f, default=str)
    os.replace(tmp, STATE_FILE)   # atomares Überschreiben


# ── Daten-Fetch ───────────────────────────────────────────────────────────────

def fetch_data_with_retry(lookback: int = LOOKBACK) -> pd.DataFrame | None:
    start = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(hours=lookback + 5)).strftime("%Y-%m-%d")
    for attempt in range(3):
        try:
            df = fetch_binance_data(start=start)
            if len(df) < 100:
                raise ValueError(f"Zu wenige Kerzen: {len(df)}")
            return df
        except Exception as e:
            wait = [5, 30, 120][attempt]
            log.warning("Fetch-Fehler (Versuch %d/3): %s — warte %ds", attempt + 1, e, wait)
            if attempt < 2:
                time.sleep(wait)
    log.error("Binance-Fetch nach 3 Versuchen fehlgeschlagen.")
    return None


def check_missing_candles(df: pd.DataFrame, last_ts_str: str | None):
    if last_ts_str is None:
        return
    last = pd.Timestamp(last_ts_str)
    # Neueste vollständige Kerze: letzte, die mindestens 1 Minute alt ist
    now_utc = pd.Timestamp.now("UTC").tz_localize(None)
    complete = df[df.index <= now_utc - pd.Timedelta(minutes=5)]
    if complete.empty:
        return
    newest = complete.index[-1]
    gap_h = (newest - last).total_seconds() / 3600
    if gap_h > 2.0:
        log.warning(
            "⚠️  FEHLENDE KERZEN: letzte verarbeitete %s, neueste vollständige %s (%.1fh Gap). "
            "Mac war wahrscheinlich im Schlaf — Systemeinstellungen → Energie → Nie schlafen prüfen!",
            last, newest, gap_h,
        )


# ── Per-Bar-Logik ─────────────────────────────────────────────────────────────
# Spiegelt backtest.py Z. 407–741 exakt.
# WICHTIG: Änderungen in backtest.py MÜSSEN hier gespiegelt werden (replay.py prüft Parität).

def process_bar(row, ts, state: dict, p: dict) -> tuple[dict, dict | None, float]:
    """
    Verarbeitet eine Kerze. Gibt (state, trade_or_None, equity_val) zurück.
    state wird in-place mutiert UND zurückgegeben.
    """
    kurs   = row["Close"]
    atr    = row["ATR"]  if not pd.isna(row["ATR"])  else 0.001 * kurs
    vola   = row["Vola"] if not pd.isna(row["Vola"]) else 0.5
    regime = row["Regime"]

    cash          = state["cash"]
    pos           = state["pos"]
    peak          = state["peak"]
    cooldown_left = state["cooldown_left"]
    equity_window = state["equity_window"]
    regime_age    = state["regime_age"]
    prev_regime   = state["prev_regime"]
    atl_peak      = state["atl_peak"]
    trade         = None

    # ── Regime-Alter ─────────────────────────────────────────────────────
    if regime == "bulle":
        if prev_regime != "bulle":
            regime_age = 0
        regime_age += 1
    else:
        regime_age = 0
    prev_regime = regime

    # ── Portfolio-Wert + Rolling DD ──────────────────────────────────────
    port_value = cash if pos is None else (cash + _mtm(pos, kurs))
    equity_window.append(port_value)
    if len(equity_window) > DD_RESET_BARS:
        equity_window.pop(0)
    peak     = max(equity_window)
    dd       = (port_value - peak) / peak
    atl_peak = max(atl_peak, port_value)

    # ── Funding (Binance Perpetual: 00:00 / 08:00 / 16:00 UTC) ──────────
    if pos is not None and pos.side == "short" and ts.hour in FUNDING_HOURS:
        cash -= pos.entry_cash * FUNDING8 * pos.hebel

    # ── SL / TP / Trail ───────────────────────────────────────────────────
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
            if pos.trail_price is not None and atr > 0:
                if regime == "bulle":
                    trail_mult = p.get("bulle_trail_mult") or 3.0
                elif regime == "extrem_bulle" and p.get("extrem_bulle_trail_mult") is not None:
                    trail_mult = p["extrem_bulle_trail_mult"]
                else:
                    trail_mult = 1.2
                new_trail       = kurs - trail_mult * atr
                pos.trail_price = max(pos.trail_price, new_trail)
                if kurs <= pos.trail_price and exit_reason is None:
                    exit_reason = "Trail"
                    exit_price  = _sell_price(pos.trail_price, vola)
        else:  # short
            if kurs >= pos.sl_price:
                exit_reason = "SL"
                exit_price  = _buy_price(pos.sl_price, vola)
            elif pos.tp_price is not None and kurs <= pos.tp_price:
                exit_reason = "TP"
                exit_price  = _buy_price(pos.tp_price, vola)
            if (regime == "seitwaerts" and pos.regime == "seitwaerts"
                    and signal_seitwaerts_cover(row) and exit_reason is None):
                exit_reason = "Cover"
                exit_price  = _buy_price(kurs, vola)
            if regime == "extrem_baer" and signal_extrem_baer_cover(row) and exit_reason is None:
                exit_reason = "Cover"
                exit_price  = _buy_price(kurs, vola)
            if (regime == "baer" and pos.regime == "baer"
                    and signal_baer_cover(row) and exit_reason is None):
                exit_reason = "Cover"
                exit_price  = _buy_price(kurs, vola)

        if exit_reason:
            proceeds = _exit_pnl(pos, exit_price)
            pnl_pct  = (proceeds - pos.entry_cash) / pos.entry_cash * 100
            trade = {
                "Entry_ts":         str(pos.candle_idx),
                "Exit_ts":          str(ts),
                "Side":             pos.side,
                "Entry_price":      pos.entry_price,
                "Exit_price":       exit_price,
                "Hebel":            pos.hebel,
                "PnL_pct":          pnl_pct,
                "Entry_cash":       pos.entry_cash,
                "Regime":           pos.regime,
                "Exit_reason":      exit_reason,
                "Trigger_id":       pos.trigger_id,
                "RSI_entry":        pos.rsi_entry,
                "Dist_ema21_entry": pos.dist_ema21_entry,
                "Regime_age_entry": pos.regime_age_entry,
                "ADX_entry":        pos.adx_entry,
                "Vola_ratio_entry": pos.vola_ratio_entry,
                "Pyramid_count":    pos.pyramid_count,
            }
            cash          += proceeds
            pos            = None
            cooldown_left  = COOLDOWN

    # ── Bulle Long-Exit via Signal ────────────────────────────────────────
    if pos is not None and pos.side == "long" and regime == "bulle":
        if signal_bulle_exit(row):
            exit_price = _sell_price(kurs, vola)
            proceeds   = _exit_pnl(pos, exit_price)
            pnl_pct    = (proceeds - pos.entry_cash) / pos.entry_cash * 100
            trade = {
                "Entry_ts":         str(pos.candle_idx),
                "Exit_ts":          str(ts),
                "Side":             "long",
                "Entry_price":      pos.entry_price,
                "Exit_price":       exit_price,
                "Hebel":            pos.hebel,
                "PnL_pct":          pnl_pct,
                "Entry_cash":       pos.entry_cash,
                "Regime":           pos.regime,
                "Exit_reason":      "Exit_signal",
                "Trigger_id":       pos.trigger_id,
                "RSI_entry":        pos.rsi_entry,
                "Dist_ema21_entry": pos.dist_ema21_entry,
                "Regime_age_entry": pos.regime_age_entry,
                "ADX_entry":        pos.adx_entry,
                "Vola_ratio_entry": pos.vola_ratio_entry,
                "Pyramid_count":    pos.pyramid_count,
            }
            cash          += proceeds
            pos            = None
            cooldown_left  = COOLDOWN

    # ── Pyramiding ────────────────────────────────────────────────────────
    if pos is not None and pos.pyramid_count < 3 and cash > 10:
        do_pyramid = False
        extra_pct  = 0.25
        if regime == "bulle" and pos.side == "long":
            do_pyramid = signal_bulle_pyramid(row, pos.entry_price)
        elif regime == "extrem_bulle" and pos.side == "long":
            do_pyramid = signal_extrem_bulle_pyramid(row, pos.entry_price)
            extra_pct  = 0.30
            if pos.pyramid_count >= 3:
                do_pyramid = False
        if do_pyramid:
            add_cash        = cash * extra_pct
            pyramid_price   = _buy_price(kurs, vola)
            new_total_cash  = pos.entry_cash + add_cash
            pos.entry_price = (pos.entry_price * pos.entry_cash + pyramid_price * add_cash) / new_total_cash
            cash           -= add_cash
            pos.entry_cash  = new_total_cash
            pos.pyramid_count += 1

    # ── Cooldown ──────────────────────────────────────────────────────────
    if cooldown_left > 0:
        cooldown_left -= 1

    # ── Neues Entry-Signal ────────────────────────────────────────────────
    if pos is None and cash > 1.0 and cooldown_left == 0:
        sig = get_signal(row, regime, p["sw_mode"], bulle_mode=p["bulle_mode"])

        if sig in (1, -2):
            is_short = (sig == -2)
            side     = "short" if is_short else "long"
            h = calc_hebel(_g(row, "ADX", 20), _g(row, "Vola_ratio", 1), regime, is_short, dd)

            if regime == "seitwaerts" and not is_short:
                h = 2.0
            if p.get("sw_lev_override") is not None and regime == "seitwaerts" and not is_short:
                h = p["sw_lev_override"]
            if regime == "baer" and is_short:
                base_h = p.get("baer_lev_override") or 8.0
                if dd < -0.25:   h = 1.0
                elif dd < -0.15: h = max(base_h * 0.5, 1.0)
                else:            h = base_h
            if regime == "bulle" and not is_short:
                _adx   = _g(row, "ADX", 0.0)
                _thresh = p.get("bulle_adx_thresh_override") or 0.0
                _hi     = p.get("bulle_lev_high_override") or 1.0
                base_h = _hi if (_thresh > 0 and _adx >= _thresh) else (p.get("bulle_lev_override") or 1.0)
                if dd < -0.25:   h = 1.0
                elif dd < -0.15: h = max(base_h * 0.5, 1.0)
                else:            h = base_h
            if p.get("extrem_bulle_lev_override") is not None and regime == "extrem_bulle" and not is_short:
                base_h = p["extrem_bulle_lev_override"]
                if dd < -0.25:   h = 1.0
                elif dd < -0.15: h = max(base_h * 0.5, 1.0)
                else:            h = base_h
            if p.get("sw_short_lev_override") is not None and regime == "seitwaerts" and is_short:
                base_h = p["sw_short_lev_override"]
                if dd < -0.25:   h = 1.0
                elif dd < -0.15: h = max(base_h * 0.5, 1.0)
                else:            h = base_h

            # Diagnose-Felder
            _trigger_id = _rsi_entry = _dist_ema21_entry = _regime_age_entry = None
            _adx_entry  = _vola_ratio_entry = None
            if regime == "bulle" and side == "long":
                _, _trigger_id    = signal_bulle_diag(row)
                _rsi_entry        = _g(row, "RSI", None)
                _ema21            = _g(row, "EMA21", None)
                _dist_ema21_entry = (kurs / _ema21 - 1) * 100 if _ema21 else None
                _regime_age_entry = regime_age
                _adx_entry        = _g(row, "ADX", None)
            elif regime == "baer" and side == "short":
                _, _trigger_id    = signal_baer_diag(row)
                _rsi_entry        = _g(row, "RSI", None)
                _vwap             = _g(row, "VWAP", None)
                _dist_ema21_entry = (kurs / _vwap - 1) * 100 if _vwap else None
                _regime_age_entry = regime_age
            elif regime == "extrem_bulle" and side == "long":
                _rsi_entry        = _g(row, "RSI", None)
                _dist_ema21_entry = _g(row, "MA200_dist", None)
                _adx_entry        = _g(row, "ADX", None)
                _vola_ratio_entry = _g(row, "Vola_ratio", None)
                _ma500            = _g(row, "MA500", 0.0)
                _regime_age_entry = 1 if (_ma500 > 0 and kurs > _ma500) else 0
            elif regime == "seitwaerts" and side == "long":
                _rsi_entry        = _g(row, "RSI", None)
                _bb_lower         = _g(row, "BB_lower", None)
                _dist_ema21_entry = (kurs / _bb_lower - 1) * 100 if _bb_lower else None
                _rsi_div_b        = bool(_g(row, "RSI_div_bull", False))
                _obv_b            = bool(_g(row, "OBV_bull", False))
                _st_b             = bool(_g(row, "ST_bull", False))
                _parts            = (["rsi"] if (_rsi_entry and _rsi_entry < 45) else []) + \
                                    (["div"] if (_rsi_div_b or _st_b) else []) + \
                                    (["obv"] if _obv_b else [])
                _trigger_id       = "+".join(_parts) if _parts else "none"
                _regime_age_entry = int(_rsi_div_b) + 2 * int(_obv_b)
                _adx_entry        = _g(row, "ADX", None)
                _vola_ratio_entry = _g(row, "Vola_ratio", None)

            sl_m, tp_m = SL_TP.get(regime, (2.0, 4.0))
            if regime == "seitwaerts" and side == "short":
                sl_m, tp_m = 2.0, None
            if p.get("sw_sl_mult") is not None and regime == "seitwaerts" and side == "long":
                sl_m = p["sw_sl_mult"]
            if p.get("sw_tp_mult") is not None and regime == "seitwaerts" and side == "long":
                tp_m = p["sw_tp_mult"]
            if p.get("extrem_bulle_sl_mult") is not None and regime == "extrem_bulle":
                sl_m = p["extrem_bulle_sl_mult"]
            if p.get("extrem_bulle_tp_mult") is not None and regime == "extrem_bulle":
                tp_m = p["extrem_bulle_tp_mult"] if p["extrem_bulle_tp_mult"] > 0 else None
            if p.get("bulle_sl_mult") is not None and regime == "bulle":
                sl_m = p["bulle_sl_mult"]
            if p.get("bulle_tp_mult") is not None and regime == "bulle":
                tp_m = p["bulle_tp_mult"]

            if side == "long":
                entry_p = _buy_price(kurs, vola)
                sl_p    = entry_p - sl_m * atr
                tp_p    = (entry_p + tp_m * atr) if tp_m else None
                _eb_tm  = p.get("extrem_bulle_trail_mult") or 1.2
                trail   = entry_p - _eb_tm * atr
                if regime == "bulle":
                    trail = entry_p - (p.get("bulle_trail_mult") or 3.0) * atr
                elif regime == "extrem_bulle":
                    trail = entry_p - _eb_tm * atr
                elif regime == "seitwaerts":
                    trail = None
            else:
                entry_p = _sell_price(kurs, vola)
                sl_p    = entry_p + sl_m * atr
                tp_p    = (entry_p - tp_m * atr) if tp_m else None
                trail   = None
            if regime == "seitwaerts" and side == "short":
                trail = entry_p + 1.0 * atr

            max_ps   = p.get("max_pos_size", 0.10)
            pos_size = calc_position_size(1, 1, _g(row, "ADX", 20),
                                          extrem=(regime in ("extrem_baer", "extrem_bulle")))
            pos_size = min(pos_size, max_ps)
            use_cash = cash * pos_size
            cash    -= use_cash

            pos = Position(
                side=side, entry_price=entry_p, entry_cash=use_cash, hebel=h,
                sl_price=sl_p, tp_price=tp_p, trail_price=trail, regime=regime,
                candle_idx=str(ts),
                trigger_id=_trigger_id, rsi_entry=_rsi_entry,
                dist_ema21_entry=_dist_ema21_entry, regime_age_entry=_regime_age_entry,
                adx_entry=_adx_entry, vola_ratio_entry=_vola_ratio_entry,
            )

    # ── State aktualisieren ───────────────────────────────────────────────
    equity_val = cash if pos is None else (cash + _mtm(pos, kurs))

    state["cash"]          = cash
    state["pos"]           = pos
    state["peak"]          = peak
    state["cooldown_left"] = cooldown_left
    state["equity_window"] = equity_window
    state["regime_age"]    = regime_age
    state["prev_regime"]   = prev_regime
    state["atl_peak"]      = atl_peak
    state["last_ts"]       = str(ts)

    return state, trade, equity_val


# ── Sleep-Logik ───────────────────────────────────────────────────────────────

def _sleep_until_next_bar():
    now    = pd.Timestamp.now("UTC").tz_localize(None)
    next_h = now.floor("h") + pd.Timedelta(hours=1)
    target = next_h + pd.Timedelta(minutes=1, seconds=5)
    wait   = (target - now).total_seconds()
    log.info("Schlafe %.0fs bis %s UTC", wait, target.strftime("%H:%M:%S"))
    time.sleep(wait)


# ── Hauptloop ─────────────────────────────────────────────────────────────────

def run():
    _setup_logging()
    log.info("=== Paper Bot gestartet ===")
    init_db()
    state = load_state()
    if state.get("live_since") is None:
        today = str(pd.Timestamp.now("UTC").tz_localize(None).normalize())
        state["live_since"] = today
        save_state(state)
        log.info("Live-Start gesetzt: %s", today)
    log.info("State geladen: cash=$%.2f, last_ts=%s, live_since=%s",
             state["cash"], state.get("last_ts"), state.get("live_since"))

    while True:
        _sleep_until_next_bar()
        log.info("--- Stündlicher Bar-Check ---")

        df_raw = fetch_data_with_retry(LOOKBACK)
        if df_raw is None:
            log.error("Fetch fehlgeschlagen — überspringe Bar.")
            continue

        check_missing_candles(df_raw, state.get("last_ts"))

        # Indikatoren + Regime + Daily Macro
        macro = compute_daily_macro(df_raw)
        df    = add_indicators(df_raw)
        df["Regime"]           = classify_regime(df)
        df["Daily_macro_bull"] = macro.reindex(df.index, method="ffill").fillna(False)

        # Alle unverarbeiteten vollständigen Kerzen verarbeiten (Catch-up nach Sleep)
        now_utc  = pd.Timestamp.now("UTC").tz_localize(None)
        df_done  = df[df.index <= now_utc - pd.Timedelta(minutes=5)]
        last_str = state.get("last_ts")
        if last_str is not None:
            df_done = df_done[df_done.index > pd.Timestamp(last_str)]

        if df_done.empty:
            log.info("Keine neuen Kerzen — überspringe.")
            continue

        live_since_ts = pd.Timestamp(state["live_since"])

        for ts, row in df_done.iterrows():
            if ts < live_since_ts:
                state["last_ts"] = str(ts)
                continue

            pos_before = state["pos"]
            state, trade, equity_val = process_bar(row, ts, state, PROD_PARAMS)
            pos_after = state["pos"]

            regime = row.get("Regime", "?")
            _log_equity(ts, equity_val, state["cash"], regime, pos_after is not None)

            if trade is not None:
                _log_trade(trade)
                log.info("EXIT  %-12s | %-14s | %s | PnL: %+.2f%% | Cash: $%.2f",
                          trade["Exit_reason"], trade["Regime"], trade["Side"],
                          trade["PnL_pct"], state["cash"])

            if pos_before is None and pos_after is not None:
                log.info("ENTRY %-5s | %-14s | entry=%.4f | hebel=%.1f× | sl=%.4f | tp=%s",
                          pos_after.side, pos_after.regime,
                          pos_after.entry_price, pos_after.hebel, pos_after.sl_price,
                          f"{pos_after.tp_price:.4f}" if pos_after.tp_price else "None")

            log.info("Bar %s | regime=%-14s | cash=$%.2f | portfolio=$%.2f | pos=%s",
                     ts, regime, state["cash"], equity_val,
                     f"{pos_after.side}@{pos_after.entry_price:.2f}" if pos_after else "None")

        save_state(state)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        log.info("Bot gestoppt (Ctrl+C).")
