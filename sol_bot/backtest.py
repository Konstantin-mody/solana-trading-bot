"""
backtest.py — Preismodell, Kostenmodell und Backtest-Engine

Preismodell: synthetisch, seed=42, ~29170 Stunden-Kerzen (Jan 2022 – Apr 2025)
"""

import numpy as np
import pandas as pd

from sol_bot.indicators import add_indicators
from sol_bot.regime     import classify_regime
from sol_bot.strategy   import (
    calc_hebel, calc_position_size, get_signal,
    signal_bulle_diag, signal_baer_diag,
    signal_bulle_exit, signal_bulle_pyramid,
    signal_extrem_bulle_pyramid, signal_seitwaerts_cover,
    signal_baer_cover, signal_extrem_baer_cover,
    SL_TP, _g
)

# ─── Kostenmodell ────────────────────────────────────────────────────────────

GEBUEHR  = 0.00075   # 0.075%
SLIPPAGE = 0.0003    # 0.03%
SPREAD   = 0.0002    # 0.02%
FUNDING8 = 0.0001    # pro 8h für Short × Hebel

STARTKAPITAL = 200.0


def _buy_price(kurs: float, vola: float) -> float:
    slip = min(SLIPPAGE + vola * 0.00003, 0.001)
    return kurs * (1 + slip + SPREAD / 2)


def _sell_price(kurs: float, vola: float) -> float:
    slip = min(SLIPPAGE + vola * 0.00003, 0.001)
    return kurs * (1 - slip - SPREAD / 2)


# ─── Preismodell ─────────────────────────────────────────────────────────────

def fetch_binance_data(start: str = "2020-11-01") -> pd.DataFrame:
    """
    Lädt echte SOL/USDT 1h-Daten direkt von der Binance REST API.
    Kein API-Key nötig. Paginiert automatisch.

    start : ISO-Datum, z.B. "2020-11-01" (SOL-Listing auf Binance)
    Liefert ~48.000 Kerzen für den vollen Zyklus 2020–heute.
    """
    import requests, time as _time

    url      = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int(pd.Timestamp.now().timestamp() * 1000)
    limit    = 1000
    all_rows = []

    ts = start_ms
    while ts < end_ms:
        params = {"symbol": "SOLUSDT", "interval": "1h",
                  "startTime": ts, "limit": limit}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_rows.extend(batch)
        ts = batch[-1][0] + 1          # nächste Kerze nach letztem Timestamp
        if len(batch) < limit:
            break
        _time.sleep(0.05)              # Binance Rate-Limit schonen

    if not all_rows:
        raise ValueError("Keine Daten von Binance erhalten.")

    df = pd.DataFrame(all_rows, columns=[
        "OpenTime", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVol", "Trades", "TakerBase", "TakerQuote", "Ignore"
    ])
    df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
    df = df.set_index("OpenTime")
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.index.name = None
    return df


def fetch_real_data(interval: str = "1h") -> pd.DataFrame:
    """
    Lädt echte SOL/USD OHLCV-Daten via yfinance.

    interval='1h'  → letzten 730 Tage (Apr 2024 – Apr 2026), ~17.000 Kerzen
    interval='1d'  → volle Historie seit Apr 2020, ~2.200 Kerzen
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance nicht installiert. `pip install yfinance`")

    if interval == "1h":
        df = yf.download("SOL-USD", period="730d", interval="1h",
                         auto_adjust=True, progress=False)
    else:
        df = yf.download("SOL-USD", start="2020-04-01", interval=interval,
                         auto_adjust=True, progress=False)

    if len(df) == 0:
        raise ValueError("Keine Daten von yfinance erhalten.")

    # MultiIndex-Spalten flachklopfen
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Timezone entfernen (Engine erwartet tz-naive)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Close"])
    return df


def make_price_data() -> pd.DataFrame:
    """
    Reproduzierbares synthetisches SOL/USD Preismodell.
    seed=42, ~29170 Stunden-Kerzen von 2022-01-01 bis ~April 2025.
    """
    np.random.seed(42)
    n = int(3.33 * 365 * 24)
    dates = pd.date_range("2022-01-01", periods=n, freq="1h")

    checkpoints = [
        170, 140, 100, 60, 35, 28, 20, 15, 14, 16, 18, 22, 20, 22, 24,
        30, 38, 55, 58, 75, 95, 110, 160, 200, 175, 155, 145, 150, 160,
        175, 200, 210, 190, 170, 140, 120, 110
    ]

    # Linearer Trend zwischen Checkpoints
    cp_indices = np.linspace(0, n - 1, len(checkpoints)).astype(int)
    trend = np.interp(np.arange(n), cp_indices, checkpoints)

    # Zufällige Volatilität
    noise = np.random.normal(0, 0.008, n)
    mult  = np.exp(np.cumsum(noise) * 0.3)
    close = trend * mult / mult[0]  # Normiert auf Startpreis

    # OHLCV generieren
    # Intracandle-Vola proportional zur close-to-close Bewegung:
    # Typisch: High-Low Range ≈ 1.5× der absoluten Candle-Bewegung
    cc_move = np.abs(np.diff(close, prepend=close[0]))
    hl_half = cc_move * 1.5 + close * 0.001  # mind. 0.1% pro Candle
    high  = close + hl_half
    low   = close - hl_half
    # Open = vorheriger Close (Lückenlos)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    volume = np.random.lognormal(mean=10, sigma=0.5, size=n) * 1000

    df = pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": volume,
    }, index=dates)

    return df


# ─── Backtest-Engine ─────────────────────────────────────────────────────────

class Position:
    __slots__ = ["side", "entry_price", "entry_cash", "hebel",
                 "sl_price", "tp_price", "trail_price", "regime",
                 "pyramid_count", "candle_idx",
                 "trigger_id", "rsi_entry", "dist_ema21_entry", "regime_age_entry",
                 "adx_entry", "vola_ratio_entry"]

    def __init__(self, side, entry_price, entry_cash, hebel,
                 sl_price, tp_price, trail_price, regime, candle_idx,
                 trigger_id=None, rsi_entry=None, dist_ema21_entry=None, regime_age_entry=None,
                 adx_entry=None, vola_ratio_entry=None):
        self.side             = side          # "long" | "short"
        self.entry_price      = entry_price
        self.entry_cash       = entry_cash
        self.hebel            = hebel
        self.sl_price         = sl_price
        self.tp_price         = tp_price
        self.trail_price      = trail_price   # None = kein Trail
        self.regime           = regime
        self.pyramid_count    = 0
        self.candle_idx       = candle_idx
        self.trigger_id       = trigger_id
        self.rsi_entry        = rsi_entry
        self.dist_ema21_entry = dist_ema21_entry
        self.regime_age_entry = regime_age_entry
        self.adx_entry        = adx_entry
        self.vola_ratio_entry = vola_ratio_entry


def _exit_pnl(pos: Position, exit_price: float) -> float:
    """Berechnet den Erlös nach Kosten (Gebühr auf Notional, Entry + Exit)."""
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
    """Mark-to-market Wert der offenen Position (ohne Gebühren, für Portfolio-Schätzung)."""
    if pos.side == "long":
        raw = (kurs - pos.entry_price) / pos.entry_price
    else:
        raw = (pos.entry_price - kurs) / pos.entry_price
    return max(pos.entry_cash + pos.entry_cash * pos.hebel * raw, 0.01)


def fetch_btc_data(start: str = "2021-08-20") -> pd.Series:
    """
    Lädt BTC/USDT 1h Binance-Daten und gibt BTC_ma200_bull (Close > MA200) zurück.
    Start früher als SOL-Daten-Anfang, damit MA200-Warmup komplett ist.
    """
    import requests, time as _time

    url      = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int(pd.Timestamp.now().timestamp() * 1000)
    limit    = 1000
    all_rows = []

    ts = start_ms
    while ts < end_ms:
        params = {"symbol": "BTCUSDT", "interval": "1h",
                  "startTime": ts, "limit": limit}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_rows.extend(batch)
        ts = batch[-1][0] + 1
        if len(batch) < limit:
            break
        _time.sleep(0.05)

    df = pd.DataFrame(all_rows, columns=[
        "OpenTime", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVol", "Trades", "TakerBase", "TakerQuote", "Ignore"
    ])
    df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
    df = df.set_index("OpenTime").astype(float)
    btc_close = df["Close"].rename("BTC_Close")
    ma200     = btc_close.rolling(200).mean()
    # Gibt DataFrame zurück: BTC_Close + BTC_ma200_bull
    result = pd.DataFrame({
        "BTC_Close":      btc_close,
        "BTC_ma200_bull": (btc_close > ma200),
    })
    return result


def compute_daily_macro(df_raw: pd.DataFrame) -> pd.Series:
    """
    Berechnet täglichen Makro-Bias aus stündlichen Daten.
    Kein API-Call nötig. daily_macro_bull = EMA21 > EMA50 auf Tagesbasis.
    """
    close_d  = df_raw["Close"].resample("1D").last().dropna()
    ema21    = close_d.ewm(span=21, adjust=False).mean()
    ema50    = close_d.ewm(span=50, adjust=False).mean()
    bull     = (ema21 > ema50).rename("Daily_macro_bull")
    # shift(1): verhindert Look-Ahead — Stunden von Tag D nutzen nur EMA-Status von Tag D-1
    return bull.shift(1).reindex(df_raw.index, method="ffill").fillna(False)


def fetch_fear_greed() -> pd.Series:
    """
    Lädt historischen Fear & Greed Index von Alternative.me (kostenlos, täglich seit 2020).
    Gibt FG_ok (F&G >= 40, kein Panik-Modus) auf stündliches Raster forward-filled zurück.
    Logik: F&G < 40 = anhaltende Panik → Mean-Reversion-Longs scheitern strukturell.
    """
    import requests
    r    = requests.get("https://api.alternative.me/fng/?limit=2000", timeout=15)
    data = r.json()["data"]
    df   = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df   = df.set_index("ts").sort_index()
    fg   = df["value"].astype(int).rename("FG_ok")
    fg   = (fg >= 40)
    idx_h = pd.date_range(start=fg.index[0], end=pd.Timestamp.now(), freq="1h")
    return fg.reindex(idx_h, method="ffill").fillna(False)


def compute_sol_btc_rs(df_raw: pd.DataFrame, btc_series: pd.Series) -> pd.Series:
    """
    SOL/BTC Relative Strength: True wenn SOL/BTC-Ratio über eigenem 50h-Durchschnitt liegt.
    Indikator: SOL hält seinen Wert vs. BTC oder outperformed → mean-reversion-Long wahrscheinlicher.
    """
    btc_aligned  = btc_series.reindex(df_raw.index, method="ffill")
    # btc_series kann BTC_ma200_bull (bool) oder BTC Close-Preise sein — wir brauchen Close
    # Wir berechnen das Ratio direkt aus df_raw.Close / BTC_Close
    # Deshalb akzeptieren wir hier eine BTC-Close-Series statt der bool-Series
    ratio        = (df_raw["Close"] / btc_aligned).rename("SOL_BTC_RS")
    rs_bull      = (ratio > ratio.rolling(50).mean()).rename("SOL_BTC_rs_bull")
    return rs_bull.fillna(False)


def fetch_funding_rate(start: str = "2021-09-01") -> pd.Series:
    """
    Lädt SOL-USDT Perpetual Funding Rate von Binance Futures API (8h-Intervall).
    Gibt funding_bull = (funding_rate < 0) zurück, forward-filled auf 1h-Raster.
    Funding < 0 bedeutet: Shorts zahlen Longs — bearish crowded, Mean-Reversion-Signal.
    """
    import requests, time as _time

    url      = "https://fapi.binance.com/fapi/v1/fundingRate"
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int(pd.Timestamp.now().timestamp() * 1000)
    limit    = 1000
    all_rows = []

    ts = start_ms
    while ts < end_ms:
        params = {"symbol": "SOLUSDT", "startTime": ts, "limit": limit}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_rows.extend(batch)
        ts = batch[-1]["fundingTime"] + 1
        if len(batch) < limit:
            break
        _time.sleep(0.05)

    df_fr = pd.DataFrame(all_rows)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], unit="ms")
    df_fr = df_fr.set_index("fundingTime")
    df_fr["fundingRate"] = df_fr["fundingRate"].astype(float)
    funding_bull = (df_fr["fundingRate"] < 0).rename("Funding_bull")
    # Forward-fill auf stündliches Raster
    idx_hourly  = pd.date_range(start=df_fr.index[0], end=pd.Timestamp.now(), freq="1h")
    funding_1h  = funding_bull.reindex(idx_hourly).ffill()
    return funding_1h


def run_backtest(df_raw: pd.DataFrame, sw_mode: str = "v13", max_pos_size: float = 0.10,
                 extra_signals: dict = None, sw_sl_mult: float = None,
                 sw_tp_mult: float = None, sw_lev_override: float = None,
                 baer_lev_override: float = None,
                 bulle_lev_override: float = None,
                 extrem_bulle_lev_override: float = None,
                 extrem_bulle_sl_mult: float = None,
                 extrem_bulle_tp_mult: float = None,
                 extrem_bulle_trail_mult: float = None,
                 bulle_sl_mult: float = None,
                 bulle_trail_mult: float = None,
                 bulle_tp_mult: float = None,
                 sw_short_lev_override: float = None,
                 track_blocked: bool = False,
                 bulle_mode: str = "default",
                 bulle_lev_high_override: float = None,
                 bulle_adx_thresh_override: float = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Führt den vollständigen Backtest durch.

    Parameter:
        df_raw  : OHLCV-DataFrame (Output von make_price_data)
        sw_mode : 'v13' | 'rsi_div' | 'off'

    Rückgabe:
        (trades_df, equity_df)
    """
    # Indikatoren berechnen (langsam wegen RSI-Divergenz-Iteration)
    df = add_indicators(df_raw)
    df["Regime"] = classify_regime(df)

    # Extra-Signale (BTC-Trend, Funding-Rate) mergen
    if extra_signals is not None and len(extra_signals) > 0:
        for col, series in extra_signals.items():
            aligned = series.reindex(df.index, method="ffill")
            df[col] = aligned.fillna(False)

    cash          = STARTKAPITAL
    pos: Position | None = None
    equity        = []
    trades        = []
    blocked_log   = []   # (pos_regime, pos_side, blocked_regime, blocked_side) wenn track_blocked
    peak          = STARTKAPITAL
    cooldown_left = 0    # Kerzen-Cooldown nach Exit, verhindert sofortigen Re-Entry
    COOLDOWN      = 5    # mind. 5 Kerzen Pause nach Exit
    DD_RESET_BARS = 90 * 24   # Rolling DD-Peak: alle 90 Tage zurücksetzen
    # Ringpuffer für Rolling-Peak (letzten 90 Tage an Portfolio-Werten)
    equity_window: list[float] = []
    regime_age    = 0    # Kerzen seit letztem Wechsel ins Bulle-Regime
    prev_regime   = None
    atl_peak      = STARTKAPITAL   # All-Time-High (für zukünftige Circuit-Breaker-Erweiterung)

    for i, (ts, row) in enumerate(df.iterrows()):
        kurs  = row["Close"]
        atr   = row["ATR"]   if not pd.isna(row["ATR"])  else 0.001 * kurs
        vola  = row["Vola"]  if not pd.isna(row["Vola"]) else 0.5
        regime = row["Regime"]

        # Regime-Alter für Bulle-Diagnose
        if regime == "bulle":
            if prev_regime != "bulle":
                regime_age = 0
            regime_age += 1
        else:
            regime_age = 0
        prev_regime = regime

        # Aktueller DD (Rolling 90-Tage-Peak statt All-Time-Peak)
        # All-Time-Peak für Circuit-Breaker separat tracken
        port_value = cash if pos is None else (
            cash + _mtm(pos, kurs)
        )
        equity_window.append(port_value)
        if len(equity_window) > DD_RESET_BARS:
            equity_window.pop(0)
        peak     = max(equity_window)
        dd       = (port_value - peak) / peak
        atl_peak = max(atl_peak, port_value)

        # ── Funding-Kosten für Short ─────────────────────────────────────
        if pos is not None and pos.side == "short" and i % 8 == 0:
            funding_cost = pos.entry_cash * FUNDING8 * pos.hebel
            cash -= funding_cost

        # ── SL / TP / Trail prüfen ───────────────────────────────────────
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
                # Trail-Update
                if pos.trail_price is not None and atr > 0:
                    if regime == "bulle":
                        trail_mult = bulle_trail_mult if bulle_trail_mult is not None else 3.0
                    elif regime == "extrem_bulle" and extrem_bulle_trail_mult is not None:
                        trail_mult = extrem_bulle_trail_mult
                    else:
                        trail_mult = 1.2
                    new_trail  = kurs - trail_mult * atr
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
                # Seitwärts Cover: nur wenn Position auch IN seitwaerts als Short geöffnet wurde
                if regime == "seitwaerts" and pos.regime == "seitwaerts" and pos.side == "short" and signal_seitwaerts_cover(row) and exit_reason is None:
                    exit_reason = "Cover"
                    exit_price  = _buy_price(kurs, vola)
                # Extrem Bär Cover — dedizierte Funktion, unabhängig vom Short-Entry-Signal
                if regime == "extrem_baer" and pos.side == "short":
                    if signal_extrem_baer_cover(row) and exit_reason is None:
                        exit_reason = "Cover"
                        exit_price  = _buy_price(kurs, vola)
                # Bär Cover — dedizierte Funktion, unabhängig vom Short-Entry-Signal
                if regime == "baer" and pos.regime == "baer" and pos.side == "short":
                    if signal_baer_cover(row) and exit_reason is None:
                        exit_reason = "Cover"
                        exit_price  = _buy_price(kurs, vola)

            if exit_reason:
                proceeds = _exit_pnl(pos, exit_price)
                pnl_pct  = (proceeds - pos.entry_cash) / pos.entry_cash * 100
                trades.append({
                    "Entry_ts":         df.index[pos.candle_idx],
                    "Exit_ts":          ts,
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
                })
                cash          += proceeds
                pos            = None
                cooldown_left  = COOLDOWN

        # ── Bulle Long-Exit ──────────────────────────────────────────────
        if pos is not None and pos.side == "long" and regime == "bulle":
            if signal_bulle_exit(row):
                exit_price = _sell_price(kurs, vola)
                proceeds   = _exit_pnl(pos, exit_price)
                pnl_pct    = (proceeds - pos.entry_cash) / pos.entry_cash * 100
                trades.append({
                    "Entry_ts":         df.index[pos.candle_idx],
                    "Exit_ts":          ts,
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
                })
                cash          += proceeds
                pos            = None
                cooldown_left  = COOLDOWN

        # ── Pyramiding ───────────────────────────────────────────────────
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
                add_cash       = cash * extra_pct
                pyramid_price  = _buy_price(kurs, vola)
                new_total_cash = pos.entry_cash + add_cash
                pos.entry_price = (pos.entry_price * pos.entry_cash + pyramid_price * add_cash) / new_total_cash
                cash           -= add_cash
                pos.entry_cash  = new_total_cash
                pos.pyramid_count += 1

        # ── Cooldown runterzählen ────────────────────────────────────────
        if cooldown_left > 0:
            cooldown_left -= 1

        # ── Blockierte Signale tracken ───────────────────────────────────
        if track_blocked and pos is not None and cooldown_left == 0:
            b_sig = get_signal(row, regime, sw_mode, bulle_mode=bulle_mode)
            if b_sig in (1, -2):
                b_side = "short" if b_sig == -2 else "long"
                if not (regime == pos.regime and b_side == pos.side):
                    blocked_log.append((pos.regime, pos.side, regime, b_side))

        # ── Neues Signal prüfen ──────────────────────────────────────────
        if pos is None and cash > 1.0 and cooldown_left == 0:
            sig = get_signal(row, regime, sw_mode, bulle_mode=bulle_mode)

            if sig in (1, -2):
                is_short = (sig == -2)
                side     = "short" if is_short else "long"
                h        = calc_hebel(_g(row, "ADX", 20), _g(row, "Vola_ratio", 1),
                                      regime, is_short, dd)
                if regime == "seitwaerts" and not is_short:
                    h = 2.0  # SW-Long: fester 2× Hebel (Sweep-Optimum)
                if sw_lev_override is not None and regime == "seitwaerts" and not is_short:
                    h = sw_lev_override
                if regime == "baer" and is_short:
                    base_h = baer_lev_override if baer_lev_override is not None else 8.0
                    if dd < -0.25:
                        h = 1.0
                    elif dd < -0.15:
                        h = max(base_h * 0.5, 1.0)
                    else:
                        h = base_h
                if regime == "bulle" and not is_short:
                    _entry_adx  = _g(row, "ADX", 0.0)
                    _adx_thresh = bulle_adx_thresh_override if bulle_adx_thresh_override is not None else 0.0
                    _high_lev   = bulle_lev_high_override   if bulle_lev_high_override   is not None else 1.0
                    if _adx_thresh > 0 and _entry_adx >= _adx_thresh:
                        base_h = _high_lev
                    else:
                        base_h = bulle_lev_override if bulle_lev_override is not None else 1.0
                    if dd < -0.25:
                        h = 1.0
                    elif dd < -0.15:
                        h = max(base_h * 0.5, 1.0)
                    else:
                        h = base_h
                if extrem_bulle_lev_override is not None and regime == "extrem_bulle" and not is_short:
                    base_h = extrem_bulle_lev_override
                    if dd < -0.25:
                        h = 1.0
                    elif dd < -0.15:
                        h = max(base_h * 0.5, 1.0)
                    else:
                        h = base_h
                if sw_short_lev_override is not None and regime == "seitwaerts" and is_short:
                    base_h = sw_short_lev_override
                    if dd < -0.25:
                        h = 1.0
                    elif dd < -0.15:
                        h = max(base_h * 0.5, 1.0)
                    else:
                        h = base_h

                # Trade-Diagnose
                _trigger_id       = None
                _rsi_entry        = None
                _dist_ema21_entry = None
                _regime_age_entry = None
                _adx_entry        = None
                _vola_ratio_entry = None
                if regime == "bulle" and side == "long":
                    _, _trigger_id = signal_bulle_diag(row)
                    _rsi_entry        = _g(row, "RSI", None)
                    _ema21            = _g(row, "EMA21", None)
                    _dist_ema21_entry = (kurs / _ema21 - 1) * 100 if _ema21 else None
                    _regime_age_entry = regime_age
                    _adx_entry        = _g(row, "ADX", None)
                elif regime == "baer" and side == "short":
                    _, _trigger_id = signal_baer_diag(row)
                    _rsi_entry        = _g(row, "RSI", None)
                    _vwap             = _g(row, "VWAP", None)
                    _dist_ema21_entry = (kurs / _vwap - 1) * 100 if _vwap else None  # Dist zu VWAP
                    _regime_age_entry = regime_age
                elif regime == "extrem_bulle" and side == "long":
                    _rsi_entry        = _g(row, "RSI", None)
                    _dist_ema21_entry = _g(row, "MA200_dist", None)   # Distanz MA200 in %
                    _adx_entry        = _g(row, "ADX", None)
                    _vola_ratio_entry = _g(row, "Vola_ratio", None)
                    _ma500            = _g(row, "MA500", 0.0)
                    # regime_age_entry: 1 = close über MA500, 0 = darunter (als Boolean-Flag)
                    _regime_age_entry = 1 if (_ma500 > 0 and kurs > _ma500) else 0
                elif regime == "seitwaerts" and side == "long":
                    _rsi_entry        = _g(row, "RSI", None)
                    _bb_lower_val     = _g(row, "BB_lower", None)
                    _dist_ema21_entry = (kurs / _bb_lower_val - 1) * 100 if _bb_lower_val else None
                    _rsi_div_b        = bool(_g(row, "RSI_div_bull", False))
                    _obv_b            = bool(_g(row, "OBV_bull",     False))
                    _st_b             = bool(_g(row, "ST_bull",      False))
                    _rsi_lo           = (_rsi_entry is not None and _rsi_entry < 45)
                    _div_ok           = _rsi_div_b or _st_b
                    _parts            = []
                    if _rsi_lo:  _parts.append("rsi")
                    if _div_ok:  _parts.append("div")
                    if _obv_b:   _parts.append("obv")
                    _trigger_id       = "+".join(_parts) if _parts else "none"
                    _regime_age_entry = int(_rsi_div_b) + 2 * int(_obv_b)  # Bitmask 0-3
                    _adx_entry        = _g(row, "ADX", None)
                    _vola_ratio_entry = _g(row, "Vola_ratio", None)

                sl_m, tp_m = SL_TP.get(regime, (2.0, 4.0))
                if regime == "seitwaerts" and side == "short":
                    sl_m, tp_m = 2.0, None  # SW-Short: eigenes SL/TP, unabhängig von SW-Long
                if sw_sl_mult is not None and regime == "seitwaerts" and side == "long":
                    sl_m = sw_sl_mult
                if sw_tp_mult is not None and regime == "seitwaerts" and side == "long":
                    tp_m = sw_tp_mult
                if extrem_bulle_sl_mult is not None and regime == "extrem_bulle":
                    sl_m = extrem_bulle_sl_mult
                if extrem_bulle_tp_mult is not None and regime == "extrem_bulle":
                    tp_m = extrem_bulle_tp_mult if extrem_bulle_tp_mult > 0 else None
                if bulle_sl_mult is not None and regime == "bulle":
                    sl_m = bulle_sl_mult
                if bulle_tp_mult is not None and regime == "bulle":
                    tp_m = bulle_tp_mult

                if side == "long":
                    entry_p = _buy_price(kurs, vola)
                    sl_p    = entry_p - sl_m * atr
                    tp_p    = (entry_p + tp_m * atr) if tp_m else None
                    _eb_tm  = extrem_bulle_trail_mult if extrem_bulle_trail_mult is not None else 1.2
                    trail   = entry_p - _eb_tm * atr  # Standard-Trail für Long (EB-override möglich)
                    if regime == "bulle":
                        _bl_tm = bulle_trail_mult if bulle_trail_mult is not None else 3.0
                        trail = entry_p - _bl_tm * atr
                    elif regime == "extrem_bulle":
                        trail = entry_p - _eb_tm * atr  # Explizit setzen (redundant, aber klar)
                    elif regime == "seitwaerts":
                        trail = None  # Kein Trail für SW Mean-Reversion
                else:
                    entry_p = _sell_price(kurs, vola)
                    sl_p    = entry_p + sl_m * atr
                    tp_p    = (entry_p - tp_m * atr) if tp_m else None
                    trail   = None  # Shorts kein Trail

                # Seitwärts Short Trail: 1.0×ATR
                if regime == "seitwaerts" and side == "short":
                    trail = entry_p + 1.0 * atr

                pos_size = calc_position_size(1, 1, _g(row, "ADX", 20),
                                              extrem=(regime in ("extrem_baer", "extrem_bulle")))
                pos_size = min(pos_size, max_pos_size)
                use_cash = cash * pos_size
                cash    -= use_cash

                pos = Position(
                    side=side,
                    entry_price=entry_p,
                    entry_cash=use_cash,
                    hebel=h,
                    sl_price=sl_p,
                    tp_price=tp_p,
                    trail_price=trail,
                    regime=regime,
                    candle_idx=i,
                    trigger_id=_trigger_id,
                    rsi_entry=_rsi_entry,
                    dist_ema21_entry=_dist_ema21_entry,
                    regime_age_entry=_regime_age_entry,
                    adx_entry=_adx_entry,
                    vola_ratio_entry=_vola_ratio_entry,
                )

        # ── Equity tracken ───────────────────────────────────────────────
        if pos is not None:
            est = _mtm(pos, kurs)
            equity.append(cash + est)
        else:
            equity.append(cash)

    # Offene Position schließen
    if pos is not None:
        last_row   = df.iloc[-1]
        last_kurs  = last_row["Close"]
        last_vola  = last_row["Vola"] if not pd.isna(last_row["Vola"]) else 0.5
        exit_price = _sell_price(last_kurs, last_vola) if pos.side == "long" else _buy_price(last_kurs, last_vola)
        proceeds   = _exit_pnl(pos, exit_price)
        pnl_pct    = (proceeds - pos.entry_cash) / pos.entry_cash * 100
        trades.append({
            "Entry_ts":         df.index[pos.candle_idx],
            "Exit_ts":          df.index[-1],
            "Side":             pos.side,
            "Entry_price":      pos.entry_price,
            "Exit_price":       exit_price,
            "Hebel":            pos.hebel,
            "PnL_pct":          pnl_pct,
            "Entry_cash":       pos.entry_cash,
            "Regime":           pos.regime,
            "Exit_reason":      "End",
            "Trigger_id":       pos.trigger_id,
            "RSI_entry":        pos.rsi_entry,
            "Dist_ema21_entry": pos.dist_ema21_entry,
            "Regime_age_entry": pos.regime_age_entry,
            "Pyramid_count":    pos.pyramid_count,
        })
        cash += proceeds

    trades_df  = pd.DataFrame(trades)
    equity_df  = pd.Series(equity, index=df.index, name="Equity")
    blocked_df = pd.DataFrame(blocked_log, columns=["pos_regime","pos_side","blocked_regime","blocked_side"]) if blocked_log else pd.DataFrame()

    return trades_df, equity_df, df, blocked_df
