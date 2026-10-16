"""
strategy.py — Signallogik je Regime, dynamischer Hebel, Position Sizing

Signal-Codes:
  -2 = Short eröffnen
  -1 = Short (schwächer, nicht genutzt)
   0 = Kein Signal
   1 = Long eröffnen
   2 = Cover / Exit

SL/TP-Multiplikatoren werden je Regime als Tupel (sl_mult, tp_mult) zurückgegeben.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Dynamischer Hebel ──────────────────────────────────────────────────────

def calc_hebel(adx: float, vola_ratio: float, regime: str,
               is_short: bool, current_dd: float) -> float:
    """Berechnet den Hebel basierend auf ADX, Vola, Regime und DD."""
    # Nan-Schutz
    if np.isnan(adx):       adx       = 20.0
    if np.isnan(vola_ratio): vola_ratio = 1.0

    # Basis aus ADX
    if adx < 20:   basis = 1.0
    elif adx < 30: basis = 1.5
    elif adx < 40: basis = 2.0
    elif adx < 50: basis = 3.0
    else:          basis = 4.0

    # Vola-Multiplikator
    if vola_ratio > 3.0:   vm = 0.25
    elif vola_ratio > 2.0: vm = 0.5
    elif vola_ratio > 1.5: vm = 0.75
    else:                  vm = 1.0

    h = basis * vm

    # Regime-Limits
    regime_max = {
        "extrem_baer":  1.0,
        "baer":         2.0,
        "seitwaerts":   3.0,
        "neutral":      2.0,
        "bulle":        4.0,
        "extrem_bulle": 4.0,
    }
    h = min(h, regime_max.get(regime, 2.0))

    # Short immer max 2×
    if is_short:
        h = min(h, 2.0)

    # Drawdown-Schutz
    if current_dd < -0.15:
        h = max(h * 0.5, 1.0)
    if current_dd < -0.25:
        h = 1.0

    return max(h, 1.0)


# ─── Position Sizing ────────────────────────────────────────────────────────

def calc_position_size(conditions_met: int, conditions_total: int,
                       adx: float, extrem: bool = False) -> float:
    """Gibt den Anteil des Kapitals zurück (0.0–1.0)."""
    if np.isnan(adx): adx = 20.0

    ratio = conditions_met / max(conditions_total, 1)
    if ratio >= 1.0:   base = 1.0
    elif ratio >= 0.67: base = 0.65
    else:               base = 0.40

    if extrem:
        base = min(base * 1.3, 1.0)
    elif adx > 45:
        base = min(base * 1.25, 1.0)
    elif adx > 35:
        base = min(base * 1.15, 1.0)

    return base


# ─── SL/TP je Regime ────────────────────────────────────────────────────────

SL_TP = {
    "extrem_baer":  (1.5, 2.0),   # TP verkürzt: 3.5→2.0 für höhere TP-Hit-Rate in echten Crashes
    "baer":         (2.0, 4.5),
    "seitwaerts":   (3.5, 10.0),   # SL 3.5×ATR, TP 10×ATR; Sweep-Optimum: Sharpe 2.36, MaxDD -20.4%
    "neutral":      (2.0, 4.0),
    "bulle":        (2.0, 4.5),
    "extrem_bulle": (1.5, 5.0),
}


# ─── Signallogik je Regime ──────────────────────────────────────────────────

def _g(row, col, default=False):
    """Sicher einen Wert aus einer Zeile holen."""
    val = row.get(col, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


def signal_extrem_baer(row) -> tuple[int, int, int]:
    """
    Gibt (signal, sl_mult_x10, tp_mult_x10) zurück.
    signal: -2=Short, 1=Long(Kapitulation), 2=Cover, 0=nichts
    """
    rsi       = _g(row, "RSI",         50.0)
    ema_bear  = _g(row, "EMA_bear",    False)
    di_bear   = _g(row, "DI_bear",     False)
    macd_neg  = _g(row, "MACD_neg",    False)
    ich_bear  = _g(row, "Ichimoku_bear", False)
    obv_bull  = _g(row, "OBV_bull",    False)
    consec    = _g(row, "Consec",      0)
    close     = _g(row, "Close",       0.0)
    vwap      = _g(row, "VWAP",        0.0)
    vol_soft  = _g(row, "vol_soft",    True)
    rsi_div_b = _g(row, "RSI_div_bull", False)
    vol_surge = _g(row, "Vol_surge",   False)
    sar_bull  = _g(row, "SAR_bull",    False)
    stoch_k   = _g(row, "Stoch_K",     50.0)
    ema_bull  = _g(row, "EMA_bull",    False)
    macd_pos  = _g(row, "MACD_pos",    False)

    # Short deaktiviert (Phase-1-Test: Long-only)

    # Long (Kapitulation)
    if rsi < 18 and (rsi_div_b or vol_surge) and sar_bull:
        return 1

    return 0


def signal_baer(row) -> int:
    ema_bear  = _g(row, "EMA_bear",     False)
    di_bear   = _g(row, "DI_bear",      False)
    macd_neg  = _g(row, "MACD_neg",     False)
    ich_bear  = _g(row, "Ichimoku_bear", False)
    obv_bull  = _g(row, "OBV_bull",     False)
    close     = _g(row, "Close",        0.0)
    vwap      = _g(row, "VWAP",         0.0)
    vol_soft  = _g(row, "vol_soft",     True)
    rsi_div_b = _g(row, "RSI_div_bull", False)
    st_bull   = _g(row, "ST_bull",      False)

    # Vetos: kein Short wenn Trend dreht
    if st_bull or rsi_div_b:
        return 0

    cond_a = ema_bear and di_bear
    cond_b = macd_neg
    cond_c = ich_bear and not obv_bull
    cond_d = close < vwap
    score  = sum([cond_a, cond_b, cond_c, cond_d])

    rsi = _g(row, "RSI", 50.0)
    # Dead-Cat-Bounce Short: A+B+C required + Kurs >= VWAP + RSI < 35 + vol_soft
    if cond_a and cond_b and cond_c and close >= vwap and rsi < 35 and vol_soft:
        return -2
    return 0


def signal_baer_diag(row) -> tuple[int, str | None]:
    """Wie signal_baer(), gibt zusätzlich Bedingungsstring zurück."""
    ema_bear  = _g(row, "EMA_bear",     False)
    di_bear   = _g(row, "DI_bear",      False)
    macd_neg  = _g(row, "MACD_neg",     False)
    ich_bear  = _g(row, "Ichimoku_bear", False)
    obv_bull  = _g(row, "OBV_bull",     False)
    close     = _g(row, "Close",        0.0)
    vwap      = _g(row, "VWAP",         0.0)
    vol_soft  = _g(row, "vol_soft",     True)
    rsi_div_b = _g(row, "RSI_div_bull", False)
    st_bull   = _g(row, "ST_bull",      False)

    if st_bull or rsi_div_b:
        return (0, None)

    cond_a = ema_bear and di_bear
    cond_b = macd_neg
    cond_c = ich_bear and not obv_bull
    cond_d = close < vwap
    score  = sum([cond_a, cond_b, cond_c, cond_d])

    if score >= 3 and vol_soft:
        label = ("A" if cond_a else "") + ("B" if cond_b else "") + ("C" if cond_c else "") + ("D" if cond_d else "")
        return (-2, label)
    return (0, None)


def signal_seitwaerts(row, sw_mode: str = "v13") -> int:
    """
    sw_mode: 'v13' = BB-Berührung, 'rsi_div' = RSI-Divergenz-Einstieg, 'off' = kein Handel
    """
    if sw_mode == "off":
        return 0

    squeeze   = _g(row, "Squeeze",     False)
    close     = _g(row, "Close",       0.0)
    bb_lower  = _g(row, "BB_lower",    0.0)
    bb_upper  = _g(row, "BB_upper",    float("inf"))
    rsi       = _g(row, "RSI",         50.0)
    st_bull   = _g(row, "ST_bull",     False)
    rsi_div_b = _g(row, "RSI_div_bull", False)
    rsi_div_r = _g(row, "RSI_div_bear", False)
    obv_bull  = _g(row, "OBV_bull",    False)
    vol_soft  = _g(row, "vol_soft",    True)
    macd_neg  = _g(row, "MACD_neg",    False)
    stoch_k   = _g(row, "Stoch_K",     50.0)
    stoch_d   = _g(row, "Stoch_D",     50.0)

    # Kein Handel bei Squeeze
    if squeeze:
        return 0

    if sw_mode == "v13":
        # Long: Kurs ≤ BB_lower + 2/3 Bedingungen
        long_score = sum([
            rsi < 45,
            st_bull or rsi_div_b,
            obv_bull,
        ])
        if close <= bb_lower and long_score >= 2 and vol_soft:
            return 1

        # Short: 4/6 Bedingungen
        # "nahe BB_oben" = Preis berührt oder übersteigt BB_upper (symmetrisch zu Long: close <= BB_lower)
        short_score = sum([
            close >= bb_upper,
            rsi > 58,
            stoch_k < stoch_d and stoch_k > 70,
            not obv_bull,
            macd_neg or rsi_div_r,
            not st_bull,
        ])
        if short_score >= 4 and vol_soft:
            return -2

    elif sw_mode == "rsi_div":
        # Seitwärts-Long deaktiviert: WR 10.5% in jedem Jahr negativ (strukturelles Problem)
        short_score = sum([
            close >= bb_upper,
            rsi > 58,
            stoch_k < stoch_d and stoch_k > 70,
            not obv_bull,
            macd_neg or rsi_div_r,
            not st_bull,
        ])
        if short_score >= 4 and vol_soft:
            return -2

    elif sw_mode in ("daily_bull", "fear_greed", "sol_btc_rs",
                     "funding_daily", "fear_daily"):
        # Makro-gefiltertes v13: Long nur wenn jeweiliger Makro-Filter aktiv
        daily_bull   = _g(row, "Daily_macro_bull", False)
        fg_ok        = _g(row, "FG_ok",            False)
        sol_btc_bull = _g(row, "SOL_BTC_rs_bull",  False)
        funding_bull = _g(row, "Funding_bull",      False)

        if sw_mode == "daily_bull":
            long_allowed = daily_bull
        elif sw_mode == "fear_greed":
            long_allowed = fg_ok
        elif sw_mode == "sol_btc_rs":
            long_allowed = sol_btc_bull
        elif sw_mode == "funding_daily":
            long_allowed = funding_bull and daily_bull
        elif sw_mode == "fear_daily":
            long_allowed = fg_ok and daily_bull

        if long_allowed:
            long_score = sum([
                rsi < 45,
                st_bull or rsi_div_b,
                obv_bull,
            ])
            if close <= bb_lower and long_score >= 2 and vol_soft:
                return 1

        # Shorts: immer erlaubt
        short_score = sum([
            close >= bb_upper,
            rsi > 58,
            stoch_k < stoch_d and stoch_k > 70,
            not obv_bull,
            macd_neg or rsi_div_r,
            not st_bull,
        ])
        if short_score >= 4 and vol_soft:
            return -2

    elif sw_mode in ("btc_trend", "funding_rate", "btc_funding"):
        # Makro-gefiltertes v13: Long nur wenn Makro-Filter aktiv, Short immer erlaubt
        btc_bull     = _g(row, "BTC_ma200_bull", False)
        funding_bull = _g(row, "Funding_bull",   False)

        if sw_mode == "btc_trend":
            long_allowed = btc_bull
        elif sw_mode == "funding_rate":
            long_allowed = funding_bull
        else:  # btc_funding
            long_allowed = btc_bull and funding_bull

        if long_allowed:
            long_score = sum([
                rsi < 45,
                st_bull or rsi_div_b,
                obv_bull,
            ])
            if close <= bb_lower and long_score >= 2 and vol_soft:
                return 1

        # Shorts: immer erlaubt (identisch zu rsi_div)
        short_score = sum([
            close >= bb_upper,
            rsi > 58,
            stoch_k < stoch_d and stoch_k > 70,
            not obv_bull,
            macd_neg or rsi_div_r,
            not st_bull,
        ])
        if short_score >= 4 and vol_soft:
            return -2

    elif sw_mode == "v15":
        # v15: Stochastik-Crossover im unteren/oberen Range-Bereich
        # Einstieg erst NACH dem Drehen des Momentums — kein fallendes Messer
        bb_width = (bb_upper - bb_lower) if (bb_upper > bb_lower) else 0.001
        bb_pos   = (close - bb_lower) / bb_width  # 0.0 = unteres Band, 1.0 = oberes Band

        stoch_cross_up   = (stoch_k > stoch_d) and (stoch_k < 45)  # Stoch dreht aufwärts, noch niedrig
        stoch_cross_down = (stoch_k < stoch_d) and (stoch_k > 55)  # Stoch dreht abwärts, noch hoch

        # Long: unteres Drittel der Range + Stoch dreht nach oben + RSI-Bestätigung
        if bb_pos < 0.30 and stoch_cross_up and rsi < 48 and (rsi_div_b or obv_bull):
            return 1

        # Short: oberes Drittel der Range + Stoch dreht nach unten + RSI-Bestätigung
        if bb_pos > 0.70 and stoch_cross_down and rsi > 52 and (rsi_div_r or not obv_bull) and vol_soft:
            return -2

    return 0


def signal_baer_cover(row) -> bool:
    """
    Cover-Signal für Bär-Short — separat geprüft vor dem Short-Entry-Signal,
    um das strukturelle Override-Problem zu vermeiden.
    """
    stoch_k  = _g(row, "Stoch_K",  50.0)
    ema_bull = _g(row, "EMA_bull", False)
    macd_pos = _g(row, "MACD_pos", False)
    sar_bull = _g(row, "SAR_bull", False)
    rsi      = _g(row, "RSI",      50.0)
    return stoch_k < 12 or (ema_bull and macd_pos and sar_bull) or rsi > 50


def signal_extrem_baer_cover(row) -> bool:
    """
    Cover-Signal für Extrem-Bär-Short — separat geprüft vor dem Short-Entry-Signal.
    """
    stoch_k  = _g(row, "Stoch_K",  50.0)
    ema_bull = _g(row, "EMA_bull", False)
    macd_pos = _g(row, "MACD_pos", False)
    rsi      = _g(row, "RSI",      50.0)
    return stoch_k < 8 or (ema_bull and macd_pos and rsi < 25)


def signal_seitwaerts_cover(row) -> bool:
    """
    Cover-Signal für Seitwärts-Short (v13).
    Stoch_K überquert 40 von unten (Crossover) UND RSI steigt (RSI > 40).
    Da wir keine vorherige Kerze sehen, nutzen wir: Stoch_K > Stoch_D (bullish Kreuz)
    UND Stoch_K im Bereich 35–55 (frisch über 40) UND RSI > 40.
    """
    stoch_k = _g(row, "Stoch_K", 50.0)
    stoch_d = _g(row, "Stoch_D", 50.0)
    rsi     = _g(row, "RSI",     50.0)
    # Stoch bullish Kreuz im mittleren Bereich + RSI erholt
    return (stoch_k > stoch_d) and (35 < stoch_k < 60) and (rsi > 40)


def signal_bulle(row) -> int:
    """
    Qualitätsgefilterte Bulle-Long-Entries (RSI-Momentum-Zone 65-70).
    """
    close          = _g(row, "Close",          0.0)
    ema21          = _g(row, "EMA21",          close)
    ma50           = _g(row, "MA50",           close)
    rsi            = _g(row, "RSI",            50.0)
    macd_hist_up   = _g(row, "MACD_hist_up",   False)
    ema21_slope    = _g(row, "EMA21_slope",    True)
    cross_ema21_up = _g(row, "cross_ema21_up", False)
    di_bull        = _g(row, "DI_bull",        False)
    obv_bull       = _g(row, "OBV_bull",       False)
    sar_bull       = _g(row, "SAR_bull",       False)
    adx            = _g(row, "ADX",            20.0)
    ma500          = _g(row, "MA500",          0.0)

    # ── Pflicht-Filter ──────────────────────────────────────────────────────
    if not di_bull:         return 0
    if not ema21_slope:     return 0
    if close < ma50 * 0.97: return 0
    if ma500 > 0 and close < ma500: return 0

    if rsi < 65 or rsi > 70: return 0   # RSI-Fenster: Momentum-Zone 65-70
    if adx < 35:             return 0   # Trend muss stark sein (Diskriminator: Winner ADX 38.6 vs Loser 33.1)

    not_extended = close < ema21 * 1.02

    # ── Trigger A: EMA21-Bounce + Momentum dreht gerade auf ─────────────────
    if cross_ema21_up and macd_hist_up and (obv_bull or sar_bull):
        return 1

    # ── Trigger C: MACD dreht auf, Kurs nah an EMA21, ADX stark ─────────────
    if not_extended and macd_hist_up and adx > 28 and obv_bull and sar_bull:
        return 1

    return 0


def signal_bulle_diag(row) -> tuple[int, str | None]:
    """Wie signal_bulle(), gibt trigger_id zurück. Lockstep mit signal_bulle() halten."""
    close          = _g(row, "Close",          0.0)
    ema21          = _g(row, "EMA21",          close)
    ma50           = _g(row, "MA50",           close)
    rsi            = _g(row, "RSI",            50.0)
    macd_hist_up   = _g(row, "MACD_hist_up",   False)
    ema21_slope    = _g(row, "EMA21_slope",    True)
    cross_ema21_up = _g(row, "cross_ema21_up", False)
    di_bull        = _g(row, "DI_bull",        False)
    obv_bull       = _g(row, "OBV_bull",       False)
    sar_bull       = _g(row, "SAR_bull",       False)
    adx            = _g(row, "ADX",            20.0)
    ma500          = _g(row, "MA500",          0.0)

    if not di_bull:         return (0, None)
    if not ema21_slope:     return (0, None)
    if close < ma50 * 0.97: return (0, None)
    if ma500 > 0 and close < ma500: return (0, None)
    if rsi < 65 or rsi > 70: return (0, None)
    if adx < 35:             return (0, None)

    not_extended = close < ema21 * 1.02

    if cross_ema21_up and macd_hist_up and (obv_bull or sar_bull):
        return (1, "A")
    if not_extended and macd_hist_up and adx > 28 and obv_bull and sar_bull:
        return (1, "C")
    return (0, None)


def signal_bulle_exit(row) -> bool:
    """Long-Exit im Bulle-Regime."""
    ich_bull = _g(row, "Ichimoku_bull", True)
    sar_bull = _g(row, "SAR_bull",      True)
    macd_neg = _g(row, "MACD_neg",      False)
    close    = _g(row, "Close",         0.0)
    vwap     = _g(row, "VWAP",          0.0)
    return not ich_bull and not sar_bull and macd_neg and close < vwap


def signal_bulle_pyramid(row, entry_price: float) -> bool:
    """Pyramiding-Signal im Bulle-Regime."""
    roc          = _g(row, "ROC",          0.0)
    adx          = _g(row, "ADX",          20.0)
    close        = _g(row, "Close",        0.0)
    ema21        = _g(row, "EMA21",        close)
    obv_bull     = _g(row, "OBV_bull",     False)
    macd_hist_up = _g(row, "MACD_hist_up", False)

    # Trend intakt + Preis über Entry + Momentum bestätigt
    return (roc > 1.5 and adx > 28 and close > ema21
            and close > entry_price and (obv_bull or macd_hist_up))


def signal_extrem_bulle(row) -> int:
    # EB-A: Entry wenn ADX < 40 (Erschöpfungsschutz bei Blowoff-Top)
    adx = _g(row, "ADX", 20)
    if adx >= 40:
        return 0
    return 1


def signal_extrem_bulle_pyramid(row, entry_price: float) -> bool:
    roc      = _g(row, "ROC",           0.0)
    atr_sp   = _g(row, "ATR_spike",     1.0)
    ich_bull = _g(row, "Ichimoku_bull", False)
    ema_bull = _g(row, "EMA_bull",      False)
    close    = _g(row, "Close",         0.0)
    return roc > 1.5 and atr_sp > 1.5 and ich_bull and ema_bull and close > entry_price


def signal_bulle_strict(row) -> int:
    """Methode 1: signal_bulle() + cross_ema21_up als Pflicht (stärkster Diskriminator)."""
    if not _g(row, "cross_ema21_up", False):
        return 0
    return signal_bulle(row)


def signal_bulle_reset(row) -> int:
    """
    Bulle-Reset: Kaufe den Erholungsbeginn nach einem Dip im Bullenmarkt.
    Entry wenn: Bulle-Regime + Tages-Makro-Bulle + DD_30d ≤ -8% + MACD_hist_up.
    MACD_hist_up = Momentum-Umkehr bestätigt → verhindert reines Messer-Fangen.
    """
    daily_macro  = _g(row, "Daily_macro_bull", False)
    dd_30d       = _g(row, "DD_30d",           0.0)
    macd_hist_up = _g(row, "MACD_hist_up",     False)

    if not daily_macro:   return 0   # Makro-Trend bärisch → kein Kauf
    if dd_30d > -8.0:     return 0   # Mindestens 8% Rücksetzer vom 30d-Hoch
    if not macd_hist_up:  return 0   # MACD dreht auf = Erholungsbeginn
    return 1


def signal_bulle_pullback(row) -> int:
    """
    Pullback-Long im Bulle-Regime: Kaufe Rücksetzer zur EMA21.
    Entry wenn: Trend intakt + Kurs nahe/unter EMA21 + MACD dreht auf + RSI in Rücksetzer-Zone (30-60).
    Breiter als signal_bulle() (kein RSI-Fenster 65-70), kein Mindest-DD nötig.
    """
    close          = _g(row, "Close",          0.0)
    ema21          = _g(row, "EMA21",          close)
    ma50           = _g(row, "MA50",           close)
    rsi            = _g(row, "RSI",            50.0)
    macd_hist_up   = _g(row, "MACD_hist_up",   False)
    ema21_slope    = _g(row, "EMA21_slope",    True)
    di_bull        = _g(row, "DI_bull",        False)
    adx            = _g(row, "ADX",            20.0)
    daily_macro    = _g(row, "Daily_macro_bull", False)

    if not daily_macro:               return 0  # Makro-Trend bärisch
    if not di_bull:                   return 0  # DI+ < DI-: Trend nicht bullisch
    if not ema21_slope:               return 0  # EMA21 fällt: kein Aufwärtstrend
    if close < ma50 * 0.97:           return 0  # Kurs zu weit unter MA50
    if adx < 20:                      return 0  # Zu wenig Trend-Stärke
    if rsi > 60 or rsi < 30:          return 0  # Nicht im Rücksetzer-Fenster
    if not macd_hist_up:              return 0  # MACD-Histogramm dreht noch nicht auf
    if close > ema21 * 1.03:          return 0  # Kurs zu weit über EMA21 (kein echter Rücksetzer)
    return 1


def signal_bulle_reset_rsi55(row) -> int:
    """signal_bulle_reset() + RSI<55 Filter — kauft nur bei echter RSI-Schwäche."""
    if _g(row, "RSI", 50.0) >= 55.0:
        return 0
    return signal_bulle_reset(row)


def signal_bulle_reset_rsi50(row) -> int:
    """signal_bulle_reset() + RSI<50 Filter — stärkere Überverkauft-Bedingung."""
    if _g(row, "RSI", 50.0) >= 50.0:
        return 0
    return signal_bulle_reset(row)


def signal_bulle_rsi3(row, rsi14_min: float = 40.0, adx_min: float = 0.0) -> int:
    """
    RSI(3)-Mikro-Dip im Bulle-Trend.
    RSI(3) < 20 = kurzfristige Stunden-Schwäche in starkem Aufwärtstrend.
    rsi14_min kontrolliert die untere RSI14-Grenze (Default 40).
    adx_min: optionaler ADX-Mindestfilter (0=deaktiviert). ADX>40: EV +1.44%/Trade.
    Diagnose Apr 2026: Bucket 50-55 strukturell verlustreich (WR=23%, -38.3%), 55-60 profitabel.
    """
    rsi3        = _g(row, "RSI3",             50.0)
    rsi14       = _g(row, "RSI",              50.0)
    daily_macro = _g(row, "Daily_macro_bull", False)
    di_bull     = _g(row, "DI_bull",          False)
    ema21_slope = _g(row, "EMA21_slope",      True)
    close       = _g(row, "Close",            0.0)
    ma50        = _g(row, "MA50",             close)
    adx         = _g(row, "ADX",              20.0)

    if not daily_macro:                    return 0
    if not di_bull:                        return 0
    if not ema21_slope:                    return 0
    if close < ma50 * 0.95:               return 0
    if rsi3 > 20:                          return 0
    if rsi14 < rsi14_min or rsi14 > 75:   return 0
    if adx_min > 0 and adx < adx_min:     return 0
    return 1


def signal_bulle_ema21_bounce(row) -> int:
    """
    EMA21-Bounce ohne RSI-Fenster-Einschränkung.
    Kauft den Moment, wenn Preis EMA21 von unten kreuzt (Rücksetzer abgeschlossen).
    Kein RSI 65-70 Filter → viel höheres n als signal_bulle().
    MACD dreht auf + ADX > 20 als Qualitätsfilter gegen Fehlkreuzungen.
    """
    close          = _g(row, "Close",            0.0)
    ma50           = _g(row, "MA50",             close)
    cross_ema21_up = _g(row, "cross_ema21_up",   False)
    di_bull        = _g(row, "DI_bull",          False)
    ema21_slope    = _g(row, "EMA21_slope",      True)
    adx            = _g(row, "ADX",              20.0)
    daily_macro    = _g(row, "Daily_macro_bull", False)
    macd_hist_up   = _g(row, "MACD_hist_up",     False)
    rsi            = _g(row, "RSI",              50.0)

    if not daily_macro:       return 0
    if not di_bull:           return 0
    if not ema21_slope:       return 0
    if close < ma50 * 0.95:  return 0
    if adx < 20:              return 0
    if not cross_ema21_up:    return 0
    if not macd_hist_up:      return 0
    if rsi > 70:              return 0
    return 1


def signal_bulle_dip(row) -> int:
    """Methode 3: Dip-Buy im Bulle-Regime — RSI-Pullback 25-52 statt Momentum-Breakout."""
    close        = _g(row, "Close",        0.0)
    ema21        = _g(row, "EMA21",        close)
    ma50         = _g(row, "MA50",         close)
    ma500        = _g(row, "MA500",        0.0)
    rsi          = _g(row, "RSI",          50.0)
    di_bull      = _g(row, "DI_bull",      False)
    ema21_slope  = _g(row, "EMA21_slope",  True)
    macd_hist_up = _g(row, "MACD_hist_up", False)
    obv_bull     = _g(row, "OBV_bull",     False)
    cross_ema21_up = _g(row, "cross_ema21_up", False)

    if not di_bull:                    return 0
    if not ema21_slope:                return 0
    if close < ma50 * 0.92:           return 0
    if ma500 > 0 and close < ma500:   return 0
    if rsi > 52 or rsi < 25:          return 0   # Dip-Zone: RSI 25-52
    if close > ema21 * 1.04:          return 0   # nicht zu weit über EMA21
    if macd_hist_up and (obv_bull or cross_ema21_up):
        return 1
    return 0


def get_signal(row, regime: str, sw_mode: str = "v13",
               bulle_mode: str = "default") -> int:
    """Haupt-Dispatcher: gibt Signal-Code für die aktuelle Kerze zurück."""
    if regime == "extrem_baer":
        return signal_extrem_baer(row)
    elif regime == "baer":
        return signal_baer(row)
    elif regime == "seitwaerts":
        # neutral_*-Modi: SW weiterhin mit daily_bull-Logik
        effective_sw = "daily_bull" if sw_mode.startswith("neutral_") else sw_mode
        return signal_seitwaerts(row, sw_mode=effective_sw)
    elif regime == "bulle":
        if bulle_mode == "off":
            return 0
        elif bulle_mode == "strict":
            return signal_bulle_strict(row)
        elif bulle_mode == "dip":
            return signal_bulle_dip(row)
        elif bulle_mode == "reset":
            return signal_bulle_reset(row)
        elif bulle_mode == "reset_rsi55":
            return signal_bulle_reset_rsi55(row)
        elif bulle_mode == "reset_rsi50":
            return signal_bulle_reset_rsi50(row)
        elif bulle_mode == "pullback":
            return signal_bulle_pullback(row)
        elif bulle_mode in ("rsi3", "rsi3_b1"):
            return signal_bulle_rsi3(row, rsi14_min=40.0)
        elif bulle_mode == "rsi3_adx35":
            return signal_bulle_rsi3(row, rsi14_min=40.0, adx_min=35.0)
        elif bulle_mode == "rsi3_adx40":
            return signal_bulle_rsi3(row, rsi14_min=40.0, adx_min=40.0)
        elif bulle_mode == "rsi3_r45":
            return signal_bulle_rsi3(row, rsi14_min=45.0)
        elif bulle_mode == "rsi3_r50":
            return signal_bulle_rsi3(row, rsi14_min=50.0)
        elif bulle_mode == "rsi3_r55":
            return signal_bulle_rsi3(row, rsi14_min=55.0)
        elif bulle_mode == "rsi3_r57":
            return signal_bulle_rsi3(row, rsi14_min=57.0)
        elif bulle_mode == "rsi3_notch":
            # Notch: excludiert genau 50-55 (schlechtester Bucket), behält 40-50 + 55-75
            rsi14 = _g(row, "RSI", 50.0)
            if 50.0 <= rsi14 < 55.0:
                return 0
            return signal_bulle_rsi3(row, rsi14_min=40.0)
        elif bulle_mode == "ema21_bounce":
            return signal_bulle_ema21_bounce(row)
        return signal_bulle(row)
    elif regime == "extrem_bulle":
        return signal_extrem_bulle(row)
    else:  # neutral
        if bulle_mode == "rsi3_b1":
            return signal_bulle_rsi3(row, rsi14_min=40.0)
        if sw_mode == "neutral_sw_short":
            sig = signal_seitwaerts(row, sw_mode="v13")
            return sig if sig < 0 else 0
        elif sw_mode == "neutral_sw_full":
            return signal_seitwaerts(row, sw_mode="v13")
        elif sw_mode == "neutral_rsi_ext":
            rsi = _g(row, "RSI", 50.0)
            if rsi < 15:
                return 1
            return 0
        return 0
