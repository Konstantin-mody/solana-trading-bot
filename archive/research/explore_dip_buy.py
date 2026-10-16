"""
explore_dip_buy.py — Explorations-Skript für Bulle-Dip-Buy-Strategie
Testet verschiedene Filter-Kombinationen für DD+RSI-basierte Entries
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sol_bot.indicators import add_indicators
from sol_bot.regime import classify_regime
from sol_bot.backtest import fetch_binance_data, compute_daily_macro

# ─── Daten laden ──────────────────────────────────────────────────────────────
def load_data():
    print("Lade Binance-Daten (API)...")
    df = fetch_binance_data(start="2021-09-01")
    print(f"  → {len(df)} Kerzen geladen")
    return df


# ─── Hauptanalyse ──────────────────────────────────────────────────────────────
def main():
    print("=== Bulle-Dip-Buy Explorations-Analyse ===\n")

    df_raw = load_data()
    print(f"Daten geladen: {len(df_raw)} Kerzen ({df_raw.index[0].date()} → {df_raw.index[-1].date()})")

    df = add_indicators(df_raw)
    regime = classify_regime(df)
    df["Regime"] = regime
    df["Daily_macro_bull"] = compute_daily_macro(df_raw)

    # 30-Tage Rolling Max + Drawdown
    df["RolMax_30d"] = df["Close"].rolling(720, min_periods=1).max()
    df["DD_30d"] = (df["Close"] / df["RolMax_30d"] - 1) * 100

    print(f"\n--- Basis-Statistiken ---")
    print(f"Bulle-Regime: {(df['Regime']=='bulle').sum()} Kerzen")
    print(f"  + Daily MacroBull: {((df['Regime']=='bulle') & df['Daily_macro_bull']).sum()} Kerzen")

    bulle_macro = (df["Regime"] == "bulle") & df["Daily_macro_bull"]
    base_n = bulle_macro.sum()

    # RSI-Verteilung im Bulle+MacroBull-Regime nach verschiedenen DD-Schwellen
    print(f"\n--- RSI-Verteilung bei DD-Schwellen ---")
    for dd_thresh in [-5, -10, -15, -20]:
        mask = bulle_macro & (df["DD_30d"] <= dd_thresh)
        n = mask.sum()
        if n > 0:
            rsi_vals = df.loc[mask, "RSI"]
            print(f"DD≤{dd_thresh}%: n={n} | RSI Median={rsi_vals.median():.1f} "
                  f"| p25={rsi_vals.quantile(0.25):.1f} | p10={rsi_vals.quantile(0.1):.1f} "
                  f"| <40: {(rsi_vals<40).sum()} | <45: {(rsi_vals<45).sum()} "
                  f"| <50: {(rsi_vals<50).sum()} | <55: {(rsi_vals<55).sum()}")

    # ── Signal-Funnel nach verschiedenen Kombinationen ───────────────────────
    print(f"\n--- Signal-Funnel (verschiedene Filter-Kombinationen) ---")
    print(f"{'Filter':<55} {'Kerzen':>7} {'Signals':>8} {'Ø/Jahr':>7}")
    print("-" * 80)

    years_total = (df.index[-1] - df.index[0]).days / 365.25

    configs = [
        ("Bulle + MacroBull",                       bulle_macro, None, None),
        ("Bulle + MacroBull + DD≤-5%",              bulle_macro, -5,  None),
        ("Bulle + MacroBull + DD≤-10%",             bulle_macro, -10, None),
        ("Bulle + MacroBull + DD≤-10% + RSI<55",   bulle_macro, -10, 55),
        ("Bulle + MacroBull + DD≤-10% + RSI<50",   bulle_macro, -10, 50),
        ("Bulle + MacroBull + DD≤-10% + RSI<45",   bulle_macro, -10, 45),
        ("Bulle + MacroBull + DD≤-10% + RSI<40",   bulle_macro, -10, 40),
        ("Bulle + MacroBull + DD≤-15%",             bulle_macro, -15, None),
        ("Bulle + MacroBull + DD≤-15% + RSI<55",   bulle_macro, -15, 55),
        ("Bulle + MacroBull + DD≤-15% + RSI<50",   bulle_macro, -15, 50),
        ("Bulle + MacroBull + DD≤-15% + RSI<45",   bulle_macro, -15, 45),
        ("Bulle + MacroBull + DD≤-20%",             bulle_macro, -20, None),
        ("Bulle + MacroBull + DD≤-20% + RSI<50",   bulle_macro, -20, 50),
    ]

    COOLDOWN_HOURS = 48

    for label, base_mask, dd_thresh, rsi_thresh in configs:
        mask = base_mask.copy()
        if dd_thresh is not None:
            mask = mask & (df["DD_30d"] <= dd_thresh)
        if rsi_thresh is not None:
            mask = mask & (df["RSI"] < rsi_thresh)

        n_candles = mask.sum()

        # Signals mit Cooldown
        signal_idx = df.index[mask]
        if len(signal_idx) == 0:
            print(f"{label:<55} {n_candles:>7} {0:>8} {0:>7.1f}")
            continue

        # Cooldown-Filterung
        signals = []
        last_sig = None
        for ts in signal_idx:
            if last_sig is None or (ts - last_sig).total_seconds() / 3600 >= COOLDOWN_HOURS:
                signals.append(ts)
                last_sig = ts

        n_signals = len(signals)
        avg_per_year = n_signals / years_total
        print(f"{label:<55} {n_candles:>7} {n_signals:>8} {avg_per_year:>7.1f}")

    # ── Jahresverteilung für vielversprechende Konfigurationen ────────────────
    print(f"\n--- Jahresverteilung (DD≤-10%, verschiedene RSI-Schwellen, 48h-Cooldown) ---")

    dd10_mask = bulle_macro & (df["DD_30d"] <= -10)

    for rsi_t in [None, 55, 50, 45]:
        rsi_label = f"RSI<{rsi_t}" if rsi_t is not None else "kein RSI-Filter"
        mask = dd10_mask.copy()
        if rsi_t is not None:
            mask = mask & (df["RSI"] < rsi_t)

        signal_idx = df.index[mask]
        signals = []
        last_sig = None
        for ts in signal_idx:
            if last_sig is None or (ts - last_sig).total_seconds() / 3600 >= COOLDOWN_HOURS:
                signals.append(ts)
                last_sig = ts

        if not signals:
            print(f"\n{rsi_label}: 0 Signals")
            continue

        sig_df = pd.DataFrame({"ts": signals})
        sig_df["year"] = sig_df["ts"].dt.year
        per_year = sig_df.groupby("year").size()

        print(f"\n{rsi_label}: n={len(signals)}")
        for yr, cnt in per_year.items():
            print(f"  {yr}: {cnt} Signals")

    # ── RSI+OBV Kombination ────────────────────────────────────────────────────
    print(f"\n--- Kombinationsfilter (DD≤-10% + RSI<50 + OBV oder RSI_div) ---")
    dd10_rsi50 = dd10_mask & (df["RSI"] < 50)
    print(f"DD≤-10% + RSI<50: {dd10_rsi50.sum()} Kerzen")
    with_obv = dd10_rsi50 & df["OBV_bull"]
    with_rsi_div = dd10_rsi50 & df["RSI_div_bull"]
    with_either = dd10_rsi50 & (df["OBV_bull"] | df["RSI_div_bull"])
    print(f"  + OBV_bull: {with_obv.sum()} Kerzen")
    print(f"  + RSI_div_bull: {with_rsi_div.sum()} Kerzen")
    print(f"  + (OBV OR RSI_div): {with_either.sum()} Kerzen")

    # Signals nach Cooldown für (OBV OR RSI_div)
    signal_idx = df.index[with_either]
    signals = []
    last_sig = None
    for ts in signal_idx:
        if last_sig is None or (ts - last_sig).total_seconds() / 3600 >= COOLDOWN_HOURS:
            signals.append(ts)
            last_sig = ts
    print(f"  + (OBV OR RSI_div) nach 48h-Cooldown: {len(signals)} Signals ({len(signals)/years_total:.1f}/Jahr)")

    # ── "Dip + Recovery"-Kombinationen ────────────────────────────────────────
    print(f"\n--- 'Dip + Recovery' Filter-Kombinationen (Bounce-Bestätigung nach Rücksetzer) ---")
    print(f"{'Filter':<65} {'Kerzen':>7} {'Signals':>8} {'Ø/Jahr':>7}")
    print("-" * 90)

    dd5_mask  = bulle_macro & (df["DD_30d"] <= -5)
    dd8_mask  = bulle_macro & (df["DD_30d"] <= -8)
    dd10_mask_raw = bulle_macro & (df["DD_30d"] <= -10)

    recovery_configs = [
        # Nur DD, kein Recovery (Referenz)
        ("DD≤-5% (Referenz)",              dd5_mask,  False, False, False, False),
        ("DD≤-8% (Referenz)",              dd8_mask,  False, False, False, False),
        ("DD≤-10% (Referenz)",             dd10_mask_raw, False, False, False, False),
        # DD + cross_ema21_up (Preis-Bounce über EMA21)
        ("DD≤-5% + cross_ema21_up",        dd5_mask,  True,  False, False, False),
        ("DD≤-8% + cross_ema21_up",        dd8_mask,  True,  False, False, False),
        ("DD≤-10% + cross_ema21_up",       dd10_mask_raw, True,  False, False, False),
        # DD + MACD dreht auf
        ("DD≤-5% + MACD_hist_up",          dd5_mask,  False, True,  False, False),
        ("DD≤-8% + MACD_hist_up",          dd8_mask,  False, True,  False, False),
        ("DD≤-10% + MACD_hist_up",         dd10_mask_raw, False, True,  False, False),
        # DD + cross_ema21_up + MACD_hist_up (echte Erholung)
        ("DD≤-5% + cross + MACD",          dd5_mask,  True,  True,  False, False),
        ("DD≤-8% + cross + MACD",          dd8_mask,  True,  True,  False, False),
        ("DD≤-10% + cross + MACD",         dd10_mask_raw, True,  True,  False, False),
        # DD + cross + MACD + OBV
        ("DD≤-5% + cross + MACD + OBV",    dd5_mask,  True,  True,  True,  False),
        ("DD≤-8% + cross + MACD + OBV",    dd8_mask,  True,  True,  True,  False),
        ("DD≤-10% + cross + MACD + OBV",   dd10_mask_raw, True, True, True, False),
        # wie signal_bulle() Trigger-A aber ohne RSI-Filter
        ("DD≤-5% + cross + MACD + (OBV|SAR)", dd5_mask, True, True, True, True),
        ("DD≤-8% + cross + MACD + (OBV|SAR)", dd8_mask, True, True, True, True),
    ]

    for (label, base_mask, use_cross, use_macd, use_obv, use_obv_or_sar) in recovery_configs:
        mask = base_mask.copy()
        if use_cross:
            mask = mask & df["cross_ema21_up"]
        if use_macd:
            mask = mask & df["MACD_hist_up"]
        if use_obv and not use_obv_or_sar:
            mask = mask & df["OBV_bull"]
        if use_obv_or_sar:
            mask = mask & (df["OBV_bull"] | df["SAR_bull"])

        n_candles = mask.sum()
        signal_idx = df.index[mask]
        signals = []
        last_sig = None
        for ts in signal_idx:
            if last_sig is None or (ts - last_sig).total_seconds() / 3600 >= COOLDOWN_HOURS:
                signals.append(ts)
                last_sig = ts
        n_signals = len(signals)
        avg_per_year = n_signals / years_total
        print(f"{label:<65} {n_candles:>7} {n_signals:>8} {avg_per_year:>7.1f}")

    # ── Jahresverteilung für vielversprechende Recovery-Kombinationen ──────────
    print(f"\n--- Jahresverteilung Recovery-Varianten (48h-Cooldown) ---")
    reco_variants = [
        ("DD≤-5% + cross + MACD",       dd5_mask  & df["cross_ema21_up"] & df["MACD_hist_up"]),
        ("DD≤-8% + cross + MACD",       dd8_mask  & df["cross_ema21_up"] & df["MACD_hist_up"]),
        ("DD≤-10% + cross + MACD",      dd10_mask_raw & df["cross_ema21_up"] & df["MACD_hist_up"]),
        ("DD≤-5% + cross + MACD + OBV", dd5_mask  & df["cross_ema21_up"] & df["MACD_hist_up"] & df["OBV_bull"]),
        ("DD≤-8% + cross + MACD + OBV", dd8_mask  & df["cross_ema21_up"] & df["MACD_hist_up"] & df["OBV_bull"]),
    ]
    for label, mask in reco_variants:
        signal_idx = df.index[mask]
        signals = []
        last_sig = None
        for ts in signal_idx:
            if last_sig is None or (ts - last_sig).total_seconds() / 3600 >= COOLDOWN_HOURS:
                signals.append(ts)
                last_sig = ts
        if not signals:
            print(f"\n{label}: 0 Signals")
            continue
        sig_df = pd.DataFrame({"ts": signals})
        sig_df["year"] = sig_df["ts"].dt.year
        per_year = sig_df.groupby("year").size()
        print(f"\n{label}: n={len(signals)} ({len(signals)/years_total:.1f}/Jahr)")
        for yr, cnt in per_year.items():
            print(f"  {yr}: {cnt}")

    print("\n=== Fertig ===")


if __name__ == "__main__":
    main()
