"""
Diagnose: Warum hat der Bot 2023 kaum Bulle-Trades gemacht?

Zeigt:
  1. Regime-Verteilung pro Jahr
  2. Filter-Waterfall für Bulle-Regime-Kerzen (welcher Filter blockiert wie viele Signale?)
  3. Extrem-Bulle-Kerzen pro Jahr (signal_extrem_bulle ist aktuell return 0)
  4. RSI-Verteilung in Bulle-Kerzen (zeigt, ob RSI 65-70 Fenster zu eng ist)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sol_bot.backtest import compute_daily_macro, fetch_binance_data
from sol_bot.indicators import add_indicators
from sol_bot.regime import classify_regime
from sol_bot.strategy import _g


def _waterfall_bulle(df_bulle: pd.DataFrame) -> None:
    n = len(df_bulle)
    if n == 0:
        print("  (keine Bulle-Kerzen)")
        return

    def pct(k):
        return f"{k:>5}  ({k / n * 100:>5.1f}%)"

    # Jeder Filter kumulativ — so wie signal_bulle() ihn anwendet
    mask = pd.Series(True, index=df_bulle.index)

    mask_di    = df_bulle["DI_bull"].fillna(False)
    blocked_di = (~mask_di).sum()

    mask = mask & mask_di
    mask_slope    = df_bulle["EMA21_slope"].fillna(False)
    blocked_slope = (mask & ~mask_slope).sum()

    mask = mask & mask_slope
    close  = df_bulle["Close"]
    ma50   = df_bulle["MA50"].fillna(0)
    mask_ma50    = close >= ma50 * 0.97
    blocked_ma50 = (mask & ~mask_ma50).sum()

    mask = mask & mask_ma50
    ma500 = df_bulle["MA500"].fillna(0)
    mask_ma500    = ~((ma500 > 0) & (close < ma500))
    blocked_ma500 = (mask & ~mask_ma500).sum()

    mask = mask & mask_ma500
    rsi         = df_bulle["RSI"].fillna(50)
    mask_rsi    = (rsi >= 65) & (rsi <= 70)
    blocked_rsi = (mask & ~mask_rsi).sum()

    mask = mask & mask_rsi
    adx         = df_bulle["ADX"].fillna(20)
    mask_adx    = adx >= 35
    blocked_adx = (mask & ~mask_adx).sum()

    passed_all = (mask & mask_adx).sum()

    print(f"  Bulle-Kerzen gesamt          : {pct(n)}")
    print(f"  Blockiert durch DI_bull=False: {pct(blocked_di)}")
    print(f"  Blockiert durch EMA21_slope  : {pct(blocked_slope)}")
    print(f"  Blockiert durch Close<MA50   : {pct(blocked_ma50)}")
    print(f"  Blockiert durch Close<MA500  : {pct(blocked_ma500)}")
    print(f"  Blockiert durch RSI außer 65-70: {pct(blocked_rsi)}")
    print(f"  Blockiert durch ADX < 35     : {pct(blocked_adx)}")
    print(f"  Bestehen alle Filter         : {pct(passed_all)}")


def _rsi_buckets_bulle(df_bulle: pd.DataFrame) -> None:
    if len(df_bulle) == 0:
        return
    rsi = df_bulle["RSI"].dropna()
    buckets = [
        (0,  30,  "RSI  0-30  (überverkauft)"),
        (30, 45,  "RSI 30-45"),
        (45, 55,  "RSI 45-55  (neutral)"),
        (55, 65,  "RSI 55-65  (Hauptverlierer-Bucket)"),
        (65, 70,  "RSI 65-70  (aktuelles Fenster)"),
        (70, 80,  "RSI 70-80"),
        (80, 100, "RSI 80-100 (überkauft)"),
    ]
    total = len(rsi)
    for lo, hi, label in buckets:
        cnt = ((rsi >= lo) & (rsi < hi)).sum()
        print(f"    {label:<40}: {cnt:>5}  ({cnt / total * 100:>5.1f}%)")


def main():
    print("Lade Binance-Daten...")
    df_raw   = fetch_binance_data(start="2021-09-01")
    daily_s  = compute_daily_macro(df_raw)

    print("Berechne Indikatoren...")
    df = add_indicators(df_raw.copy())
    df["Daily_macro_bull"] = daily_s.reindex(df.index, method="ffill").fillna(False)

    print("Klassifiziere Regimes...")
    df["Regime"] = classify_regime(df)

    years = sorted(df.index.year.unique())

    # ─── 1. Regime-Verteilung pro Jahr ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. REGIME-VERTEILUNG PRO JAHR")
    print("=" * 70)

    regimes_ordered = ["extrem_baer", "baer", "seitwaerts", "neutral", "bulle", "extrem_bulle"]
    header = f"{'Jahr':<6}" + "".join(f"  {r[:8]:>9}" for r in regimes_ordered)
    print(header)
    print("-" * 70)

    for yr in years:
        dy = df[df.index.year == yr]
        total = len(dy)
        row = f"{yr:<6}"
        for reg in regimes_ordered:
            cnt = (dy["Regime"] == reg).sum()
            row += f"  {cnt / total * 100:>8.1f}%"
        print(row)

    # ─── 2. Filter-Waterfall Bulle-Kerzen pro Jahr ───────────────────────────
    print("\n" + "=" * 70)
    print("2. FILTER-WATERFALL: BULLE-REGIME-KERZEN PRO JAHR")
    print("=" * 70)

    for yr in years:
        df_bulle = df[(df.index.year == yr) & (df["Regime"] == "bulle")]
        print(f"\n  --- {yr} ---")
        _waterfall_bulle(df_bulle)

    # ─── 3. RSI-Verteilung in Bulle-Kerzen ───────────────────────────────────
    print("\n" + "=" * 70)
    print("3. RSI-VERTEILUNG IN BULLE-KERZEN (nach DI_bull + EMA21_slope Filter)")
    print("=" * 70)

    for yr in years:
        df_bulle = df[
            (df.index.year == yr)
            & (df["Regime"] == "bulle")
            & df["DI_bull"].fillna(False)
            & df["EMA21_slope"].fillna(False)
        ]
        n = len(df_bulle)
        if n == 0:
            print(f"\n  {yr}: keine Kerzen nach DI_bull+EMA21_slope Filter")
            continue
        print(f"\n  {yr}  (n={n} nach DI_bull+EMA21_slope):")
        _rsi_buckets_bulle(df_bulle)

    # ─── 4. Extrem-Bulle pro Jahr ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. EXTREM-BULLE-KERZEN PRO JAHR  (signal_extrem_bulle = return 0 ← deaktiviert)")
    print("=" * 70)

    for yr in years:
        dy     = df[df.index.year == yr]
        total  = len(dy)
        eb_cnt = (dy["Regime"] == "extrem_bulle").sum()
        b_cnt  = (dy["Regime"] == "bulle").sum()
        print(f"  {yr}:  extrem_bulle={eb_cnt:>5} ({eb_cnt/total*100:.1f}%)   "
              f"bulle={b_cnt:>5} ({b_cnt/total*100:.1f}%)")

    # ─── 5. 2023 Detail: Monatliche Regime-Verteilung ────────────────────────
    print("\n" + "=" * 70)
    print("5. 2023 MONATLICH — wo war der Anstieg (+918% BaH)?")
    print("=" * 70)

    df_2023 = df[df.index.year == 2023]
    print(f"  {'Monat':<8}" + "".join(f"  {r[:8]:>9}" for r in regimes_ordered))
    print("  " + "-" * 68)
    for m in range(1, 13):
        dm = df_2023[df_2023.index.month == m]
        if len(dm) == 0:
            continue
        total = len(dm)
        row = f"  {m:>02}-2023"
        for reg in regimes_ordered:
            cnt = (dm["Regime"] == reg).sum()
            row += f"  {cnt / total * 100:>8.1f}%"
        print(row)

    # ─── 6. Bulle-Kerzen 2023: ohne RSI- und ADX-Filter ─────────────────────
    print("\n" + "=" * 70)
    print("6. BULLE-KERZEN 2023 OHNE RSI/ADX-FILTER  (potenzielle Trades)")
    print("=" * 70)

    df_bulle_2023 = df[
        (df.index.year == 2023)
        & (df["Regime"] == "bulle")
        & df["DI_bull"].fillna(False)
        & df["EMA21_slope"].fillna(False)
        & (df["Close"] >= df["MA50"].fillna(0) * 0.97)
    ]
    n = len(df_bulle_2023)
    print(f"  Kerzen nach Pflicht-Filter (ohne RSI+ADX): {n}")
    if n > 0:
        rsi = df_bulle_2023["RSI"].dropna()
        adx = df_bulle_2023["ADX"].dropna()
        print(f"  RSI  — Median: {rsi.median():.1f}  Min: {rsi.min():.1f}  Max: {rsi.max():.1f}")
        print(f"  ADX  — Median: {adx.median():.1f}  Min: {adx.min():.1f}  Max: {adx.max():.1f}")

        # Wie viele Kerzen hätten bei erweitertem RSI-Fenster gefeuert?
        for rsi_lo, rsi_hi in [(45, 70), (50, 75), (55, 72), (60, 72), (65, 70)]:
            cnt = ((df_bulle_2023["RSI"] >= rsi_lo) & (df_bulle_2023["RSI"] <= rsi_hi)).sum()
            adx_ok = ((df_bulle_2023["RSI"] >= rsi_lo) & (df_bulle_2023["RSI"] <= rsi_hi)
                      & (df_bulle_2023["ADX"] >= 35)).sum()
            print(f"  RSI {rsi_lo}-{rsi_hi}: {cnt:>4} Kerzen  (davon ADX≥35: {adx_ok})")


if __name__ == "__main__":
    main()
