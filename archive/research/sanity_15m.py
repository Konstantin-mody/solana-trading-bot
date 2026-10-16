"""
sanity_15m.py — Cost-Sanity-Check für einen möglichen 15min-Bot

Prüft 4 triviale Baseline-Strategien auf Binance 15min SOL/USDT-Daten.
Entscheidungsschwelle: Brutto-Edge >= 0.5%/Trade UND Netto-Edge >= 0.25%/Trade
Falls keine Strategie die Schwelle erreicht → 15min-Bot nicht wirtschaftlich.

Aufruf: python3 sanity_15m.py
"""

import time
import requests
import numpy as np
import pandas as pd

# ─── Kostenmodell (identisch zu sol_bot/backtest.py) ─────────────────────────
GEBUEHR  = 0.00075
SLIPPAGE = 0.0003
SPREAD   = 0.0002
ROUNDTRIP = 2 * (GEBUEHR + SLIPPAGE) + SPREAD   # ~0.23%

# Entscheidungsschwellen
MIN_GROSS_EDGE = 0.0050   # 0.5% Brutto pro Trade
MIN_NET_EDGE   = 0.0025   # 0.25% Netto pro Trade (= breakeven nach Kosten)
MIN_TRADES     = 500      # Mindest-n für Aussagekraft


# ─── Daten laden ─────────────────────────────────────────────────────────────

def fetch_15m(start: str = "2021-09-01") -> pd.DataFrame:
    url      = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms   = int(pd.Timestamp.now().timestamp() * 1000)
    rows     = []

    ts = start_ms
    while ts < end_ms:
        r = requests.get(url, params={
            "symbol": "SOLUSDT", "interval": "15m",
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
    return df


# ─── Indikatoren ─────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]

    # RSI(5)
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=4, adjust=False).mean()
    avg_l = loss.ewm(com=4, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["RSI5"] = 100 - 100 / (1 + rs)

    # EMA5 / EMA13
    df["EMA5"]  = c.ewm(span=5,  adjust=False).mean()
    df["EMA13"] = c.ewm(span=13, adjust=False).mean()

    # VWAP (rollend über 96 Kerzen = 1 Tag)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol     = df["Volume"]
    df["VWAP96"] = (typical * vol).rolling(96).sum() / vol.rolling(96).sum()

    # Bollinger Bands (20, 2)
    df["BB_mid"] = c.rolling(20).mean()
    std20        = c.rolling(20).std()
    df["BB_lo"]  = df["BB_mid"] - 2 * std20
    df["BB_hi"]  = df["BB_mid"] + 2 * std20

    # ATR(14) für Normalisierung
    prev_c = c.shift(1)
    tr     = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_c).abs(),
        (df["Low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(com=13, adjust=False).mean()

    return df


# ─── Generischer Simulator ────────────────────────────────────────────────────

def simulate(df: pd.DataFrame, entries: pd.Series, exits: pd.Series,
             side: str = "long") -> dict:
    """
    entries/exits: Boolean-Series (True = Signal auf dieser Kerze)
    Einstieg zum nächsten Open, Ausstieg zum nächsten Open nach Exit-Signal.
    Kein Halten über offene Position (Entry ignoriert während Position offen).
    """
    trades = []
    in_pos = False
    entry_price = 0.0

    for i in range(1, len(df) - 1):
        if not in_pos and entries.iloc[i]:
            entry_price = df["Close"].iloc[i]   # Näherung: Close der Signalkerze
            in_pos = True
            continue
        if in_pos and exits.iloc[i]:
            exit_price = df["Close"].iloc[i]
            if side == "long":
                gross = (exit_price - entry_price) / entry_price
            else:
                gross = (entry_price - exit_price) / entry_price
            net = gross - ROUNDTRIP
            trades.append({"gross": gross, "net": net})
            in_pos = False

    if not trades:
        return {"n": 0, "wr": 0, "avg_gross": 0, "avg_net": 0, "sum_net": 0}

    t = pd.DataFrame(trades)
    return {
        "n":         len(t),
        "wr":        (t["net"] > 0).mean() * 100,
        "avg_gross": t["gross"].mean() * 100,
        "avg_net":   t["net"].mean() * 100,
        "sum_net":   t["net"].sum() * 100,
    }


# ─── Strategie-Definitionen ───────────────────────────────────────────────────

def run_strategies(df: pd.DataFrame) -> list[dict]:
    results = []

    # S1: RSI(5) Oversold Mean-Reversion (Long)
    entry = (df["RSI5"] < 15) & (df["RSI5"].shift(1) >= 15)
    exit_ = (df["RSI5"] > 60)
    r = simulate(df, entry, exit_, "long")
    results.append({"name": "S1: RSI5<15 Long (exit RSI>60)", **r})

    # S2: RSI(5) Overbought Mean-Reversion (Short)
    entry = (df["RSI5"] > 85) & (df["RSI5"].shift(1) <= 85)
    exit_ = (df["RSI5"] < 40)
    r = simulate(df, entry, exit_, "short")
    results.append({"name": "S2: RSI5>85 Short (exit RSI<40)", **r})

    # S3: VWAP Dip-Buy (Long, nur wenn Preis > 1.5% unter VWAP)
    deviation = (df["VWAP96"] - df["Close"]) / df["VWAP96"]
    entry = (deviation > 0.015) & (deviation.shift(1) <= 0.015)
    exit_ = (df["Close"] >= df["VWAP96"])
    r = simulate(df, entry, exit_, "long")
    results.append({"name": "S3: VWAP-Dip >1.5% Long (exit @VWAP)", **r})

    # S4: Bollinger-Band-Bounce (Long)
    entry = (df["Close"] < df["BB_lo"]) & (df["Close"].shift(1) >= df["BB_lo"])
    exit_ = (df["Close"] > df["BB_mid"])
    r = simulate(df, entry, exit_, "long")
    results.append({"name": "S4: BB-Unterkante Long (exit @BB-mid)", **r})

    # S5: EMA5 > EMA13 Crossover (Long Momentum)
    cross_up   = (df["EMA5"] > df["EMA13"]) & (df["EMA5"].shift(1) <= df["EMA13"].shift(1))
    cross_down = (df["EMA5"] < df["EMA13"]) & (df["EMA5"].shift(1) >= df["EMA13"].shift(1))
    r = simulate(df, cross_up, cross_down, "long")
    results.append({"name": "S5: EMA5/EMA13 Crossover Long", **r})

    # S6: RSI(5) Oversold + Preis > VWAP (Trendbedingung)
    entry = (df["RSI5"] < 20) & (df["Close"] > df["VWAP96"])
    exit_ = (df["RSI5"] > 55)
    r = simulate(df, entry, exit_, "long")
    results.append({"name": "S6: RSI5<20 + über VWAP Long", **r})

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Lade Binance 15min SOL/USDT-Daten ab 2021-09-01 …")
    df = fetch_15m("2021-09-01")
    df = add_indicators(df)
    df = df.dropna()

    candles   = len(df)
    days      = candles / 96
    daterange = f"{df.index[0].date()} → {df.index[-1].date()}"
    print(f"  {candles:,} Kerzen ({days:.0f} Tage) | {daterange}\n")

    results = run_strategies(df)

    PASS = "✅"
    FAIL = "❌"
    print(f"{'Strategie':<42} {'n':>5} {'WR%':>6} {'Brutto%':>8} {'Netto%':>7} {'ΣNetto%':>8}  Status")
    print("─" * 100)

    any_pass = False
    for r in results:
        if r["n"] == 0:
            print(f"  {r['name']:<40}  {'–':>5}  –  keine Trades")
            continue
        gross_ok = r["avg_gross"] >= MIN_GROSS_EDGE * 100
        net_ok   = r["avg_net"]   >= MIN_NET_EDGE   * 100
        n_ok     = r["n"]         >= MIN_TRADES
        ok       = gross_ok and net_ok and n_ok
        status   = PASS if ok else FAIL
        if ok:
            any_pass = True
        print(
            f"  {r['name']:<40}  {r['n']:>5}  {r['wr']:>5.1f}%"
            f"  {r['avg_gross']:>+7.3f}%  {r['avg_net']:>+6.3f}%"
            f"  {r['sum_net']:>+7.1f}%  {status}"
        )

    print()
    print(f"Roundtrip-Kosten pro Trade: {ROUNDTRIP*100:.3f}%")
    print(f"Schwellen: Brutto ≥ {MIN_GROSS_EDGE*100:.2f}%/Trade, Netto ≥ {MIN_NET_EDGE*100:.2f}%/Trade, n ≥ {MIN_TRADES}")
    print()
    if any_pass:
        print("ERGEBNIS: ✅  Mindestens eine Baseline erfüllt die Schwelle.")
        print("  → 15min-Bot ist konzeptionell machbar. Tagesrate prüfen:")
        for r in results:
            if r["n"] >= MIN_TRADES:
                rate = r["n"] / days
                print(f"    {r['name']}: {rate:.1f} Trades/Tag")
    else:
        print("ERGEBNIS: ❌  Keine Baseline erreicht Brutto ≥ 0.5%/Trade + Netto ≥ 0.25%/Trade.")
        print("  → 15min standalone Bot ist unter aktuellem Kostenmodell nicht wirtschaftlich.")
        print("  → Empfehlung: 15min nur als Timing-Layer im Hierarchischen Combinator.")


if __name__ == "__main__":
    main()
