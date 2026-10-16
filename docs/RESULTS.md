# SOL Bot — Backtest-Ergebnisse & Versionshistorie

> **Primäre Datenquelle (aktuell):** Binance 1h SOL/USD ab 2021-09-01 (`--data binance`)
> Alle Entscheidungen seit v14a auf Binance-Daten. Synthetische Daten nur für historische Referenz.

---

## Aktuelle Ergebnisse — Binance 1h, Vollzyklus 2021–2026 (Stand: Apr 2026, RSI>50 Cover)

> `python3 run_backtest.py --data binance`  (Default: `--mode daily_bull`)
> Startkapital $200, Binance 2021-09-01 → Apr 2026 (~40.600 Kerzen)
> Aktive Parameter: Bär 6×, Bulle 1×, **EB 2× + ADX<40-Filter**, **Bär-Cover RSI>50**

| Kennzahl | Wert |
|---|---|
| Endkapital | **$707.82** (Start: $200) |
| Rendite | **+253.9%** |
| Sharpe Ratio | **1.35** |
| Max. Drawdown | **-27.3%** (2.7pp Puffer) |
| Long WR | 31.3% (n=115) |
| Short WR | 65.3% (n=101) |
| Ø Hebel | 3.38× |

### Jahresperformance

| Jahr | Bot | BaH | Outperf. |
|---|---|---|---|
| 2021 | +9.9% | +59.0% | -49.1% |
| 2022 | **+21.2%** | -94.2% | **+115.4%** |
| 2023 | **+50.4%** | +918.2% | -867.8% |
| 2024 | **+42.6%** | +85.7% | -43.1% |
| 2025 | **+33.6%** | -35.3% | **+68.9%** |
| 2026 | -7.2% | -29.6% | +22.4% |

### Regime-Beiträge

| Regime | Side | N | WR | ΣPnL | Ø Hebel |
|---|---|---|---|---|---|
| baer | short | 87 | 67.8% | +357.5% | 5.83× |
| seitwaerts | long | 37 | 35.1% | +1.2% | 2.00× |
| seitwaerts | short | 14 | 50.0% | +5.0% | 1.07× |
| bulle | long | 20 | 40.0% | +0.1% | 1.00× |
| extrem_bulle | long | 58 | 25.9% | -35.5% | 1.97× |

### Regime-Beiträge nach Jahr

#### Bär-Short (nach RSI>50 Cover-Trigger)
| Jahr | N | WR | ΣPnL |
|---|---|---|---|
| 2021 | 4 | 75.0% | +42.7% |
| 2022 | 35 | 65.7% | +187.5% |
| 2023 | 12 | 58.3% | -40.3% |
| 2024 | 13 | 69.2% | +109.9% |
| 2025 | 15 | 80.0% | +116.3% |
| 2026 | 8 | 62.5% | -58.6% |

#### Extrem-Bulle (Long)
| Jahr | N | WR | ΣPnL | $-Beitrag |
|---|---|---|---|---|
| 2021 | 7 | 14.3% | -23.4% | -$17 |
| 2022 | 12 | 16.7% | -29.8% | -$3 |
| 2023 | 17 | 35.3% | +20.1% | +$93 |
| 2024 | 7 | 28.6% | -6.8% | -$8 |
| 2025 | 13 | 30.8% | +8.0% | +$91 |
| 2026 | 2 | 0.0% | -3.6% | -$4 |

#### Seitwärts-Long
| Jahr | N | WR | ΣPnL | $-Beitrag |
|---|---|---|---|---|
| 2021 | 2 | 50.0% | +16.0% | +$28 |
| 2022 | 0 | — | — | — |
| 2023 | 12 | 33.3% | +15.0% | +$42 |
| 2024 | 15 | 40.0% | +6.6% | +$112 |
| 2025 | 8 | 25.0% | -36.4% | +$10 |
| 2026 | 0 | — | — | — |

#### Bulle-Long
| Jahr | N | WR | ΣPnL | $-Beitrag |
|---|---|---|---|---|
| 2022 | 2 | 50.0% | +2.4% | +$5 |
| 2023 | 2 | 50.0% | +0.8% | +$1 |
| 2024 | 5 | 40.0% | +0.5% | +$5 |
| 2025 | 7 | 42.9% | -1.4% | +$9 |
| 2026 | 4 | 25.0% | -2.3% | -$9 |

### Live-Code-Gates (Stand: Apr 2026, post RSI>50)

1. ✅ Seitwärts profitabel (WR 39.2%, EV +0.12%, ΣPnL positiv)
2. ✅ Extrem-Bär kein negativer ΣPnL-Beitrag (0 Trades auf Binance)
3. ✅ Bulle-Long ΣPnL-Beitrag positiv (WR 40.0%, n=20, ΣPnL +0.1% — ⚠️ sehr fragil)
4. ✅ Max Drawdown < 30% (aktuell **-27.3%** — 2.7pp Puffer)
5. ✅ Binance-Daten-Backtest validiert

### Offene Strukturprobleme

- **EB Trail zu eng (1.2×ATR):** 37/58 Exits via Trail, WR 10.8%, ΣPnL -99.8%. TP-Exits WR 100%, ΣPnL +108% — Entries funktionieren, Trail schneidet Gewinne ab. Fix: Trail-Widening 1.2→2.0×ATR in `backtest.py` (High-Stakes).
- **Bär-Short in Recovery:** 2023 (ΣPnL -40.3%) und 2026 (ΣPnL -58.6%) — RSI>50 verbessert beide Jahre, aber strukturell immer noch verlustreich in Aufwärtsphasen nach Regime-Wechsel.
- **Bär-Short Outlier 2026-02-06:** -51.7% Einzeltrade (ATR=4.3%, Hebel 6×). Fast-Bounce — RSI>50 greift nicht rechtzeitig. Accepted recovery cost.

---

## Ergebnisse vor RSI>50 Cover (Apr 2026, EB=2× + ADX<40, kein RSI-Cover)

> Historische Referenz. Alles identisch, nur `signal_baer_cover()` ohne RSI-Bedingung.

| Kennzahl | Wert |
|---|---|
| Rendite | +222.5% |
| Sharpe | 1.25 |
| Max. Drawdown | -29.8% (⚠️ 0.2pp Puffer) |
| Bär WR | 66.7%, ΣPnL +266.4% |

## Ergebnisse vor EB-Aktivierung (Mai 2026, nach Bug-Fixes, EB=1×)

> Historische Referenz. EB deaktiviert (Hebel 1×), alle anderen Parameter gleich.

| Kennzahl | Wert |
|---|---|
| Rendite | ~+126% |
| Sharpe | ~0.92 |
| Max. Drawdown | -28.4% |
| 2023 Bot | ~+7% |

---

## Bug-Fix Isolationstest (Mai 2026)

---

## Bug-Fix Isolationstest (Mai 2026)

> Zeigt den Beitrag jedes Bugs zur Performanz-Differenz.

| Variante | Endkapital | Rendite | Sharpe | MaxDD |
|---|---|---|---|---|
| A: Beide Bugs aktiv (≈ alter Code) | $6.617 | +3.208% | 2.43 | -23.5% |
| B: Nur Fee-Fix (Pyramiding noch alt) | $5.189 | +2.494% | 2.28 | -24.2% |
| C: Nur Pyramiding-Fix (Fees noch alt) | $556 | +178% | 0.88 | -33.6% |
| D: Beide Fixes — aktueller Stand | **$435** | **+117%** | **0.71** | **-34.6%** |

**Behobene Bugs:**
1. **Gebühren auf Notional** (statt auf entry_cash): Binance berechnet 0.075% auf Positionsgröße × Hebel, nicht auf Kapital. Bei 8× Hebel war die Fee ~8× zu niedrig. Beitrag: −$1.428.
2. **Pyramiding Weighted-Avg Entry Price**: entry_price wurde nie aktualisiert wenn nachgekauft wurde. PnL wurde gegen den ursprünglichen (niedrigen) Einstiegspreis berechnet statt gegen den gewichteten Durchschnitt → 2–3× Aufblähung. Beitrag: −$6.061.
3. **Bär-Cover Regime-Check**: pos.regime=="baer" Prüfung fehlte (Minor).

---

## Abgeschlossene Optimierungen

| Bereich | Entscheidung | Begründung |
|---|---|---|
| SW-Short | $+29 als Bonus akzeptiert, keine weitere Optimierung | n=14 zu dünn, Makro-Split zeigt kein Muster |
| Bulle-Long RSI 70-80 | Abgelehnt | WR 0%, n=2 |
| Bär-Long | Deaktiviert | WR zu gering |
| Bulle-Short | Deaktiviert | WR 34%, negativer EV |
| Multi-Position | Nicht implementiert | 36 blockierte Signale in 5 Jahren, 34/36 Long→Long, nicht lohnenswert |

---

## Historische Ergebnisse (vor Bug-Fixes — veraltet, nur Referenz)

> **ACHTUNG: Alle Zahlen unten sind mit fehlerhafter Backtest-Engine entstanden.**
> Gebühren und Pyramiding waren falsch berechnet. Nicht als Entscheidungsgrundlage verwenden.

| Kennzahl | Wert (buggy) |
|---|---|
| Endkapital | ~~$4.750–$6.704~~ |
| Rendite | ~~+2275% – +3252%~~ |
| Sharpe Ratio | ~~2.42–2.45~~ |
| Max. Drawdown | ~~-23.3% – -23.4%~~ |

### Versions-Einzel-Verbesserungen (Binance-Daten, chronologisch, mit altem Bug)

| Änderung | Auswirkung (damals gemessen, buggy) |
|---|---|
| DI/EMA-Regime statt MA200 | +20% Gesamt, 2022: +81%, 2024: +81% besser |
| Bär-Short (Dead-Cat-Bounce: A+B+C + VWAP + RSI<35) | 87 Shorts, WR 69%, +114% ΣPnL |
| RSI-Filter Bulle 38-70 → 65-70 | Hauptverlierer RSI 55-65 entfernt, +80% Outperf. |
| 10% PosSize statt 100% | Kelly-Effekt: $872/Sharpe 1.47 vs $435/Sharpe 0.58 |
| Bulle-Long Trail (3×ATR) | SL-Rate 55%→31%, Sharpe 1.47→1.56 |
| Daily Macro Bull (EMA21>EMA50 1d) als SW-Long-Filter | 2022 komplett geblockt |
| SW-Long SL=3.5×ATR + TP=10×ATR + Hebel=2× (2D-Grid) | Sharpe 1.56→2.35 |
| Bär-Short Hebel 8× (DD-geschützt) | Sharpe 2.36→2.42 |
| Bulle-Long Hebel 5× (DD-geschützt) | Sharpe 2.42→2.45 |

---

## Ältere Ergebnisse (Yahoo Finance, veraltet — nur historische Referenz)

> **VERALTET.** Diese Ergebnisse basieren auf Yahoo Finance 1h-Daten (Apr 2024 – Apr 2026).

### A/B-Test — alle Varianten (echte Yahoo-Daten)

| Variante | Rendite | Sharpe | Max DD |
|---|---|---|---|
| v13 Baseline | +9.6% | 0.38 | -45.3% |
| **v14a RSI-Divergenz** | **+190.5%** | **1.13** | **-39.2%** |
| v14b SW deaktiviert | +74.7% | 0.76 | -38.2% |
| v15 SW verbessert | +5.6% | 0.35 | -43.8% |

---

## Versionshistorie

| Version | Änderung | Rendite (Spec) | Max DD |
|---|---|---|---|
| v1 | MA-Crossover 20/50 | -47.6% | -53.4% |
| v5 | Alle Indikatoren | -13.8% | -15.5% |
| v7 | 5-Regime-System | -13.5% | -16.0% |
| v8 (1h, 2J) | Zeitrahmenwechsel | +102.2% | -14.5% |
| v11 | Trail Bulle 3×ATR | +296.3% | -15.1% |
| v12 | Dynamischer Hebel 1-4× | +4.376% | -18.4% |
| v13 | SW-Fix: BB_mid TP, Cover@40 | +1.260% | -43.8% |
| v14b | SW deaktiviert | +14.275% | -18.6% |
| v14a+Regime+Bär-Short+Filter+Trail+SW-Opt | +1617% (Binance, buggy) | -20.4% |
| **Bug-Fix: Notional-Fees + Pyramiding-Avg** | **+117% (korrekt)** | **-34.6%** |
| **EB 2× + ADX<40 + Hebel-Sweep + Bär-Cover RSI>50** | **+253.9%** | **-27.3%** |
