# SOL Trading Bot — Arbeitsanweisung

> Diese Datei wird von Claude Code automatisch geladen. Detailinfos → `docs/`.
> Nach jeder größeren Änderung diese Datei updaten.

---

## Projekt auf einen Blick

| Feld | Wert |
|---|---|
| **Ziel** | Algorithmischer SOL/USD Bot mit positivem Backtest über vollen Zyklus, dann live |
| **Zeitrahmen** | 1h-Kerzen (~17.000–29.000 Kerzen) |
| **Startkapital** | $200 |
| **Aktuelle Version** | v14a + DI/EMA-Regime + Bär-Short + RSI-65-70-Filter + **ADX>35 Bulle-Entry-Filter** + 10% PosSize + Bulle-Trail + Daily-Macro-Bull SW-Filter + SW-Long SL 3.5×ATR / TP 10×ATR / Hebel 2× + Bär-Short Hebel **6×** + Bulle-Long Hebel **1× (ADX<40) / 2× (ADX≥40) + TP 10×ATR + Trail 4×ATR** + **Extrem-Bulle Hebel 2× + ADX<40 Entry-Filter + TP 10×ATR** + **Bug-Fixes: Notional-Fees, Pyramiding-Weighted-Avg** + **Bär-Cover RSI>50 Exit-Trigger** + **RSI3-Bulle-Long** + **Neutral SW-Short** + **B-Lever ADX≥40 → 2×** |
| **Status** | Vollzyklus (2021–2026, `neutral_sw_short` Default): **+647.3% Rendite** (Binance), Sharpe 1.80, Max DD -19.9% (10.1pp Puffer). Bulle-Long: WR=31.9%, n=69, BL-ΣPnL%(2022+)=+18.7%. Alle 5 Gates ✅. |
| **Nächster Schritt** | **Paper Trading läuft** (seit 2026-05-03, 4 Wochen). paper_bot.py via launchd, $200, parity ✅. Danach: Live auf VPS. |
| **1d-System** | 3 Engines (1d/4h/1w) aus $200: 1d=+57.8%/Sh 0.72, 4h=+40.3%/Sh 0.62, 1w=+405%/Sh 1.17/IS fertig |
| **1d-Bot Gates** | Gate 5 ✅ (OOS n=20). **Paper Trading läuft seit 2026-05-11** (via launchd). Endet ~2026-06-08. |
| **Cross-TF Macro-Filter** | VERWORFEN: 1W-MA50 als Filter -20pp auf 1d, -21pp auf 4h. 1W bleibt eigenständige Engine. |
| **15min-Bot** | **ENDGÜLTIG VERWORFEN (Mai 2026)**: 9 Iterationen, 5 Ansätze (Mean Reversion, RSI-Dip, Breakout/Donchian, **S/R-Double-Test**). Alle scheitern. S/R: WR=27%, Ø=-0.404%, Abort-Gate ausgelöst. Grundproblem: 0.21% Fixkosten/Round-Trip strukturell unvereinbar mit 15min-Edge. **Kein weiteres Tuning.** $200 → 1h-Bot oder 1d-System. |

---

## Multi-Timeframe-Architektur (Mai 2026)

**1h-Bot** bleibt eigenständig ($200). **1d-System** = 3 Engines aus $200 Gesamtkapital:

| Bot | Ordner | TF | Kapital | Status |
|---|---|---|---|---|
| 1h-Bot | `sol_bot/` | 1h | $200 eigen | **Live (Paper Trading)** — nicht anfassen |
| 1d-Bot | `sol_bot_1d/` | 1d | $90 (45%) | **Kein Neutral, MA200-Tiebreaker. Rendite +57.8%/Sh 0.72** |
| 4h-Bot | `sol_bot_4h/` | 4h | $50 (25%) | **v1 fertig. Rendite +40.3%/Sh 0.62/OOS-Sh 0.90** |
| 1w-Bot | `sol_bot_1w/` | 1w | $60 (30%) | **EMA10/21-Cross + MA50-Filter, Long-only, Hebel 2×, SL 1.5×ATR. IS: +405%/Sh 1.17/MaxDD -22.1%** |
| 15min-Bot | `sol_bot_15m/` | 15min | $200 eigen | **ENDGÜLTIG VERWORFEN (Mai 2026)** — 9 Iterationen (inkl. S/R-Double-Test), Abort-Gate ausgelöst, 0.21% Fixkosten strukturell nicht kompatibel mit 15min-Edge |

**Geteiltes Kostenmodell:** `cost_model.py` — importiert von allen Bots.
**Funding:** 1d=3×/Kerze, 4h=alle 2 Kerzen, 1w=21×/Kerze, 1h=alle 8 Kerzen, 15min=alle 32 Kerzen.

**Cross-TF Macro-Filter: VERWORFEN (Mai 2026)**
- 1W-MA50 als Richtungsfilter für 1d/4h getestet: 1d Rendite -20pp, 4h -21pp
- Zu viele profitable Gegen-Trend-Rücksetzer blockiert → kein Nutzen
- 1W läuft als eigenständige Trade-Engine (wenn Signale profitabel)

**Nächste Schritte:**
- Combined Paper Trading endet ~2026-06-08 → dann Live auf VPS (alle 3 zusammen)
- 15min: **GESCHLOSSEN** — kein weiteres Tuning
- 4h/1w Verbesserungen: erst nach Paper Trading Ende, falls gewünscht vor Live-Start

**Wenn du am 1d-System arbeitest:** `sol_bot_1d/`, `sol_bot_4h/`, `sol_bot_1w/` + jeweilige Runner.
**1h-Bot:** Nicht anfassen — läuft im Paper Trading.

## Dateistruktur

```
trading bot/
├── CLAUDE.md                    ← Diese Datei (immer aktuell halten)
├── cost_model.py                ← Geteiltes Kostenmodell (GEBUEHR, SLIPPAGE, FUNDING8, …)
├── run_backtest.py              ← 1h-Bot: python3 run_backtest.py --data binance
├── run_backtest_1d.py           ← 1d-Bot: [--wf] [--regime_sweep] [--macro_filter]
├── run_backtest_4h.py           ← 4h-Bot: [--wf] [--macro_filter]
├── run_backtest_1w.py           ← 1w-Bot: [--wf]
├── run_backtest_15m.py          ← 15min-Bot: [--wf] [--regime_sweep]
├── paper_bot.py                 ← 1h Paper Trading (läuft via launchd, nicht anfassen)
├── paper_state.json             ← Live-State des Paper Bots
├── paper_trades.db              ← Trade-History SQLite
├── replay.py                    ← Parity-Test Paper Bot vs. Backtest (1h)
├── paper_bot_combined.py        ← Combined 1d+4h+1w Paper Trading (läuft via launchd seit 2026-05-11)
├── paper_state_combined.json    ← Live-State des Combined Paper Bots
├── paper_trades_combined.db     ← Trade-History Combined (SQLite)
├── replay_combined.py           ← Parity-Test Combined Paper Bot vs. Backtest ✅ PASSED (397 Trades, Δ=$0)
├── view_trades.py               ← Trade-Visualisierung
│
├── research/
│   ├── test_sanity.py           ← 1h Sanity-Tests (nach jeder backtest.py-Änderung!)
│   └── test_sanity_1d.py        ← 1d Sanity-Tests
│
├── sol_bot/                     ← 1h-Bot (LIVE — nicht ändern ohne Advisor)
│   ├── indicators.py            ← ~62 Indikatoren (MA, EMA, RSI, MACD, ADX, BB, VWAP, …)
│   ├── regime.py                ← 5-Regime-Klassifizierung mit MIN_REGIME=4 Hysterese
│   ├── strategy.py              ← Signallogik je Regime, Hebel-Calc, Position Sizing
│   └── backtest.py              ← Preismodell (seed=42), Kostenmodell, Backtest-Engine
│
├── sol_bot_1d/                  ← 1d-Bot ($90, 45% des 1d-Systems)
│   ├── indicators.py            ← EMA10/21/50, MA50/200, RSI3, ADX, ATR, OBV, Fib
│   ├── regime.py                ← 2-Regime (bull/bear, kein Neutral), MA200-Tiebreaker
│   ├── strategy.py              ← RSI3<50 Bull-Long + RSI3>65 Bear-Short, SL/TP 1×/8×ATR
│   └── backtest.py              ← Funding 3×/Kerze, COOLDOWN=2, PosSize=20%, macro_regime/bear_trail_mult param
│
├── sol_bot_4h/                  ← 4h-Bot ($50, 25% des 1d-Systems)
│   ├── indicators.py            ← Gleiche Perioden wie 1d (auf 4h: EMA10=40h, MA200=800h)
│   ├── regime.py                ← 2-Regime (bull/bear), MA200-Tiebreaker, MIN_REGIME=6
│   ├── strategy.py              ← Gleiche Signallogik wie 1d
│   └── backtest.py              ← Funding alle 2 Kerzen, COOLDOWN=12, macro_regime param
│
├── sol_bot_1w/                  ← 1w-Bot ($60, 30% des 1d-Systems)
│   ├── indicators.py            ← MA50/200, RSI14, ADX, ATR auf Weekly
│   ├── regime.py                ← 2-Regime, MA50-weekly als Bull/Bear-Schalter
│   ├── strategy.py              ← RSI14-basiert (Signale brauchen Tuning — user hat Ideen)
│   └── backtest.py              ← Funding 21×/Kerze, COOLDOWN=4 Wochen, Start ab 2020-01
│
├── sol_bot_15m/                 ← 15min-Bot ($200 eigen, Tuning offen)
│   ├── indicators.py            ← RSI5/14, EMA5/13/21/50, VWAP96, ATR, ADX, Daily-Macro
│   ├── regime.py                ← 3-Regime (bull/bear/neutral), MIN_REGIME=8
│   ├── strategy.py              ← S6-Basis: RSI5<25 + VWAP + Daily-Macro, SL 1.5×ATR, TP 4×ATR
│   └── backtest.py              ← Funding alle 32 Kerzen, MAX_DAILY_TRADES=5
│
├── docs/                        ← Dokumentation (STRATEGY, RESULTS, ROADMAP)
└── archive/                     ← Alte Dateien + abgeschlossene Research-Skripte
    └── research/                ← 20 abgeschlossene Analyse-Skripte (archiviert Mai 2026)
```

---

## Schnellstart

```bash
# 1h-Bot (live, nicht anfassen)
python3 run_backtest.py --data binance                   # Produktionslauf

# 1d-System
python3 run_backtest_1d.py --wf                          # Walk-Forward (IS vs. OOS)
python3 run_backtest_1d.py --macro_filter                # Vergleich mit/ohne 1W-Filter
python3 run_backtest_1d.py --bear_trail_sweep            # Bear-Trail Sweep {None,1.5,2.0,2.5,3.0,4.0} [VERWORFEN — ATR zu groß]
python3 run_backtest_4h.py --wf                          # 4h Walk-Forward
python3 run_backtest_4h.py --macro_filter                # 4h Vergleich mit/ohne 1W-Filter
python3 run_backtest_1w.py --wf                          # 1w Walk-Forward

# 15min-Bot
python3 run_backtest_15m.py --wf                         # Walk-Forward
python3 run_backtest_15m.py --regime_sweep               # MIN_REGIME {4,8,16} vergleichen

# Diagnose / Sanity
python3 research/test_sanity.py                          # 1h Sanity (nach backtest.py-Änderung)
python3 research/test_sanity_1d.py                       # 1d Sanity
```

---

## Advisor-Workflow

Advisor kostet Budget (voller Transcript pro Aufruf). Nur einsetzen, wo er Wert schafft.

### Triage: Welche Stufe?

| Stufe | Wann | Advisor |
|---|---|---|
| **Trivial** | Doku, Umbenennung, Formatierung, Kommentare | Keiner |
| **Normal** | Einzel-Datei-Änderung (Indikator, Signal, Filter) | 1× vorher, 1× nachher |
| **High-Stakes** | Regime-Logik, Backtest-Engine, Live-Code-Vorbereitung | Schleife (max 3 Runden) |

### Stufe Normal (Standard)

1. `advisor()` aufrufen mit geplantem Vorgehen → Plan anpassen
2. Umsetzen
3. `advisor()` aufrufen mit fertigem Code → Blocker sofort beheben
4. Fertig (kein weiterer Advisor, außer neue Blocker tauchen auf)

### Stufe High-Stakes (Schleife)

1. `advisor()` mit Plan → anpassen
2. `advisor()` mit angepasstem Plan → weiter anpassen, **bis keine Blocker mehr** (max 3 Runden)
3. Umsetzen
4. `advisor()` mit fertigem Code → Blocker beheben
5. `advisor()` erneut → Schleife endet wenn **keine Blocker mehr** (max 2 weitere Runden)

**Stopp-Kriterium:** Keine neuen **Blocker** — nicht "keine Verbesserungen". Kosmetische Vorschläge des Advisors können ignoriert werden.

**Mandatory High-Stakes:** Jede Änderung, die eine Live-Code-Readiness-Gate berührt (Regime-WR, Drawdown-Schwelle, Signal-Logik die alle Trades betrifft).

**Mandatory High-Stakes: jede Änderung an `backtest.py`** — Die Backtest-Engine ist der kritischste Teil. Ein Bug hier macht alle Optimierungen wertlos (Lektion Mai 2026: Pyramiding + Fee-Bug haben Endkapital 15× aufgebläht).

### Pflicht-Sanity-Checks nach jeder backtest.py-Änderung

```bash
python3 research/test_sanity.py   # Muss "13/13 Bestanden" zeigen — sonst nicht weitermachen
```

---

## Wichtige Entscheidungen (nicht nochmal diskutieren)

| Entscheidung | Begründung |
|---|---|
| 1h-Kerzen (nicht 15min) | 15min → 30% Kostenlast, Overtrading |
| Hebel dynamisch 1×–4× | ADX + Vola-basiert, Short max 2× |
| v14a (RSI-Div) als Baseline | Synthetisch besser als v14b, echt +190.5% (Apr 2024–Apr 2026 Fenster) |
| Regime: DI/EMA statt MA200 | MA200 ist level-basiert, bleibt 2023 nach FTX-Crash das ganze Jahr Bär. DI/EMA reagiert auf Richtung: +20% Gesamtverbesserung (2022: +81%, 2024: +81% besser). 2023 bleibt strukturell schwach — jeder Trendklassifizierer liest Post-Crash-Recovery falsch |
| signal_baer() Veto: ST_bull + RSI_div_bull | Shorts werden blockiert wenn Supertrend bullisch oder RSI-Divergenz bullisch — Schutz vor Shorts gegen Trendumkehr |
| signal_bulle(): Trigger B entfernt, not_extended 1.06→1.02 | Trigger B (WR 27%, ΣPnL negativ) entfernt. Dist-Filter von 6% auf 2% getightnet — war Hauptverlustbringer (dist>2%: WR 21-27%, ΣPnL -181%). Bringt +17% Outperformance, 2023 von -51.8% auf -15.5% |
| signal_bulle() RSI-Filter: 38-70 → 65-70 | Diagnose: RSI 55-65 war Hauptverlierer (73 Trades, WR 23.3%, ΣPnL -76.8%). RSI 65-70 einzig profitabler Bucket (WR 44.4%, ΣPnL +21.8%), cross-year robust in 4/5 Jahren. Outperformance -16.8% → +63.9%. RISIKO: Nur ~37 Bulle-Trades über 5 Jahre — statistisch fragil; extrem_bulle dominiert das Ergebnis. |
| Bär-Short reaktiviert (Dead-Cat-Bounce) | Dead-Cat-Bounce-Filter: A+B+C + close>=VWAP + RSI<45. 303 Shorts, WR 57.8%. Ohne Shorts war bot Long-only, nun +134% Outperf. (100% PosSize-Basis). |
| 10% Initial-Entry-PosSize (nicht 100%) | Sweep 10–100%: 10% → $872/+335%/Sharpe 1.47/MaxDD -18.7%; 100% → $435/+117%/Sharpe 0.58/MaxDD -50.5%. Kelly-Effekt: Überbetten vernichtet Compound. Pyramiding weiterhin aktiv (Entry 10% → bis ~60% über 3 Pyramiden). |
| Bär-Cover NICHT beschleunigen | Schnellerer Cover → sofortiger Re-Entry (gleiche Bedingungen aktiv) → mehr Trades, mehr Kosten, schlechter. Problem liegt im Re-Entry, nicht im Exit. |
| Bär-Cover RSI>50 Exit-Trigger (Apr 2026) | In Recovery-Jahren (2023/2026) springt Preis 2×ATR bevor EMA/MACD/SAR bullisch drehen → SL feuert vor Cover. RSI steigt von <35 (Entry) schneller auf >50 als Trend-Indikatoren reagieren. Sensitivity-Sweep 45/48/50/52/55: RSI>50 ist robustes globales Optimum (glatte monotone Kurve). RSI>45 schlechter (Cover zu früh → Re-entry sofort). Ergebnis: +222.5%→+253.9%, Sharpe 1.25→1.35, MaxDD -29.8%→-27.3% (0.2pp→2.7pp Puffer). Offener Outlier: 2026-02-06 -51.7% (ATR=4.3%, Fast-Bounce, RSI>50 zu langsam → "accepted recovery cost"). |
| Bulle-Shorts deaktiviert | WR 34% → negativer EV |
| Bär-Longs deaktiviert | WR zu gering, jetzt wirklich aus |
| Extrem-Bulle ADX<40 Entry-Filter + Hebel 2× + TP 10×ATR (Apr 2026) | Vorherige Hypothesen verworfen: MA500-Filter, RSI>72-Filter, Trail deaktivieren, Trail-Widening (Sweep 1.5×–3.0×: alle Gate 4-verletzend). **TP-Widening (Apr 2026):** TP-Sweep {5–12}×ATR mit Trail=1.2×(ON): Plateau TP=10–12 (10 ist Peak), Gate 4 stets ✅, MaxDD −27.3% unverändert. Ergebnis: +253.9%→+467.5%, Sharpe 1.35→1.58, EB-$ $164→$402. TP=10× in `run_backtest.py` Defaults eingetragen. **Caveat:** 82% des EB-$ aus 2023 ($329 von $402) — TP=10× löst nicht die strukturelle EB-Schwäche, sondern lässt Recovery-Year-Winner weiter laufen. In Blowoff-Phasen (2021/2022: -$15/-$3) bleibt EB negativ. Erwarteter Vorteil bedingt auf Recovery-Zyklen. Weitere Entry-Filter verboten: n=57 zu dünn (→ Overfitting). |
| Bulle-Long Trail aktiviert (3×ATR) | Trail bei Entry auf 3×ATR gesetzt (identisch mit Update-Multiplikator). SL-Rate halbiert (55% → 31%), 2025 von -4.1% auf +13.6% ΣPnL verbessert. Sharpe 1.47 → 1.56, MaxDD -18.7% → -16.8%. RSI 70-80 verworfen (WR 0%, ΣPnL -4.9%). Entry-Filter-Optimierung abgeschlossen: n=38 zu dünn für weiteres Overfitting. |
| Gate 2 (Extrem-Bär) geschlossen und umformuliert | Auf Binance-Daten: 0 Trades in extrem_baer (359 Kerzen, 0.9%, nur 2025–2026). FTX-Crash 2022 triggert Regime nicht. WR-basierte Gate war undefiniert (n=0) und widersprach User-Priorität (ΣPnL > WR). Signal-Tuning wäre Overfitting einer Marktphase. Neue Gate-Bedingung: kein negativer ΣPnL-Beitrag → bei 0 Trades erfüllt. |
| SW-Short nicht weiter optimierbar | n=14 über 5 Jahre strukturell zu dünn. Diagnose: kein Makro-Splitting ergibt stabiles Muster (bull=True vs False: WR identisch 50%). $+29 als Bonus akzeptiert. |
| Kein Live-Code bis alle Regime profitabel | — |
| Binance 1h-Daten für alle Entscheidungen | Nicht synthetisch, nicht Yahoo — immer `--data binance` |
| Binance-Start: `2021-09-01` | Voller Zyklus ab ATH-Anlauf 2021 (inkl. FTX-Crash, Erholung, neuer Zyklus). Daten vor 2022 dienen nur als Vollständigkeitsreferenz — **Optimierungsentscheidungen immer auf Basis der Daten ab 2022** (danach andere Marktstruktur + Liquidität). |
| SW-Long SL/TP/Hebel optimiert | 2D-Grid-Sweep: SL 3.5×ATR + TP 10×ATR + Hebel 2× = Sweep-Optimum. Sharpe 2.36, MaxDD -20.4%. TP=10× senkt MaxDD (Kapital schneller frei), Hebel=2× amplifiziert Gewinne. 4×ATR-SL verworfen wegen MaxDD -22% ohne TP. |
| Bär-Short Hebel 6× (DD-geschützt) | Post-Bug-Fix-Sweep (Mai 2026): 8× → MaxDD -34.6%, Gate 4 verletzt. 2D-Sweep (Bulle×Bär × EB1×): Bär 6× + EB1× → MaxDD -28.4%, Sharpe 0.92. Bär 8× verletzt Gate 4 auch bei Bulle 1×. Defaults in `run_backtest.py` (nicht `backtest.py`), da backtest.py-Fallback-Werte 8× bleiben für direkte API-Aufrufe. |
| Bulle-Long Hebel 1× Baseline / 2× ADX≥40 (B-Lever) | Post-Bug-Fix: WR 31.4% = negativer EV bei Einheits-Hebel. RSI3-Signal verbessert auf n=70. ADX≥40-Bucket (n=19) hat EV +1.44%/Trade → 2× sinnvoll. Unter ADX<40: WR 27.5%, negativer EV → bleibt bei 1×. Uniform-Erhöhung auf 2× würde MaxDD verschlechtern. Gate 3: ΣPnL%(2022+)=+24.6% ✅. Siehe "B-Lever" Entscheidung. |
| Extrem-Bulle Hebel 2× (Apr 2026, war 1×) | Post-Bug-Fix: EB Hebel auf 1× (Mai 2026) wegen ΣPnL=-121.2% bei Ø1.93×. Jetzt 2× mit ADX<40-Filter: 2023 +44.7% vs. 1× ~+14%. 3× verletzt Gate 4 via Compounding-Effekt (SW-Long Pyramiding greift auf Equity-Peak auf). Defaults in `run_backtest.py`. |
| Daily Macro Bull (EMA21>EMA50 1d) als SW-Long-Filter | Eliminiert 2022 komplett (0 Trades). `Daily_macro_bull` extra_signal via `compute_daily_macro(df_raw)`. Basis für alle `neutral_*`-Modi (die automatisch auf `daily_bull` für SW mappen). |
| Compounding-Warnung SW-Long | $3414 Endkap. ~40-50% pfadabhängig. Realistischer Erwartungswert neuer Zyklus: $1500-2200. Robuste Metriken: Sharpe 2.35, EV +3.15%/Trade, 4/5 Jahre positiv. |
| Kein Chart-Output | Token sparen |
| **Bug-Fix: Notional-Fees + Pyramiding (Mai 2026)** | Drei kritische Bugs behoben: (1) Gebühren wurden auf entry_cash statt Notional (cash×hebel) berechnet → 8×–16× Unterzählung bei hohem Hebel. (2) Pyramiding: entry_price blieb am Original statt gewichtetem Durchschnitt → Gewinne 2–3× aufgebläht. (3) Bär-Cover prüfte nicht pos.regime. **Auswirkung:** Endkapital korrigiert von $6617 → $435, Sharpe 2.43 → 0.71. Isolationstest: Fee-Bug −$1428, Pyramiding-Bug −$6061. Alle alten Zahlen in RESULTS.md sind historisch/veraltet. |
| **Produktions-Defaults in `run_backtest.py`, nicht `backtest.py`** | Hebel-Overrides (Bär 6×, Bulle 1×, **EB 2×**) + **EB TP 10×ATR** werden in `run_backtest.py` als explizite Parameter übergeben. `backtest.py`-Fallbacks (8×/5×, EB-TP=5×) bleiben für direkte API-Aufrufe — wer `run_backtest()` direkt importiert, muss Overrides selbst übergeben. |
| **Bulle-Long ADX>35 Entry-Filter (Mai 2026)** | Diskriminator-Analyse auf 39 Trades: Winner-Median ADX=38.6 vs Loser-Median=33.1 (+5.6 Diff). Filter reduziert n 39→21, WR 30.8%→42.9%, ΣPnL -25%→+3.25%. 4/5 Jahre positiv. ⚠️ Risiko: ADX-Threshold in-sample (selbe Daten, keine Walk-Forward-Validierung). **[Überlagert durch B-Lever — ADX-Diskriminator wird jetzt als Dosing-Signal verwendet, nicht als Entry-Filter]** |
| **Bär-Short RSI-Untergrenze (rsi_min=25) — VERWORFEN (Mai 2026)** | Trade-Level: RSI<25-Bucket n=9, WR=38%, ΣPnL=-45.7% → verlockend. Portfolio-Sweep (rsi_min ∈ 20–30): rsi_min=25 ist lokales Peak (+7.5pp, Sharpe +0.01), aber rsi_min=26 ist −25.5pp schlechter als Baseline. Threshold-Swing (33pp/RSI-Punkt) > Gewinn (7.5pp). Path-Dependenz hat Trade-Level-Effekt 6× überschätzt. n=9 in-sample, fat-tail-getrieben. Kill-Evidence: Non-Monotonie beweist Overfitting auf Pfad-Noise, kein robustes Signal. |
| **Bulle-Long B-Lever: ADX≥40 → 2× (Mai 2026)** | ADX-konditionaler Hebel in `backtest.py`: 2× wenn ADX≥40 bei Entry, sonst 1×. Alle 70 Trades bleiben erhalten (kein Filter-Effekt). EV ADX≥40: +1.44%/Trade → mit 2×: +2.88%/Trade. Ergebnis: Endkapital $1274→$1494, Rendite +537%→+647%, Sharpe 1.73→1.77 (besser als Baseline), MaxDD -17.2%→-21.7% (+4.5pp Kosten). Gate 3: BL-ΣPnL%(2022+) -0.9%→+24.6% ✅. **Caveats:** (1) 11/19 ADX≥40-Trades aus 2023 (Recovery-Year-Dominanz wie EB). (2) Threshold ADX≥40 in-sample (n=19). (3) MaxDD +4.5pp = Preis für 2×. **DD-Schutz:** Beide Hebel-Tiers fallen auf 1× bei dd<-0.15. Pyramiding erbt pos.hebel (Variante A). Parameter: `bulle_lev_high_override=2.0, bulle_adx_thresh_override=40.0`. **Achtung: RSI3 ohne B-Lever verletzt Gate 4 (-34.8%). B-Lever ist nicht nur Rendite-Booster, sondern Drawdown-Schutz. Nie ohne B-Lever deployen.** |
| **Connors RSI-2-System — VERWORFEN (Mai 2026)** | Entry: Close>SMA(200) + RSI(2)<thresh; Exit: RSI(2)>90 OR signal_bulle_exit(). Threshold-Sweep {5,8,10,12,15} monoton (Sh 1.42→0.81) — kein Overfitting-Spike wie rsi_min=25, sondern strukturelles Edge-Defizit. Beste Variante t5: +171%/Sh 1.42/MaxDD -19.7% vs Produktion +647%/Sh 1.77/-21.7%. BL-ΣPnL%(2022+) trade-level positiv (+210%), konvertiert nicht zu Portfolio-Rendite (Pfad-Dependenz + 2.3× Cost-Drag aus höherem Trade-Count). Ursache: Mean-Reversion-Exits cutten Trend-Runner für SOL zu früh. Connors ist für niedrigvolatile US-Equities mit kurzen Recovery-Zyklen — SOL läuft in monatelangen Trends. |
| **Bulle-Long TP 10×ATR + Trail 4×ATR (Apr 2026)** | TP-Sweep {4.5, 6, 8, 10, 12}×ATR auf default + M4:R55 mit Trail=4×ATR. **Default:** TP=10× optimiert → 2024 +0.5%→+10.8% (Runner laufen aus), Gesamt ΣPnL +0.1%→+4.6%, Rendite +467.5%→+477.8%. Trail-Update-Bug gleichzeitig behoben (hardcoded 3.0 → parametrisiert). **M4:R55:** TP=6× leicht besser, aber TP≥8× verschlechtert (Winner erreichen TP8× nicht via Trail gestoppt) — M4:R55 strukturell nicht rettbar. Defaults in `run_backtest.py`: `bulle_tp_mult=10.0, bulle_trail_mult=4.0`. |
| **Look-Ahead-Bug: compute_daily_macro (Mai 2026)** | `compute_daily_macro` in `backtest.py` hat `.shift(1)` gefehlt: `resample("1D").last()` erzeugt Index "Tag D 00:00" mit dem 23:xx-Close von Tag D — beim `reindex(ffill)` bekam die 00:00-Kerze von Tag D bereits den EMA-Status von Tag D's Ende (23h Look-Ahead). Fix: `bull.shift(1).reindex(...)`. Impact akademisch klein: Sharpe 1.77→1.80, MaxDD -21.7%→-19.9% (besser!), BL-ΣPnL%(2022+) +24.6%→+18.7% (Gate 3 bleibt ✅). Alle 5 Gates weiterhin ✅. Note: `fetch_fear_greed` hat gleichen Pattern, aber nur in Research-Modi aktiv — Tech-Debt. `compute_sol_btc_rs` war sauber (1h-native, kein Daily-Resampling). |
| **Bulle-Long Signal: RSI3-Mikro-Dip (Apr 2026)** | `signal_bulle_rsi3()` ersetzt `signal_bulle()` als Produktions-Default. RSI(3)<20 + RSI14∈[40,75] + daily_macro + di_bull + ema21_slope + close≥MA50×0.95. **Diagnose:** n=70 (vs. 19 baseline), WR=32.9%, ΣPnL%(gesamt)=-25.0% (davon 2021=-24.1%, Vollständigkeitsreferenz), **ΣPnL%(2022+)=-0.9% (≈break-even)**, BL-$=+$94 (vs. $69 baseline). Portfolio: +500.3%/Sharpe 1.68/MaxDD -16.9% (alles besser als Baseline). **RSI14-Bucket-Analyse:** 50-55-Bucket strukturell verlustreich (WR=23%, -38.3%), 55-60 profitabel (+14.4%). Notch-Filter (ex 50-55) ergibt Gate3✅ aber MaxDD=-30.0% (Gate4❌). ≥55-Filter: Gate3✅ aber n=15 (unter Baseline). → RSI3 solo mit rsi14_min=40 als robustem Kompromiss gewählt. **Gate 3 neu:** auf 2022+-Basis (2021=Vollständigkeitsreferenz, keine Optimierungsbasis per Projekt-Regel). |
| **1d-Bot signal_bull_entry: RSI3<50, MA200-Filter entfernt, OBV behalten (Mai 2026)** | 2×2-Isolation MA200×OBV: OBV trägt Edge (Sh +0.05 allein), MA200 nicht (+3.0% mit MA200+OBV vs. +4.6% ohne MA200). RSI3-Full-Sweep {30..60}: Plateau 50–55 (RSI3<50 = RSI3<52 identisch, RSI3<54=RSI3<55 identisch), Inflektion bei 56, Kollaps bei 58. RSI3<50 = konservativer Plateau-Eintritt. Ergebnis: Rendite +3.0%→+17.3%, Sharpe 0.12→0.44, MaxDD -11.6%→-8.1%, Bull-n 16→30, WR 44%→57%. **Caveats:** (1) IS-ΣPnL 80% aus 2023 (Recovery-Year-Dominanz). (2) OOS-neutralität (Bull-OOS ΣPnL=-0.4%, n=8) — kein OOS-positiv-Beweis. (3) OOS-"Verbesserung" von -30.2%→-0.4% hängt an Aug-2025-Cluster (+31.3pp aus 2 Trades, 5 Tage auseinander). (4) Strategie-Typ ändert sich: nicht mehr RSI3-Dip-Buy sondern Trend-Continuation (RSI3<50 auf 1d = Trend-Pause, nicht struktureller Tief). RSI3<55 verworfen — OOS+20.2% war Single-Cluster-Artefakt (selber Aug-2025-Cluster). |
| **1d-Bot SL_TP["bull"]: 2.0×ATR → 1.0×ATR (Mai 2026)** | SL-Sweep {0.5..2.5}×ATR zeigte monotone OOS-Verbesserung (0.06→0.85 OOS-Sharpe). Advisor warnte: Monotonie = mögliches Slot-Rotation-Artefakt. Fate-Analyse durchgeführt: für SL=1.0× sind 87% der neuen SL-Exits genuine Verlierer bei 2.0× (6/8 LOSER, nur 1 WINNER entgangen, 10.5pp). Für SL=0.5× 76% genuine Verlierer aber 4 WINNER entgangen (+59.7pp) + 0.5×ATR ≈ 3.5% liegt im Intraday-Noise-Floor von SOL (3-5%), Close-only-Backtest blind für Wickouts. SL=1.0×ATR gewählt: OOS Bull ΣPnL -0.4%→+30.7%, OOS n=8 identisch (gleiche Trades, kleinere Verluste), Sharpe 0.44→0.48, Rendite +17.3%→+18.7%, MaxDD -8.1%→-8.7%. Nur Bull-SL geändert; Bear-SL bleibt 2.0× (noch nicht validiert, Baustelle A). |
| **1d-Bot Bear-SL 2.0×ATR → 1.0×ATR (Mai 2026)** | Fate-Analyse: 100% der neuen SL-Exits bei 1.0× sind genuine Verlierer (4/4 LOSER, 0 WINNER entgangen). Bear ΣPnL +45.2%→+93.9%. Rendite +24.6%→+31.0%, Sharpe 0.62→0.76, MaxDD -8.5%→-7.0%. |
| **1d-Bot Bear-Regime: Tuning abgeschlossen (Mai 2026)** | Bear-ΣPnL aktuell +66.4% (IS +35.7%, OOS +30.7%), n=29, WR=45%. ADX-Filter (≥25) und MA200=NaN-Block beide verworfen: ADX non-monoton (Overfitting-Muster wie rsi_min=25), NaN-Block trifft 50% Winner-Rate (keine echte Discriminator-Eigenschaft). Die CLAUDE.md-Note "-105pp unbehandelt" war Pre-SL-Tightening — nach 1.0×ATR-SL sind die 3 Großverluste von -37.7/-30.9/-36.9 → -14.3/-11.7/-18.9. Bear-Regime mature, kein weiterer Tuning-Versuch. |
| **1d-Bot Bug-Fixes: IS/OOS-Label + Funding-Timing (Mai 2026)** | (1) IS/OOS-Label verwendete Exit-Timestamp statt Entry-Timestamp (`backtest.py:214`): Trade eröffnet in IS, geschlossen in OOS → fälschlicherweise als OOS gezählt. Fix: `df.index[pos.candle_idx] < WF_SPLIT`. Kein numerischer Effekt in aktuellen Daten (kein Trade überbrückte 2025-01-01), aber konzeptuell korrekt. (2) Equity-Snapshot und DD-Schutz berechneten Portfolio-Wert vor Funding-Abzug. Fix: Funding-Block vor Equity-Berechnung verschoben. Numerischer Effekt vernachlässigbar (<$0.02/Tag). |
| **1d-Bot Position Size 10% → 20% (Mai 2026)** | Sweep {5%..100%}: Sharpe konstant (0.78–0.82), kein Edge-Cliff. Gate 4 (MaxDD<-15%): 20% = -12.7% ✅, 25% = -15.7% ❌. 20% verdoppelt Rendite (+28.5%→+62.0%) bei gleichem Sharpe (0.79). Default in `sol_bot_1d/backtest.py`: `max_pos_size=0.20`. |
| **1d-Bot Hebel/PosSize-Sweep — VERWORFEN (Mai 2026)** | Bull-Hebel 2.5× und PosSize 22% bestehen formal alle Gates, aber MaxDD bei 2.5× exakt -15.0% (0pp Puffer). Gate 8 (500%) strukturell unerreichbar — Decke des 1d-Bots liegt bei ~80% ohne fundamentale Strategie-Änderung. Kein Nutzen. Baseline bleibt Bull=2.0×, PosSize=20%. |
| **1d-Bot Neutral-Regime: 5/5 Hypothesen gescheitert — GESCHLOSSEN (Mai 2026)** | Hypothesen: (A) RSI3<25 Long, (B) RSI3<30+Fib Long, (C) RSI3<25+OBV Long, (D) ADX-Compression-Breakout, (E) EMA_bear+RSI3>Threshold Short. Alle scheitern: OOS-ΣPnL negativ in allen Varianten, Rendite sinkt unter Baseline. ADX-Compression: strukturell unmöglich wegen 14-Perioden Wilder-ADX (4–6 Wochen zum Abschwächen), niemals echt komprimiert vor Regime-Flip. Short-Hypothese: OOS -17% bis -31%, beide Sub-Perioden (2025/2026) negativ. **Kein weiterer Neutral-Versuch.** |
| **1d-Bot Cooldown 3 → 2 Tage (Mai 2026)** | Sweep {0,1,2,3}: CD=2 einzige Alternative die alle 4 Gates besteht. OOS Bull ΣPnL +30.7%→+58.7% (+28pp). Zwei neue OOS-Trades: (1) 2025-08-25 Entry, +28.5% Cover, klare Bull-Continuation nach RSI3-Reset (gap=2d nach vorherigem Exit) ✅. (2) 2025-09-19 Entry, -10% SL, valides Entry-Signal aber unglückliches Regime-Flip 2 Tage später ✅ (Entry-Logik korrekt). Netto: +18.5pp aus 2 neuen Trades. IS leicht verschlechtert (-4.8pp), OOS deutlich besser (+28pp). Sanity 18/18. Default in `sol_bot_1d/backtest.py`: `COOLDOWN = 2`. |
| **1d-Bot Hebel-Optimierung: Bull=2.0×, Bear=1.5× (Mai 2026)** | Sweep BULL_MULTS=[1.0,1.5,2.0,2.5,3.0], BEAR_MULTS=[1.0,1.5,2.0,2.5,3.0,4.0]. Bull-OOS-Sharpe flacht bei 2.0× ab (0.74 identisch bei 2.0×/2.5×/3.0×) — kein struktureller OOS-Gewinn durch höheres Leverage. Bull bleibt 2.0×. Bear: Sharpe-Peak bei 1.5× (Sh=0.81) durch lineare Funding-Kosten-Skalierung — ab 2.0× sinkt Sharpe wieder. Finales Ergebnis: Rendite +28.5%, Sharpe 0.79, MaxDD -6.5%. |
| **1d-Bot signal_bear_entry: MA200-Filter Close < MA200 (Mai 2026)** | Diskriminator-Analyse Bear Verlierer vs. Gewinner: Median Close/MA200 Verlierer=-12.7% vs. Gewinner=-46.2%. 6/19 Verlierer über MA200 (32%) vs. 1/16 Gewinner (6%). Sweep MA200-Faktor {1.10, 1.05, 1.00, 0.95, 0.90, 0.80}: Peak bei 1.00 und 1.05 (beide OOS-Bear +37.8% identisch) — 1.00 gewählt (keine freien Parameter). Cliff bei 0.95: +37.8% → -8.4% OOS, erklärt durch 2025-03-02 (+46.2% OOS-Winner, ratio=0.970) der im [0.95-1.00]-Band liegt und bei 0.95 eliminiert wird. Ergebnis: Rendite +18.7%→+24.6%, Sharpe 0.40→0.62, Bear-WR 46%→54%, IS-Bear -16.2%→+7.4%, OOS-Bear +15.1%→+37.8%. **Scope:** Nur der "über-MA200"-Subbug gefixt. Ungelöst: 2023-01-02 (-37.7%, -61% unter MA200, RSI3=85 Recovery-Rally-Start), 2023-06-19 (-30.9% SL, -18% unter MA200), 2022-01-31 (-36.9%, MA200=NaN). Ca. -105pp Verlust-ΣPnL in Bear bleibt unbehandelt. |
| **1d-Bot Bear-Trail (ATR-basiert) — VERWORFEN (Mai 2026)** | Hypothese: Trailing Stop für Short-Positionen sichert profitable Float. Implementierung: `bear_trail_mult`-Parameter in `run_backtest()`, trail_price=ep+mult×ATR, min-following. Sweep {None, 1.5, 2.0, 2.5, 3.0, 4.0}. **Kill-Evidence:** 1d-ATR = 7–16% des Preises. Trail 2.0×ATR = initial 14–31% über Entry. Typische Bear-Trade-MFE (Close-basiert) = 3–15%. Ergebnis: Trail≥2.0× feuert bei 0/36 Bear-Trades. Trail 1.5× feuert 2× und beschädigt IS-Bear von +32.9% auf +14.0% (−18.9pp) für nur +2.1pp OOS-Gain — zwei IS-Winner werden vorzeitig gecuttet. Trace-Beispiel: 2022-06-15, ep=34.74, ATR=5.40 (15.5%), Trail initial=45.53 (+31.1%), Price nie nah genug — Cover feuert bei Close=38.15 (DI_bull). **Ursache strukturell:** Close-only-Backtest + 1d-ATR-Größe. MFE via Low-Candle überschätzt tatsächliche Close-basierte Bewegung (Wickouts sichtbar in OHLC, aber Backtest sieht nur Close). **Code behalten:** `bear_trail_mult=None` Default = identisches Verhalten, kein Regression-Risiko. Kein weiterer ATR-basierter Trail-Versuch für Bear. |

---

## Live-Code-Gates (gesperrt bis erfüllt)

> **Stand Apr 2026:** Bär 6×, Bulle 1×, Extrem-Bulle **2×** + ADX<40-Filter. Korrekte Gebühren auf Notional + Pyramiding Weighted-Avg.

1. ✅ Seitwärts profitabel (WR 39.2%, EV +0.12%, ΣPnL positiv)
2. ✅ Extrem-Bär kein negativer ΣPnL-Beitrag (0 Trades auf Binance → kein Drawdown-Beitrag)
3. ✅ Bulle-Long ΣPnL-Beitrag positiv (WR=31.9%, n=69, ΣPnL%(2022+)=+18.7% — RSI3-Signal + TP 10×ATR + Trail 4×ATR + B-Lever ADX≥40→2×; Gate auf 2022+-Basis, da 2021=Vollständigkeitsreferenz)
4. ✅ Max Drawdown < 30% (aktuell **-19.9%**, 10.1pp Puffer)
5. ✅ Binance-Daten-Backtest validiert (alle Entscheidungen seit v14a auf `--data binance`)

---

## 1d-Bot Live-Gates (gesperrt bis erfüllt)

> **Stand Mai 2026:** Bull=2.0×, Bear=1.5×, SL beide 1.0×ATR. WF-Split: IS 2021-09→2024-12, OOS 2025+.
> **Benchmark Gate 7:** 1h-Bot Bull OOS 2025+: n=13, ΣPnL=+12.6% → **+0.97%/Trade** (gemessen Mai 2026).

1. ✅ OOS Bull ΣPnL > 0% (aktuell +58.7%, n=10)
2. ✅ OOS Bear ΣPnL > 0% (aktuell +30.7%, n=9)
3. ✅ OOS-Sharpe ≥ 0.5 (aktuell 0.74+)
4. ✅ MaxDD gesamt < -15% (aktuell ~-8%)
5. ❌ OOS n ≥ 20 (aktuell 19 — ca. 2–4 Wochen / ~1 Trade fehlt)
6. ✅ Kein Extrem-Regime getraded (by design — keine Signale implementiert)
7. ✅ 1d Bull OOS ΣPnL%/Trade > 1h Bull OOS ΣPnL%/Trade (>3.84% > 0.97%)

8. ❌ Gesamtrendite ≥ 500% über 5 Jahre (aktuell +63.1% — 1h-Bot Referenz: +647.3%)

**Nach Erfüllung aller Gates:** 4 Wochen Paper Trading, danach Live auf VPS (analog 1h-Bot).

---

## Kostenmodell (fix, nicht ändern)

```
Gebühr:   0.075% pro Trade
Slippage: 0.03%
Spread:   0.02%
Funding:  0.01% alle 8h (nur Short × Hebel)
```

---

## Regime-Verteilung (synthetisch, Referenz)

| Regime | Anteil | Trades aktiv? |
|---|---|---|
| Seitwärts | ~41% | Ja (RSI-Div) |
| Neutral | ~26% | Nein (0 Trades) |
| Bulle | ~16% | Ja |
| Bär | ~15% | Ja |
| Extrem Bär | ~0.8% | Ja |
| Extrem Bulle | ~0.8% | Ja |
