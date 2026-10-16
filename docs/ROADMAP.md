# SOL Bot — Roadmap & Offene Baustellen

> Priorisierte Liste: was kommt als nächstes, was ist offen, was ist der Horizont.
> Status nach jeder Session updaten.

---

## Status-Übersicht (Stand: Apr 2026)

| Phase | Status |
|---|---|
| v13 implementieren & validieren | ✅ Erledigt |
| v14 Seitwärts A/B-Test | ✅ Erledigt → v14a RSI-Div ist Baseline |
| DI/EMA-Regime statt MA200 | ✅ Erledigt → +20% Gesamt, 2022/2024 je +81% besser |
| Bär-Short (Dead-Cat-Bounce) | ✅ Erledigt → 87 Shorts, WR 67%, ΣPnL +266% |
| RSI-Filter Bulle-Long 38-70 → 65-70 | ✅ Erledigt → Hauptverlierer RSI 55-65 entfernt |
| 10% PosSize statt 100% | ✅ Erledigt → Kelly-Effekt |
| Bulle-Long Trail (3×ATR at Entry) | ✅ Erledigt → SL-Rate 55%→31% |
| Bulle-Long ADX>35 Entry-Filter | ✅ Erledigt → WR 31%→43%, ΣPnL +3.25%, n=21 |
| Binance-Daten als Standard | ✅ Alle Entscheidungen auf `--data binance` |
| Daily Macro Bull als SW-Long-Filter | ✅ Erledigt → 2022 komplett geblockt |
| SW-Long SL/TP/Hebel Grid-Sweep | ✅ Erledigt → SL=3.5×ATR, TP=10×ATR, Hebel=2× |
| SW-Short optimieren | ✅ Abgeschlossen → n=14 zu dünn, $+3 als Bonus akzeptiert |
| **Extrem-Bulle aktivieren (EB-A)** | ✅ **Erledigt Apr 2026** → ADX<40-Filter + Hebel 2×, 2023 +7%→+44.7% |
| **EB TP-Widening** | ✅ **Erledigt Apr 2026** → TP=10×ATR, +253.9%→+467.5%, Sharpe 1.35→1.58, EB-$ $164→$402 |
| **Alle Live-Code-Gates** | ✅ **Alle 5 Gates erfüllt** (Gate 4 mit 2.7pp Puffer nach RSI>50) |
| **Bär-Cover RSI>50 Exit-Trigger** | ✅ **Erledigt Apr 2026** → +222.5%→+253.9%, Sharpe 1.25→1.35, MaxDD -29.8%→-27.3% |
| **EB TP-Widening** | ✅ **Erledigt Apr 2026** → TP=10×ATR (Plateau 10–12), +467.5%, Sharpe 1.58, Gate 4 ✅ |
| **Live-Trading-Vorbereitung** | ⬜ Noch nicht begonnen |

---

## Live-Code-Gates (Stand: Apr 2026)

1. ✅ Seitwärts profitabel (WR 39.2%, EV +0.12%, $+194)
2. ✅ Extrem-Bär kein negativer ΣPnL-Beitrag (0 Trades auf Binance)
3. ✅ Bulle-Long ΣPnL-Beitrag positiv (WR 40.0%, n=20, ΣPnL +0.08% — ⚠️ sehr fragil)
4. ✅ Max Drawdown < 30% (aktuell **-27.3%** — 2.7pp Puffer, nach RSI>50 Cover)
5. ✅ Binance-Daten-Backtest validiert

---

## Offene Optimierungen (priorisiert)

### ~~Priorität 1: EB Trail-Widening~~ (Abgeschlossen via TP-Widening)
- **Lösung:** Trail-Widening-Sweep (1.5×–3.0×) alle Gate 4-verletzend → verworfen.
- **Stattdessen:** TP-Widening: TP=10×ATR mit Trail=1.2×(ON). Plateau-Kurve TP=10–12 (10=Peak), Gate 4 stets ✅.
- **Ergebnis:** +253.9%→+467.5%, Sharpe 1.35→1.58, MaxDD −27.3% (unverändert), EB-$ $164→$402.
- **Caveat:** 82% des EB-$ aus 2023 — TP=10× lässt Recovery-Winner laufen, behebt nicht strukturelle EB-Schwäche in Blowoff-Phasen.

### Priorität 1: Bulle-Long Gate 3 stabilisieren
- **Problem:** ΣPnL +0.1% bei n=20 — ein schlechtes Jahr kippt Gate 3. 2026 WR nur 25%.
- **Hypothese:** `cross_ema21_up` als zusätzlicher Entry-Filter (22% bei Winnern, 0% bei Losern in alter Analyse)
- **Risiko:** n=20 noch dünner nach Filter → Overfitting

### Priorität 2: Bär-Short Recovery-Filter
- **Problem:** 2023 (-$20) und 2026 (-$55) — Bot shortet in Aufwärtsphasen nach Regime-Wechsel
- **Hypothese:** Regime-Age-Filter oder Macro-Bull-Signal als Veto für Bär-Shorts
- **Risiko:** Weniger Shorts → Bär-Beitrag sinkt

---

## Nächste Schritte: Live-Trading-Vorbereitung

> Erst starten wenn EB Trail-Widening abgeschlossen und Gate 4 Puffer > 2pp.

### Schritt 1: Paper Trading auf Binance Testnet
- Binance Futures Testnet API (kostenlos, gleiche REST-API wie Produktion)
- Websocket 1h-Kerzen → `signal → order` Pipeline
- Gleiche `sol_bot`-Paketlogik, kein neuer Code in der Strategie

### Schritt 2: Infra & Monitoring
- Telegram-Alert bei Trade-Signal
- Tägliches PnL-Reporting
- Cooldown-Mechanismus sicherstellen

### Schritt 3: Live mit echtem Kapital ($200)
- Erst nach 4 Wochen Paper-Trading ohne kritische Bugs
- Position Size und Hebel identisch zum Backtest

---

## Entschiedene Fragen (nicht nochmal diskutieren)

| Frage | Entscheidung | Begründung |
|---|---|---|
| 1h vs. 15min? | 1h | 15min → 30% Kostenlast, Overtrading |
| Synthetisch vs. Binance? | Binance | Alle Entscheidungen auf echten Daten |
| Bulle-Shorts? | Deaktiviert | WR 34% → negativer EV |
| Bär-Longs? | Deaktiviert | WR zu gering |
| Chart-Output? | Kein Chart | Token sparen |
| Extrem-Bär Signal-Tuning? | Abgelehnt | n=0 auf Binance, Overfitting einer Marktphase |
| Bulle-Long RSI 70-80? | Abgelehnt | WR 0%, n=2, ΣPnL -4.9% |
| EB Entry-Filter (ADX<30, Vola≥2, RSI≥80)? | Abgelehnt | n=58 zu dünn, Overfitting. ADX<40 ist bereits aktiv und load-bearing. |
| SW-Short weitere Optimierung? | Abgelehnt | n=14 zu dünn, Makro-Split zeigt kein Muster |
| Multi-Position? | Abgelehnt | 36 blockierte Signale in 5 Jahren, nicht lohnenswert |
| EB Lever 3×? | Abgelehnt | MaxDD -31.6% via Compounding-Effekt (SW-Long Pyramiding auf hohem Peak) |
