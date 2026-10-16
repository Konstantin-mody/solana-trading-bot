# Bulle-Signal — Kandidaten zum Ausprobieren

> Status: Brainstorming, kein Backtest validiert.
> Kein Code anfassen bevor Backtest-Evidenz vorliegt.
> Alle Indikatoren-Spalten beziehen sich auf `indicators.py` (was schon da ist).

---

## Warum das aktuelle Signal zu wenig Trades macht

Das aktuelle `signal_bulle()` wartet auf RSI **65–70** — also einen Momentum-Peak.
Das ist ein kleines 5-Punkte-Fenster, das in einem Jahr vielleicht 3–8× trifft.
Zusätzlich ADX > 35 macht es noch enger (n=21 über 5 Jahre).

**Wichtige Einschränkung aus Erfahrung:** RSI 55–65 war empirisch der Hauptverlierer
(WR 23%, ΣPnL −76%). Wenn ein Kandidat unten bei RSI 45–65 einsteigt, muss
das explizit getestet werden — blindes Lockern des RSI-Fensters ist nicht neutral.

---

## Hebel-Schwellen (bevor ein Hebel festgelegt wird)

Hebel ist keine Geschmacksfrage sondern eine Funktion von WR und R:R.
Max DD hängt davon ab wie oft und wie groß die Verlierer sind.

| Hebel | Benötigt WR | Benötigt R:R | Kommentar |
|------:|:-----------:|:------------:|-----------|
| 1–2×  | > 40%       | > 1.0        | Defensiv, Gate-kompatibel |
| 3×    | > 45%       | > 1.2        | Grenzbereich bei kurzen Drawdown-Clustern |
| 5×    | > 52%       | > 1.5        | Nur tragbar wenn Signale unkorreliert |
| 7×    | > 58%       | > 1.8        | Erfordert empirischen Nachweis, kein Wunsch |

**Fazit:** Hebel-Entscheidung erst nach gemessener WR und R:R im Backtest.
5–7× ist das Ziel — aber der Weg dorthin geht über ein besseres Signal, nicht über
höheren Hebel auf das jetzige Signal.

---

## Kandidat A — Pullback zur EMA21 (Trend-Continuation)

**Logik:** In einem Aufwärtstrend zieht der Preis regelmäßig zur EMA21 zurück,
bevor er weiterläuft. Entry beim Bounce, nicht beim Momentum-Peak.
Das ist der Gegenentwurf zum aktuellen Signal.

**Vorteil gegenüber RSI 65–70:** Viel mehr Entries (in 2023 wären das ~30–60 statt 3–5),
bessere Entry-Preise, kleinerer SL-Abstand → besseres R:R.

**Bedingungen:**
- `DI_bull` = True (Trend-Bestätigung)
- `EMA21_slope` = True (EMA21 steigt)
- `Close` zwischen `EMA21 * 0.98` und `EMA21 * 1.03` (Preis nah an EMA21)
- `RSI` in [45, 62] (zurückgefallen, aber nicht überverkauft)
- `MACD_hist_up` = True ODER `SAR_bull` = True (Momentum dreht wieder auf)
- `OBV_bull` = True (Volumen bestätigt)
- `ADX` > 20 (milder Trend, nicht >35)

**Indikatoren vorhanden:** `DI_bull`, `EMA21`, `EMA21_slope`, `RSI`, `MACD_hist_up`,
`SAR_bull`, `OBV_bull`, `ADX` — alles in `indicators.py`, kein Neubau nötig.

**Neues nötig:** Nichts.

**Risiko:** RSI 45–62 schließt den bekannten Verlierer-Bucket 55–65 ein.
Bucket-Analyse beim Backtest zwingend.

**Hebel-Empfehlung (nach Backtest):** Start mit 2–3×, Erhöhung wenn WR > 50% bestätigt.

---

## Kandidat B — MACD-Crossover in bestätigtem Uptrend

**Logik:** MACD-Linie kreuzt Signal-Linie nach oben = klassisches Momentum-Signal.
In Bullphasen ist das ein früher, gut messbarer Entry.

**Bedingungen:**
- `Close` > `MA50` (Preis über mittelfristigem Trend)
- `EMA_bull` = True (EMA21 > EMA50 > EMA200-Alignment)
- MACD-Crossover: `MACD_hist` gerade von negativ auf positiv (`MACD_hist_up` True
  UND vorherige Kerze negativ — braucht 1-Kerzen-Lag)
- `RSI` > 45 (nicht überverkauft)
- `DI_bull` = True
- `ADX` > 22

**Indikatoren vorhanden:** `MACD_hist`, `MACD_hist_up`, `MA50`, `EMA_bull`,
`RSI`, `DI_bull`, `ADX`.

**Neues nötig:** 1-Kerzen-Lag für den echten Crossover (derzeit kein `MACD_cross_up`
Boolean in `indicators.py`). Einfacher Einzeiler: `df["MACD_cross_up"] = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) <= 0)`.

**Risiko:** MACD-Crossover ist bekannt für Whipsaws in seitwärts-volatilen Märkten.
Wird häufig in Range-Phasen falsch getriggert → Regime-Filter (nur in `bulle`) ist Pflicht.

**Hebel-Empfehlung:** 2–4× nach Bestätigung WR > 48%.

---

## Kandidat C — Supertrend-Flip + Volumen-Bestätigung

**Logik:** Supertrend wechselt von bearish auf bullish = klarer Trendumschwung.
Mit Volumen-Bestätigung ist das ein starkes, seltenes aber hohes-Konfidenz-Signal.

**Bedingungen:**
- `ST_bull` True UND vorherige Kerze `ST_bull` False (Flip-Kerze)
- `RSI` > 48
- `Vol_surge` = True ODER `Volume` > 20-Perioden-Ø × 1.3
- `DI_bull` = True
- `Close` > `MA50`

**Indikatoren vorhanden:** `ST_bull`, `RSI`, `Vol_surge`, `DI_bull`, `MA50`.

**Neues nötig:** Supertrend-Flip-Boolean: `df["ST_flip_bull"] = df["ST_bull"] & ~df["ST_bull"].shift(1).fillna(False)`.
`Vol_surge` ist bereits in `indicators.py`.

**Erwartete Tradehäufigkeit:** Selten (~5–15/Jahr in Bullphasen), aber sehr hohe
Aussagekraft. Eher Ergänzung zu Kandidat A als Ersatz.

**Hebel-Empfehlung:** Kann höher sein (3–5×) bei gutem R:R wegen Signalqualität.
Aber n wird klein bleiben — statistisch fragil, ähnlich wie jetzt.

---

## Kandidat D — Stochastik-Bounce im Uptrend (Mean-Reversion + Trend)

**Logik:** Stochastik fällt in Bullentrend auf <35 (kurzfristig überverkauft) und
kreuzt dann nach oben. Klassischer "Dip kaufen im Trend" Einstieg.

**Bedingungen:**
- `EMA_bull` = True (mittelfristiger Aufwärtstrend)
- `Stoch_K` < 35 auf der Vorkerze, jetzt `Stoch_K` > `Stoch_D` (Crossover aufwärts)
- `RSI` in [38, 58]
- `Close` > `EMA21` × 0.96 (nicht zu weit gefallen)
- `DI_bull` = True

**Indikatoren vorhanden:** `Stoch_K`, `Stoch_D`, `EMA_bull`, `RSI`, `EMA21`, `DI_bull`.

**Neues nötig:** Stoch-Crossover braucht 1-Lag. `df["Stoch_cross_up"] = (df["Stoch_K"] > df["Stoch_D"]) & (df["Stoch_K"].shift(1) <= df["Stoch_D"].shift(1))`.

**Risiko:** Ähnlich wie Kandidat A — RSI-Bereich 45–58 war historisch Verlustzone
in dieser Strategie. Zwingend per Bucket analysieren.

**Hebel-Empfehlung:** Erst nach Backtest. Intuitiv 2–3× (Mean-Reversion hat
engere SL → geringere R:R-Basis für hohen Hebel).

---

## Kombination empfohlen zum ersten Test

**Primär:** Kandidat A (Pullback EMA21) als Basis-Entry — gibt die Handelsfrequenz.
**Ergänzung:** Kandidat C (Supertrend-Flip) als Hochkonfidenz-Entry — kann höheren Hebel tragen.

Die beiden kombiniert decken zwei Entry-Typen ab:
Trend-Continuation (A) + Trend-Initiierung (C).

---

## Was nicht probieren (gelernte Lektionen)

| Idee | Warum nicht |
|------|-------------|
| RSI >70 entries | WR 0%, n=2 — bereits verworfen |
| Hebel direkt auf 5–7× setzen | Ohne WR-Evidenz verletzt das Gate 4 fast sicher |
| MA500-Filter | Falsche Richtung (bereits als Hypothese verworfen) |

---

---

# Extrem-Bulle — Kandidaten zum Ausprobieren

> Ziel: frühzeitig in die Top-Bullmarkt-Phase einsteigen, mit engen Stops
> die Gains bei Drawdowns schützen. Hebel max 3×.
>
> Stand: `signal_extrem_bulle()` ist komplett deaktiviert (`return 0`).
> Drei Hypothesen wurden bereits verworfen: MA500-Filter, RSI>72-Filter,
> Trail deaktivieren. Das strukturelle Problem (Blow-off 2021–2022) ist ungelöst.

---

## Was das Extrem-Bulle-Regime ist (und wann es zu spät ist)

Das Regime feuert wenn ≥2 von: RSI>72, ATR_spike>1.8, Vola_ratio>1.8, Consec≥5
— UND Preis >5% über MA200. Das bedeutet: der Bot erkennt EB erst mitten in der
Euphorie. Entry an diesem Punkt hat zwei mögliche Ausgänge:
1. Fortsetzung (Blow-off läuft weiter) → Gewinne
2. Sofortiger Reversal → großer Verlust durch hohe Vola

Der entscheidende Datenpunkt aus dem letzten Backtest:
- `pyramid_count=0` Trades: WR ~0% (sofortiger Reversal, keine Bestätigung)
- `pyramid_count=3` Trades: WR 41.9%, ΣPnL +23.5%

Das heißt: **ein Trade der sich nicht bestätigt (kein Pyramid) ist strukturell ein Verlierer.**
Die Strategie muss entweder früher einsteigen oder schnell qualifizieren ob der Trade läuft.

---

## Warum enge Stops bei Extrem-Bulle schwierig sind

ATR ist im EB-Regime groß (hohe Vola). Ein 1.5×ATR-Stop bei ATR=$3 = $4.50 SL.
Das ist prozentual enger als ein ruhiger Markt, aber absolut trotzdem viel.
Fester %-Stop (z.B. −8%) könnte besser passen als ATR-Multiples.

---

## Kandidat EB-A — Regime-Flip Entry (erste EB-Kerze)

**Logik:** Einsteigen bei der **allerersten Kerze** im Extrem-Bulle-Regime,
wenn die vorherige Kerze noch `bulle` war. Das ist der frühestmögliche Einstieg —
bevor das Regime vollständig "heiß" ist.

**Bedingungen:**
- Aktuelle Kerze = erstes EB nach ≥1 Bulle-Kerze (Regime-Flip)
- `RSI` < 80 (noch nicht im vollen Blow-off)
- `MACD_hist_up` = True (Momentum steigt noch)
- `OBV_bull` = True (Volumen bestätigt)

**Stop-Variante:** Fester %-Trailing von Peak: −8% (nicht ATR-basiert,
weil ATR im EB zu groß ist für enge Stops).

**Exit:** Entweder Stop oder nach 48 Kerzen (2 Tage) Force-Exit wenn kein Pyramid.
Begründung: pyramid_count=0 = struktureller Verlierer → früh absägen.

**Hebel:** 2–3× (User-Ziel). Erst 3× wenn WR EB-A > 45%.

**Indikatoren vorhanden:** `RSI`, `MACD_hist_up`, `OBV_bull`, Regime-Series.
**Neues nötig:** Regime-Flip-Boolean (1-Kerzen-Lag auf Regime-Spalte).

---

## Kandidat EB-B — Bestätigter Entry (nach 6–12h im Regime)

**Logik:** Nicht bei der ersten EB-Kerze einsteigen, sondern erst wenn das Regime
6–12 Kerzen (6–12h) bestätigt ist. Der Preis hat sich damit "bewiesen" —
kurzfristige False-Positives werden herausgefiltert.

**Bedingungen:**
- Regime seit ≥6 Kerzen EB (Hysterese-Bestätigung)
- `RSI` zwischen 70 und 82 (Momentum vorhanden, aber nicht Spitze)
- `Close` > `EMA21` (Preis über kurzfristigem Trend)
- `DI_bull` = True
- `ADX` > 30

**Stop-Variante:** 2×ATR-Trailing vom Hochpunkt (enger als bisherige 3×ATR).
Zusätzlich: wenn Position nach 24h noch kein Pyramid → auf Break-Even setzen.

**Hebel:** Start mit 2×, Erhöhung auf 3× nur wenn 1. Pyramid erfolgt ist.

**Indikatoren vorhanden:** Alle bereits da.
**Neues nötig:** Kerzen-Zähler für Regime-Dauer im Backtest-Loop.

---

## Kandidat EB-C — Pyramiding-Gate als Primärfilter

**Logik:** Aus den Daten wissen wir: `pyramid_count=3` hat WR 41.9% und ΣPnL +23.5%.
Die Strategie dreht das um: kleines Initial-Entry (5% Kapital), und der Trade
existiert nur weiter wenn sich er innerhalb von 8h positiv entwickelt (+2%).
Ohne Bestätigung: automatischer Exit.

**Bedingungen Entry:**
- EB-Regime aktiv
- `RSI` 68–84
- `Close` > `EMA21`

**Bestätigungs-Gate (nach 8h):**
- Wenn `Close` < `Entry_price × 1.02` → sofortiger Exit (Trade hat sich nicht bestätigt)
- Wenn bestätigt → Pyramid auf 20%, dann 40%

**Stop:** −6% vom jeweils aktuellen Hochpunkt (Peak-Trailing).

**Hebel:** 3× — aber nur auf bestätigte Position (nach Gate).

**Indikatoren vorhanden:** Alle da.
**Neues nötig:** Bestätigungs-Gate-Logik im Backtest-Loop (nicht in `strategy.py`
sondern in `backtest.py` — High-Stakes, Sanity-Tests danach Pflicht).

---

## Vergleich der EB-Kandidaten

| Kandidat | Entry-Zeitpunkt | Stop-Typ | Erwartete n/Jahr | Hebel |
|----------|----------------|----------|:----------------:|:-----:|
| EB-A (Flip) | Frühest möglich | −8% Peak-Trailing | 5–15 | 2–3× |
| EB-B (Bestätigt) | 6–12h nach EB | 2×ATR-Trailing | 3–8 | 2→3× |
| EB-C (Pyramid-Gate) | EB + 8h Bestätigung | −6% Peak | 3–6 | 3× bestätigt |

**Empfehlung für ersten Test:** EB-A + EB-C kombiniert.
EB-A gibt frühen Entry mit Force-Exit wenn kein Pyramid.
EB-C ist konzeptuell dasselbe aber sauberer implementiert.

---

## Was strukturell ungelöst bleibt

Der Blow-off 2021–2022 ist nicht RSI-differenzierbar und nicht MA-differenzierbar
(wurde bereits getestet). Das bedeutet: jede EB-Strategie hat ein inhärentes
Risiko bei echten Blow-off-Tops (SOL $250 → $10). Enge Stops reduzieren den
Schaden, lösen das Problem aber nicht vollständig.

Max DD von 30% bleibt Gate 4. Bei 3× Hebel und −8% Stop pro Trade ist ein
Cluster von 3–4 EB-Verlierern in Folge = ca. −10–15% Gesamt-DD.
Gate-kompatibel, aber knapp bei schlechtem Timing.

---

---

# Pyramiding mit eskalierendem Hebel

> Erst implementieren wenn Backtests wieder laufen. Berührt `backtest.py` → High-Stakes.

## Aktueller Stand

Pyramiding ist für Bulle und Extrem-Bulle eingebaut (max 3×, je 25–30% Restkapital).
Der Hebel (`pos.hebel`) wird beim initialen Entry berechnet und bleibt für den
gesamten Trade **fix** — auch nach Pyramids.

## Idee: Hebel pro Pyramid erhöhen

Wenn der Trade sich bestätigt hat (Pyramid-Bedingung erfüllt = Trend läuft),
den Hebel für die neue Tranche höher ansetzen als den Basis-Hebel.

**Beispiel Bulle:**
- Entry: 10% Kapital, 2× Hebel
- Pyramid 1 (bestätigt): 25% Restkapital, 3× Hebel
- Pyramid 2 (nochmals bestätigt): 25% Restkapital, 4× Hebel

**Warum sinnvoll:** Erhöhtes Risiko nur wenn Trade bereits im Gewinn.
Kleiner initialer Drawdown-Beitrag, Gewinne werden stärker amplifiziert.

## Technische Voraussetzung (Falle beachten)

`pos.hebel` wird auf `pos.entry_cash` angewendet. Einfaches Überschreiben
wäre mathematisch falsch (rückwirkende Hebeländerung auf alten Teil).

**Korrekte Lösung:** Hebel als gewichteten Durchschnitt fortschreiben,
analog zur bestehenden `entry_price`-Logik:

```python
new_total_cash  = pos.entry_cash + add_cash
pos.hebel       = (pos.hebel * pos.entry_cash + new_hebel * add_cash) / new_total_cash
pos.entry_cash  = new_total_cash
```

`new_hebel` = z.B. `min(pos.hebel + 1.0, max_hebel_regime)` oder
neu berechnet via `calc_hebel(row)` zum Zeitpunkt des Pyramids.

## Was danach getestet werden muss

- ΣPnL Bulle-Long mit vs. ohne eskalierendem Hebel
- Max DD: eskalierender Hebel erhöht Verlust bei Reversal nach Pyramid
- Gate 4 (MaxDD < 30%) nach Änderung neu prüfen
- `test_sanity.py` muss 13/13 zeigen (da `backtest.py` geändert wird)
