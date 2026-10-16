# SOL Trading Bot — Projektstand

## Meta
- **Ziel:** Algorithmischer SOL/USD Bot, positiver Backtest über kompletten Zyklus, dann live
- **Zeitrahmen:** 1h-Kerzen
- **Backtest-Zeitraum:** Jan 2022 – Apr 2025 (kompletter Zyklus)
- **Startkapital:** $200
- **Aktuelle Version:** v13
- **Status:** Positiv (+1.260%), Seitwärts-EV noch negativ

---

## Technischer Stack
- Python, simulierte Preisdaten (kein yfinance), pandas, numpy
- Preismodell: `checkpoints * exp(cumsum(normal(0,0.008)) * 0.3)`, seed=42
- Kosten: Gebühr 0.075%, Slippage 0.03%, Spread 0.02%, Funding alle 8h

---

## Preismodell (Reproduzierbar)
```python
np.random.seed(42)
n = int(3.33 * 365 * 24)  # ~29170 Kerzen
dates = pd.date_range("2022-01-01", periods=n, freq="1h")
checkpoints_prices = [170,140,100,60,35,28,20,15,14,16,18,22,20,22,24,
                      30,38,55,58,75,95,110,160,200,175,155,145,150,160,
                      175,200,210,190,170,140,120,110]
# SOL: ATH 2022 → FTX-Crash → Erholung 2023 → ATH 2024 → Korrektur 2025
# Marktrenditen: 2022: -87.7%, 2023: +701.7%, 2024: +23.5%, 2025: -50.1%
```

---

## Indikatoren (vollständige Liste)

### Trend
- MA20, MA50, MA200
- EMA8, EMA13, EMA21 (Ribbon)
- MACD (12/26/9) + Histogramm
- Parabolic SAR (AF=0.02, max=0.2)
- Supertrend (2.5×ATR)
- Ichimoku (Tenkan 9, Kijun 26)
- ROC (10 Perioden)

### Momentum
- RSI (14) + Divergenz bull/bear (5-Kerzen-Lookback)
- Stochastic K/D (14/3)
- DI+ / DI- / ADX (14)

### Volatilität
- ATR (14)
- ATR_MA50 → ATR_spike = ATR/ATR_MA50
- Vola (20-Perioden Rolling Std)
- Vola_5 / Vola_50 → Vola_ratio = Vola_5/Vola_50

### Volumen
- OBV + OBV_MA (EMA 20) → OBV_bull
- Volume, Vol_MA20, Vol_ratio_raw, Vol_surge (>2×)

### Bänder
- Bollinger Bands (20, ±2σ) + BB_mid
- Keltner Channel (20 EMA, 1.5×ATR)
- Squeeze = BB innerhalb KC

### Sonstiges
- VWAP (täglich reset)
- Consec (consecutive up/down candles)
- MA200_dist = (Close - MA200) / MA200 * 100

---

## 5-Regime-System

```python
MIN_REGIME = 4  # Mindest-Kerzen bevor Wechsel

def regime_fn(row):
    ma200d = (Close - MA200) / MA200 * 100
    crash = [RSI<25, ATR_spike>1.8, Vola_ratio>2.0, Consec<=-5]
    if sum(crash)>=2 and ma200d<-5:  return "extrem_baer"
    euph  = [RSI>72, ATR_spike>1.8, Vola_ratio>1.8, Consec>=5]
    if sum(euph)>=2  and ma200d>+5:  return "extrem_bulle"
    if ADX>22 and ma200d<-3:         return "baer"
    elif ADX>25 and ma200d>+2:       return "bulle"
    elif ADX<22 and Vola<1.2:        return "seitwaerts"
    return "neutral"
```

**Verteilung im 2022-2025 Backtest:**
| Regime | Kerzen | Anteil |
|---|---|---|
| Seitwärts | 11.245 | 38.5% |
| Neutral | 9.847 | 33.8% |
| Bulle | 3.867 | 13.3% |
| Bär | 3.784 | 13.0% |
| Extrem Bär | 224 | 0.8% |
| Extrem Bulle | 203 | 0.7% |

---

## Signallogik je Regime

### EXTREM BÄR
- **Short:** 3/4 von [EMA bearish+DI bearish, MACD neg, Ichimoku bear/OBV fällt, Consec≤−3] + Kurs<VWAP → signal=-2
- **Long (Kapitulation):** RSI<18 + (RSI_div_bull oder Vol_surge) + SAR_bull → signal=1
- **Cover:** Stoch<8 oder (EMA bullish + MACD pos + RSI<25) → signal=2
- **SL:** 1.5×ATR, **TP:** 3.5×ATR

### BÄR
- **Short:** Score aus [EMA+DI bearish, MACD neg, Ichimoku bear/OBV fällt, Kurs<VWAP] ≥3 + vol_soft → signal=-2
- **Long:** Score aus [RSI<28, RSI_div_bull, SAR_bull, Kurs<BB_unten] ≥2 + vol_soft → signal=1
- **Cover:** Stoch<12 oder (EMA bullish + MACD pos + SAR bull) → signal=2
- **SL:** 2.0×ATR, **TP:** 4.5×ATR

### SEITWÄRTS ⚠️ (Baustelle — EV noch -0.25%)
- **Kein Handel bei Squeeze**
- **Long:** Kurs≤BB_unten + 2/3 von [RSI<45, ST_bull/RSI_div_bull, OBV_bull] + vol_soft
- **Short:** 4/6 von [nahe BB_oben, RSI>58, Stoch dreht von oben, OBV fällt, MACD neg/RSI_div_bear, ST bearish] + vol_soft
- **SL:** 2.5×ATR (v13, war 2.0)
- **TP:** BB_mid (v13, Mean-Reversion-Ziel, war 4.5×ATR)
- **SW-Cover Short:** Stoch kreuzt 40 von unten + RSI steigt (v13, war Stoch 30)
- **Trail Short:** 1.0×ATR (eng), Trail Long: 1.2×ATR

**v13 SW-Diagnose:**
| Kennzahl | v12 | v13 |
|---|---|---|
| WR | 45.5% | 46.3% |
| Ø Gewinn | +0.55% | +0.53% |
| Ø Verlust | −1.29% | −0.93% |
| R:R | 0.43 | 0.57 |
| EV | −0.30% | −0.25% |

### BULLE
- **Long:** Ichimoku_bull+MACD pos + SAR_bull+EMA_bull + ROC>0+OBV_bull + vol_soft + Kurs>VWAP → 3/3
- **Long (schwächer):** erste zwei Bedingungen + vol_soft → 2/3
- **Long (ADX):** Ichimoku_bull + SAR_bull + ADX>35 + OBV_bull + vol_soft
- **Pyramiding:** ROC steigend + ROC>1.5 + ADX>28 + Ichimoku_bull + vol_soft + Kurs>Entry → +25% Cash (max 3×)
- **Short (Trendbruch):** ALLE 3: [kein Ichimoku+MACD neg+EMA bearish] + [kein SAR+Kurs<VWAP] + [(ROC<-2/RSI_div_bear)+Kurs<MA50] + vol_soft
- **Long-Exit:** kein Ichimoku + kein SAR + MACD neg + Kurs<VWAP
- **SL:** 2.0×ATR, **TP:** 4.5×ATR, **Trail:** 3.0×ATR (v11, war 1.5)

### EXTREM BULLE
- **Long:** 2/4 von [EMA+Ichimoku bull, MACD pos+ROC>2, Consec≥4, OBV_bull] + Kurs>VWAP
- **Pyramiding:** ROC>1.5 + ATR_spike>1.5 + Ichimoku_bull + EMA_bull + Kurs>Entry → +30% (max 4×)
- **Short (Blow-off):** RSI>85 + MACD neg + Consec≥8 + OBV fällt
- **SL:** 1.5×ATR, **TP:** 5.0×ATR

---

## Dynamischer Hebel (v12)

```python
def calc_hebel(adx, vola_ratio, regime, is_short, current_dd):
    # Basis aus ADX
    basis = {<20: 1.0, <30: 1.5, <40: 2.0, <50: 3.0, >=50: 4.0}[adx]
    
    # Vola-Multiplikator
    vm = {>3.0: 0.25, >2.0: 0.5, >1.5: 0.75, else: 1.0}[vola_ratio]
    
    h = basis * vm
    
    # Regime-Limits
    extrem_baer:  max 1.0×
    baer:         max 2.0×
    seitwaerts:   max 1.5×  (v13, war 2.0)
    bulle/xbulle: max 4.0×
    
    if is_short:    h = min(h, 2.0)  # Short immer max 2×
    if dd < -0.15:  h = max(h*0.5, 1.0)
    if dd < -0.25:  h = 1.0
    return max(h, 1.0)
```

---

## Position Sizing

```python
def psize_fn(bedinungen_erfüllt, gesamt, adx, extrem=False):
    ratio = bedingungen/gesamt
    base = 1.0 if ratio>=1.0 else (0.65 if ratio>=0.67 else 0.40)
    if extrem: return min(base*1.3, 1.0)
    if adx>45: return min(base*1.25, 1.0)
    elif adx>35: return min(base*1.15, 1.0)
    return base
```

---

## Kostenmodell

```python
GEBUEHR  = 0.00075   # 0.075% pro Trade
SLIPPAGE = 0.0003    # 0.03%
SPREAD   = 0.0002    # 0.02%
FUNDING8 = 0.0001    # alle 8h für Short-Positionen × Hebel

def bk(kurs, vola):  # buy
    return kurs * (1 + min(SLIPPAGE + vola*0.00003, 0.001) + SPREAD/2)
def sk(kurs, vola):  # sell
    return kurs * (1 - min(SLIPPAGE + vola*0.00003, 0.001) - SPREAD/2)
```

---

## Exit-Logik mit Hebel

```python
# Long-Exit
raw_pnl = (exit_kurs - entry) / entry
leveraged_pnl = raw_pnl * hebel
proceeds = pos_cash * (1 + leveraged_pnl) * (1 - GEBUEHR)
cash += max(proceeds, 0.01)

# Short-Exit
raw_pnl = (entry - exit_kurs) / entry
leveraged_pnl = raw_pnl * hebel
proceeds = pos_cash * (1 + leveraged_pnl) * (1 - GEBUEHR)
cash += max(proceeds, 0.01)
```

---

## Ergebnis-Historie (alle Versionen)

| Version | Änderung | Rendite | Outperf. | DD | SW-WR |
|---|---|---|---|---|---|
| v1 | MA-Crossover 20/50 | −47.6% | +11.2% | −53.4% | — |
| v2 | +ADX Filter | −34.2% | +24.6% | −35.0% | — |
| v4b | +SL/TP (1.5/3.5×ATR) | −26.6% | +32.2% | −29.1% | — |
| v5 | Alle Indikatoren | −13.8% | +44.9% | −15.5% | — |
| v6 | Long+Short alle Regime | −15.4% | +43.4% | −16.0% | — |
| v7 | 5-Regime-System | −13.5% | +45.2% | −16.0% | — |
| v8 (15min) | Shorts gefixt | −24.1% | +34.6% | −25.1% | — |
| **v8 (1h, 2J)** | **Zeitrahmenwechsel** | **+102.2%** | — | −14.5% | — |
| v9 | Bulle optimiert, Shorts deaktiviert | +121.8% | — | −13.1% | — |
| v10 | Extrem-Regime gelockert | +126.8% | — | −15.5% | 44.9% |
| v11 | Trail Bulle 3×ATR + SW-Cover@30 | +296.3% | — | −15.1% | 55.3% |
| v12 | Dynamischer Hebel 1×-4× | +4375.7% (2J) | — | −18.4% | — |
| **v12 (3.3J)** | **Kompletter Zyklus** | **+1234.9%** | **+1269.7%** | −43.8% | 45.5% |
| **v13** | **SW-Fix: BB_mid TP, SL 2.5×, Cover@40** | **+1259.8%** | **+1294.7%** | −43.8% | 46.3% |

---

## v13 Aktuelle Ergebnisse (Jan 2022 – Apr 2025)

| Kennzahl | Wert |
|---|---|
| Endkapital | $2.719.70 |
| Rendite | +1.259.8% |
| Buy & Hold | −34.9% |
| Outperformance | +1.294.7% |
| Sharpe Ratio | 3.35 |
| Max. Drawdown | −43.8% |
| Trades/Tag | 0.8 |
| Long/Short | 292 / 183 |
| Long WR | 48.6% |
| Short WR | 76.5% |
| Ø Hebel | 1.46× |

**Jahresperformance:**
| Jahr | Bot | BaH | Outperf. |
|---|---|---|---|
| 2022 | −33.9% | −87.7% | +53.7% |
| 2023 | +1025.3% | +701.7% | +323.5% |
| 2024 | +100.8% | +23.5% | +77.3% |
| 2025 | −9.1% | −50.1% | +41.0% |

**Win Rate je Regime:**
| Regime | WR | Long | Short | Ø Hebel |
|---|---|---|---|---|
| Extrem Bär | 25.0% | 0 | 15 | 1.0× |
| Bär | 58.5% | 122 | 120 | 1.3× |
| Seitwärts | 46.3% | 6 | 48 | 1.1× |
| Bulle | 66.0% | 156 | 0 | 1.9× |
| Extrem Bulle | 62.5% | 8 | 0 | 2.1× |

---

## Offene Baustellen

1. **Seitwärts EV noch negativ (−0.25%)** — größte offene Baustelle (38.5% aller Kerzen)
   - Optionen: anderer Einstieg (RSI-Divergenz-basiert statt BB-Berührung) ODER Seitwärts komplett deaktivieren
2. **Max Drawdown −43.8%** — hauptsächlich 2022-Crash, Short-Signale noch nicht aggressiv genug in Extrem-Bär-Phasen
3. **Extrem-Bär WR 25%** — nur 15 Short-Trades, noch zu wenig Daten

---

## Wichtige Entscheidungen

- **Kein Live-Code** bis Bot stabil profitabel ist
- **1h-Kerzen** (15min hatte 30% Kostenlast → Overtrading)
- **Kein Chart/Balkendiagramm** — nur Tabellen (Token sparen)
- **Dynamischer Hebel** max 4× (nicht fix), ADX+Vola bestimmen
- **Shorts im Bulle deaktiviert** (WR war 34% → negativer EV)
- **Seitwärts TP = BB_mid** (Mean-Reversion-Ziel, v13)
- **Trail im Bulle = 3×ATR** (lässt Trends laufen, v11)

---

## Nächste Schritte (offen)

- Seitwärts-EV positiv machen: RSI-Divergenz-Einstieg ODER deaktivieren
- Extrem-Bär Short-Strategie verbessern (WR 25% → Ziel >50%)
- Wenn alle Regime profitabel → Live-Code schreiben
