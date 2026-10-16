# SOL Bot — Vollständige Strategie-Spezifikation

> Referenz für alle Signal-Parameter. Änderungen hier → immer auch in `strategy.py` nachziehen und `CLAUDE.md` updaten.

---

## Preismodell

```python
np.random.seed(42)
n = int(3.33 * 365 * 24)  # ~29.170 Stunden-Kerzen
dates = pd.date_range("2022-01-01", periods=n, freq="1h")
checkpoints = [170,140,100,60,35,28,20,15,14,16,18,22,20,22,24,
               30,38,55,58,75,95,110,160,200,175,155,145,150,160,
               175,200,210,190,170,140,120,110]
# Trend = lineare Interpolation zwischen Checkpoints
# Noise = exp(cumsum(normal(0, 0.008)) * 0.3)
# OHLC: High/Low = close ± 1.5 × |ΔClose| + 0.1%
```

---

## Indikatoren (`sol_bot/indicators.py`)

### Trend
| Indikator | Parameter | Spalten |
|---|---|---|
| Simple MA | 20, 50, 200 | `MA20`, `MA50`, `MA200` |
| EMA-Ribbon | 8, 13, 21 | `EMA8`, `EMA13`, `EMA21` |
| MACD | 12/26/9 | `MACD_line`, `MACD_signal`, `MACD_hist` |
| Parabolic SAR | AF=0.02, max=0.2 | `SAR` |
| Supertrend | 2.5×ATR | `Supertrend`, `ST_bull` |
| Ichimoku | Tenkan 9, Kijun 26 | `Tenkan`, `Kijun`, `Senkou_A`, `Senkou_B` |
| ROC | 10 Perioden | `ROC` |

### Momentum
| Indikator | Parameter | Spalten |
|---|---|---|
| RSI | 14 (Wilder) | `RSI` |
| RSI-Divergenz | 5-Kerzen Lookback | `RSI_div_bull`, `RSI_div_bear` |
| Stochastik | 14/3 | `Stoch_K`, `Stoch_D` |
| ADX/DI | 14 (Wilder) | `DI_plus`, `DI_minus`, `ADX` |

### Volatilität
| Indikator | Parameter | Spalten |
|---|---|---|
| ATR | 14 (Wilder) | `ATR` |
| ATR-Ratio | ATR/ATR_MA50 | `ATR_MA50`, `ATR_spike` |
| Vola | Rolling Std × 100 | `Vola` (20), `Vola_5`, `Vola_50`, `Vola_ratio` |

### Volumen
| Indikator | Parameter | Spalten |
|---|---|---|
| OBV | kumulativ | `OBV`, `OBV_MA` (EMA20), `OBV_bull` |
| Volumen-Ratio | Vol/Vol_MA20 | `Vol_MA20`, `Vol_ratio_raw`, `Vol_surge` (>2×) |

### Bänder
| Indikator | Parameter | Spalten |
|---|---|---|
| Bollinger Bands | 20, ±2σ | `BB_upper`, `BB_mid`, `BB_lower` |
| Keltner Channel | 20 EMA, 0.7×ATR* | `KC_upper`, `KC_lower` |
| Squeeze | BB innerhalb KC | `Squeeze` |

> *KC-Multiplikator 0.7 statt Standard 1.5 — angepasst an synthetische Daten (ATR ~0.78% des Preises)

### Sonstiges
| Indikator | Berechnung | Spalte |
|---|---|---|
| VWAP | (H+L+C)/3 × Vol, tägl. Reset | `VWAP` |
| Consecutive | Aufeinanderfolgende Up/Down-Kerzen | `Consec` (+up, -down) |
| MA200-Abstand | (Close-MA200)/MA200×100 | `MA200_dist` |

### Composite-Signale (automatisch berechnet)
```
EMA_bull     = EMA8 > EMA13 > EMA21
EMA_bear     = EMA8 < EMA13 < EMA21
DI_bull      = DI+ > DI-
DI_bear      = DI- > DI+
MACD_pos     = MACD_hist > 0
MACD_neg     = MACD_hist < 0
Ichimoku_bull = Close > Senkou_A & Close > Senkou_B & Tenkan > Kijun
Ichimoku_bear = Close < Senkou_A & Close < Senkou_B
SAR_bull     = Close > SAR
ST_bull      = Supertrend bullish
vol_soft     = NOT Vol_surge
```

---

## 5-Regime-System (`sol_bot/regime.py`)

```python
MIN_REGIME = 4  # Mindest-Kerzen bevor Wechsel (Hysterese)

# Priorität: extrem_baer > extrem_bulle > baer > bulle > seitwaerts > neutral

extrem_baer:  ≥2 von [RSI<25, ATR_spike>1.8, Vola_ratio>2.0, Consec≤-5]  + MA200_dist < -5
extrem_bulle: ≥2 von [RSI>72, ATR_spike>1.8, Vola_ratio>1.8, Consec≥5]   + MA200_dist > +5
baer:         ADX>22  AND  MA200_dist < -3
bulle:        ADX>25  AND  MA200_dist > +2
seitwaerts:   ADX<22  AND  Vola < 1.2
neutral:      Fallback
```

---

## Dynamischer Hebel (`sol_bot/strategy.py`)

```python
# Basis aus ADX
ADX < 20  → 1.0×
ADX < 30  → 1.5×
ADX < 40  → 2.0×
ADX < 50  → 3.0×
ADX ≥ 50  → 4.0×

# Vola-Multiplikator
Vola_ratio > 3.0 → × 0.25
Vola_ratio > 2.0 → × 0.50
Vola_ratio > 1.5 → × 0.75
sonst            → × 1.00

# Regime-Limits
extrem_baer:  max 1.0×
baer:         max 2.0×
seitwaerts:   max 1.5×
neutral:      max 2.0×
bulle:        max 4.0×
extrem_bulle: max 4.0×

# Globale Caps
Short immer: max 2.0×
DD < -15%:   Hebel × 0.5 (min 1.0×)
DD < -25%:   Hebel = 1.0×
```

---

## Position Sizing

```python
ratio = conditions_met / conditions_total
ratio ≥ 1.00  → 100% des verfügbaren Kapitals
ratio ≥ 0.67  → 65%
sonst         → 40%

ADX > 45  → × 1.25 (cap 100%)
ADX > 35  → × 1.15 (cap 100%)
Extrem-Regime → × 1.30 (cap 100%)
```

---

## SL/TP je Regime

| Regime | SL | TP | Trail |
|---|---|---|---|
| Extrem Bär | 1.5×ATR | 3.5×ATR | — |
| Bär | 2.0×ATR | 4.5×ATR | — |
| Seitwärts | 2.5×ATR | BB_mid (dynamisch) | Long 1.2×ATR, Short 1.0×ATR |
| Neutral | 2.0×ATR | 4.0×ATR | — |
| Bulle | 2.0×ATR | 4.5×ATR | 3.0×ATR (Long) |
| Extrem Bulle | 1.5×ATR | 5.0×ATR | — |

---

## Signallogik je Regime

### EXTREM BÄR
- **Short** (signal=-2): 3/4 von [EMA_bear+DI_bear, MACD_neg, Ich_bear+OBV↓, Consec≤-3] + Close<VWAP + vol_soft
- **Long** (Kapitulation): RSI<18 + (RSI_div_bull ODER Vol_surge) + SAR_bull
- **Cover**: Stoch_K<8 ODER (EMA_bull + MACD_pos + RSI<25)

### BÄR
- **Short**: 3/4 von [EMA_bear+DI_bear, MACD_neg, Ich_bear+OBV↓, Close<VWAP] + vol_soft
- **Long**: 2/4 von [RSI<28, RSI_div_bull, SAR_bull, Close<BB_lower] + vol_soft
- **Cover**: Stoch_K<12 ODER (EMA_bull + MACD_pos + SAR_bull)

### SEITWÄRTS *(v14b: deaktiviert)*
- **Kein Handel bei Squeeze**
- **Long** (v13): Close≤BB_lower + 2/3 von [RSI<45, ST_bull/RSI_div_bull, OBV_bull] + vol_soft
- **Short** (v13): 4/6 von [Close≥BB_upper, RSI>58, Stoch↓ von >70, OBV↓, MACD_neg/RSI_div_bear, ST_bear] + vol_soft
- **Cover Short**: Stoch_K>Stoch_D AND 35<Stoch_K<60 AND RSI>40

### BULLE
- **Long (voll)**: Ich_bull+MACD_pos UND SAR_bull+EMA_bull UND ROC>0+OBV_bull + vol_soft + Close>VWAP
- **Long (schwächer)**: Ich_bull+MACD_pos UND SAR_bull+EMA_bull + vol_soft
- **Long (ADX)**: Ich_bull + SAR_bull + ADX>35 + OBV_bull + vol_soft
- **Pyramiding**: ROC>1.5 + ADX>28 + Ich_bull + vol_soft + Close>Entry → +25% Cash (max 3×)
- **Short (Trendbruch)**: ALLE 3: [¬Ich+MACD_neg+EMA_bear] + [¬SAR+Close<VWAP] + [(ROC<-2/RSI_div_bear)+Close<MA50] + vol_soft
- **Exit**: ¬Ich_bull + ¬SAR_bull + MACD_neg + Close<VWAP

### EXTREM BULLE
- **Long**: 2/4 von [EMA_bull+Ich_bull, MACD_pos+ROC>2, Consec≥4, OBV_bull] + Close>VWAP + vol_soft
- **Pyramiding**: ROC>1.5 + ATR_spike>1.5 + Ich_bull + EMA_bull + Close>Entry → +30% (max 4×)
- **Short (Blow-off)**: RSI>85 + MACD_neg + Consec≥8 + OBV↓

---

## Kostenmodell

```python
GEBUEHR  = 0.00075   # 0.075% pro Trade (Entry + Exit)
SLIPPAGE = 0.0003    # 0.03% (vola-adjustiert)
SPREAD   = 0.0002    # 0.02%
FUNDING8 = 0.0001    # 0.01% alle 8h für Short × Hebel

buy_price  = kurs × (1 + min(SLIPPAGE + vola×0.00003, 0.001) + SPREAD/2)
sell_price = kurs × (1 - min(SLIPPAGE + vola×0.00003, 0.001) - SPREAD/2)

# PnL-Berechnung (leveraged)
raw_pnl      = (exit - entry) / entry  # long; (entry - exit) / entry für short
leveraged_pnl = raw_pnl × hebel
proceeds      = pos_cash × (1 + leveraged_pnl) × (1 - GEBUEHR)
cash         += max(proceeds, 0.01)
```
