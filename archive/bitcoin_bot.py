# ============================================================
#  Bitcoin Trading Bot – 100-Tage MA Analyse
#  Testet MA20/100, MA50/100 und reinen MA100 Vergleich
# ============================================================

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. EINSTELLUNGEN ────────────────────────────────────────
STARTKAPITAL   = 1000
ZEITRAUM_START = "2020-01-01"   # Längerer Zeitraum für bessere Analyse
ZEITRAUM_ENDE  = "2024-12-31"

# ── 2. DATEN LADEN ──────────────────────────────────────────
print("⏳ Lade Bitcoin-Kursdaten (2020-2024)...")
daten = yf.download("BTC-USD", start=ZEITRAUM_START, end=ZEITRAUM_ENDE)
daten.columns = daten.columns.get_level_values(0)
preise = daten[["Close"]].copy()
print(f"✅ {len(preise)} Handelstage geladen.\n")

# ── 3. ALLE MAs BERECHNEN ───────────────────────────────────
preise["MA20"]  = preise["Close"].rolling(20).mean()
preise["MA50"]  = preise["Close"].rolling(50).mean()
preise["MA100"] = preise["Close"].rolling(100).mean()
preise["MA200"] = preise["Close"].rolling(200).mean()

# ── 4. BACKTEST FUNKTION ────────────────────────────────────
def backtest(kauf_fn, verkauf_fn):
    kapital = STARTKAPITAL
    btc     = 0.0
    markt   = False
    trades  = []
    port    = []

    for datum, z in preise.iterrows():
        kurs = float(z["Close"])
        fehlend = any(pd.isna(z[col]) for col in ["MA20","MA50","MA100","MA200"])
        if fehlend:
            port.append(kapital)
            continue

        if kauf_fn(z) and not markt:
            btc    = kapital / kurs
            kapital = 0.0
            markt  = True
            trades.append({"Datum": datum, "Aktion": "KAUF", "Kurs": round(kurs, 0)})

        elif verkauf_fn(z) and markt:
            kapital = btc * kurs
            btc    = 0.0
            markt  = False
            trades.append({"Datum": datum, "Aktion": "VERKAUF", "Kurs": round(kurs, 0)})

        port.append(btc * kurs if markt else kapital)

    letzter = float(preise["Close"].iloc[-1])
    if markt:
        kapital = btc * letzter
    rendite = ((kapital - STARTKAPITAL) / STARTKAPITAL) * 100
    return kapital, rendite, len(trades), port

# ── 5. STRATEGIEN DEFINIEREN ────────────────────────────────

# Strategie A: Preis über/unter MA100 (einfachste Variante)
# Kaufe wenn Preis > MA100, verkaufe wenn Preis < MA100
a_k, a_r, a_t, a_p = backtest(
    lambda z: z["Close"] > z["MA100"],
    lambda z: z["Close"] < z["MA100"]
)

# Strategie B: MA20 kreuzt MA100 (laut Grayscale-Studie optimal)
b_k, b_r, b_t, b_p = backtest(
    lambda z: z["MA20"] > z["MA100"],
    lambda z: z["MA20"] < z["MA100"]
)

# Strategie C: MA50 kreuzt MA100
c_k, c_r, c_t, c_p = backtest(
    lambda z: z["MA50"] > z["MA100"],
    lambda z: z["MA50"] < z["MA100"]
)

# Strategie D: Golden Cross MA50/MA200 (klassisch)
d_k, d_r, d_t, d_p = backtest(
    lambda z: z["MA50"] > z["MA200"],
    lambda z: z["MA50"] < z["MA200"]
)

# Buy-and-Hold
btc_bah    = STARTKAPITAL / float(preise["Close"].iloc[200])
bah_wert   = btc_bah * float(preise["Close"].iloc[-1])
bah_rendite = ((bah_wert - STARTKAPITAL) / STARTKAPITAL) * 100

# ── 6. ERGEBNISSE ───────────────────────────────────────────
print("=" * 62)
print("                  BACKTEST ERGEBNISSE (2020-2024)")
print("=" * 62)
print(f"  A) Preis > MA100      : {a_k:>10.2f} $  ({a_r:>+.2f} %)  |  {a_t} Trades")
print(f"  B) MA20  kreuzt MA100 : {b_k:>10.2f} $  ({b_r:>+.2f} %)  |  {b_t} Trades")
print(f"  C) MA50  kreuzt MA100 : {c_k:>10.2f} $  ({c_r:>+.2f} %)  |  {c_t} Trades")
print(f"  D) MA50  kreuzt MA200 : {d_k:>10.2f} $  ({d_r:>+.2f} %)  |  {d_t} Trades")
print(f"  Buy-and-Hold          : {bah_wert:>10.2f} $  ({bah_rendite:>+.2f} %)")
print("=" * 62)

beste = max([("A: Preis>MA100",a_r),("B: MA20/MA100",b_r),
             ("C: MA50/MA100",c_r),("D: MA50/MA200",d_r)], key=lambda x: x[1])
print(f"\n  🏆 Beste Strategie: {beste[0]} mit {beste[1]:+.2f}%")

# ── 7. CHART ────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
gs  = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1.5])

# Chart 1: Preis + alle MAs
ax1 = fig.add_subplot(gs[0])
ax1.plot(preise.index, preise["Close"],  color="gray",   linewidth=0.8, label="BTC Preis", alpha=0.8)
ax1.plot(preise.index, preise["MA20"],   color="blue",   linewidth=1.2, label="MA 20T",  alpha=0.9)
ax1.plot(preise.index, preise["MA50"],   color="orange", linewidth=1.2, label="MA 50T",  alpha=0.9)
ax1.plot(preise.index, preise["MA100"],  color="red",    linewidth=2.0, label="MA 100T ★", alpha=1.0)
ax1.plot(preise.index, preise["MA200"],  color="purple", linewidth=1.2, label="MA 200T", alpha=0.9)

# MA100 Zone einfärben
ax1.fill_between(preise.index, preise["Close"], preise["MA100"],
    where=(preise["Close"] > preise["MA100"]), alpha=0.08, color="green", label="Preis über MA100")
ax1.fill_between(preise.index, preise["Close"], preise["MA100"],
    where=(preise["Close"] < preise["MA100"]), alpha=0.08, color="red",   label="Preis unter MA100")

ax1.set_ylabel("Preis (USD)")
ax1.set_title("Bitcoin 2020–2024: MA100 im Vergleich zu anderen Moving Averages", fontsize=13)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

# Chart 2: Abstand Preis zu MA100 in %
ax2 = fig.add_subplot(gs[1], sharex=ax1)
abstand = ((preise["Close"] - preise["MA100"]) / preise["MA100"] * 100)
ax2.bar(preise.index, abstand, color=["green" if v > 0 else "red" for v in abstand],
        alpha=0.6, width=1)
ax2.axhline(y=0, color="black", linewidth=0.8)
ax2.axhline(y=20,  color="orange", linewidth=0.8, linestyle="--", alpha=0.7)
ax2.axhline(y=-20, color="orange", linewidth=0.8, linestyle="--", alpha=0.7)
ax2.set_ylabel("Abstand MA100 (%)")
ax2.set_title("Wie weit ist der Preis vom MA100 entfernt?  (Orange = ±20% Zone)")
ax2.grid(True, alpha=0.3)

# Chart 3: Portfolio-Vergleich
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(preise.index, a_p, color="blue",   linewidth=1.5, label=f"A: Preis>MA100 ({a_r:+.1f}%)")
ax3.plot(preise.index, b_p, color="red",    linewidth=2.0, label=f"B: MA20/MA100 ({b_r:+.1f}%) ★")
ax3.plot(preise.index, c_p, color="orange", linewidth=1.5, label=f"C: MA50/MA100 ({c_r:+.1f}%)")
ax3.plot(preise.index, d_p, color="purple", linewidth=1.5, label=f"D: MA50/MA200 ({d_r:+.1f}%)")
ax3.axhline(y=STARTKAPITAL, color="gray", linestyle="--", linewidth=1, label="Startkapital")
ax3.set_ylabel("Portfolio-Wert (USD)")
ax3.set_xlabel("Datum")
ax3.set_title("Portfolio-Vergleich aller 4 Strategien")
ax3.legend(loc="upper left", fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("backtest_ergebnis.png", dpi=150)
plt.show()
print("\n📊 Chart gespeichert als 'backtest_ergebnis.png'")
