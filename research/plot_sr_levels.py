"""
research/plot_sr_levels.py — Visuelle Validierung der S/R-Level-Detektion.

Verwendet die letzten ~500 Kerzen BTC/USDT 15min.
Output: research/sr_levels.png

Ausführen:
  python3 research/plot_sr_levels.py
  python3 research/plot_sr_levels.py --symbol SOLUSDT --bars 300
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from sol_bot_15m.backtest import fetch_binance_15m
from sol_bot_15m.indicators import add_indicators
from sol_bot_15m.levels import LevelTracker


def plot_sr(df: pd.DataFrame, tracker: LevelTracker, n_bars: int, out_path: str) -> None:
    tail = df.iloc[-n_bars:].copy()
    tail_start = len(df) - n_bars

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # ── Candlesticks ──────────────────────────────────────────────────────────
    bar_w = 0.6
    xs = np.arange(len(tail))
    for k, (_, row) in enumerate(tail.iterrows()):
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.plot([k, k], [l, h], color=color, linewidth=0.7, zorder=2)
        rect = plt.Rectangle(
            (k - bar_w / 2, min(o, c)), bar_w, abs(c - o),
            color=color, zorder=3,
        )
        ax.add_patch(rect)

    # ── S/R-Level einzeichnen (Stand: Ende des Fensters) ─────────────────────
    end_bar = len(df) - 1
    confirmed = tracker.get_level_state_at(end_bar, df, min_touches=2)

    # Nur Level innerhalb des sichtbaren Preisfensters (±5% Puffer)
    vis_lo = tail["Low"].min()  * 0.95
    vis_hi = tail["High"].max() * 1.05
    confirmed = [l for l in confirmed if vis_lo <= l.price <= vis_hi]

    for lvl in confirmed:
        color  = "#ff6b6b" if lvl.level_type == "resistance" else "#51cf66"
        lw     = 1.5 if lvl.touch_count >= 3 else 0.8
        alpha  = 0.9 if lvl.touch_count >= 3 else 0.5
        ax.axhline(lvl.price, color=color, linewidth=lw, alpha=alpha,
                   linestyle="--", zorder=4)
        ax.text(
            len(tail) - 0.5, lvl.price,
            f"  {lvl.price:.1f}  ({lvl.touch_count}×)",
            color=color, fontsize=7, va="center", alpha=alpha,
        )

    # ── Touch-Punkte markieren ────────────────────────────────────────────────
    for lvl in confirmed:
        color = "#ff6b6b" if lvl.level_type == "resistance" else "#51cf66"
        for touch_bar, kind in lvl.touches:
            k = touch_bar - tail_start
            if 0 <= k < len(tail):
                marker = "v" if lvl.level_type == "resistance" else "^"
                ms = 8 if kind == "touch" else 5
                ax.plot(k, lvl.price, marker=marker, color=color,
                        markersize=ms, alpha=0.85, zorder=5)

    # ── Achsen / Labels ───────────────────────────────────────────────────────
    step = max(1, len(tail) // 10)
    tick_xs = xs[::step]
    tick_ls = [tail.index[k].strftime("%m-%d %H:%M") for k in tick_xs]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_ls, rotation=30, ha="right", fontsize=8, color="#cccccc")
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    symbol = "BTC" if "BTC" in df.attrs.get("symbol", "BTC") else df.attrs.get("symbol", "BTC")
    ax.set_title(
        f"S/R Level Detection — {symbol}/USDT 15min  |  letzte {n_bars} Kerzen  |  "
        f"{tail.index[0].date()} → {tail.index[-1].date()}\n"
        f"rot = Resistance  |  grün = Support  |  (n×) = Touch-Count  |  "
        f"Marker: ▼/▲ = Touch, kleiner = Swing-Ursprung",
        color="#eeeeee", fontsize=10,
    )
    ax.set_ylabel("Preis (USDT)", color="#cccccc")
    ax.yaxis.label.set_color("#cccccc")

    res_patch = mpatches.Patch(color="#ff6b6b", label="Resistance")
    sup_patch = mpatches.Patch(color="#51cf66", label="Support")
    ax.legend(handles=[res_patch, sup_patch], loc="upper left",
              facecolor="#2a2a4a", labelcolor="#eeeeee", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gespeichert: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--bars",   type=int, default=500,
                        help="Anzahl Kerzen im Plot (default 500)")
    args = parser.parse_args()

    sym_label = args.symbol.replace("USDT", "")
    print(f"Lade {sym_label}/USDT 15min-Daten …")
    df_raw = fetch_binance_15m("2021-09-01", symbol=args.symbol)
    df_raw.attrs["symbol"] = sym_label
    print(f"  {len(df_raw):,} Kerzen geladen")

    print("Berechne Indikatoren …")
    df = add_indicators(df_raw)

    print("Erkenne S/R-Level (full scan) …")
    tracker = LevelTracker()
    tracker.build_full(df)

    all_levels   = tracker.levels
    active_conf  = tracker.get_active_levels(min_touches=2)
    active_all   = [l for l in all_levels if l.active]

    print(f"\n  Swings gesamt erkannt:     {len(all_levels)}")
    print(f"  Davon noch aktiv:          {len(active_all)}")
    print(f"  Bestätigt (≥2 Touches):    {len(active_conf)}")

    if active_conf:
        print("\n  Bestätigte aktive Level (Preis / Typ / Touches):")
        for lvl in sorted(active_conf, key=lambda x: x.price, reverse=True)[:20]:
            print(f"    {lvl.price:10.2f}  {lvl.level_type:<12}  {lvl.touch_count}×  "
                  f"(born {lvl.born_ts.date()})")

    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(out_dir, "sr_levels.png")
    print(f"\nErzeuge Plot ({args.bars} Kerzen) …")
    plot_sr(df, tracker, args.bars, out_path)


if __name__ == "__main__":
    main()
