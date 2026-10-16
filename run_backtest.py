"""
run_backtest.py — Einstiegspunkt für den SOL Trading Bot Backtest

Verwendung:
    python run_backtest.py                        # v13 Baseline, synthetische Daten
    python run_backtest.py --ab                   # A/B-Test Seitwärts-Modi
    python run_backtest.py --mode rsi_div         # RSI-Divergenz-Modus
    python run_backtest.py --data real            # echte SOL 1h-Daten (730 Tage)
    python run_backtest.py --data daily           # echte SOL Tagesdaten (seit 2020)
    python run_backtest.py --data real --ab       # A/B-Test auf echten Daten
"""

import sys
import argparse
import numpy as np
import pandas as pd

from sol_bot.backtest import (run_backtest, make_price_data, fetch_real_data, fetch_binance_data,
                               fetch_btc_data, fetch_funding_rate, fetch_fear_greed,
                               compute_daily_macro, compute_sol_btc_rs, STARTKAPITAL)


# ─── Reporting ───────────────────────────────────────────────────────────────

def sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    return (rets.mean() / rets.std()) * np.sqrt(8760)  # annualisiert auf Stunden


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd   = (equity - peak) / peak
    return dd.min() * 100


def print_report(label: str, trades_df: pd.DataFrame,
                 equity: pd.Series, df_full: pd.DataFrame):
    final    = equity.iloc[-1]
    rendite  = (final - STARTKAPITAL) / STARTKAPITAL * 100
    bah_end  = df_full["Close"].iloc[-1]
    bah_ret  = (bah_end / df_full["Close"].iloc[0] - 1) * 100
    outperf  = rendite - bah_ret
    sr       = sharpe(equity)
    dd       = max_drawdown(equity)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Endkapital     : ${final:>10.2f}")
    print(f"  Rendite        : {rendite:>+.1f}%")
    print(f"  Buy & Hold     : {bah_ret:>+.1f}%")
    print(f"  Outperformance : {outperf:>+.1f}%")
    print(f"  Sharpe Ratio   : {sr:.2f}")
    print(f"  Max. Drawdown  : {dd:.1f}%")

    if len(trades_df) == 0:
        print("  Keine Trades.")
        return

    n_days  = (equity.index[-1] - equity.index[0]).days
    tpd     = len(trades_df) / max(n_days, 1)
    n_long  = (trades_df["Side"] == "long").sum()
    n_short = (trades_df["Side"] == "short").sum()
    wr_long  = (trades_df[trades_df["Side"] == "long"]["PnL_pct"] > 0).mean() * 100 if n_long else 0
    wr_short = (trades_df[trades_df["Side"] == "short"]["PnL_pct"] > 0).mean() * 100 if n_short else 0
    avg_h    = trades_df["Hebel"].mean()

    print(f"  Trades/Tag     : {tpd:.1f}")
    print(f"  Long / Short   : {n_long} / {n_short}")
    print(f"  Long WR        : {wr_long:.1f}%")
    print(f"  Short WR       : {wr_short:.1f}%")
    print(f"  Ø Hebel        : {avg_h:.2f}×")

    # Jahresperformance
    print(f"\n  Jahresperformance:")
    print(f"  {'Jahr':<6} | {'Bot':>8} | {'BaH':>8} | {'Outperf.':>10}")
    print(f"  {'-'*38}")
    for year in sorted(equity.index.year.unique()):
        eq_y   = equity[equity.index.year == year]
        cl_y   = df_full["Close"][df_full.index.year == year]
        if len(eq_y) < 2 or len(cl_y) < 2:
            continue
        bot_r  = (eq_y.iloc[-1] / eq_y.iloc[0] - 1) * 100
        bah_r  = (cl_y.iloc[-1] / cl_y.iloc[0] - 1) * 100
        out_r  = bot_r - bah_r
        print(f"  {year:<6} | {bot_r:>+7.1f}% | {bah_r:>+7.1f}% | {out_r:>+9.1f}%")

    # Win Rate je Regime
    print(f"\n  Win Rate je Regime:")
    print(f"  {'Regime':<15} | {'WR':>6} | {'Long':>5} | {'Short':>5} | {'Ø Hebel':>7}")
    print(f"  {'-'*50}")
    for reg in ["extrem_baer", "baer", "seitwaerts", "neutral", "bulle", "extrem_bulle"]:
        sub = trades_df[trades_df["Regime"] == reg]
        if len(sub) == 0:
            continue
        wr  = (sub["PnL_pct"] > 0).mean() * 100
        nl  = (sub["Side"] == "long").sum()
        ns  = (sub["Side"] == "short").sum()
        ah  = sub["Hebel"].mean()
        print(f"  {reg:<15} | {wr:>5.1f}% | {nl:>5} | {ns:>5} | {ah:>6.2f}×")

    # Seitwärts-Detail (EV)
    sw = trades_df[trades_df["Regime"] == "seitwaerts"]
    if len(sw) > 0:
        ev     = sw["PnL_pct"].mean()
        avg_w  = sw[sw["PnL_pct"] > 0]["PnL_pct"].mean() if (sw["PnL_pct"] > 0).any() else 0
        avg_l  = sw[sw["PnL_pct"] < 0]["PnL_pct"].mean() if (sw["PnL_pct"] < 0).any() else 0
        rr     = abs(avg_w / avg_l) if avg_l != 0 else 0
        print(f"\n  Seitwärts Detail:")
        print(f"    EV     : {ev:>+.2f}%")
        print(f"    Ø Win  : {avg_w:>+.2f}%   Ø Loss: {avg_l:>+.2f}%   R:R: {rr:.2f}")

    # Regime-Verteilung
    if df_full is not None and "Regime" in df_full.columns:
        print(f"\n  Regime-Verteilung:")
        total = len(df_full)
        for reg, cnt in df_full["Regime"].value_counts().items():
            print(f"    {reg:<15}: {cnt:>6}  ({cnt/total*100:.1f}%)")

    print()


# ─── Drawdown-Diagnose ───────────────────────────────────────────────────────

def print_dd_diag(trades_df: pd.DataFrame, equity: pd.Series):
    """Lokalisiert den Max-DD Peak/Trough und analysiert Trades im DD-Fenster."""
    if len(equity) == 0:
        return

    print(f"\n{'='*60}")
    print(f"  DRAWDOWN-DIAGNOSE")
    print(f"{'='*60}")

    # Peak & Trough des maximalen Drawdowns finden
    cum_peak = equity.cummax()
    dd_curve = (equity - cum_peak) / cum_peak

    idx_trough = dd_curve.idxmin()
    idx_peak   = equity.loc[:idx_trough].idxmax()
    max_dd_val = dd_curve.min() * 100

    eq_peak   = equity.loc[idx_peak]
    eq_trough = equity.loc[idx_trough]
    duration  = (idx_trough - idx_peak).days

    print(f"\n  Max-DD Fenster:")
    print(f"    Peak    : {idx_peak.date()}  ${eq_peak:.2f}")
    print(f"    Trough  : {idx_trough.date()}  ${eq_trough:.2f}")
    print(f"    Max DD  : {max_dd_val:+.1f}%")
    print(f"    Dauer   : {duration} Tage")

    # Trades im Peak→Trough-Fenster
    if len(trades_df) == 0:
        return

    in_window = trades_df[
        (trades_df["Entry_ts"] >= idx_peak) &
        (trades_df["Entry_ts"] <= idx_trough)
    ].copy()

    print(f"\n  Trades im DD-Fenster (n={len(in_window)}, "
          f"außerhalb: {len(trades_df)-len(in_window)}):")
    print(f"  {'Regime':<15} | {'Side':<5} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*52}")
    for reg in ["extrem_baer", "baer", "seitwaerts", "bulle", "extrem_bulle"]:
        for side in ["long", "short"]:
            sub = in_window[(in_window["Regime"] == reg) & (in_window["Side"] == side)]
            if len(sub) == 0:
                continue
            wr   = (sub["PnL_pct"] > 0).mean() * 100
            spnl = sub["PnL_pct"].sum()
            print(f"  {reg:<15} | {side:<5} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # Jahr-Aufschlüsselung des DD-Fensters
    if len(in_window) > 0:
        in_window["Year"] = pd.to_datetime(in_window["Entry_ts"]).dt.year
        print(f"\n  Nach Jahr im DD-Fenster:")
        print(f"  {'Jahr':<6} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
        print(f"  {'-'*34}")
        for yr in sorted(in_window["Year"].unique()):
            sub  = in_window[in_window["Year"] == yr]
            wr   = (sub["PnL_pct"] > 0).mean() * 100
            spnl = sub["PnL_pct"].sum()
            print(f"  {yr:<6} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # R:R-Analyse für Bär-Shorts: In-Window vs. Out-of-Window
    def _rr_stats(subset, label):
        if len(subset) == 0:
            return
        wins  = subset[subset["PnL_pct"] > 0]["PnL_pct"]
        loss  = subset[subset["PnL_pct"] < 0]["PnL_pct"]
        wr    = len(wins) / len(subset) * 100
        aw    = wins.mean() if len(wins) > 0 else 0
        al    = loss.mean() if len(loss) > 0 else 0
        rr    = abs(aw / al) if al != 0 else float("inf")
        ev    = subset["PnL_pct"].mean()
        sl_p  = (subset["Exit_reason"] == "SL").mean() * 100 if "Exit_reason" in subset.columns else 0
        tp_p  = (subset["Exit_reason"] == "TP").mean() * 100 if "Exit_reason" in subset.columns else 0
        cv_p  = (subset["Exit_reason"] == "Cover").mean() * 100 if "Exit_reason" in subset.columns else 0
        dur   = (pd.to_datetime(subset["Exit_ts"]) - pd.to_datetime(subset["Entry_ts"])).dt.total_seconds() / 3600
        print(f"\n  {label} (n={len(subset)}):")
        print(f"    WR={wr:.1f}%  EV={ev:+.2f}%  Ø-Win={aw:+.2f}%  Ø-Loss={al:+.2f}%  R:R={rr:.2f}")
        print(f"    SL={sl_p:.1f}%  TP={tp_p:.1f}%  Cover={cv_p:.1f}%  Ø-Dauer={dur.mean():.1f}h")

    bs_all    = trades_df[(trades_df["Regime"] == "baer") & (trades_df["Side"] == "short")]
    bs_in     = in_window[(in_window["Regime"] == "baer") & (in_window["Side"] == "short")]
    bs_out    = bs_all[~bs_all.index.isin(in_window.index)]
    print(f"\n  Bär-Short R:R-Vergleich:")
    _rr_stats(bs_out,  "Außerhalb DD-Fenster")
    _rr_stats(bs_in,   "Im DD-Fenster (Jun25–Apr26)")

    # Trades unter DD-Protection-Schwelle (-25%)
    # Equity beim Entry approximieren via nächstem bekanntem Wert
    dd_at_entry = []
    for _, tr in trades_df.iterrows():
        ts = tr["Entry_ts"]
        if ts in dd_curve.index:
            dd_at_entry.append(dd_curve.loc[ts] * 100)
        else:
            idx_pos = dd_curve.index.searchsorted(ts)
            dd_at_entry.append(dd_curve.iloc[min(idx_pos, len(dd_curve)-1)] * 100)
    trades_df = trades_df.copy()
    trades_df["DD_at_entry"] = dd_at_entry

    protected = trades_df[trades_df["DD_at_entry"] < -25.0]
    if len(protected) > 0:
        wr_prot   = (protected["PnL_pct"] > 0).mean() * 100
        spnl_prot = protected["PnL_pct"].sum()
        print(f"\n  Trades bei DD < -25% (Protection aktiv): n={len(protected)}")
        print(f"    WR    : {wr_prot:.1f}%")
        print(f"    ΣPnL  : {spnl_prot:+.1f}%")
        by_reg = protected.groupby("Regime").apply(
            lambda g: pd.Series({
                "N": len(g),
                "WR": (g["PnL_pct"] > 0).mean() * 100,
                "ΣPnL": g["PnL_pct"].sum()
            })
        )
        for reg, r2 in by_reg.iterrows():
            print(f"    {reg:<15}: N={int(r2['N'])}, WR={r2['WR']:.1f}%, ΣPnL={r2['ΣPnL']:+.1f}%")
    else:
        print(f"\n  Keine Trades bei DD < -25% — Protection wurde nicht überwältigt.")

    print()


# ─── Bär-Short Diagnose ──────────────────────────────────────────────────────

def print_baer_diag(trades_df: pd.DataFrame):
    bs = trades_df[
        (trades_df["Regime"] == "baer") &
        (trades_df["Side"] == "short") &
        trades_df["Trigger_id"].notna()
    ].copy()
    if len(bs) == 0:
        return

    print(f"\n{'='*60}")
    print(f"  BÄR-SHORT DIAGNOSE  (n={len(bs)})")
    print(f"{'='*60}")

    # ── Score-Kombination ────────────────────────────────────────────────────
    print(f"\n  Häufigste Bedingungskombinationen:")
    combos = bs["Trigger_id"].value_counts().head(10)
    print(f"  {'Bedingungen':<10} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*36}")
    for combo, n in combos.items():
        sub  = bs[bs["Trigger_id"] == combo]
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {combo:<10} | {n:>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # ── RSI-Buckets ──────────────────────────────────────────────────────────
    rsi_bins   = [20, 35, 45, 55, 65, 80]
    rsi_labels = ["20–35", "35–45", "45–55", "55–65", "65–80"]
    bs["RSI_bucket"] = pd.cut(bs["RSI_entry"], bins=rsi_bins, labels=rsi_labels, right=True)
    print(f"\n  WR nach RSI-Bucket bei Entry:")
    print(f"  {'Bucket':<8} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*36}")
    for lb in rsi_labels:
        sub = bs[bs["RSI_bucket"] == lb]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {lb:<8} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # ── Dist-VWAP-Buckets ────────────────────────────────────────────────────
    dist_bins   = [-15, -5, -2, 0, 2, 5]
    dist_labels = ["<-5%", "-5–-2%", "-2–0%", "0–2%", "2–5%"]
    bs["Dist_bucket"] = pd.cut(bs["Dist_ema21_entry"], bins=dist_bins, labels=dist_labels, right=True)
    print(f"\n  WR nach Distanz VWAP bei Entry:")
    print(f"  {'Distanz':<8} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*36}")
    for lb in dist_labels:
        sub = bs[bs["Dist_bucket"] == lb]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {lb:<8} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # ── Jahresaufschlüsselung ────────────────────────────────────────────────
    bs["Year"] = pd.to_datetime(bs["Entry_ts"]).dt.year
    print(f"\n  Bär-Short nach Jahr:")
    print(f"  {'Jahr':<6} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'RSI<35':>7} | {'RSI35-45':>9}")
    print(f"  {'-'*52}")
    for yr in sorted(bs["Year"].unique()):
        sub   = bs[bs["Year"] == yr]
        wr    = (sub["PnL_pct"] > 0).mean() * 100
        spnl  = sub["PnL_pct"].sum()
        n_low = (sub["RSI_entry"] <= 35).sum()
        n_mid = ((sub["RSI_entry"] > 35) & (sub["RSI_entry"] <= 45)).sum()
        print(f"  {yr:<6} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}% | {n_low:>7} | {n_mid:>9}")

    # ── RSI<35 vs RSI35-45 nach Jahr (ΣPnL-Split) ───────────────────────────
    print(f"\n  ΣPnL% nach RSI-Bucket × Jahr:")
    print(f"  {'Jahr':<6} | {'RSI<35 ΣPnL':>12} | {'RSI35-45 ΣPnL':>14} | {'RSI<35 WR':>10} | {'RSI35-45 WR':>12}")
    print(f"  {'-'*62}")
    for yr in sorted(bs["Year"].unique()):
        sub   = bs[bs["Year"] == yr]
        low   = sub[sub["RSI_entry"] <= 35]
        mid   = sub[(sub["RSI_entry"] > 35) & (sub["RSI_entry"] <= 45)]
        l_pnl = low["PnL_pct"].sum()  if len(low) > 0 else float("nan")
        m_pnl = mid["PnL_pct"].sum()  if len(mid) > 0 else float("nan")
        l_wr  = (low["PnL_pct"] > 0).mean() * 100 if len(low) > 0 else float("nan")
        m_wr  = (mid["PnL_pct"] > 0).mean() * 100 if len(mid) > 0 else float("nan")
        l_str = f"{l_pnl:>+.1f}%(n={len(low)})" if not pd.isna(l_pnl) else "  —"
        m_str = f"{m_pnl:>+.1f}%(n={len(mid)})" if not pd.isna(m_pnl) else "  —"
        l_wr_str = f"{l_wr:.1f}%" if not pd.isna(l_wr) else "—"
        m_wr_str = f"{m_wr:.1f}%" if not pd.isna(m_wr) else "—"
        print(f"  {yr:<6} | {l_str:>12} | {m_str:>14} | {l_wr_str:>10} | {m_wr_str:>12}")

    # ── Exit-Grund-Verteilung ────────────────────────────────────────────────
    print(f"\n  Exit-Grund-Verteilung (Gesamt):")
    print(f"  {'Exit':<12} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'ØW%':>7} | {'ØL%':>7} | {'Ø-Dauer h':>10}")
    print(f"  {'-'*64}")
    bs["Dauer_h"] = (pd.to_datetime(bs["Exit_ts"]) - pd.to_datetime(bs["Entry_ts"])).dt.total_seconds() / 3600
    for reason in sorted(bs["Exit_reason"].dropna().unique()):
        sub  = bs[bs["Exit_reason"] == reason]
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        ow   = sub[sub["PnL_pct"] > 0]["PnL_pct"].mean() if (sub["PnL_pct"] > 0).any() else float("nan")
        ol   = sub[sub["PnL_pct"] < 0]["PnL_pct"].mean() if (sub["PnL_pct"] < 0).any() else float("nan")
        dur  = sub["Dauer_h"].mean()
        ow_s = f"{ow:>+6.1f}%" if not pd.isna(ow) else "    n/a"
        ol_s = f"{ol:>+6.1f}%" if not pd.isna(ol) else "    n/a"
        print(f"  {reason:<12} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}% | {ow_s} | {ol_s} | {dur:>9.1f}h")

    # ── R:R-Gesamtstatistik ──────────────────────────────────────────────────
    wins  = bs[bs["PnL_pct"] > 0]
    loss  = bs[bs["PnL_pct"] < 0]
    aw    = wins["PnL_pct"].mean() if len(wins) > 0 else float("nan")
    al    = loss["PnL_pct"].mean() if len(loss) > 0 else float("nan")
    rr    = abs(aw / al) if (not pd.isna(aw) and not pd.isna(al) and al != 0) else float("nan")
    ev    = bs["PnL_pct"].mean()
    print(f"\n  R:R-Gesamtstatistik:")
    print(f"    EV={ev:+.2f}%  Ø-Win={aw:+.2f}%  Ø-Loss={al:+.2f}%  R:R={rr:.2f}")
    print(f"    TP-Rate={len(bs[bs['Exit_reason']=='TP'])/len(bs)*100:.1f}%  "
          f"Cover-Rate={len(bs[bs['Exit_reason']=='Cover'])/len(bs)*100:.1f}%  "
          f"SL-Rate={len(bs[bs['Exit_reason']=='SL'])/len(bs)*100:.1f}%")

    print()


# ─── Bulle-Long Diagnose ─────────────────────────────────────────────────────

def print_bulle_diag(trades_df: pd.DataFrame):
    bl = trades_df[
        (trades_df["Regime"] == "bulle") &
        (trades_df["Side"] == "long") &
        trades_df["Trigger_id"].notna()
    ].copy()
    if len(bl) == 0:
        return

    print(f"\n{'='*60}")
    print(f"  BULLE-LONG DIAGNOSE  (n={len(bl)})")
    print(f"{'='*60}")

    # ── Per Trigger ──────────────────────────────────────────────────────────
    print(f"\n  Per Trigger:")
    print(f"  {'Trig':<5} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'ØW%':>7} | {'ØL%':>7} | {'SL%':>6}")
    print(f"  {'-'*58}")
    for tid in ["A", "B", "C"]:
        sub = bl[bl["Trigger_id"] == tid]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        ow   = sub[sub["PnL_pct"] > 0]["PnL_pct"].mean() if (sub["PnL_pct"] > 0).any() else float("nan")
        ol   = sub[sub["PnL_pct"] < 0]["PnL_pct"].mean() if (sub["PnL_pct"] < 0).any() else float("nan")
        sl_p = (sub["Exit_reason"] == "SL").mean() * 100
        print(f"  {tid:<5} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}% | {ow:>+6.1f}% | {ol:>+6.1f}% | {sl_p:>5.1f}%")

    # ── Exit-Grund je Trigger ────────────────────────────────────────────────
    print(f"\n  Exit-Grund je Trigger (Anteil %):")
    reasons = bl["Exit_reason"].unique()
    header  = f"  {'Trig':<5}" + "".join(f" | {r:>10}" for r in sorted(reasons))
    print(header)
    print(f"  {'-'*len(header)}")
    for tid in ["A", "B", "C"]:
        sub = bl[bl["Trigger_id"] == tid]
        if len(sub) == 0:
            continue
        row_str = f"  {tid:<5}"
        for r in sorted(reasons):
            pct = (sub["Exit_reason"] == r).mean() * 100
            row_str += f" | {pct:>9.1f}%"
        print(row_str)

    # ── RSI-Buckets (gesamt + per Trigger) ──────────────────────────────────
    rsi_bins   = [38, 45, 55, 65, 70, 80]
    rsi_labels = ["38\u201345", "45\u201355", "55\u201365", "65\u201370", "70\u201380"]
    bl["RSI_bucket"] = pd.cut(bl["RSI_entry"], bins=rsi_bins, labels=rsi_labels, right=True)
    print(f"\n  WR nach RSI-Bucket:")
    print(f"  {'Bucket':<8} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'ØW%':>7} | {'ØL%':>7} | {'A':>4} | {'C':>4}")
    print(f"  {'-'*64}")
    for lb in rsi_labels:
        sub = bl[bl["RSI_bucket"] == lb]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        ow   = sub[sub["PnL_pct"] > 0]["PnL_pct"].mean() if (sub["PnL_pct"] > 0).any() else float("nan")
        ol   = sub[sub["PnL_pct"] < 0]["PnL_pct"].mean() if (sub["PnL_pct"] < 0).any() else float("nan")
        na   = (sub["Trigger_id"] == "A").sum()
        nc   = (sub["Trigger_id"] == "C").sum()
        print(f"  {lb:<8} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}% | {ow:>+6.1f}% | {ol:>+6.1f}% | {na:>4} | {nc:>4}")

    # ── Dist-EMA21-Buckets ───────────────────────────────────────────────────
    dist_bins   = [-10, 0, 2, 4, 6, 15]
    dist_labels = ["<0%", "0–2%", "2–4%", "4–6%", ">6%"]
    bl["Dist_bucket"] = pd.cut(bl["Dist_ema21_entry"], bins=dist_bins, labels=dist_labels, right=True)
    print(f"\n  WR nach Distanz EMA21 bei Entry:")
    print(f"  {'Distanz':<8} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*36}")
    for lb in dist_labels:
        sub = bl[bl["Dist_bucket"] == lb]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {lb:<8} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # ── Regime-Alter-Buckets ─────────────────────────────────────────────────
    age_bins   = [0, 10, 30, 9999]
    age_labels = ["<10", "10–30", "30+"]
    bl["Age_bucket"] = pd.cut(bl["Regime_age_entry"], bins=age_bins, labels=age_labels, right=True)
    print(f"\n  WR nach Regime-Alter bei Entry:")
    print(f"  {'Alter':<8} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*36}")
    for lb in age_labels:
        sub = bl[bl["Age_bucket"] == lb]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {lb:<8} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # ── Jahres-Aufschlüsselung RSI 65-70 ────────────────────────────────────
    bl["Year"] = pd.to_datetime(bl["Entry_ts"]).dt.year
    rsi_top = bl[bl["RSI_entry"] > 65]
    print(f"\n  RSI 65-70 Trades nach Jahr:")
    print(f"  {'Jahr':<6} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*34}")
    for yr in sorted(bl["Year"].unique()):
        sub = rsi_top[rsi_top["Year"] == yr]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {yr:<6} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # ── Jahres-Aufschlüsselung alle Bulle-Long-Trades ───────────────────────
    print(f"\n  Alle Bulle-Long nach Jahr:")
    print(f"  {'Jahr':<6} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*34}")
    for yr in sorted(bl["Year"].unique()):
        sub = bl[bl["Year"] == yr]
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {yr:<6} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # ── Extrem-Bulle nach Jahr ───────────────────────────────────────────────
    eb = trades_df[
        (trades_df["Regime"] == "extrem_bulle") &
        (trades_df["Side"] == "long")
    ].copy()
    if len(eb) > 0:
        eb["Year"] = pd.to_datetime(eb["Entry_ts"]).dt.year
        print(f"\n  Extrem-Bulle-Long nach Jahr (n={len(eb)}):")
        print(f"  {'Jahr':<6} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
        print(f"  {'-'*34}")
        for yr in sorted(eb["Year"].unique()):
            sub = eb[eb["Year"] == yr]
            wr   = (sub["PnL_pct"] > 0).mean() * 100
            spnl = sub["PnL_pct"].sum()
            print(f"  {yr:<6} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    print()


# ─── Extrem-Bulle Diagnose ───────────────────────────────────────────────────

def print_extrem_bulle_diag(trades_df: pd.DataFrame):
    eb = trades_df[
        (trades_df["Regime"] == "extrem_bulle") &
        (trades_df["Side"] == "long") &
        trades_df["ADX_entry"].notna()
    ].copy()
    if len(eb) == 0:
        print("\n  Keine Extrem-Bulle-Long-Trades mit Diagnostic-Daten.")
        return

    eb["Year"] = pd.to_datetime(eb["Entry_ts"]).dt.year

    print(f"\n{'='*60}")
    print(f"  EXTREM-BULLE DIAGNOSE  (n={len(eb)})")
    print(f"{'='*60}")

    # ── MA200_dist-Buckets (Schlüssel-Indikator) ─────────────────────────────
    ma_bins   = [-200, 0, 20, 40, 80, 150, 500]
    ma_labels = ["<0%", "0–20%", "20–40%", "40–80%", "80–150%", ">150%"]
    eb["MA200_bucket"] = pd.cut(eb["Dist_ema21_entry"], bins=ma_bins, labels=ma_labels, right=True)
    print(f"\n  WR nach MA200-Distanz bei Entry:")
    print(f"  {'Distanz':<9} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'ØW%':>7} | {'ØL%':>7}")
    print(f"  {'-'*56}")
    for lb in ma_labels:
        sub = eb[eb["MA200_bucket"] == lb]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        ow   = sub[sub["PnL_pct"] > 0]["PnL_pct"].mean() if (sub["PnL_pct"] > 0).any() else float("nan")
        ol   = sub[sub["PnL_pct"] < 0]["PnL_pct"].mean() if (sub["PnL_pct"] < 0).any() else float("nan")
        print(f"  {lb:<9} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}% | {ow:>+6.1f}% | {ol:>+6.1f}%")

    # ── Vola_ratio-Buckets ───────────────────────────────────────────────────
    vr_bins   = [0, 1.0, 1.5, 2.0, 3.0, 100]
    vr_labels = ["<1.0", "1.0–1.5", "1.5–2.0", "2.0–3.0", ">3.0"]
    eb["Vola_bucket"] = pd.cut(eb["Vola_ratio_entry"], bins=vr_bins, labels=vr_labels, right=True)
    print(f"\n  WR nach Vola-Ratio bei Entry:")
    print(f"  {'Vola':<9} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*36}")
    for lb in vr_labels:
        sub = eb[eb["Vola_bucket"] == lb]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {lb:<9} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # ── ADX-Buckets ──────────────────────────────────────────────────────────
    adx_bins   = [0, 20, 30, 40, 50, 100]
    adx_labels = ["<20", "20–30", "30–40", "40–50", ">50"]
    eb["ADX_bucket"] = pd.cut(eb["ADX_entry"], bins=adx_bins, labels=adx_labels, right=True)
    print(f"\n  WR nach ADX bei Entry:")
    print(f"  {'ADX':<8} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*34}")
    for lb in adx_labels:
        sub = eb[eb["ADX_bucket"] == lb]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {lb:<8} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # ── RSI-Buckets ──────────────────────────────────────────────────────────
    rsi_bins   = [0, 50, 65, 72, 80, 90, 100]
    rsi_labels = ["<50", "50–65", "65–72", "72–80", "80–90", ">90"]
    eb["RSI_bucket"] = pd.cut(eb["RSI_entry"], bins=rsi_bins, labels=rsi_labels, right=True)
    print(f"\n  WR nach RSI bei Entry:")
    print(f"  {'RSI':<8} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*34}")
    for lb in rsi_labels:
        sub = eb[eb["RSI_bucket"] == lb]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {lb:<8} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # ── MA200_dist-Jahres-Kreuzanalyse ───────────────────────────────────────
    print(f"\n  MA200_dist nach Jahr (Median & Max):")
    print(f"  {'Jahr':<6} | {'N':>4} | {'Median MA200%':>13} | {'Max MA200%':>10} | {'WR':>6}")
    print(f"  {'-'*50}")
    for yr in sorted(eb["Year"].unique()):
        sub = eb[eb["Year"] == yr]
        if "Dist_ema21_entry" in sub.columns and sub["Dist_ema21_entry"].notna().any():
            med  = sub["Dist_ema21_entry"].median()
            mx   = sub["Dist_ema21_entry"].max()
            wr   = (sub["PnL_pct"] > 0).mean() * 100
            print(f"  {yr:<6} | {len(sub):>4} | {med:>+12.1f}% | {mx:>+9.1f}% | {wr:>5.1f}%")

    # ── Jahr × RSI Cross-Tab ─────────────────────────────────────────────────
    print(f"\n  Jahr × RSI-Bucket Cross-Tab (N / ΣPnL%):")
    # Labels müssen exakt mit pd.cut-Ausgabe übereinstimmen (En-Dash)
    xt_labels = ["<50", "50\u201365", "65\u201372", "72\u201380", "80\u201390"]
    header = f"  {'Jahr':<6} | {'N':>4} | {'WR':>6}"
    for rc in xt_labels:
        header += f" | {rc:>8}"
    print(header)
    print(f"  {'-'*70}")
    for yr in sorted(eb["Year"].unique()):
        sub = eb[eb["Year"] == yr]
        wr  = (sub["PnL_pct"] > 0).mean() * 100
        row_str = f"  {yr:<6} | {len(sub):>4} | {wr:>5.1f}%"
        for lb in xt_labels:
            count = (sub["RSI_bucket"] == lb).sum() if "RSI_bucket" in sub.columns else 0
            row_str += f" | {count:>8}"
        print(row_str)

    # ── Exit-Diagnose ─────────────────────────────────────────────────────────
    print(f"\n  Exit-Diagnose (Exit-Grund-Verteilung):")
    print(f"  {'Exit':<12} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'ØW%':>7} | {'ØL%':>7}")
    print(f"  {'-'*56}")
    for reason in sorted(eb["Exit_reason"].dropna().unique()):
        sub  = eb[eb["Exit_reason"] == reason]
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        ow   = sub[sub["PnL_pct"] > 0]["PnL_pct"].mean() if (sub["PnL_pct"] > 0).any() else float("nan")
        ol   = sub[sub["PnL_pct"] < 0]["PnL_pct"].mean() if (sub["PnL_pct"] < 0).any() else float("nan")
        print(f"  {reason:<12} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}% | {ow:>+6.1f}% | {ol:>+6.1f}%")

    # Exit-Diagnose nach Jahr
    print(f"\n  Exit-Diagnose nach Jahr (SL-Anteil %):")
    print(f"  {'Jahr':<6} | {'N':>4} | {'WR':>6} | {'SL%':>6} | {'TP%':>6} | {'Trail%':>7} | {'ΣPnL%':>8}")
    print(f"  {'-'*56}")
    for yr in sorted(eb["Year"].unique()):
        sub  = eb[eb["Year"] == yr]
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        sl_p = (sub["Exit_reason"] == "SL").mean() * 100
        tp_p = (sub["Exit_reason"] == "TP").mean() * 100
        tr_p = (sub["Exit_reason"] == "Trail").mean() * 100
        print(f"  {yr:<6} | {len(sub):>4} | {wr:>5.1f}% | {sl_p:>5.1f}% | {tp_p:>5.1f}% | {tr_p:>6.1f}% | {spnl:>+7.1f}%")

    # ── MA500 Filter-Analyse (Regime_age_entry = 1 wenn close > MA500) ────────
    if "Regime_age_entry" in eb.columns and eb["Regime_age_entry"].notna().any():
        above = eb[eb["Regime_age_entry"] == 1]
        below = eb[eb["Regime_age_entry"] == 0]
        print(f"\n  MA500-Check (close vs. MA500 bei Entry):")
        print(f"  {'Status':<15} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
        print(f"  {'-'*40}")
        if len(above) > 0:
            wr_a = (above["PnL_pct"] > 0).mean() * 100
            print(f"  {'close > MA500':<15} | {len(above):>4} | {wr_a:>5.1f}% | {above['PnL_pct'].sum():>+7.1f}%")
        if len(below) > 0:
            wr_b = (below["PnL_pct"] > 0).mean() * 100
            print(f"  {'close < MA500':<15} | {len(below):>4} | {wr_b:>5.1f}% | {below['PnL_pct'].sum():>+7.1f}%")
        # Nach Jahr aufschlüsseln für close < MA500
        if len(below) > 0:
            below_y = below.copy()
            below_y["Year2"] = pd.to_datetime(below_y["Entry_ts"]).dt.year
            print(f"\n  Close < MA500 nach Jahr:")
            for yr2 in sorted(below_y["Year2"].unique()):
                sub2 = below_y[below_y["Year2"] == yr2]
                wr2  = (sub2["PnL_pct"] > 0).mean() * 100
                print(f"    {yr2}: n={len(sub2)}, WR={wr2:.1f}%, ΣPnL={sub2['PnL_pct'].sum():+.1f}%")

    print()


# ─── SW-Long Diagnose ────────────────────────────────────────────────────────

def print_sw_long_diag(trades_df: pd.DataFrame):
    sw = trades_df[
        (trades_df["Regime"] == "seitwaerts") &
        (trades_df["Side"]   == "long") &
        trades_df["RSI_entry"].notna()
    ].copy()
    if len(sw) == 0:
        print("  Keine SW-Long-Trades mit Diagnose-Daten.")
        return

    wr_all  = (sw["PnL_pct"] > 0).mean() * 100
    ev_all  = sw["PnL_pct"].mean()
    print(f"\n{'='*60}")
    print(f"  SW-LONG DIAGNOSE  (n={len(sw)}, WR={wr_all:.1f}%, EV={ev_all:+.2f}%)")
    print(f"{'='*60}")

    # RSI-Buckets
    rsi_bins   = [0, 25, 30, 35, 40, 45, 55]
    rsi_labels = ["<25", "25-30", "30-35", "35-40", "40-45", "45-55"]
    sw["RSI_bucket"] = pd.cut(sw["RSI_entry"], bins=rsi_bins, labels=rsi_labels, right=True)
    print(f"\n  WR nach RSI-Bucket bei Entry:")
    print(f"  {'Bucket':<8} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'ØW%':>7} | {'ØL%':>7}")
    print(f"  {'-'*52}")
    for lb in rsi_labels:
        sub = sw[sw["RSI_bucket"] == lb]
        if len(sub) == 0:
            continue
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        ow   = sub[sub["PnL_pct"] > 0]["PnL_pct"].mean() if (sub["PnL_pct"] > 0).any() else float("nan")
        ol   = sub[sub["PnL_pct"] < 0]["PnL_pct"].mean() if (sub["PnL_pct"] < 0).any() else float("nan")
        print(f"  {lb:<8} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}% | {ow:>+6.1f}% | {ol:>+6.1f}%")

    # Score-Kombination (Trigger_id)
    print(f"\n  WR nach Score-Kombination (Entry-Bedingungen):")
    print(f"  {'Kombo':<12} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*36}")
    for combo in sorted(sw["Trigger_id"].dropna().unique()):
        sub  = sw[sw["Trigger_id"] == combo]
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {combo:<12} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # OBV/RSI_div-Bitmask (Regime_age_entry: 0=keiner, 1=div, 2=obv, 3=div+obv)
    mask_labels = {0: "keiner", 1: "div_only", 2: "obv_only", 3: "div+obv"}
    if sw["Regime_age_entry"].notna().any():
        print(f"\n  WR nach OBV/RSI_div-Kombo:")
        print(f"  {'Kombo':<10} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
        print(f"  {'-'*34}")
        for mval, mname in mask_labels.items():
            sub = sw[sw["Regime_age_entry"] == mval]
            if len(sub) == 0:
                continue
            wr   = (sub["PnL_pct"] > 0).mean() * 100
            spnl = sub["PnL_pct"].sum()
            print(f"  {mname:<10} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # Distanz BB_lower
    if sw["Dist_ema21_entry"].notna().any():
        dist_bins   = [-15, -3, -1.5, -0.5, 0, 2]
        dist_labels = ["<-3%", "-3–-1.5%", "-1.5–-0.5%", "-0.5–0%", ">0%"]
        sw["Dist_bucket"] = pd.cut(sw["Dist_ema21_entry"], bins=dist_bins, labels=dist_labels, right=True)
        print(f"\n  WR nach Distanz BB_lower (neg=unter BB_lower):")
        print(f"  {'Distanz':<12} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
        print(f"  {'-'*38}")
        for lb in dist_labels:
            sub = sw[sw["Dist_bucket"] == lb]
            if len(sub) == 0:
                continue
            wr   = (sub["PnL_pct"] > 0).mean() * 100
            spnl = sub["PnL_pct"].sum()
            print(f"  {lb:<12} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # Exit-Grund
    print(f"\n  Exit-Grund-Verteilung:")
    print(f"  {'Exit':<12} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'ØW%':>7} | {'ØL%':>7}")
    print(f"  {'-'*56}")
    for reason in sorted(sw["Exit_reason"].dropna().unique()):
        sub  = sw[sw["Exit_reason"] == reason]
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        ow   = sub[sub["PnL_pct"] > 0]["PnL_pct"].mean() if (sub["PnL_pct"] > 0).any() else float("nan")
        ol   = sub[sub["PnL_pct"] < 0]["PnL_pct"].mean() if (sub["PnL_pct"] < 0).any() else float("nan")
        print(f"  {reason:<12} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}% | {ow:>+6.1f}% | {ol:>+6.1f}%")

    # Jahresaufschlüsselung
    sw["Year"] = pd.to_datetime(sw["Entry_ts"]).dt.year
    print(f"\n  SW-Long nach Jahr:")
    print(f"  {'Jahr':<6} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
    print(f"  {'-'*34}")
    for yr in sorted(sw["Year"].unique()):
        sub  = sw[sw["Year"] == yr]
        wr   = (sub["PnL_pct"] > 0).mean() * 100
        spnl = sub["PnL_pct"].sum()
        print(f"  {yr:<6} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

    # Haltezeit-Verteilung
    sw["Dauer_h"] = (pd.to_datetime(sw["Exit_ts"]) - pd.to_datetime(sw["Entry_ts"])).dt.total_seconds() / 3600
    winners = sw[sw["PnL_pct"] > 0]
    losers  = sw[sw["PnL_pct"] <= 0]
    print(f"\n  Haltezeit (Stunden):")
    print(f"  {'Gruppe':<10} | {'N':>4} | {'Median h':>9} | {'Ø h':>7} | {'Min h':>7} | {'Max h':>7}")
    print(f"  {'-'*52}")
    for grp, sub in [("Winner", winners), ("Loser", losers), ("Gesamt", sw)]:
        if len(sub) == 0:
            continue
        print(f"  {grp:<10} | {len(sub):>4} | {sub['Dauer_h'].median():>8.1f}h | "
              f"{sub['Dauer_h'].mean():>6.1f}h | {sub['Dauer_h'].min():>6.1f}h | {sub['Dauer_h'].max():>6.1f}h")

    # Winner vs. Loser Median-Vergleich
    print(f"\n  Winner (n={len(winners)}) vs. Loser (n={len(losers)}) — Median:")
    print(f"  {'Metrik':<20} | {'Winner':>10} | {'Loser':>10}")
    print(f"  {'-'*46}")
    for col, label in [("RSI_entry", "RSI"),
                       ("Dist_ema21_entry", "Dist BB_lower%"),
                       ("ADX_entry", "ADX"),
                       ("Vola_ratio_entry", "Vola_ratio"),
                       ("Dauer_h", "Haltezeit (h)")]:
        if col in sw.columns and sw[col].notna().any():
            wm = winners[col].median() if len(winners) > 0 else float("nan")
            lm = losers[col].median()  if len(losers)  > 0 else float("nan")
            ws = f"{wm:>+.1f}" if not pd.isna(wm) else "n/a"
            ls = f"{lm:>+.1f}" if not pd.isna(lm) else "n/a"
            print(f"  {label:<20} | {ws:>10} | {ls:>10}")

    print()


# ─── Regime-Detail-Analyse ───────────────────────────────────────────────────

def print_regime_detail(trades_df: pd.DataFrame, startkapital: float = 200.0):
    """Vollständige Regime-Aufschlüsselung: Trades, WR, ΣPnL%, $-Gewinn."""
    if len(trades_df) == 0:
        print("  Keine Trades.")
        return

    regime_order = ["extrem_bulle", "bulle", "seitwaerts", "neutral", "baer", "extrem_baer"]
    regime_label = {
        "extrem_bulle": "Extrem-Bulle",
        "bulle":        "Bulle",
        "seitwaerts":   "Seitwärts",
        "neutral":      "Neutral",
        "baer":         "Bär",
        "extrem_baer":  "Extrem-Bär",
    }

    print(f"\n{'='*75}")
    print(f"  REGIME-DETAILANALYSE  (Startkapital: ${startkapital:.0f})")
    print(f"  Hinweis: $-Beitrag = realisierter Gewinn je Trade (mit Compounding), Summe ≈ Endkapital−${startkapital:.0f}")
    print(f"{'='*75}")
    print(f"  {'Regime':<14} | {'Side':<5} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'$-Beitrag':>10}")
    print(f"  {'-'*65}")

    total_dollar = 0.0
    for reg in regime_order:
        for side in ["long", "short"]:
            sub = trades_df[(trades_df["Regime"] == reg) & (trades_df["Side"] == side)]
            if len(sub) == 0:
                continue
            wr      = (sub["PnL_pct"] > 0).mean() * 100
            spnl    = sub["PnL_pct"].sum()
            dollar  = (sub["PnL_pct"] / 100 * sub["Entry_cash"]).sum() if "Entry_cash" in sub.columns else float("nan")
            total_dollar += dollar if not pd.isna(dollar) else 0
            label   = regime_label.get(reg, reg)
            d_str   = f"${dollar:>+.2f}" if not pd.isna(dollar) else "   n/a"
            print(f"  {label:<14} | {side:<5} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}% | {d_str:>10}")

    print(f"  {'-'*65}")
    print(f"  {'GESAMT':<14} | {'':5} | {len(trades_df):>4} | {'':6} | {'':8} | ${total_dollar:>+9.2f}")
    print(f"  Sanity-Check: Σ$-Beiträge = ${total_dollar:>+.2f}  |  Erwartetes Endkapital ≈ ${startkapital + total_dollar:.2f}")

    # Jahresaufschlüsselung je Regime
    trades_df = trades_df.copy()
    trades_df["Year"] = pd.to_datetime(trades_df["Entry_ts"]).dt.year

    print(f"\n  JAHRESAUFSCHLÜSSELUNG JE REGIME:")
    for reg in regime_order:
        sub_reg = trades_df[trades_df["Regime"] == reg]
        if len(sub_reg) == 0:
            continue
        label = regime_label.get(reg, reg)
        print(f"\n  {label}:")
        print(f"  {'Jahr':<6} | {'Side':<5} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'$-Gewinn':>10}")
        print(f"  {'-'*52}")
        for yr in sorted(sub_reg["Year"].unique()):
            for side in ["long", "short"]:
                sub = sub_reg[(sub_reg["Year"] == yr) & (sub_reg["Side"] == side)]
                if len(sub) == 0:
                    continue
                wr     = (sub["PnL_pct"] > 0).mean() * 100
                spnl   = sub["PnL_pct"].sum()
                dollar = (sub["PnL_pct"] / 100 * sub["Entry_cash"]).sum() if "Entry_cash" in sub.columns else float("nan")
                d_str  = f"${dollar:>+.2f}" if not pd.isna(dollar) else "   n/a"
                print(f"  {yr:<6} | {side:<5} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}% | {d_str:>10}")
    print()


def _print_block_diag(blocked_df, trades_df):
    """Zeigt blockierte Signale gruppiert nach Positionskontext und blockiertem Signal."""
    import pandas as pd

    # Bekannte WR je Regime/Side aus tatsächlichen Trades
    wr_lookup = {}
    if len(trades_df) > 0:
        for (reg, side), grp in trades_df.groupby(["Regime", "Side"]):
            wr_lookup[(reg, side)] = (len(grp), (grp["PnL_pct"] > 0).mean() * 100)

    total_blocked = len(blocked_df)
    same_regime   = (blocked_df["pos_regime"] == blocked_df["blocked_regime"]).sum()
    diff_regime   = total_blocked - same_regime

    print(f"\n{'='*70}")
    print(f"  BLOCKIERTE SIGNALE — Single-Position-Diagnose")
    print(f"{'='*70}")
    print(f"  Gesamt blockiert (anderes Regime/Side): {diff_regime}")
    print(f"  Davon gleicher Regime (ignoriert):      {same_regime}")
    print()
    print(f"  {'Aktive Pos.':>20} | {'Blockiertes Signal':>22} | {'N geblockt':>10} | {'Hist. WR':>8}")
    print(f"  {'-'*68}")

    cross = blocked_df[blocked_df["pos_regime"] != blocked_df["blocked_regime"]]
    grouped = cross.groupby(["pos_regime","pos_side","blocked_regime","blocked_side"]).size().reset_index(name="count")
    grouped = grouped.sort_values("count", ascending=False)

    for _, row in grouped.iterrows():
        pos_lbl  = f"{row['pos_regime']} {row['pos_side']}"
        blk_lbl  = f"{row['blocked_regime']} {row['blocked_side']}"
        n        = row["count"]
        hist_wr  = wr_lookup.get((row["blocked_regime"], row["blocked_side"]))
        wr_s     = f"{hist_wr[1]:.1f}% (n={hist_wr[0]})" if hist_wr else "unbekannt"
        print(f"  {pos_lbl:>20} | {blk_lbl:>22} | {n:>10} | {wr_s:>8}")
    print()
    pct = diff_regime / len(trades_df) * 100 if len(trades_df) > 0 else 0
    print(f"  → {diff_regime} blockierte Signale = {pct:.1f}% der tatsächlich genommenen Trades")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SOL Bot Backtest")
    parser.add_argument("--mode", choices=["v13", "rsi_div", "off", "v15",
                                           "btc_trend", "funding_rate", "btc_funding",
                                           "daily_bull", "fear_greed", "sol_btc_rs",
                                           "funding_daily", "fear_daily",
                                           "neutral_sw_short", "neutral_sw_full", "neutral_rsi_ext",
                                           ],
                        default="neutral_sw_short",
                        help="Seitwärts-Modus (alle Modi)")
    parser.add_argument("--ab",   action="store_true",
                        help="A/B-Test: alle 4 Seitwärts-Modi vergleichen")
    parser.add_argument("--diag", action="store_true",
                        help="Bulle-Long-Diagnose: Trigger A/B/C + RSI/Dist/Alter-Buckets")
    parser.add_argument("--dd", action="store_true",
                        help="Drawdown-Diagnose: Peak/Trough-Lokalisierung + Trade-Attribution")
    parser.add_argument("--eb_diag", action="store_true",
                        help="Extrem-Bulle-Diagnose: MA200-dist, Vola-Ratio, ADX, RSI-Buckets")
    parser.add_argument("--pos_sweep", action="store_true",
                        help="Position-Size-Sweep: teste 10%%-100%% Kapitaleinsatz")
    parser.add_argument("--regime_detail", action="store_true",
                        help="Regime-Detailanalyse mit $-Gewinn je Regime/Jahr")
    parser.add_argument("--sw_compare", action="store_true",
                        help="SW-Long Vergleichstabelle: alle Makro-Filter-Ansätze vs. Baseline")
    parser.add_argument("--sw_diag", action="store_true",
                        help="SW-Long Diagnose: RSI-Buckets, Score-Kombo, OBV/div, Winner/Loser-Split")
    parser.add_argument("--sw_sl_sweep", action="store_true",
                        help="SW-Long SL-Sweep: teste ATR-Multiplikatoren 1.0×–3.0× für Seitwärts-Long")
    parser.add_argument("--sw_grid", action="store_true",
                        help="SW-Long 2D-Grid: Phase A SL×TP, Phase B beste Combos × Hebel")
    parser.add_argument("--block_diag", action="store_true",
                        help="Diagnose: Wie viele Signale werden durch Single-Position blockiert?")
    parser.add_argument("--baer_lev_sweep", action="store_true",
                        help="Bär-Short Hebel-Sweep: teste 2× bis 6× für Bär-Shorts")
    parser.add_argument("--eb_trail_sweep", action="store_true",
                        help="EB Trail-Widening-Sweep: teste 1.2× bis 3.0× für Extrem-Bulle Trail-Multiplikator")
    parser.add_argument("--eb_grid", action="store_true",
                        help="EB No-Trail SL×TP-Grid: 4×4 Grid (SL ∈ {1.5,2.0,2.5,3.0} × TP ∈ {5,7,10,None})")
    parser.add_argument("--bulle_sweep", action="store_true",
                        help="Bulle-Long 3-Methoden-Sweep: default vs. strict (cross_ema21_up) vs. off vs. dip-buy")
    parser.add_argument("--rsi3_diag", action="store_true",
                        help="RSI3 Diagnose: Exit-Typ × Win/Loss × Jahr-Aufschlüsselung")
    parser.add_argument("--neutral_sweep", action="store_true",
                        help="Neutral-Regime-Sweep: SW-Short / SW-Full / RSI-Extreme in neutral")
    parser.add_argument("--sw_adx_diag", action="store_true",
                        help="SW-Long ADX-Diagnose: Winner vs. Loser ADX-Verteilung + Bucket-Analyse")
    parser.add_argument("--wf", action="store_true",
                        help="Konsistenz-Probe (Pseudo-OOS): Performance per Zeitfenster + Naive-OOS (rsi3 vs default auf 2024+)")
    parser.add_argument("--diag2026", action="store_true",
                        help="2026-Regime-Diagnose: Regime/Monat, Preisbewegung, alle 2026-Trades im Detail")
    parser.add_argument("--data", choices=["synthetic", "real", "daily", "binance"], default="synthetic",
                        help="Datenquelle: synthetic (seed=42) | real (Yahoo 1h, 730 Tage) | daily (Yahoo 1d, seit 2020) | binance (Binance 1h, seit Nov 2020)")
    args = parser.parse_args()

    if args.data == "binance":
        print("Lade echte SOL/USDT 1h-Daten von Binance (seit Jan 2023, ~29.000 Kerzen) ...")
        df_raw = fetch_binance_data(start="2021-09-01")
        data_label = "Binance (1h, Apr 2024–heute)"
    elif args.data == "real":
        print("Lade echte SOL/USD 1h-Daten von Yahoo Finance (letzten 730 Tage) ...")
        df_raw = fetch_real_data(interval="1h")
        data_label = "Yahoo (1h, 730 Tage)"
    elif args.data == "daily":
        print("Lade echte SOL/USD Tagesdaten von Yahoo Finance (seit Apr 2020) ...")
        df_raw = fetch_real_data(interval="1d")
        data_label = "Yahoo (1d, seit 2020)"
    else:
        print("Erstelle synthetische Preisdaten (seed=42) ...")
        df_raw = make_price_data()
        data_label = "Synthetisch (seed=42)"

    print(f"  {len(df_raw)} Kerzen, {df_raw.index[0].date()} – {df_raw.index[-1].date()}  [{data_label}]")

    print("Berechne Indikatoren (kann 1-2 Minuten dauern) ...")

    if args.sw_compare:
        # ── SW-Long Makro-Filter Vergleich ──────────────────────────────────
        print("\nLade BTC-Daten ...")
        btc_df      = fetch_btc_data(start="2021-08-20")
        btc_ma200   = btc_df["BTC_ma200_bull"]
        btc_close   = btc_df["BTC_Close"]
        print("Lade Funding-Rate-Daten ...")
        funding_s   = fetch_funding_rate(start="2021-09-01")
        print("Lade Fear & Greed Index ...")
        fg_s        = fetch_fear_greed()
        print("Berechne Daily Macro Bull ...")
        daily_s     = compute_daily_macro(df_raw)
        print("Berechne SOL/BTC Relative Strength ...")
        sol_btc_s   = compute_sol_btc_rs(df_raw, btc_close)

        compare_modes = [
            ("Baseline: rsi_div (SW-Long aus)",  "rsi_div",       {}),
            ("v13 (alle SW-Longs, ungefiltert)", "v13",            {}),
            ("BTC-Trend (BTC > MA200)",           "btc_trend",     {"BTC_ma200_bull": btc_ma200}),
            ("Funding-Rate (Rate < 0)",           "funding_rate",  {"Funding_bull": funding_s}),
            ("BTC + Funding",                     "btc_funding",   {"BTC_ma200_bull": btc_ma200,
                                                                     "Funding_bull": funding_s}),
            ("Daily Macro Bull (EMA21>EMA50 1d)", "daily_bull",    {"Daily_macro_bull": daily_s}),
            ("Fear & Greed (F&G >= 40)",          "fear_greed",    {"FG_ok": fg_s}),
            ("SOL/BTC Rel. Strength",             "sol_btc_rs",    {"SOL_BTC_rs_bull": sol_btc_s}),
            ("Funding + Daily Bull",              "funding_daily", {"Funding_bull": funding_s,
                                                                     "Daily_macro_bull": daily_s}),
            ("F&G + Daily Bull",                  "fear_daily",    {"FG_ok": fg_s,
                                                                     "Daily_macro_bull": daily_s}),
        ]

        # Ergebnisse sammeln (einmalig je Ansatz)
        compare_results = []
        all_trades_by_label = {}
        for label, mode, extra in compare_modes:
            print(f"  Starte: {label} ...")
            trades, equity, df_full, _ = run_backtest(df_raw.copy(), sw_mode=mode,
                                                    extra_signals=extra if extra else None,
                                                    bulle_mode="rsi3",
                                                    baer_lev_override=6.0,
                                                    bulle_lev_override=1.0,
                                                    bulle_tp_mult=10.0,
                                                    bulle_trail_mult=4.0,
                                                    extrem_bulle_lev_override=2.0,
                                                    extrem_bulle_tp_mult=10.0)
            all_trades_by_label[label] = trades
            final   = equity.iloc[-1]
            r       = (final - STARTKAPITAL) / STARTKAPITAL * 100
            sr      = sharpe(equity)
            dd      = max_drawdown(equity)
            bah_ret = (df_full["Close"].iloc[-1] / df_full["Close"].iloc[0] - 1) * 100
            outperf = r - bah_ret

            sw_long  = trades[(trades["Regime"] == "seitwaerts") & (trades["Side"] == "long")]  if len(trades) > 0 else pd.DataFrame()
            sw_short = trades[(trades["Regime"] == "seitwaerts") & (trades["Side"] == "short")] if len(trades) > 0 else pd.DataFrame()
            sw_n_l  = len(sw_long)
            sw_n_s  = len(sw_short)
            sw_wr_l = (sw_long["PnL_pct"] > 0).mean() * 100 if sw_n_l > 0 else float("nan")
            sw_ev_l = sw_long["PnL_pct"].mean()            if sw_n_l > 0 else float("nan")
            sw_sp_l = sw_long["PnL_pct"].sum()             if sw_n_l > 0 else float("nan")

            # Jahr-Konsistenz SW-Long: in wie vielen Jahren ΣPnL positiv?
            yr_consist = 0
            yr_count   = 0
            if sw_n_l > 0:
                sw_long_c = sw_long.copy()
                sw_long_c["Year"] = pd.to_datetime(sw_long_c["Entry_ts"]).dt.year
                for yr in sorted(sw_long_c["Year"].unique()):
                    sub  = sw_long_c[sw_long_c["Year"] == yr]
                    spnl = sub["PnL_pct"].sum()
                    yr_count  += 1
                    if spnl > 0:
                        yr_consist += 1

            compare_results.append({
                "label":      label,
                "final":      final,
                "rendite":    r,
                "outperf":    outperf,
                "sharpe":     sr,
                "maxdd":      dd,
                "sw_n_l":     sw_n_l,
                "sw_n_s":     sw_n_s,
                "sw_wr_l":    sw_wr_l,
                "sw_ev_l":    sw_ev_l,
                "sw_sp_l":    sw_sp_l,
                "yr_consist": f"{yr_consist}/{yr_count}" if yr_count > 0 else "—",
            })

        # Jahres-Detail je Ansatz ausgeben
        print("\n" + "="*75)
        print("  SW-LONG JAHRESDETAIL JE ANSATZ")
        print("="*75)
        for label, mode, extra in compare_modes:
            trades = all_trades_by_label[label]
            sw_long = trades[(trades["Regime"] == "seitwaerts") & (trades["Side"] == "long")] if len(trades) > 0 else pd.DataFrame()
            print(f"\n  {label} (SW-Long n={len(sw_long)}):")
            if len(sw_long) == 0:
                print("    — keine SW-Long-Trades —")
                continue
            sw_long = sw_long.copy()
            sw_long["Year"] = pd.to_datetime(sw_long["Entry_ts"]).dt.year
            print(f"    {'Jahr':<6} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
            print(f"    {'-'*30}")
            for yr in sorted(sw_long["Year"].unique()):
                sub  = sw_long[sw_long["Year"] == yr]
                wr   = (sub["PnL_pct"] > 0).mean() * 100
                spnl = sub["PnL_pct"].sum()
                print(f"    {yr:<6} | {len(sub):>4} | {wr:>5.1f}% | {spnl:>+7.1f}%")

        # Haupt-Vergleichstabelle
        print("\n" + "="*110)
        print("  SW-LONG MAKRO-FILTER VERGLEICH")
        print("="*110)
        hdr = (f"  {'Ansatz':<36} | {'Endkap.':>9} | {'Rendite':>8} | {'Outperf.':>9} | "
               f"{'Sharpe':>7} | {'MaxDD':>7} | {'SW-L N':>6} | {'SW-S N':>6} | "
               f"{'SW-L WR':>7} | {'SW-L EV':>7} | {'SW-L ΣPnL':>9} | {'J-Konst.':>8}")
        print(hdr)
        print(f"  {'-'*108}")
        for cr in compare_results:
            sw_wr_s = f"{cr['sw_wr_l']:>6.1f}%" if not pd.isna(cr['sw_wr_l']) else "   n/a"
            sw_ev_s = f"{cr['sw_ev_l']:>+6.2f}%" if not pd.isna(cr['sw_ev_l']) else "   n/a"
            sw_sp_s = f"{cr['sw_sp_l']:>+8.1f}%" if not pd.isna(cr['sw_sp_l']) else "     n/a"
            print(f"  {cr['label']:<36} | ${cr['final']:>8.2f} | {cr['rendite']:>+7.1f}% | "
                  f"{cr['outperf']:>+8.1f}% | {cr['sharpe']:>7.2f} | {cr['maxdd']:>+6.1f}% | "
                  f"{cr['sw_n_l']:>6} | {cr['sw_n_s']:>6} | {sw_wr_s:>7} | "
                  f"{sw_ev_s:>7} | {sw_sp_s:>9} | {cr['yr_consist']:>8}")
        print()
        return

    if args.ab:
        modes = [("v13 Baseline",              "v13"),
                 ("v14a RSI-Divergenz-Entry",   "rsi_div"),
                 ("v14b Seitwärts deaktiviert", "off"),
                 ("v15 SW verbessert",          "v15")]
    else:
        label_suffix = f"[{data_label}]"
        modes = [(f"{args.mode} {label_suffix}", args.mode)]

    if args.pos_sweep:
        print(f"\n{'='*60}")
        print(f"  POSITION-SIZE SWEEP (mode={args.mode})")
        print(f"{'='*60}")
        print(f"  {'PosSize':>8} | {'Endkap.':>9} | {'Rendite':>8} | {'Outperf':>8} | {'Sharpe':>7} | {'MaxDD':>7}")
        print(f"  {'-'*65}")
        sizes = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
        for ps in sizes:
            trades, equity, df_full, _ = run_backtest(df_raw.copy(), sw_mode=args.mode, max_pos_size=ps)
            final   = equity.iloc[-1]
            r       = (final - STARTKAPITAL) / STARTKAPITAL * 100
            bah_ret = (df_full["Close"].iloc[-1] / df_full["Close"].iloc[0] - 1) * 100
            out     = r - bah_ret
            sr      = sharpe(equity)
            dd      = max_drawdown(equity)
            print(f"  {ps*100:>7.0f}% | ${final:>8.2f} | {r:>+7.1f}% | {out:>+7.1f}% | {sr:>7.2f} | {dd:>+6.1f}%")
        print()
        return

    # Extra-Signale für Makro-Filter-Modi vorbereiten
    _macro_modes = {"btc_trend", "funding_rate", "btc_funding",
                    "daily_bull", "fear_greed", "sol_btc_rs",
                    "funding_daily", "fear_daily",
                    "neutral_sw_short", "neutral_sw_full", "neutral_rsi_ext"}
    _extra_cache: dict = {}
    if args.mode in _macro_modes:
        if args.mode in ("btc_trend", "btc_funding", "sol_btc_rs"):
            print("Lade BTC-Daten ...")
            btc_df_s  = fetch_btc_data(start="2021-08-20")
            _extra_cache["BTC_ma200_bull"] = btc_df_s["BTC_ma200_bull"]
            _extra_cache["BTC_Close"]      = btc_df_s["BTC_Close"]
        if args.mode in ("funding_rate", "btc_funding", "funding_daily"):
            print("Lade Funding-Rate-Daten ...")
            _extra_cache["Funding_bull"] = fetch_funding_rate(start="2021-09-01")
        if args.mode in ("daily_bull", "funding_daily", "fear_daily",
                         "neutral_sw_short", "neutral_sw_full", "neutral_rsi_ext"):
            print("Berechne Daily Macro Bull ...")
            _extra_cache["Daily_macro_bull"] = compute_daily_macro(df_raw)
        if args.mode in ("fear_greed", "fear_daily"):
            print("Lade Fear & Greed Index ...")
            _extra_cache["FG_ok"] = fetch_fear_greed()
        if args.mode == "sol_btc_rs":
            print("Berechne SOL/BTC Relative Strength ...")
            _extra_cache["SOL_BTC_rs_bull"] = compute_sol_btc_rs(df_raw, _extra_cache["BTC_Close"])

    if args.sw_sl_sweep:
        print(f"\n{'='*100}")
        print(f"  SW-LONG SL-SWEEP  (sw_mode=neutral_sw_short, Bär 6×, EB 2×, bulle=rsi3)")
        print(f"{'='*100}")
        print(f"  {'SL-Mult':>8} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | "
              f"{'SL-Rate':>7} | {'TP-Rate':>7} | {'Avg-Los':>8} | "
              f"{'Endkap.':>9} | {'Sharpe':>7} | {'MaxDD':>7}")
        print(f"  {'-'*95}")
        sl_results = []
        for sl_mult in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
            t, eq, _, _ = run_backtest(df_raw.copy(), sw_mode="neutral_sw_short",
                                       extra_signals=_extra_cache if _extra_cache else None,
                                       baer_lev_override=6.0,
                                       bulle_lev_override=1.0,
                                       extrem_bulle_lev_override=2.0,
                                       extrem_bulle_tp_mult=10.0,
                                       bulle_mode="rsi3",
                                       bulle_tp_mult=10.0,
                                       bulle_trail_mult=4.0,
                                       sw_sl_mult=sl_mult)
            swl = t[(t["Regime"] == "seitwaerts") & (t["Side"] == "long")] if len(t) > 0 else pd.DataFrame()
            swl_n  = len(swl)
            swl_wr = (swl["PnL_pct"] > 0).mean() * 100 if swl_n > 0 else float("nan")
            swl_sp = swl["PnL_pct"].sum() if swl_n > 0 else float("nan")
            sl_rate = (swl["Exit_reason"] == "SL").mean() * 100 if swl_n > 0 and "Exit_reason" in swl.columns else float("nan")
            tp_rate = (swl["Exit_reason"] == "TP").mean() * 100 if swl_n > 0 and "Exit_reason" in swl.columns else float("nan")
            avg_los = swl[swl["PnL_pct"] <= 0]["PnL_pct"].mean() if swl_n > 0 and (swl["PnL_pct"] <= 0).any() else float("nan")
            final  = eq.iloc[-1]
            sr     = sharpe(eq)
            dd     = max_drawdown(eq)
            wr_s   = f"{swl_wr:>5.1f}%" if not pd.isna(swl_wr) else "   n/a"
            sp_s   = f"{swl_sp:>+7.1f}%" if not pd.isna(swl_sp) else "    n/a"
            slr_s  = f"{sl_rate:>6.1f}%" if not pd.isna(sl_rate) else "   n/a"
            tpr_s  = f"{tp_rate:>6.1f}%" if not pd.isna(tp_rate) else "   n/a"
            als_s  = f"{avg_los:>+7.2f}%" if not pd.isna(avg_los) else "     n/a"
            marker = " ◄" if sl_mult == 3.5 else ""
            print(f"  {sl_mult:>7.1f}× | {swl_n:>4} | {wr_s:>6} | {sp_s:>8} | "
                  f"{slr_s:>7} | {tpr_s:>7} | {als_s:>8} | "
                  f"${final:>8.2f} | {sr:>7.2f} | {dd:>+6.1f}%{marker}")
            sl_results.append((sl_mult, swl))

        # Per-Jahr-Aufschlüsselung
        print(f"\n  Per-Jahr:")
        yrs = sorted(sl_results[0][1]["Entry_ts"].pipe(pd.to_datetime).dt.year.unique()) if len(sl_results[0][1]) > 0 else []
        if yrs:
            yr_hdr = "  " + f"{'SL':>4}" + " | " + " | ".join(f"{y:>14}" for y in yrs)
            print(yr_hdr)
            print(f"  {'-'*max(60, len(yr_hdr)-2)}")
            for sl_mult, swl in sl_results:
                if len(swl) == 0:
                    continue
                swl = swl.copy()
                swl["Year"] = pd.to_datetime(swl["Entry_ts"]).dt.year
                yr_parts = []
                for y in yrs:
                    gy = swl[swl["Year"] == y]
                    if len(gy) == 0:
                        yr_parts.append(f"{'n=0':>14}")
                    else:
                        sp = gy["PnL_pct"].sum()
                        n  = len(gy)
                        wr = (gy["PnL_pct"] > 0).mean() * 100
                        yr_parts.append(f"n={n} {wr:>4.0f}% {sp:>+6.1f}%")
                marker = " ◄" if sl_mult == 3.5 else ""
                print(f"  {sl_mult:>3.1f}× | " + " | ".join(yr_parts) + marker)
        print()
        return

    if args.sw_grid:
        def _sw_run(sl, tp, lev):
            t, eq, _, _b = run_backtest(df_raw.copy(), sw_mode=args.mode,
                                    extra_signals=_extra_cache if _extra_cache else None,
                                    sw_sl_mult=sl, sw_tp_mult=tp, sw_lev_override=lev)
            swl = t[(t["Regime"] == "seitwaerts") & (t["Side"] == "long")] if len(t) > 0 else pd.DataFrame()
            n   = len(swl)
            wr  = (swl["PnL_pct"] > 0).mean() * 100 if n > 0 else float("nan")
            sp  = swl["PnL_pct"].sum()                if n > 0 else float("nan")
            fin = eq.iloc[-1]
            sr  = sharpe(eq)
            dd  = max_drawdown(eq)
            return n, wr, sp, fin, sr, dd

        def _fmt(n, wr, sp, fin, sr, dd):
            wr_s = f"{wr:>5.1f}%" if not pd.isna(wr) else "  n/a"
            sp_s = f"{sp:>+8.1f}%" if not pd.isna(sp) else "     n/a"
            return f"{n:>4} | {wr_s} | {sp_s} | ${fin:>8.2f} | {sr:>5.2f} | {dd:>+6.1f}%"

        hdr = f"{'N':>4} | {'WR':>6} | {'ΣPnL%':>9} | {'Endkap.':>9} | {'Sharpe':>5} | {'MaxDD':>6}"

        # ── Phase A: SL × TP Grid ──────────────────────────────────────────
        print(f"\n{'='*75}")
        print(f"  SW-LONG PHASE A — SL × TP Grid  (Hebel: dynamisch)")
        print(f"{'='*75}")
        sl_vals = [2.5, 3.0, 3.5]
        tp_vals = [None, 8.0, 10.0, 12.0]
        tp_labels = {None: "  None", 8.0: "  8×", 10.0: " 10×", 12.0: " 12×"}
        print(f"  {'SL':>5} | {'TP':>5} | {hdr}")
        print(f"  {'-'*72}")
        phase_a_results = []
        for sl in sl_vals:
            for tp in tp_vals:
                res = _sw_run(sl, tp, None)
                marker = " ◄" if sl == 3.0 and tp is None else ""
                print(f"  {sl:>4.1f}× | {tp_labels[tp]} | {_fmt(*res)}{marker}")
                phase_a_results.append((sl, tp, res))
        print()

        # ── Phase B: Hebel-Sweep auf Basis bester Phase-A-Kombos ──────────
        # Beste 3 nach Sharpe ermitteln
        scored = sorted(phase_a_results, key=lambda x: x[2][4], reverse=True)[:3]
        lev_vals = [1.0, 1.5, 2.0, 3.0]

        print(f"{'='*75}")
        print(f"  SW-LONG PHASE B — Hebel-Sweep  (beste 3 SL/TP-Combos)")
        print(f"{'='*75}")
        for sl, tp, _ in scored:
            tp_lbl = tp_labels[tp] if tp in tp_labels else f"{tp}×"
            print(f"\n  SL={sl}× / TP={tp_lbl.strip()}")
            print(f"  {'Hebel':>6} | {hdr}")
            print(f"  {'-'*68}")
            for lev in lev_vals:
                res = _sw_run(sl, tp, lev)
                marker = " ◄" if lev == 1.0 else ""
                print(f"  {lev:>5.1f}× | {_fmt(*res)}{marker}")
        print()
        return

    if args.baer_lev_sweep:
        print(f"\n{'='*75}")
        print(f"  BÄR-SHORT HEBEL-SWEEP  (SL=2×ATR, TP=4.5×ATR, Cover-Signal aktiv)")
        print(f"{'='*75}")
        hdr_b = (f"{'Bär-N':>6} | {'Bär-WR':>7} | {'Bär-ΣPnL%':>10} | "
                 f"{'Bär-$':>8} | {'Sharpe':>7} | {'MaxDD':>7} | "
                 f"{'DD-2022':>8} | {'Cover%':>7} | {'SL%':>5}")
        print(f"  {'Hebel':>6} | {hdr_b}")
        print(f"  {'-'*82}")
        for lev in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
            t, eq, df_f, _ = run_backtest(df_raw.copy(), sw_mode=args.mode,
                                          extra_signals=_extra_cache if _extra_cache else None,
                                          sw_sl_mult=3.5, sw_tp_mult=10.0,
                                          bulle_mode="rsi3", bulle_tp_mult=10.0, bulle_trail_mult=4.0,
                                          bulle_lev_override=1.0, extrem_bulle_lev_override=2.0,
                                          extrem_bulle_tp_mult=10.0,
                                          baer_lev_override=lev)
            bs   = t[(t["Regime"] == "baer") & (t["Side"] == "short")] if len(t) > 0 else pd.DataFrame()
            n_b  = len(bs)
            wr_b = (bs["PnL_pct"] > 0).mean() * 100 if n_b > 0 else float("nan")
            spnl = bs["PnL_pct"].sum() if n_b > 0 else 0.0
            dol  = eq.iloc[-1] - STARTKAPITAL
            # Sharpe
            eq_vals  = eq.values
            ret_s    = pd.Series(eq_vals).pct_change().dropna()
            sharpe_v = (ret_s.mean() / ret_s.std() * (8760 ** 0.5)) if ret_s.std() > 0 else float("nan")
            # MaxDD gesamt
            roll_max = pd.Series(eq_vals).cummax()
            dd_ser   = (pd.Series(eq_vals) - roll_max) / roll_max
            max_dd   = dd_ser.min() * 100
            # MaxDD 2022
            mask_2022 = (eq.index >= "2022-01-01") & (eq.index < "2023-01-01")
            eq_2022   = eq[mask_2022]
            if len(eq_2022) > 1:
                rm22  = eq_2022.cummax()
                dd22  = ((eq_2022 - rm22) / rm22).min() * 100
            else:
                dd22 = float("nan")
            # Cover/SL-Rate
            cov_r = len(bs[bs["Exit_reason"] == "Cover"]) / n_b * 100 if n_b > 0 else float("nan")
            sl_r  = len(bs[bs["Exit_reason"] == "SL"])    / n_b * 100 if n_b > 0 else float("nan")
            bstar = " ◄" if lev == 2.0 else ""
            dd22_s = f"{dd22:>+6.1f}%" if not pd.isna(dd22) else "    n/a"
            print(f"  {lev:>5.1f}× | {n_b:>6} | {wr_b:>6.1f}% | {spnl:>+9.1f}% | "
                  f"${dol:>7,.0f} | {sharpe_v:>6.2f} | {max_dd:>+6.1f}% | "
                  f"{dd22_s} | {cov_r:>6.1f}% | {sl_r:>4.1f}%{bstar}")
        print()
        return

    if args.eb_grid:
        sl_vals  = [1.5, 2.0, 2.5, 3.0]
        tp_vals  = [5.0, 7.0, 10.0, 0]   # 0 = kein TP (Sentinel)
        tp_lbls  = {5.0: " 5×ATR", 7.0: " 7×ATR", 10.0: "10×ATR", 0: "  None"}
        TRAIL_OFF = 99.0  # Trail effektiv deaktiviert (99 ATR unter Einstieg)

        print(f"\n{'='*95}")
        print(f"  EB NO-TRAIL SL×TP-GRID  (Bär 6×, Bulle 1×, EB 2×, ADX<40, Trail=off, mode={args.mode})")
        print(f"{'='*95}")
        hdr = (f"{'SL':>6} | {'TP':>7} | {'EB-N':>5} | {'EB-WR':>6} | {'EB-ΣPnL%':>9} | "
               f"{'EB-TP%':>7} | {'EB-$':>7} | {'Rendite':>8} | {'Sharpe':>7} | {'MaxDD':>7} | {'Gate4':>6}")
        print(f"  {hdr}")
        print(f"  {'-'*100}")

        for sl in sl_vals:
            for tp in tp_vals:
                t, eq, df_f, _ = run_backtest(
                    df_raw.copy(), sw_mode=args.mode,
                    extra_signals=_extra_cache if _extra_cache else None,
                    baer_lev_override=6.0,
                    bulle_lev_override=1.0, bulle_mode="rsi3",
                    bulle_tp_mult=10.0, bulle_trail_mult=4.0,
                    extrem_bulle_lev_override=2.0,
                    extrem_bulle_sl_mult=sl,
                    extrem_bulle_tp_mult=tp,
                    extrem_bulle_trail_mult=TRAIL_OFF,
                )
                eb     = t[(t["Regime"] == "extrem_bulle") & (t["Side"] == "long")] if len(t) > 0 else pd.DataFrame()
                eb_n   = len(eb)
                eb_wr  = (eb["PnL_pct"] > 0).mean() * 100 if eb_n > 0 else float("nan")
                eb_sp  = eb["PnL_pct"].sum() if eb_n > 0 else 0.0
                eb_tp_n = len(eb[eb["Exit_reason"] == "TP"]) if eb_n > 0 else 0
                eb_tp_r = eb_tp_n / eb_n * 100 if eb_n > 0 else float("nan")
                eb_dol = (eb["PnL_pct"] / 100 * eb["Entry_cash"]).sum() if eb_n > 0 else 0.0
                rendite = (eq.iloc[-1] - STARTKAPITAL) / STARTKAPITAL * 100
                sr      = sharpe(eq)
                dd      = max_drawdown(eq)
                gate4_ok = dd >= -30.0
                gate4_s  = f"{dd+30:>+5.1f}pp" if gate4_ok else f"❌{dd+30:>+4.1f}pp"
                # Baseline-Marker: originale Einstellungen (SL=1.5, TP=5.0, Trail aktiv)
                is_base = (sl == 1.5 and tp == 5.0)
                marker  = " ◄" if is_base else ""
                tp_s    = tp_lbls[tp]
                print(f"  {sl:>5.1f}× | {tp_s} | {eb_n:>5} | {eb_wr:>5.1f}% | {eb_sp:>+8.1f}% | "
                      f"{eb_tp_r:>6.1f}% | ${eb_dol:>5,.0f} | {rendite:>+7.1f}% | {sr:>6.2f} | "
                      f"{dd:>+6.1f}% | {gate4_s}{marker}")
            print(f"  {'·'*100}")
        print()
        return

    if args.eb_trail_sweep:
        print(f"\n{'='*90}")
        print(f"  EXTREM-BULLE TRAIL-WIDENING-SWEEP  (Bär 6×, Bulle 1×, EB 2×, ADX<40-Filter, mode={args.mode})")
        print(f"{'='*90}")
        print(f"  {'Trail':>6} | {'EB-N':>5} | {'EB-WR':>6} | {'EB-ΣPnL%':>9} | {'EB-Trail-N':>10} | "
              f"{'EB-Trail-WR':>11} | {'EB-$':>7} | {'Rendite':>8} | {'Sharpe':>7} | {'MaxDD':>7} | {'DD-Puf':>7}")
        print(f"  {'-'*95}")
        for tm in [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
            t, eq, df_f, _ = run_backtest(df_raw.copy(), sw_mode=args.mode,
                                          extra_signals=_extra_cache if _extra_cache else None,
                                          baer_lev_override=6.0,
                                          bulle_lev_override=1.0, bulle_mode="rsi3",
                                          bulle_tp_mult=10.0, bulle_trail_mult=4.0,
                                          extrem_bulle_lev_override=2.0,
                                          extrem_bulle_tp_mult=10.0,
                                          extrem_bulle_trail_mult=tm)
            eb = t[(t["Regime"] == "extrem_bulle") & (t["Side"] == "long")] if len(t) > 0 else pd.DataFrame()
            eb_n      = len(eb)
            eb_wr     = (eb["PnL_pct"] > 0).mean() * 100 if eb_n > 0 else float("nan")
            eb_sp     = eb["PnL_pct"].sum() if eb_n > 0 else 0.0
            eb_trail  = eb[eb["Exit_reason"] == "Trail"] if eb_n > 0 else pd.DataFrame()
            eb_tr_n   = len(eb_trail)
            eb_tr_wr  = (eb_trail["PnL_pct"] > 0).mean() * 100 if eb_tr_n > 0 else float("nan")
            start_cap = eq.iloc[0]
            final_cap = eq.iloc[-1]
            eb_dol    = (eb["PnL_pct"] / 100 * eb["Entry_cash"]).sum() if eb_n > 0 else 0.0
            rendite   = (final_cap - STARTKAPITAL) / STARTKAPITAL * 100
            sr        = sharpe(eq)
            dd        = max_drawdown(eq)
            dd_puffer = -30.0 - dd
            bstar     = " ◄" if tm == 1.2 else ""
            eb_wr_s   = f"{eb_wr:>5.1f}%" if not pd.isna(eb_wr) else "  n/a"
            eb_tr_wr_s = f"{eb_tr_wr:>9.1f}%" if not pd.isna(eb_tr_wr) else "     n/a"
            print(f"  {tm:>5.1f}× | {eb_n:>5} | {eb_wr_s} | {eb_sp:>+8.1f}% | {eb_tr_n:>10} | "
                  f"{eb_tr_wr_s} | ${eb_dol:>5,.0f} | {rendite:>+7.1f}% | {sr:>6.2f} | "
                  f"{dd:>+6.1f}% | {dd_puffer:>+6.1f}pp{bstar}")
        print()
        return

    if args.rsi3_diag:
        print(f"\n{'='*100}")
        print(f"  RSI3 DIAGNOSE  (bulle_mode=rsi3, Bär 6×, EB 2×, TP 10×, Trail 4×, mode={args.mode})")
        print(f"{'='*100}")
        t, eq, _, _ = run_backtest(df_raw.copy(), sw_mode=args.mode,
                                   extra_signals=_extra_cache if _extra_cache else None,
                                   baer_lev_override=6.0,
                                   bulle_lev_override=1.0,
                                   extrem_bulle_lev_override=2.0,
                                   extrem_bulle_tp_mult=10.0,
                                   bulle_mode="rsi3",
                                   bulle_trail_mult=4.0,
                                   bulle_tp_mult=10.0,
                                   bulle_lev_high_override=2.0,
                                   bulle_adx_thresh_override=40.0)
        bl = t[(t["Regime"] == "bulle") & (t["Side"] == "long")].copy() if len(t) > 0 else pd.DataFrame()
        if len(bl) == 0:
            print("  Keine Bulle-Long-Trades gefunden.")
            return
        bl["Year"] = pd.to_datetime(bl["Entry_ts"]).dt.year
        bl["Win"]  = bl["PnL_pct"] > 0

        # ── 1. Gesamt-Summary ────────────────────────────────────────────────
        n_tot  = len(bl)
        wr_tot = bl["Win"].mean() * 100
        sp_tot = bl["PnL_pct"].sum()
        sp_22p = bl[bl["Year"] >= 2022]["PnL_pct"].sum()
        avg_w  = bl[bl["Win"]]["PnL_pct"].mean()
        avg_l  = bl[~bl["Win"]]["PnL_pct"].mean()
        dol    = (bl["PnL_pct"] / 100 * bl["Entry_cash"]).sum()
        print(f"\n  Gesamt:  n={n_tot}  WR={wr_tot:.1f}%  ΣPnL%={sp_tot:+.1f}%  ΣPnL%(2022+)={sp_22p:+.1f}%")
        print(f"  Avg-Win={avg_w:+.2f}%  Avg-Loss={avg_l:+.2f}%  BL-$={dol:+,.0f}")

        # ── 2. Per-Jahr ΣPnL% + Win/Loss-Count ──────────────────────────────
        print(f"\n  {'Jahr':<6} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'ΣPnL%(Σ so weit)':>17} | {'Avg-Win':>8} | {'Avg-Loss':>9}")
        print(f"  {'-'*75}")
        cumulative = 0.0
        for yr, g in bl.groupby("Year"):
            n_yr   = len(g)
            wr_yr  = g["Win"].mean() * 100
            sp_yr  = g["PnL_pct"].sum()
            cumulative += sp_yr
            aw_yr  = g[g["Win"]]["PnL_pct"].mean() if g["Win"].any() else float("nan")
            al_yr  = g[~g["Win"]]["PnL_pct"].mean() if (~g["Win"]).any() else float("nan")
            aw_s   = f"{aw_yr:+.2f}%" if not pd.isna(aw_yr) else "   n/a"
            al_s   = f"{al_yr:+.2f}%" if not pd.isna(al_yr) else "    n/a"
            print(f"  {yr:<6} | {n_yr:>4} | {wr_yr:>5.1f}% | {sp_yr:>+7.1f}% | {cumulative:>+16.1f}% | {aw_s:>8} | {al_s:>9}")

        # ── 3. Exit-Typ × Jahr ───────────────────────────────────────────────
        print(f"\n  Exit-Typ-Aufschlüsselung:")
        print(f"  {'Jahr':<6} | {'Exit':<14} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
        print(f"  {'-'*55}")
        for yr, g_yr in bl.groupby("Year"):
            for ex, g_ex in g_yr.groupby("Exit_reason"):
                n_e  = len(g_ex)
                wr_e = g_ex["Win"].mean() * 100
                sp_e = g_ex["PnL_pct"].sum()
                print(f"  {yr:<6} | {str(ex):<14} | {n_e:>4} | {wr_e:>5.1f}% | {sp_e:>+7.1f}%")

        # ── 4. RSI14-Bucket-Analyse ──────────────────────────────────────────
        if "RSI_entry" in bl.columns:
            print(f"\n  RSI14-Bucket bei Entry:")
            print(f"  {'RSI14-Bucket':<16} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'BL-$':>7}")
            print(f"  {'-'*55}")
            bins   = [0, 45, 50, 55, 60, 65, 70, 75, 100]
            labels = ["<45","45-50","50-55","55-60","60-65","65-70","70-75","75+"]
            bl["RSI14_bin"] = pd.cut(bl["RSI_entry"], bins=bins, labels=labels, right=False)
            for bkt, g_b in bl.groupby("RSI14_bin", observed=True):
                n_b   = len(g_b)
                wr_b  = g_b["Win"].mean() * 100
                sp_b  = g_b["PnL_pct"].sum()
                dol_b = (g_b["PnL_pct"] / 100 * g_b["Entry_cash"]).sum()
                print(f"  {str(bkt):<16} | {n_b:>4} | {wr_b:>5.1f}% | {sp_b:>+7.1f}% | ${dol_b:>5,.0f}")
        print()
        return

    if args.bulle_sweep:
        print(f"\n{'='*110}")
        print(f"  BULLE-LONG SWEEP  (Bär 6×, EB 2×, TP 10×, mode={args.mode})")
        print(f"{'='*110}")
        # Spalten: BL-N = Trades, BL-ΣPnL% = gesamt (bulle+neutral für _b1), G3-bl% = bulle-only (Gate 3)
        print(f"  {'Methode':<26} | {'BL-N':>5} | {'BL-WR':>6} | {'BL-ΣPnL%':>9} | {'G3-bl%':>8} | {'BL-$':>7} | "
              f"{'Rendite':>8} | {'Sharpe':>7} | {'MaxDD':>7}")
        print(f"  {'-'*105}")

        def _bulle_row(label, bulle_m, bulle_sl=None, bulle_trail=None, bulle_tp=None):
            t, eq, _, _ = run_backtest(df_raw.copy(), sw_mode=args.mode,
                                       extra_signals=_extra_cache if _extra_cache else None,
                                       baer_lev_override=6.0,
                                       bulle_lev_override=1.0,
                                       extrem_bulle_lev_override=2.0,
                                       extrem_bulle_tp_mult=10.0,
                                       bulle_mode=bulle_m,
                                       bulle_sl_mult=bulle_sl,
                                       bulle_trail_mult=bulle_trail,
                                       bulle_tp_mult=bulle_tp)
            is_b1 = bulle_m.endswith("_b1")
            if is_b1:
                bl = t[t["Regime"].isin(["bulle", "neutral"]) & (t["Side"] == "long")] if len(t) > 0 else pd.DataFrame()
            else:
                bl = t[(t["Regime"] == "bulle") & (t["Side"] == "long")] if len(t) > 0 else pd.DataFrame()
            bl_bulle = t[(t["Regime"] == "bulle") & (t["Side"] == "long")] if len(t) > 0 else pd.DataFrame()
            bl_n     = len(bl)
            bl_wr    = (bl["PnL_pct"] > 0).mean() * 100 if bl_n > 0 else float("nan")
            bl_sp    = bl["PnL_pct"].sum() if bl_n > 0 else 0.0
            bl_sp_g3 = bl_bulle["PnL_pct"].sum() if len(bl_bulle) > 0 else 0.0
            bl_dol   = (bl["PnL_pct"] / 100 * bl["Entry_cash"]).sum() if bl_n > 0 else 0.0
            final    = eq.iloc[-1]
            rend     = (final - STARTKAPITAL) / STARTKAPITAL * 100
            sr       = sharpe(eq)
            dd       = max_drawdown(eq)
            wr_s     = f"{bl_wr:>5.1f}%" if not pd.isna(bl_wr) else "  n/a"
            g3_flag  = "✅" if bl_sp_g3 > 0 else "❌"
            print(f"  {label:<26} | {bl_n:>5} | {wr_s} | {bl_sp:>+8.1f}% | {bl_sp_g3:>+6.1f}%{g3_flag} | ${bl_dol:>5,.0f} | "
                  f"{rend:>+7.1f}% | {sr:>6.2f} | {dd:>+6.1f}%")
            return t, eq, bl, bl_bulle

        # Sweep: RSI3 mit RSI14-Untergrenze-Varianten + Referenzen
        # Diagnose (Apr 2026): Bucket 50-55 ist Giftbehälter (WR=23%, ΣPnL=-38.3%)
        sweep_variants = [
            # key,            label,                       mode,            sl,   trail, tp
            ("base",          "default (Baseline)",        "default",       None, 4.0,   10.0),
            ("rsi3",          "RSI3 (RSI14≥40)",           "rsi3",          None, 4.0,   10.0),
            ("rsi3_adx35",    "RSI3 ADX≥35",               "rsi3_adx35",    None, 4.0,   10.0),
            ("rsi3_adx40",    "RSI3 ADX≥40",               "rsi3_adx40",    None, 4.0,   10.0),
            ("rsi3_r45",      "RSI3 RSI14≥45",             "rsi3_r45",      None, 4.0,   10.0),
            ("rsi3_r50",      "RSI3 RSI14≥50",             "rsi3_r50",      None, 4.0,   10.0),
            ("rsi3_notch",    "RSI3 Notch (ex 50-55)",     "rsi3_notch",    None, 4.0,   10.0),
            ("rsi3_r55",      "RSI3 RSI14≥55",             "rsi3_r55",      None, 4.0,   10.0),
            ("rsi3_r57",      "RSI3 RSI14≥57",             "rsi3_r57",      None, 4.0,   10.0),
            ("ema21b",        "EMA21-Bounce",              "ema21_bounce",  None, 4.0,   10.0),
        ]
        all_trades = {}     # bulle+neutral (für B1 korrekt)
        all_trades_bl = {}  # bulle-only (Gate 3)
        for key, disp_label, bm, bsl, btr, btp in sweep_variants:
            t_res, _, bl_res, bl_bulle_res = _bulle_row(disp_label, bm, bulle_sl=bsl, bulle_trail=btr, bulle_tp=btp)
            if len(bl_res) > 0:
                bl_c = bl_res.copy()
                bl_c["Year"] = pd.to_datetime(bl_c["Entry_ts"]).dt.year
                all_trades[key] = bl_c.groupby("Year")["PnL_pct"].sum().to_dict()
            else:
                all_trades[key] = {}
            if len(bl_bulle_res) > 0:
                bl_b = bl_bulle_res.copy()
                bl_b["Year"] = pd.to_datetime(bl_b["Entry_ts"]).dt.year
                all_trades_bl[key] = bl_b.groupby("Year")["PnL_pct"].sum().to_dict()
            else:
                all_trades_bl[key] = {}

        # Per-Jahr-Aufschlüsselung
        _pj_cols = ["base", "rsi3", "rsi3_adx35", "rsi3_adx40", "rsi3_r50", "rsi3_notch", "rsi3_r55"]
        print(f"\n  Per-Jahr Bulle(+neutral)-ΣPnL% je Methode:")
        print(f"  {'Jahr':<6}", end="")
        for lbl in _pj_cols:
            print(f" | {lbl:>16}", end="")
        print()
        print(f"  {'-'*100}")
        all_years = sorted(set(y for d in all_trades.values() for y in d))
        for yr in all_years:
            print(f"  {yr:<6}", end="")
            for lbl in _pj_cols:
                val = all_trades[lbl].get(yr, float("nan"))
                s = f"{val:>+8.1f}%" if not pd.isna(val) else "         n/a"
                print(f" | {s:>16}", end="")
            print()
        print()
        return

    if args.neutral_sweep:
        print(f"\n{'='*105}")
        print(f"  NEUTRAL-REGIME SWEEP  (Bär 6×, EB 2×, bulle_mode=rsi3, mode=daily_bull→neutral_*)")
        print(f"{'='*105}")
        print(f"  {'Variante':<26} | {'N-n':>4} | {'N-WR':>6} | {'N-ΣPnL%':>9} | {'N-$':>6} | "
              f"{'Rendite':>8} | {'Sharpe':>7} | {'MaxDD':>7}")
        print(f"  {'-'*95}")

        def _neutral_row(label, sw_m):
            t, eq, _, _ = run_backtest(df_raw.copy(), sw_mode=sw_m,
                                       extra_signals=_extra_cache if _extra_cache else None,
                                       baer_lev_override=6.0,
                                       bulle_lev_override=1.0,
                                       extrem_bulle_lev_override=2.0,
                                       extrem_bulle_tp_mult=10.0,
                                       bulle_mode="rsi3",
                                       bulle_tp_mult=10.0,
                                       bulle_trail_mult=4.0)
            nn = t[t["Regime"] == "neutral"].copy() if len(t) > 0 else pd.DataFrame()
            n_n   = len(nn)
            n_wr  = (nn["PnL_pct"] > 0).mean() * 100 if n_n > 0 else float("nan")
            n_sp  = nn["PnL_pct"].sum() if n_n > 0 else 0.0
            n_dol = (nn["PnL_pct"] / 100 * nn["Entry_cash"]).sum() if n_n > 0 else 0.0
            final = eq.iloc[-1]
            rend  = (final - STARTKAPITAL) / STARTKAPITAL * 100
            sr    = sharpe(eq)
            dd    = max_drawdown(eq)
            wr_s  = f"{n_wr:>5.1f}%" if not pd.isna(n_wr) else "  n/a"
            print(f"  {label:<26} | {n_n:>4} | {wr_s} | {n_sp:>+8.1f}% | ${n_dol:>4,.0f} | "
                  f"{rend:>+7.1f}% | {sr:>6.2f} | {dd:>+6.1f}%")
            return t, nn

        neutral_variants = [
            ("Baseline (off)",      "daily_bull"),
            ("SW-Short in neutral", "neutral_sw_short"),
            ("SW-Full in neutral",  "neutral_sw_full"),
            ("RSI<15 Long",         "neutral_rsi_ext"),
        ]
        for lbl, sw_m in neutral_variants:
            _neutral_row(lbl, sw_m)
        print()
        return

    if args.sw_adx_diag:
        print(f"\n{'='*90}")
        print(f"  SW-LONG ADX-DIAGNOSE  (sw_mode=neutral_sw_short, Bär 6×, EB 2×, bulle=rsi3)")
        print(f"{'='*90}")
        t, eq, df_full, _ = run_backtest(df_raw.copy(), sw_mode="neutral_sw_short",
                                        extra_signals=_extra_cache if _extra_cache else None,
                                        baer_lev_override=6.0,
                                        bulle_lev_override=1.0,
                                        extrem_bulle_lev_override=2.0,
                                        extrem_bulle_tp_mult=10.0,
                                        bulle_mode="rsi3",
                                        bulle_tp_mult=10.0,
                                        bulle_trail_mult=4.0)
        sl = t[(t["Regime"] == "seitwaerts") & (t["Side"] == "long")].copy() if len(t) > 0 else pd.DataFrame()
        if len(sl) == 0:
            print("  Keine SW-Long-Trades gefunden.")
            return
        sl["Year"] = pd.to_datetime(sl["Entry_ts"]).dt.year
        sl["Win"]  = sl["PnL_pct"] > 0

        # ── 1. Winner vs. Loser Median ADX ──────────────────────────────────
        winners = sl[sl["Win"]]["ADX_entry"]
        losers  = sl[~sl["Win"]]["ADX_entry"]
        print(f"\n  Gesamt:  n={len(sl)}  WR={sl['Win'].mean()*100:.1f}%  ΣPnL%={sl['PnL_pct'].sum():+.1f}%")
        print(f"\n  ADX bei Entry — Winner vs. Loser:")
        print(f"    Winner  (n={len(winners)}):  Median={winners.median():.1f}  Mean={winners.mean():.1f}  Min={winners.min():.1f}  Max={winners.max():.1f}")
        print(f"    Loser   (n={len(losers)}):  Median={losers.median():.1f}  Mean={losers.mean():.1f}  Min={losers.min():.1f}  Max={losers.max():.1f}")
        print(f"    Diff (Winner−Loser Median): {winners.median() - losers.median():+.1f}")

        # ── 2. ADX-Bucket-Analyse (Gesamt) ───────────────────────────────────
        print(f"\n  ADX-Buckets (Gesamt):")
        print(f"  {'Bucket':<12} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'Avg-Win':>8} | {'Avg-Loss':>9}")
        print(f"  {'-'*60}")
        bins   = [0, 20, 25, 30, 35, 999]
        labels = ["<20", "20-25", "25-30", "30-35", ">35"]
        sl["ADX_bin"] = pd.cut(sl["ADX_entry"], bins=bins, labels=labels, right=False)
        for bkt, g in sl.groupby("ADX_bin", observed=True):
            if len(g) == 0:
                continue
            n_b  = len(g)
            wr_b = g["Win"].mean() * 100
            sp_b = g["PnL_pct"].sum()
            aw_b = g[g["Win"]]["PnL_pct"].mean() if g["Win"].any() else float("nan")
            al_b = g[~g["Win"]]["PnL_pct"].mean() if (~g["Win"]).any() else float("nan")
            aw_s = f"{aw_b:+.2f}%" if not pd.isna(aw_b) else "    n/a"
            al_s = f"{al_b:+.2f}%" if not pd.isna(al_b) else "     n/a"
            print(f"  {str(bkt):<12} | {n_b:>4} | {wr_b:>5.1f}% | {sp_b:>+7.1f}% | {aw_s:>8} | {al_s:>9}")

        # ── 3. Per-Jahr: N, WR, ΣPnL%, Median-ADX ───────────────────────────
        print(f"\n  Per-Jahr (inkl. Median-ADX):")
        print(f"  {'Jahr':<6} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'Med-ADX':>8}")
        print(f"  {'-'*45}")
        for yr, g in sl.groupby("Year"):
            n_y  = len(g)
            wr_y = g["Win"].mean() * 100
            sp_y = g["PnL_pct"].sum()
            md_y = g["ADX_entry"].median()
            print(f"  {yr:<6} | {n_y:>4} | {wr_y:>5.1f}% | {sp_y:>+7.1f}% | {md_y:>7.1f}")

        # ── 4. 2025-Detail: ADX-Bucket ───────────────────────────────────────
        sl25 = sl[sl["Year"] == 2025]
        if len(sl25) > 0:
            print(f"\n  2025-Detail ADX-Buckets (n={len(sl25)}):")
            print(f"  {'Bucket':<12} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'Win-PnL':>8} | {'Los-PnL':>9}")
            print(f"  {'-'*65}")
            for bkt, g in sl25.groupby("ADX_bin", observed=True):
                if len(g) == 0:
                    continue
                n_b  = len(g)
                wr_b = g["Win"].mean() * 100
                sp_b = g["PnL_pct"].sum()
                aw_b = g[g["Win"]]["PnL_pct"].mean() if g["Win"].any() else float("nan")
                al_b = g[~g["Win"]]["PnL_pct"].mean() if (~g["Win"]).any() else float("nan")
                aw_s = f"{aw_b:+.2f}%" if not pd.isna(aw_b) else "    n/a"
                al_s = f"{al_b:+.2f}%" if not pd.isna(al_b) else "     n/a"
                print(f"  {str(bkt):<12} | {n_b:>4} | {wr_b:>5.1f}% | {sp_b:>+7.1f}% | {aw_s:>8} | {al_s:>9}")

        # ── 5. Bedingungsanalyse: welche v13-Bedingungen hatten Winner vs Loser ─
        cond_cols = ["RSI", "ST_bull", "RSI_div_bull", "OBV_bull"]
        available = [c for c in cond_cols if c in df_full.columns]
        if available:
            df_idx = df_full[available].copy()
            df_idx.index = pd.to_datetime(df_idx.index)
            sl["Entry_ts_idx"] = pd.to_datetime(sl["Entry_ts"])
            # Nächster verfügbarer Index-Zeitstempel je Trade
            joined = sl.copy()
            for col in available:
                joined[col + "_at_entry"] = joined["Entry_ts_idx"].map(
                    lambda ts, c=col: df_idx[c].asof(ts) if ts >= df_idx.index[0] else None
                )

            print(f"\n  Bedingungserfüllung bei Entry (v13-Score-Conditions):")
            print(f"  {'Bedingung':<22} | {'W-Rate':>7} | {'L-Rate':>7} | {'Diff':>6}")
            print(f"  {'-'*50}")

            w = joined[joined["Win"]]
            l = joined[~joined["Win"]]

            if "RSI_at_entry" in joined.columns:
                w_r = (w["RSI_at_entry"] < 45).mean() * 100
                l_r = (l["RSI_at_entry"] < 45).mean() * 100
                print(f"  {'RSI < 45':<22} | {w_r:>6.1f}% | {l_r:>6.1f}% | {w_r-l_r:>+5.1f}pp")
            if "ST_bull_at_entry" in joined.columns:
                w_r = w["ST_bull_at_entry"].mean() * 100
                l_r = l["ST_bull_at_entry"].mean() * 100
                print(f"  {'ST_bull':<22} | {w_r:>6.1f}% | {l_r:>6.1f}% | {w_r-l_r:>+5.1f}pp")
            if "RSI_div_bull_at_entry" in joined.columns:
                w_r = w["RSI_div_bull_at_entry"].mean() * 100
                l_r = l["RSI_div_bull_at_entry"].mean() * 100
                print(f"  {'RSI_div_bull':<22} | {w_r:>6.1f}% | {l_r:>6.1f}% | {w_r-l_r:>+5.1f}pp")
            if "OBV_bull_at_entry" in joined.columns:
                w_r = w["OBV_bull_at_entry"].mean() * 100
                l_r = l["OBV_bull_at_entry"].mean() * 100
                print(f"  {'OBV_bull':<22} | {w_r:>6.1f}% | {l_r:>6.1f}% | {w_r-l_r:>+5.1f}pp")

            # Kombination: ST_bull AND OBV_bull (stärkste Kombination)
            if "ST_bull_at_entry" in joined.columns and "OBV_bull_at_entry" in joined.columns:
                joined["both_bull"] = joined["ST_bull_at_entry"] & joined["OBV_bull_at_entry"]
                w_r = joined[joined["Win"]]["both_bull"].mean() * 100
                l_r = joined[~joined["Win"]]["both_bull"].mean() * 100
                print(f"  {'ST_bull AND OBV_bull':<22} | {w_r:>6.1f}% | {l_r:>6.1f}% | {w_r-l_r:>+5.1f}pp")

            # Score-Verteilung: wie viele Bedingungen erfüllten Winner vs Loser im Schnitt?
            score_cols = []
            if "RSI_at_entry" in joined.columns:
                joined["cond_rsi"]    = joined["RSI_at_entry"] < 45
                score_cols.append("cond_rsi")
            if "ST_bull_at_entry" in joined.columns:
                joined["cond_st"]     = joined["ST_bull_at_entry"].astype(bool)
                score_cols.append("cond_st")
            if "OBV_bull_at_entry" in joined.columns:
                joined["cond_obv"]    = joined["OBV_bull_at_entry"].astype(bool)
                score_cols.append("cond_obv")
            if score_cols:
                joined["score"] = joined[score_cols].sum(axis=1)
                print(f"\n  Score-Verteilung bei Entry (max {len(score_cols)}):")
                print(f"  {'Score':<8} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8}")
                print(f"  {'-'*35}")
                for sc, g in joined.groupby("score"):
                    n_s  = len(g)
                    wr_s = g["Win"].mean() * 100
                    sp_s = g["PnL_pct"].sum()
                    print(f"  {sc:<8} | {n_s:>4} | {wr_s:>5.1f}% | {sp_s:>+7.1f}%")

        # ── 6. Exit_reason × Win/Loss × Jahr ─────────────────────────────────
        if "Exit_reason" in sl.columns:
            print(f"\n  Exit_reason × Jahr:")
            print(f"  {'Jahr':<6} | {'Exit':<12} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'Avg-PnL':>8}")
            print(f"  {'-'*60}")
            for yr, gy in sl.groupby("Year"):
                for ex, gex in gy.groupby("Exit_reason"):
                    n_e  = len(gex)
                    wr_e = gex["Win"].mean() * 100
                    sp_e = gex["PnL_pct"].sum()
                    av_e = gex["PnL_pct"].mean()
                    print(f"  {yr:<6} | {str(ex):<12} | {n_e:>4} | {wr_e:>5.1f}% | {sp_e:>+7.1f}% | {av_e:>+7.2f}%")

            # Exit_reason Gesamt
            print(f"\n  Exit_reason Gesamt:")
            print(f"  {'Exit':<12} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'Avg-Win':>8} | {'Avg-Los':>8}")
            print(f"  {'-'*60}")
            for ex, gex in sl.groupby("Exit_reason"):
                n_e  = len(gex)
                wr_e = gex["Win"].mean() * 100
                sp_e = gex["PnL_pct"].sum()
                aw_e = gex[gex["Win"]]["PnL_pct"].mean() if gex["Win"].any() else float("nan")
                al_e = gex[~gex["Win"]]["PnL_pct"].mean() if (~gex["Win"]).any() else float("nan")
                aw_s = f"{aw_e:+.2f}%" if not pd.isna(aw_e) else "    n/a"
                al_s = f"{al_e:+.2f}%" if not pd.isna(al_e) else "    n/a"
                print(f"  {str(ex):<12} | {n_e:>4} | {wr_e:>5.1f}% | {sp_e:>+7.1f}% | {aw_s:>8} | {al_s:>8}")

        # ── 7. Vola_ratio_entry: Winner vs. Loser ─────────────────────────────
        if "Vola_ratio_entry" in sl.columns and sl["Vola_ratio_entry"].notna().any():
            vw = sl[sl["Win"]]["Vola_ratio_entry"]
            vl = sl[~sl["Win"]]["Vola_ratio_entry"]
            print(f"\n  Vola_ratio_entry — Winner vs. Loser:")
            print(f"    Winner (n={len(vw)}): Median={vw.median():.3f}  Mean={vw.mean():.3f}  P25={vw.quantile(0.25):.3f}  P75={vw.quantile(0.75):.3f}")
            print(f"    Loser  (n={len(vl)}): Median={vl.median():.3f}  Mean={vl.mean():.3f}  P25={vl.quantile(0.25):.3f}  P75={vl.quantile(0.75):.3f}")
            print(f"    Diff (Winner−Loser Median): {vw.median() - vl.median():+.3f}")
            # Vola-Buckets
            vr_bins   = [0, 0.8, 1.0, 1.2, 1.5, 999]
            vr_labels = ["<0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5", ">1.5"]
            sl["Vola_bin"] = pd.cut(sl["Vola_ratio_entry"], bins=vr_bins, labels=vr_labels, right=False)
            print(f"\n  Vola_ratio-Buckets:")
            print(f"  {'Bucket':<12} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'Avg-PnL':>8}")
            print(f"  {'-'*50}")
            for bkt, g in sl.groupby("Vola_bin", observed=True):
                if len(g) == 0:
                    continue
                n_b  = len(g)
                wr_b = g["Win"].mean() * 100
                sp_b = g["PnL_pct"].sum()
                av_b = g["PnL_pct"].mean()
                print(f"  {str(bkt):<12} | {n_b:>4} | {wr_b:>5.1f}% | {sp_b:>+7.1f}% | {av_b:>+7.2f}%")

        # ── 8. Dist_ema21_entry: Winner vs. Loser ─────────────────────────────
        if "Dist_ema21_entry" in sl.columns and sl["Dist_ema21_entry"].notna().any():
            dw = sl[sl["Win"]]["Dist_ema21_entry"]
            dl = sl[~sl["Win"]]["Dist_ema21_entry"]
            print(f"\n  Dist_ema21_entry — Winner vs. Loser (in %):")
            print(f"    Winner (n={len(dw)}): Median={dw.median():.2f}%  Mean={dw.mean():.2f}%  P25={dw.quantile(0.25):.2f}%  P75={dw.quantile(0.75):.2f}%")
            print(f"    Loser  (n={len(dl)}): Median={dl.median():.2f}%  Mean={dl.mean():.2f}%  P25={dl.quantile(0.25):.2f}%  P75={dl.quantile(0.75):.2f}%")
            print(f"    Diff (Winner−Loser Median): {dw.median() - dl.median():+.2f}pp")
            # Dist-Buckets
            dist_bins2   = [-999, -10, -5, -2, 0, 2, 999]
            dist_labels2 = ["<-10%", "-10..-5%", "-5..-2%", "-2..0%", "0..2%", ">2%"]
            sl["Dist_bin2"] = pd.cut(sl["Dist_ema21_entry"], bins=dist_bins2, labels=dist_labels2, right=True)
            print(f"\n  Dist_ema21-Buckets:")
            print(f"  {'Bucket':<12} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'Avg-PnL':>8}")
            print(f"  {'-'*50}")
            for bkt, g in sl.groupby("Dist_bin2", observed=True):
                if len(g) == 0:
                    continue
                n_b  = len(g)
                wr_b = g["Win"].mean() * 100
                sp_b = g["PnL_pct"].sum()
                av_b = g["PnL_pct"].mean()
                print(f"  {str(bkt):<12} | {n_b:>4} | {wr_b:>5.1f}% | {sp_b:>+7.1f}% | {av_b:>+7.2f}%")

        print()
        return

    if args.wf:
        def _wf_run(df_slice, bmode="rsi3"):
            t, eq, _, _ = run_backtest(
                df_slice.copy(), sw_mode="neutral_sw_short",
                extra_signals=_extra_cache if _extra_cache else None,
                baer_lev_override=6.0, bulle_lev_override=1.0,
                extrem_bulle_lev_override=2.0, extrem_bulle_tp_mult=10.0,
                bulle_mode=bmode, bulle_tp_mult=10.0, bulle_trail_mult=4.0,
                bulle_lev_high_override=2.0, bulle_adx_thresh_override=40.0)
            return t, eq

        def _wf_metrics(t, eq, label):
            n  = len(t)
            if len(eq) < 2:
                return label, n, float("nan"), float("nan"), float("nan"), float("nan"), {}
            start_eq = eq.iloc[0]; end_eq = eq.iloc[-1]
            rendite  = (end_eq / start_eq - 1) * 100
            sr       = sharpe(eq)
            dd       = max_drawdown(eq)
            regime_stats = {}
            if n > 0:
                for reg, side in [("seitwaerts","long"),("extrem_bulle","long"),
                                   ("baer","short"),("bulle","long"),("neutral","short"),("seitwaerts","short")]:
                    sub = t[(t["Regime"]==reg) & (t["Side"]==side)]
                    if len(sub) == 0:
                        continue
                    lbl2 = f"{reg[:4]}-{side[:1]}"
                    regime_stats[lbl2] = (len(sub),
                                          (sub["PnL_pct"]>0).mean()*100,
                                          sub["PnL_pct"].sum())
            return label, n, rendite, sr, dd, end_eq, regime_stats

        # ── Teil 1: Konsistenz-Probe ───────────────────────────────────────────────
        print(f"\n{'='*115}")
        print(f"  KONSISTENZ-PROBE  (⚠ Nicht OOS — Parameter auf 2021-2026 optimiert. Zeigt: Perioden-Konsistenz.)")
        print(f"  Params: sw_mode=neutral_sw_short, bulle_mode=rsi3, Bär 6×, EB 2×+TP10×, Bulle 1×+TP10×+Trail4×")
        print(f"{'='*115}")

        windows = [
            ("IS: voll (2021–)",      None,         None),
            ("Periode: 2022–2023",    "2022-01-01", "2023-12-31"),
            ("Periode: 2023",         "2023-01-01", "2023-12-31"),
            ("Periode: 2024",         "2024-01-01", "2024-12-31"),
            ("Periode: 2025",         "2025-01-01", "2025-12-31"),
            ("Periode: 2024+",        "2024-01-01", None),
            ("Periode: 2026 (partial)","2026-01-01", None),
        ]

        rows = []
        for lbl, start, end in windows:
            df_w = df_raw.copy()
            if start:
                df_w = df_w[df_w.index >= start]
            if end:
                df_w = df_w[df_w.index <= end]
            t_w, eq_w = _wf_run(df_w)
            rows.append(_wf_metrics(t_w, eq_w, lbl))

        hdr = f"  {'Periode':<28} | {'N':>5} | {'Rendite':>9} | {'Sharpe':>7} | {'MaxDD':>7} | {'Endkap.':>9}"
        print(hdr)
        print(f"  {'-'*100}")
        for lbl, n, rend, sr, dd, end_eq, rstat in rows:
            rend_s = f"{rend:>+8.1f}%" if not pd.isna(rend) else "       n/a"
            sr_s   = f"{sr:>7.2f}"     if not pd.isna(sr)   else "    n/a"
            dd_s   = f"{dd:>+6.1f}%"   if not pd.isna(dd)   else "   n/a"
            eq_s   = f"${end_eq:>8.2f}" if not pd.isna(end_eq) else "      n/a"
            marker = " ◄" if lbl.startswith("IS") else ""
            print(f"  {lbl:<28} | {n:>5} | {rend_s:>9} | {sr_s:>7} | {dd_s:>7} | {eq_s:>9}{marker}")

        # Per-Regime für 2024+ und Voll
        print(f"\n  Per-Regime $-Beitrag:")
        regime_order = ["seit-l","extr-l","baer-s","bull-l","neut-s","seit-s"]
        hdr2 = f"  {'Periode':<28} | " + " | ".join(f"{r:>12}" for r in regime_order)
        print(hdr2)
        print(f"  {'-'*100}")
        for lbl, n, rend, sr, dd, end_eq, rstat in rows:
            if not rstat:
                continue
            parts = []
            for r in regime_order:
                if r in rstat:
                    nn, wr, sp = rstat[r]
                    parts.append(f"n={nn} {wr:>3.0f}% {sp:>+5.1f}%")
                else:
                    parts.append(f"{'—':>12}")
            print(f"  {lbl:<28} | " + " | ".join(parts))

        # ── Teil 2: Naive OOS — rsi3 vs. default auf 2024+ ────────────────────────
        print(f"\n{'='*115}")
        print(f"  NAIVE OOS-PROBE  (2024+ mit rsi3 vs. default-Bulle — misst RSI3-Optimierungs-Robustheit)")
        print(f"{'='*115}")
        df_2024 = df_raw[df_raw.index >= "2024-01-01"].copy()
        rows2 = []
        for bmode in ["rsi3", "default"]:
            t_b, eq_b = _wf_run(df_2024, bmode=bmode)
            rows2.append(_wf_metrics(t_b, eq_b, f"2024+ bulle={bmode}"))

        print(f"  {'Variante':<28} | {'N':>5} | {'Rendite':>9} | {'Sharpe':>7} | {'MaxDD':>7} | {'Endkap.':>9}")
        print(f"  {'-'*80}")
        for lbl, n, rend, sr, dd, end_eq, rstat in rows2:
            rend_s = f"{rend:>+8.1f}%" if not pd.isna(rend) else "       n/a"
            sr_s   = f"{sr:>7.2f}"     if not pd.isna(sr)   else "    n/a"
            dd_s   = f"{dd:>+6.1f}%"   if not pd.isna(dd)   else "   n/a"
            eq_s   = f"${end_eq:>8.2f}" if not pd.isna(end_eq) else "      n/a"
            print(f"  {lbl:<28} | {n:>5} | {rend_s:>9} | {sr_s:>7} | {dd_s:>7} | {eq_s:>9}")
            if rstat:
                for r in regime_order:
                    if r in rstat:
                        nn, wr, sp = rstat[r]
                        print(f"    {r:<10} n={nn:>3} WR={wr:>4.0f}% ΣPnL={sp:>+6.1f}%")
        print()
        return

    if args.diag2026:
        print(f"\n{'='*100}")
        print(f"  2026-REGIME-DIAGNOSE  (Jan–Apr 2026)")
        print(f"{'='*100}")

        # Volles Backtest mit df_full für Regime-Zeitreihe
        t_full, eq_full, df_full, _ = run_backtest(
            df_raw.copy(), sw_mode="neutral_sw_short",
            extra_signals=_extra_cache if _extra_cache else None,
            baer_lev_override=6.0, bulle_lev_override=1.0,
            extrem_bulle_lev_override=2.0, extrem_bulle_tp_mult=10.0,
            bulle_mode="rsi3", bulle_tp_mult=10.0, bulle_trail_mult=4.0,
            bulle_lev_high_override=2.0, bulle_adx_thresh_override=40.0)

        # ── 1. Regime-Verteilung per Monat (2025-Q4 + 2026) ──────────────────────
        print(f"\n  1. Regime-Verteilung per Monat (2025-Q4 bis heute)")
        if "Regime" in df_full.columns:
            df_reg = df_full.copy()
            df_reg.index = pd.to_datetime(df_reg.index)
            df_2025q4 = df_reg[df_reg.index >= "2025-10-01"]
            df_2025q4 = df_2025q4.copy()
            df_2025q4["YM"] = df_2025q4.index.to_period("M")
            regime_cols = ["baer", "seitwaerts", "neutral", "bulle", "extrem_bulle", "extrem_baer"]
            hdr_reg = f"  {'Monat':<10} | {'Kerzen':>7} | " + " | ".join(f"{r[:8]:>8}" for r in regime_cols)
            print(hdr_reg)
            print(f"  {'-'*80}")
            for ym, grp in df_2025q4.groupby("YM"):
                total = len(grp)
                parts = []
                for r in regime_cols:
                    cnt = (grp["Regime"] == r).sum()
                    pct = cnt / total * 100 if total > 0 else 0
                    parts.append(f"{pct:>6.0f}%" if cnt > 0 else f"{'—':>7}")
                print(f"  {str(ym):<10} | {total:>7} | " + " | ".join(parts))
        else:
            print("  ⚠ Regime-Spalte nicht in df_full")

        # ── 2. Preis-Übersicht 2026 (Monatsdurchschnitt close) ───────────────────
        print(f"\n  2. Preis-Übersicht 2026 (Monats-Eröffnung, -Schluss, Bewegung)")
        if "Close" in df_full.columns:
            df_26 = df_full[df_full.index >= "2026-01-01"].copy()
            df_26.index = pd.to_datetime(df_26.index)
            df_26["YM"] = df_26.index.to_period("M")
            print(f"  {'Monat':<10} | {'Open':>8} | {'Close':>8} | {'Min':>8} | {'Max':>8} | {'Bewegung':>9}")
            print(f"  {'-'*65}")
            for ym, grp in df_26.groupby("YM"):
                op  = grp["Close"].iloc[0]
                cl  = grp["Close"].iloc[-1]
                mn  = grp["Close"].min()
                mx  = grp["Close"].max()
                mv  = (cl / op - 1) * 100
                print(f"  {str(ym):<10} | {op:>8.2f} | {cl:>8.2f} | {mn:>8.2f} | {mx:>8.2f} | {mv:>+8.1f}%")
        else:
            print("  ⚠ Close-Spalte nicht in df_full")

        # ── 3. Alle 2026-Trades im Detail ─────────────────────────────────────────
        print(f"\n  3. Alle 2026-Trades im Detail")
        if len(t_full) > 0:
            t26 = t_full[pd.to_datetime(t_full["Entry_ts"]).dt.year == 2026]
            if len(t26) == 0:
                print("  Keine 2026-Trades")
            else:
                print(f"  {'Entry':>19} | {'Exit':>19} | {'Reg':<6} | {'Side':<5} | "
                      f"{'Entry_P':>8} | {'Exit_P':>8} | {'PnL%':>7} | {'Exit_reason':<12} | {'ATR_entry':>9}")
                print(f"  {'-'*105}")
                for _, row in t26.iterrows():
                    et  = str(row.get("Entry_ts", ""))[:19]
                    xt  = str(row.get("Exit_ts",  ""))[:19]
                    reg = str(row.get("Regime",   ""))[:6]
                    sid = str(row.get("Side",     ""))[:5]
                    ep  = row.get("Entry_price", float("nan"))
                    xp  = row.get("Exit_price",  float("nan"))
                    pnl = row.get("PnL_pct",     float("nan"))
                    exr = str(row.get("Exit_reason", ""))[:12]
                    atr = row.get("ATR_entry", float("nan"))
                    ep_s  = f"{ep:>8.2f}"  if not pd.isna(ep)  else "     n/a"
                    xp_s  = f"{xp:>8.2f}"  if not pd.isna(xp)  else "     n/a"
                    pnl_s = f"{pnl:>+6.2f}%" if not pd.isna(pnl) else "    n/a"
                    atr_s = f"{atr:>9.3f}"  if not pd.isna(atr) else "      n/a"
                    print(f"  {et:>19} | {xt:>19} | {reg:<6} | {sid:<5} | "
                          f"{ep_s} | {xp_s} | {pnl_s} | {exr:<12} | {atr_s}")

        # ── 4. Bär-Short Statistik: 2024 vs 2025 vs 2026 ─────────────────────────
        print(f"\n  4. Bär-Short Jahres-Vergleich")
        if len(t_full) > 0:
            bs = t_full[(t_full["Regime"] == "baer") & (t_full["Side"] == "short")].copy()
            bs["Year"] = pd.to_datetime(bs["Entry_ts"]).dt.year
            print(f"  {'Jahr':<6} | {'N':>4} | {'WR':>6} | {'ΣPnL%':>8} | {'Avg-Win':>8} | {'Avg-Los':>8} | {'SL-Rate':>7} | {'Cover-Rate':>10}")
            print(f"  {'-'*75}")
            for yr, gy in bs.groupby("Year"):
                n_y   = len(gy)
                wr_y  = (gy["PnL_pct"] > 0).mean() * 100
                sp_y  = gy["PnL_pct"].sum()
                aw_y  = gy[gy["PnL_pct"] > 0]["PnL_pct"].mean() if (gy["PnL_pct"] > 0).any() else float("nan")
                al_y  = gy[gy["PnL_pct"] <= 0]["PnL_pct"].mean() if (gy["PnL_pct"] <= 0).any() else float("nan")
                slr   = (gy["Exit_reason"] == "SL").mean() * 100 if "Exit_reason" in gy.columns else float("nan")
                cvr   = (gy["Exit_reason"] == "Cover").mean() * 100 if "Exit_reason" in gy.columns else float("nan")
                aw_s  = f"{aw_y:>+7.2f}%" if not pd.isna(aw_y) else "     n/a"
                al_s  = f"{al_y:>+7.2f}%" if not pd.isna(al_y) else "     n/a"
                slr_s = f"{slr:>6.1f}%"   if not pd.isna(slr)  else "   n/a"
                cvr_s = f"{cvr:>9.1f}%"   if not pd.isna(cvr)  else "      n/a"
                print(f"  {yr:<6} | {n_y:>4} | {wr_y:>5.1f}% | {sp_y:>+7.1f}% | {aw_s} | {al_s} | {slr_s} | {cvr_s}")
        print()
        return

    results = {}
    for label, mode in modes:
        print(f"\n  Starte: {label} ...")
        trades, equity, df_full, blocked = run_backtest(df_raw.copy(), sw_mode=mode,
                                               extra_signals=_extra_cache if _extra_cache else None,
                                               baer_lev_override=6.0,
                                               bulle_lev_override=1.0,
                                               extrem_bulle_lev_override=2.0,
                                               extrem_bulle_tp_mult=10.0,
                                               bulle_mode="rsi3",
                                               bulle_tp_mult=10.0,
                                               bulle_trail_mult=4.0,
                                               bulle_lev_high_override=2.0,
                                               bulle_adx_thresh_override=40.0,
                                               track_blocked=args.block_diag)
        results[label] = (trades, equity, df_full)
        print_report(label, trades, equity, df_full)
        if args.regime_detail and len(trades) > 0:
            print_regime_detail(trades)
        if args.diag and len(trades) > 0:
            print_bulle_diag(trades)
            print_baer_diag(trades)
        if args.eb_diag and len(trades) > 0:
            print_extrem_bulle_diag(trades)
        if args.dd and len(trades) > 0:
            print_dd_diag(trades, equity)
        if args.sw_diag and len(trades) > 0:
            print_sw_long_diag(trades)
        if args.block_diag and len(blocked) > 0:
            _print_block_diag(blocked, trades)

    if args.ab and len(results) > 1:
        print("\n" + "="*60)
        print("  A/B-VERGLEICH SEITWÄRTS")
        print("="*60)
        print(f"  {'Variante':<30} | {'Rendite':>8} | {'Sharpe':>7} | {'Max DD':>7} | {'SW-EV':>7}")
        print(f"  {'-'*65}")
        for label, (trades, equity, df_full) in results.items():
            final   = equity.iloc[-1]
            r       = (final - STARTKAPITAL) / STARTKAPITAL * 100
            sr      = sharpe(equity)
            dd      = max_drawdown(equity)
            sw      = trades[trades["Regime"] == "seitwaerts"] if len(trades) > 0 else pd.DataFrame()
            sw_ev   = sw["PnL_pct"].mean() if len(sw) > 0 else float("nan")
            print(f"  {label:<30} | {r:>+7.1f}% | {sr:>7.2f} | {dd:>+6.1f}% | {sw_ev:>+6.2f}%")
        print()


if __name__ == "__main__":
    main()
