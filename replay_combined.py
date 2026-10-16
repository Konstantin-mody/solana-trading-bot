"""
replay_combined.py — Paritätstest: run_backtest_combined vs. paper_bot_combined

Prüft ob paper_bot_combined.process_bar() für die gleichen historischen Daten
exakt dieselben Trades produziert wie run_backtest_combined().

Verwendung:
    python3 replay_combined.py                      # vollständig ab 2021-09-01
    python3 replay_combined.py --start 2024-01-01   # nur ab 2024 (schneller)
    python3 replay_combined.py --end   2025-01-01   # nur IS-Periode
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from run_backtest_combined import run_backtest_combined, TOTAL_CAPITAL
from sol_bot_4h.backtest import fetch_binance_4h
from sol_bot_1d.backtest import fetch_binance_1d
from sol_bot_1w.backtest import fetch_binance_1w

import paper_bot_combined as pbc

TOL_PNL   = 0.005   # %-Punkte — erlaubte Float-Abweichung pro Trade
TOL_PRICE = 0.01    # absolut — erlaubte Preisabweichung


# ─── Replay via process_bar ───────────────────────────────────────────────────

def _replay(df_4h_raw, df_1d_raw, df_1w_raw):
    """Repliziert den Backtest sequenziell durch process_bar()-Aufrufe."""
    df_4h, df_1d_on_4h, df_1w_on_4h = pbc.build_aligned_frames(
        df_4h_raw, df_1d_raw, df_1w_raw
    )

    state = {
        "cash":      pbc.TOTAL_CAPITAL,
        "pos_4h":    None,
        "pos_1d":    None,
        "pos_1w":    None,
        "cd_4h":     0,
        "cd_1d":     0,
        "cd_1w":     0,
        "virt_4h":   pbc.INIT_BOT,
        "virt_1d":   pbc.INIT_BOT,
        "virt_1w":   pbc.INIT_BOT,
        "win_4h":    [pbc.INIT_BOT],
        "win_1d":    [pbc.INIT_BOT],
        "win_1w":    [pbc.INIT_BOT],
        "last_ts":   None,
        "live_since": None,
    }

    trades  = []
    eq_vals = []

    for i, (ts, row_4h) in enumerate(df_4h.iterrows()):
        row_1d = df_1d_on_4h.iloc[i]
        row_1w = df_1w_on_4h.iloc[i]

        closed = pbc.process_bar(ts, row_4h, row_1d, row_1w, state)

        for t in closed:
            trades.append({
                "bot":         t["bot"],
                "Entry_ts":    pd.Timestamp(t["entry_ts"]),
                "Exit_ts":     pd.Timestamp(t["exit_ts"]),
                "Side":        t["side"],
                "Entry_price": t["ep"],
                "Exit_price":  t["xp"],
                "Hebel":       t["hebel"],
                "PnL_pct":     t["pnl_pct"],
                "Entry_cash":  t["entry_cash"],
                "Regime":      t["regime"],
                "Exit_reason": t["reason"],
            })

        # Equity-Snapshot für Schluss-Vergleich
        kurs = float(row_4h["Close"])
        pv   = state["cash"]
        for pk in ("pos_4h", "pos_1d", "pos_1w"):
            pos = state[pk]
            if pos is not None:
                pv += pbc._mtm(pos.entry_cash, pos.hebel,
                               pos.entry_price, kurs, pos.side)
        eq_vals.append({"ts": ts, "equity": pv})

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_rp = pd.DataFrame(eq_vals).set_index("ts")["equity"] if eq_vals else pd.Series()
    return trades_df, equity_rp, state


# ─── Normierung ───────────────────────────────────────────────────────────────

def _sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(["bot", "Entry_ts", "Exit_ts"]).reset_index(drop=True)


# ─── Vergleich ────────────────────────────────────────────────────────────────

def compare(bt: pd.DataFrame, rp: pd.DataFrame,
            eq_bt: pd.Series, eq_rp: pd.Series) -> bool:
    ok = True

    # ── Trade-Anzahl ──────────────────────────────────────────────────────────
    print(f"\n  Trades  BT={len(bt)}   RP={len(rp)}")
    for bot in ["4h", "1d", "1w"]:
        nb = len(bt[bt["bot"] == bot]) if not bt.empty else 0
        nr = len(rp[rp["bot"] == bot]) if not rp.empty else 0
        tag = "✅" if nb == nr else "❌"
        print(f"    {tag} {bot}  BT={nb} vs RP={nr}")

    if len(bt) != len(rp):
        print("\n  ❌ ANZAHL-DIVERGENZ — weitere Vergleiche übersprungen.")
        return False

    # ── Trade-by-Trade ────────────────────────────────────────────────────────
    divergences = []
    for idx in range(len(bt)):
        b = bt.iloc[idx]
        r = rp.iloc[idx]
        diffs = {}

        if b["bot"] != r["bot"]:
            diffs["bot"] = (b["bot"], r["bot"])
        if b["Entry_ts"] != r["Entry_ts"]:
            diffs["Entry_ts"] = (b["Entry_ts"], r["Entry_ts"])
        if b["Exit_ts"] != r["Exit_ts"]:
            diffs["Exit_ts"] = (b["Exit_ts"], r["Exit_ts"])
        if b["Side"] != r["Side"]:
            diffs["Side"] = (b["Side"], r["Side"])
        if abs(b["PnL_pct"] - r["PnL_pct"]) > TOL_PNL:
            diffs["PnL_pct"] = (f"{b['PnL_pct']:.4f}%", f"{r['PnL_pct']:.4f}%")
        if abs(b["Entry_price"] - r["Entry_price"]) > TOL_PRICE:
            diffs["Entry_price"] = (f"{b['Entry_price']:.4f}", f"{r['Entry_price']:.4f}")
        if abs(b["Exit_price"] - r["Exit_price"]) > TOL_PRICE:
            diffs["Exit_price"] = (f"{b['Exit_price']:.4f}", f"{r['Exit_price']:.4f}")
        if b["Exit_reason"] != r["Exit_reason"]:
            diffs["Exit_reason"] = (b["Exit_reason"], r["Exit_reason"])

        if diffs:
            divergences.append((idx, b, r, diffs))

    if not divergences:
        print(f"\n  ✅ Alle {len(bt)} Trades identisch (Toleranz PnL ±{TOL_PNL}%)")
    else:
        ok = False
        print(f"\n  ❌ {len(divergences)} Trade-Divergenzen:")
        for idx, b, r, diffs in divergences[:20]:
            print(f"\n    Trade #{idx+1}  {b['bot']} | {b['Entry_ts'].date()} → "
                  f"{b['Exit_ts'].date()} | {b['Side']} | {b.get('Regime','')}")
            for field, (bv, rv) in diffs.items():
                print(f"      {field:14s}  BT={bv}   RP={rv}")
        if len(divergences) > 20:
            print(f"    … und {len(divergences)-20} weitere.")

    # ── Equity-Vergleich (Gesamt-Endwert) ────────────────────────────────────
    if not eq_bt.empty and not eq_rp.empty:
        end_bt = eq_bt.iloc[-1]
        end_rp = eq_rp.iloc[-1]
        diff   = abs(end_bt - end_rp)
        tag    = "✅" if diff < 0.10 else "❌"
        print(f"\n  {tag} Equity-End  BT=${end_bt:.4f}   RP=${end_rp:.4f}   Δ=${diff:.4f}")
        if diff >= 0.10:
            ok = False

        # ΣPnL-Vergleich
        bt_ret = (end_bt / eq_bt.iloc[0] - 1) * 100
        rp_ret = (end_rp / eq_rp.iloc[0] - 1) * 100
        print(f"     Rendite       BT={bt_ret:+.1f}%   RP={rp_ret:+.1f}%")

    return ok


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paritätstest Backtest vs. process_bar")
    parser.add_argument("--start", default="2021-09-01")
    parser.add_argument("--end",   default=None)
    args = parser.parse_args()

    print(f"Lade Daten ab {args.start} …")
    df_4h_raw = fetch_binance_4h(args.start)
    df_1d_raw = fetch_binance_1d(args.start)
    df_1w_raw = fetch_binance_1w(args.start)

    if args.end:
        end_ts    = pd.Timestamp(args.end)
        df_4h_raw = df_4h_raw[df_4h_raw.index <= end_ts]
        df_1d_raw = df_1d_raw[df_1d_raw.index <= end_ts]
        df_1w_raw = df_1w_raw[df_1w_raw.index <= end_ts]

    n4, n1, nw = len(df_4h_raw), len(df_1d_raw), len(df_1w_raw)
    print(f"  4h: {n4} Kerzen  ({df_4h_raw.index[0].date()} → {df_4h_raw.index[-1].date()})")
    print(f"  1d: {n1} Kerzen  ({df_1d_raw.index[0].date()} → {df_1d_raw.index[-1].date()})")
    print(f"  1w: {nw} Kerzen  ({df_1w_raw.index[0].date()} → {df_1w_raw.index[-1].date()})")

    # Funding-Alignment prüfen: erster Bar muss bei 00:00 UTC liegen
    first_hour = df_4h_raw.index[0].hour
    if first_hour != 0:
        print(f"\n  ⚠️  WARNUNG: Erster 4h-Bar ist {first_hour}:00 UTC (nicht 00:00).")
        print("     Index-basiertes Funding (Backtest) ≠ Wall-Clock-Funding (Paper Bot).")
        print("     Funding-Abweichungen möglich.")
    else:
        print(f"  Funding-Alignment: ✅ (Start 00:00 UTC → Index- = Wall-Clock-Funding)")

    print("\nRun backtest_combined …")
    trades_bt, eq_bt = run_backtest_combined(df_4h_raw, df_1d_raw, df_1w_raw)
    print(f"  Trades: {len(trades_bt)}   "
          f"Equity: ${eq_bt.iloc[0]:.2f} → ${eq_bt.iloc[-1]:.2f}")

    print("\nRepliziere via process_bar …")
    trades_rp, eq_rp, final_state = _replay(df_4h_raw, df_1d_raw, df_1w_raw)
    print(f"  Trades: {len(trades_rp)}   "
          f"Equity: ${eq_rp.iloc[0]:.2f} → ${eq_rp.iloc[-1]:.2f}"
          if not eq_rp.empty else f"  Trades: {len(trades_rp)}")

    bt = _sort(trades_bt)
    rp = _sort(trades_rp)

    print("\n" + "─" * 64)
    ok = compare(bt, rp, eq_bt, eq_rp)
    print("─" * 64)

    if ok:
        print("\n✅  PARITY PASSED — paper_bot_combined.process_bar ist backtest-identisch.\n")
    else:
        print("\n❌  PARITY FAILED — Divergenzen gefunden. Bot nicht starten!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
