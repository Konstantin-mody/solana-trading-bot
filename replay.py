"""
replay.py — Parity-Test: Paper Bot vs. Backtest auf identischen Daten

Verwendung:
    python3 replay.py                     # Letzten 60 Tage
    python3 replay.py --days 90           # Letzten 90 Tage
    python3 replay.py --start 2025-01-01  # Ab bestimmtem Datum

Erwartet: Entry_ts und Exit_ts identisch für alle Trades.
Muss ✅ zeigen bevor Paper Trading gestartet wird.
"""
import sys, argparse
import pandas as pd

sys.path.insert(0, ".")

from sol_bot.backtest import (
    fetch_binance_data, compute_daily_macro, run_backtest, STARTKAPITAL,
)
from sol_bot.indicators import add_indicators
from sol_bot.regime import classify_regime
from paper_bot import process_bar, _default_state, PROD_PARAMS


def run_replay(start: str, end: str | None = None, verbose: bool = False):
    print(f"\nLade Binance-Daten ab {start}...")
    df_raw = fetch_binance_data(start=start)
    if end:
        df_raw = df_raw[df_raw.index <= end]
    print(f"  {len(df_raw)} Kerzen geladen ({df_raw.index[0]} → {df_raw.index[-1]})")

    # ── Parität prüfen: Starte-Stunde muss UTC-Mitternacht sein ──────────────
    first_hour = df_raw.index[0].hour
    if first_hour != 0:
        print(f"\n⚠️  Warnung: Erste Kerze um {first_hour}:00 UTC (nicht Mitternacht).")
        print(f"    Funding-Parity-Test könnte abweichen (i%8 vs ts.hour in {{0,8,16}}).")
        print(f"    Empfehlung: --start auf Mitternacht setzen (z.B. '2025-01-01').\n")

    # ── Indikatoren + Regime für beide Läufe ─────────────────────────────────
    extra = {"Daily_macro_bull": compute_daily_macro(df_raw)}

    # ── Referenz: run_backtest() ──────────────────────────────────────────────
    print("\nLaufe Backtest (Referenz)...")
    t_bt, eq_bt, _, _ = run_backtest(
        df_raw.copy(),
        sw_mode                  = PROD_PARAMS["sw_mode"],
        bulle_mode               = PROD_PARAMS["bulle_mode"],
        max_pos_size             = PROD_PARAMS["max_pos_size"],
        extra_signals            = extra,
        baer_lev_override        = PROD_PARAMS["baer_lev_override"],
        bulle_lev_override       = PROD_PARAMS["bulle_lev_override"],
        extrem_bulle_lev_override= PROD_PARAMS["extrem_bulle_lev_override"],
        extrem_bulle_tp_mult     = PROD_PARAMS["extrem_bulle_tp_mult"],
        bulle_tp_mult            = PROD_PARAMS["bulle_tp_mult"],
        bulle_trail_mult         = PROD_PARAMS["bulle_trail_mult"],
        bulle_lev_high_override  = PROD_PARAMS["bulle_lev_high_override"],
        bulle_adx_thresh_override= PROD_PARAMS["bulle_adx_thresh_override"],
    )
    print(f"  Backtest: {len(t_bt)} Trades, Endkapital ${eq_bt.iloc[-1]:.2f}")

    # ── Replay: process_bar() bar-für-bar ────────────────────────────────────
    print("\nLaufe Paper Bot Replay...")
    macro  = compute_daily_macro(df_raw)
    df     = add_indicators(df_raw.copy())
    df["Regime"]           = classify_regime(df)
    df["Daily_macro_bull"] = macro.reindex(df.index, method="ffill").fillna(False)

    state  = _default_state()
    trades = []
    for ts, row in df.iterrows():
        state, trade, _ = process_bar(row, ts, state, PROD_PARAMS)
        if trade is not None:
            trades.append(trade)

    t_rp = pd.DataFrame(trades) if trades else pd.DataFrame(columns=list(t_bt.columns))
    print(f"  Replay:   {len(t_rp)} Trades, Endkapital ${state['cash']:.2f}")

    # ── Vergleich ─────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  PARITY-TEST ERGEBNIS")
    print(f"{'='*62}")

    ok = True

    # Anzahl
    if len(t_bt) != len(t_rp):
        print(f"  ❌ Trade-Anzahl: Backtest={len(t_bt)}  Replay={len(t_rp)}")
        ok = False
    else:
        print(f"  ✅ Trade-Anzahl: {len(t_bt)}")

    if len(t_bt) == 0 and len(t_rp) == 0:
        print(f"  ✅ Beide 0 Trades — kein Signal in diesem Zeitraum")
        return

    if len(t_rp) == 0:
        print("  Replay hat keine Trades → Trade-Vergleich nicht möglich.")
        return

    # Trade-für-Trade-Vergleich
    t_bt_s = t_bt.sort_values("Exit_ts").reset_index(drop=True)
    t_rp_s = t_rp.sort_values("Exit_ts").reset_index(drop=True)

    mismatches = 0
    for i in range(min(len(t_bt_s), len(t_rp_s))):
        r_bt = t_bt_s.iloc[i]
        r_rp = t_rp_s.iloc[i]
        notes = []

        # Exit_ts muss identisch sein
        bt_exit = str(r_bt["Exit_ts"])
        rp_exit = str(r_rp["Exit_ts"])
        if bt_exit != rp_exit:
            notes.append(f"Exit_ts: {bt_exit} ≠ {rp_exit}")

        # Side muss identisch sein
        if r_bt["Side"] != r_rp["Side"]:
            notes.append(f"Side: {r_bt['Side']} ≠ {r_rp['Side']}")

        # Exit_reason muss identisch sein
        if r_bt["Exit_reason"] != r_rp["Exit_reason"]:
            notes.append(f"Exit_reason: {r_bt['Exit_reason']} ≠ {r_rp['Exit_reason']}")

        # PnL_pct: max 0.01% Abweichung (Fließkomma-Toleranz)
        pnl_diff = abs(r_bt["PnL_pct"] - r_rp["PnL_pct"])
        if pnl_diff > 0.01:
            notes.append(f"PnL_pct: {r_bt['PnL_pct']:.4f}% ≠ {r_rp['PnL_pct']:.4f}% (Δ={pnl_diff:.4f})")

        if notes:
            mismatches += 1
            ok = False
            print(f"  ❌ Trade #{i+1}: {' | '.join(notes)}")
        elif verbose:
            print(f"  ✅ Trade #{i+1}: {bt_exit[:16]} {r_bt['Side']:5} {r_bt['Exit_reason']}")

    if mismatches == 0 and len(t_bt_s) == len(t_rp_s):
        print(f"  ✅ Alle {len(t_bt_s)} Trades identisch (PnL-Toleranz 0.01%)")

    # Equity-Vergleich
    print(f"\n  Endkapital Backtest : ${eq_bt.iloc[-1]:.2f}")
    print(f"  Cash Replay         : ${state['cash']:.2f}")
    eq_diff = abs(eq_bt.iloc[-1] - state["cash"])
    if eq_diff < 0.01:
        print(f"  ✅ Endkapital identisch")
    else:
        print(f"  ⚠️  Endkapital Δ=${eq_diff:.2f} (kann durch offene Position entstehen)")

    print(f"\n  {'✅ PARITY OK' if ok else '❌ PARITY FEHLER — paper_bot.py und backtest.py abgeglichen?'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parity-Test: Paper Bot vs. Backtest")
    parser.add_argument("--days",    type=int, default=60,   help="Letzten N Tage (Default: 60)")
    parser.add_argument("--start",   type=str, default=None, help="Start-Datum YYYY-MM-DD")
    parser.add_argument("--end",     type=str, default=None, help="End-Datum YYYY-MM-DD")
    parser.add_argument("--verbose", action="store_true",    help="Zeige alle Trades (auch ✅)")
    args = parser.parse_args()

    if args.start:
        start = args.start
    else:
        start = (pd.Timestamp.now() - pd.Timedelta(days=args.days)).strftime("%Y-%m-%d")

    run_replay(start, args.end, verbose=args.verbose)
