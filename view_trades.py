"""
view_trades.py — Trade-Viewer für den Paper Bot

Verwendung:
    python3 view_trades.py status          # Aktueller Status (Cash, Position, Rendite)
    python3 view_trades.py list [n]        # Letzte n Trades (Default: 20)
    python3 view_trades.py show <id>       # Ein Trade im Detail
    python3 view_trades.py equity          # Equity-Zusammenfassung + Jahres-Breakdown
    python3 view_trades.py regimes         # Trades nach Regime gruppiert
"""
import sys, sqlite3, json
from pathlib import Path
import pandas as pd

DB    = "paper_trades.db"
STATE = "paper_state.json"


def _load_trades(limit=None) -> pd.DataFrame:
    if not Path(DB).exists():
        return pd.DataFrame()
    con = sqlite3.connect(DB)
    q   = "SELECT * FROM trades ORDER BY id DESC"
    if limit:
        q += f" LIMIT {limit}"
    df = pd.read_sql_query(q, con)
    con.close()
    return df


def _load_equity() -> pd.DataFrame:
    if not Path(DB).exists():
        return pd.DataFrame()
    con = sqlite3.connect(DB)
    df  = pd.read_sql_query("SELECT * FROM equity_log ORDER BY ts", con)
    con.close()
    return df


def _load_state() -> dict:
    if not Path(STATE).exists():
        return {}
    with open(STATE, "r") as f:
        return json.load(f)


# ── Befehle ───────────────────────────────────────────────────────────────────

def cmd_status():
    state = _load_state()
    if not state:
        print("Kein State-File gefunden. Bot noch nicht gestartet?")
        return

    cash    = state.get("cash", 0)
    pos     = state.get("pos")
    eq_df   = _load_equity()
    equity  = eq_df.iloc[-1]["portfolio"] if len(eq_df) > 0 else cash
    rendite = (equity / 200.0 - 1) * 100
    trades  = _load_trades()

    print(f"\n{'='*52}")
    print(f"  PAPER BOT STATUS")
    print(f"{'='*52}")
    print(f"  Portfolio   : ${equity:.2f}  ({rendite:+.1f}%)")
    print(f"  Cash        : ${cash:.2f}")
    print(f"  Trades      : {len(trades)}")
    print(f"  Letzter Bar : {state.get('last_ts', 'N/A')}")

    if len(trades) > 0:
        wr  = (trades["PnL_pct"] > 0).mean() * 100
        avg = trades["PnL_pct"].mean()
        print(f"  WR gesamt   : {wr:.1f}%   Ø PnL: {avg:+.2f}%")

    if pos:
        print(f"\n  OFFENE POSITION:")
        print(f"    Side      : {pos.get('side')}")
        print(f"    Entry_ts  : {pos.get('candle_idx')}")
        print(f"    Entry     : ${pos.get('entry_price', 0):.4f}")
        print(f"    Cash risk : ${pos.get('entry_cash', 0):.2f}")
        print(f"    Hebel     : {pos.get('hebel', 1):.1f}×")
        print(f"    SL        : ${pos.get('sl_price', 0):.4f}")
        tp = pos.get('tp_price')
        print(f"    TP        : ${tp:.4f}" if tp else "    TP        : None")
        print(f"    Regime    : {pos.get('regime')}")
        print(f"    Pyramids  : {pos.get('pyramid_count', 0)}")
    else:
        print(f"\n  Keine offene Position.")


def cmd_list(n: int = 20):
    df = _load_trades(limit=n)
    if df.empty:
        print("Keine Trades vorhanden.")
        return

    print(f"\n{'='*95}")
    print(f"  {'ID':>4} | {'Exit-Datum':>19} | {'Side':>5} | {'Regime':>14} | "
          f"{'PnL%':>7} | {'Hebel':>5} | {'Grund':<12} | {'Pyramids'}")
    print(f"  {'-'*95}")
    for _, row in df.iterrows():
        print(f"  {int(row['id']):>4} | {str(row['Exit_ts'])[:19]:>19} | "
              f"{row['Side']:>5} | {str(row['Regime'])[:14]:>14} | "
              f"{row['PnL_pct']:>+7.2f}% | {row['Hebel']:>4.1f}× | "
              f"{str(row['Exit_reason']):<12} | {int(row['Pyramid_count']) if row['Pyramid_count'] is not None else 0}")

    wr  = (df["PnL_pct"] > 0).mean() * 100
    avg = df["PnL_pct"].mean()
    print(f"\n  {len(df)} Trades | WR: {wr:.1f}% | Ø PnL: {avg:+.2f}%")


def cmd_show(trade_id: int):
    if not Path(DB).exists():
        print("Kein DB-File.")
        return
    con = sqlite3.connect(DB)
    row = pd.read_sql_query(f"SELECT * FROM trades WHERE id = {trade_id}", con)
    con.close()

    if row.empty:
        print(f"Trade #{trade_id} nicht gefunden.")
        return

    r = row.iloc[0]
    print(f"\n{'='*52}")
    print(f"  TRADE #{trade_id}")
    print(f"{'='*52}")
    for col in row.columns:
        val = r[col]
        if pd.isna(val):
            val_str = "—"
        elif isinstance(val, float):
            val_str = f"{val:.4f}" if abs(val) < 10000 else f"{val:.2f}"
        else:
            val_str = str(val)
        print(f"  {col:<22}: {val_str}")

    # PnL in Dollar
    if not pd.isna(r["PnL_pct"]) and not pd.isna(r["Entry_cash"]):
        pnl_dollar = r["Entry_cash"] * r["PnL_pct"] / 100.0
        print(f"  {'PnL (Dollar)':<22}: ${pnl_dollar:+.2f}")


def cmd_equity():
    eq_df = _load_equity()
    if eq_df.empty:
        print("Keine Equity-Daten.")
        return

    eq_df["ts"] = pd.to_datetime(eq_df["ts"])
    series = pd.to_numeric(eq_df["portfolio"])

    roll_max = series.expanding().max()
    dd       = ((series - roll_max) / roll_max * 100).min()
    rendite  = (series.iloc[-1] / 200.0 - 1) * 100
    n_bars   = len(series)
    n_days   = (eq_df["ts"].iloc[-1] - eq_df["ts"].iloc[0]).total_seconds() / 86400

    print(f"\n{'='*52}")
    print(f"  EQUITY ZUSAMMENFASSUNG")
    print(f"{'='*52}")
    print(f"  Startkapital  : $200.00")
    print(f"  Aktuell       : ${series.iloc[-1]:.2f}")
    print(f"  Rendite       : {rendite:+.1f}%")
    print(f"  Max. Drawdown : {dd:.1f}%")
    print(f"  Laufzeit      : {n_days:.1f} Tage  ({n_bars} Bars)")

    print(f"\n  Jahresperformance:")
    print(f"  {'Jahr':<6} | {'Start':>8} | {'Ende':>8} | {'Rendite':>8}")
    print(f"  {'-'*38}")
    for yr in sorted(eq_df["ts"].dt.year.unique()):
        sub = eq_df[eq_df["ts"].dt.year == yr]
        s   = pd.to_numeric(sub["portfolio"])
        pct = (s.iloc[-1] / s.iloc[0] - 1) * 100
        print(f"  {yr:<6} | ${s.iloc[0]:>7.2f} | ${s.iloc[-1]:>7.2f} | {pct:>+7.1f}%")


def cmd_regimes():
    df = _load_trades()
    if df.empty:
        print("Keine Trades.")
        return

    print(f"\n{'='*68}")
    print(f"  {'Regime':>15} | {'Side':>5} | {'n':>4} | {'WR%':>6} | {'ΣPnL%':>8} | {'Ø Hebel':>7}")
    print(f"  {'-'*68}")
    for (regime, side), grp in df.groupby(["Regime", "Side"]):
        wr  = (grp["PnL_pct"] > 0).mean() * 100
        tot = grp["PnL_pct"].sum()
        avh = grp["Hebel"].mean()
        print(f"  {str(regime)[:15]:>15} | {side:>5} | {len(grp):>4} | "
              f"{wr:>5.1f}% | {tot:>+8.1f}% | {avh:>6.2f}×")

    wr_all  = (df["PnL_pct"] > 0).mean() * 100
    tot_all = df["PnL_pct"].sum()
    print(f"\n  Gesamt: n={len(df)}, WR={wr_all:.1f}%, ΣPnL={tot_all:+.1f}%")


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "status":
        cmd_status()
    elif args[0] == "list":
        n = int(args[1]) if len(args) > 1 else 20
        cmd_list(n)
    elif args[0] == "show":
        if len(args) < 2:
            print("Verwendung: python3 view_trades.py show <id>")
        else:
            cmd_show(int(args[1]))
    elif args[0] == "equity":
        cmd_equity()
    elif args[0] == "regimes":
        cmd_regimes()
    else:
        print(__doc__)
