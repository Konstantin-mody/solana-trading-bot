"""
levels.py — Look-Ahead-freie S/R-Level-Detektion für den 15min-Bot.

Konzept:
  - Swing-High/-Low mit N Bars beidseitig bestätigt (kein Look-Ahead: rechte Seite
    liegt N Bars in der Vergangenheit relativ zum aktuellen Bar)
  - Nahe Swings werden zu einem Level geclustert
  - Jeder Preisannäherung zählt als Touch; Break deaktiviert das Level
  - Mindestens MIN_TOUCHES nötig damit ein Level "bestätigt" gilt

Verwendung:
  tracker = LevelTracker()
  for i in range(len(df)):
      tracker.process_bar(i, df)
  active = tracker.get_active_levels()          # alle bestätigten aktiven Level
  state  = tracker.get_level_state_at(i, df)    # nur was bei Bar i bekannt war
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd


# ─── Default-Parameter ────────────────────────────────────────────────────────
SWING_N        = 5      # Bars links/rechts für Swing-Bestätigung
CLUSTER_PCT    = 0.003  # Swings innerhalb 0.3% → selbes Level
TOUCH_PCT      = 0.002  # Preis innerhalb 0.2% vom Level = Touch
BREAK_PCT      = 0.0015 # Close 0.15% jenseits des Levels = Level gebrochen
TOUCH_COOLDOWN = 8      # Mindestabstand (Bars) zwischen zwei Touches
MIN_TOUCHES    = 2      # Swing selbst (Touch 1) + mindestens 1 Rückkehr


# ─── Datenstruktur ────────────────────────────────────────────────────────────
@dataclass
class SRLevel:
    price:          float
    level_type:     str                            # "resistance" | "support"
    born_bar:       int                            # Bar-Index des ursprünglichen Swings
    born_ts:        pd.Timestamp
    touches:        List[Tuple[int, str]] = field(default_factory=list)  # [(bar, "swing"|"touch"|"break")]
    active:         bool = True
    last_touch_bar: int  = -999

    @property
    def touch_count(self) -> int:
        return sum(1 for _, k in self.touches if k != "break")

    @property
    def last_test_ts(self) -> pd.Timestamp | None:
        non_break = [(b, k) for b, k in self.touches if k != "break"]
        return None  # placeholder — timestamp lookup needs df


# ─── Tracker ──────────────────────────────────────────────────────────────────
class LevelTracker:
    def __init__(
        self,
        swing_n:        int   = SWING_N,
        cluster_pct:    float = CLUSTER_PCT,
        touch_pct:      float = TOUCH_PCT,
        break_pct:      float = BREAK_PCT,
        touch_cooldown: int   = TOUCH_COOLDOWN,
        min_touches:    int   = MIN_TOUCHES,
    ):
        self.swing_n        = swing_n
        self.cluster_pct    = cluster_pct
        self.touch_pct      = touch_pct
        self.break_pct      = break_pct
        self.touch_cooldown = touch_cooldown
        self.min_touches    = min_touches
        self.levels: List[SRLevel] = []

    # ── Hauptschleife ─────────────────────────────────────────────────────────
    def process_bar(self, i: int, df: pd.DataFrame) -> None:
        """
        Verarbeite Bar i ohne Look-Ahead.
        Swing bei j = i - swing_n: rechte N Bars [j+1 .. i] liegen in der Vergangenheit.
        """
        n = self.swing_n
        j = i - n

        if j < n:   # zu wenige Bars links
            return

        high_j = df["High"].iloc[j]
        low_j  = df["Low"].iloc[j]

        # ── Swing-Detektion ───────────────────────────────────────────────────
        left_h  = df["High"].iloc[j - n : j]
        right_h = df["High"].iloc[j + 1 : i + 1]
        left_l  = df["Low"].iloc[j - n : j]
        right_l = df["Low"].iloc[j + 1 : i + 1]

        if len(right_h) == n:
            is_swing_high = (high_j > left_h.max()) and (high_j > right_h.max())
            is_swing_low  = (low_j  < left_l.min()) and (low_j  < right_l.min())
        else:
            is_swing_high = is_swing_low = False

        if is_swing_high:
            self._add_or_cluster(high_j, "resistance", j, df.index[j])
        if is_swing_low:
            self._add_or_cluster(low_j,  "support",    j, df.index[j])

        # ── Bestehende Level mit aktuellem Bar aktualisieren ──────────────────
        self._update_levels(i, df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i])

    # ── Interne Hilfsmethoden ─────────────────────────────────────────────────
    def _add_or_cluster(
        self, price: float, level_type: str, bar: int, ts: pd.Timestamp
    ) -> None:
        for lvl in self.levels:
            if not lvl.active or lvl.level_type != level_type:
                continue
            if abs(price - lvl.price) / lvl.price <= self.cluster_pct:
                lvl.price = (lvl.price + price) / 2   # Gewichtetes Mittel
                return
        self.levels.append(SRLevel(
            price=price, level_type=level_type,
            born_bar=bar, born_ts=ts,
            touches=[(bar, "swing")],
            last_touch_bar=bar,
        ))

    def _update_levels(
        self, i: int, high: float, low: float, close: float
    ) -> None:
        for lvl in self.levels:
            if not lvl.active:
                continue
            p = lvl.price

            if lvl.level_type == "resistance":
                # Break zuerst prüfen
                if close > p * (1 + self.break_pct):
                    lvl.touches.append((i, "break"))
                    lvl.active = False
                    continue
                # Touch: High nähert sich Level von unten
                if (high >= p * (1 - self.touch_pct)
                        and i - lvl.last_touch_bar >= self.touch_cooldown):
                    lvl.touches.append((i, "touch"))
                    lvl.last_touch_bar = i

            else:  # support
                if close < p * (1 - self.break_pct):
                    lvl.touches.append((i, "break"))
                    lvl.active = False
                    continue
                if (low <= p * (1 + self.touch_pct)
                        and i - lvl.last_touch_bar >= self.touch_cooldown):
                    lvl.touches.append((i, "touch"))
                    lvl.last_touch_bar = i

    # ── Öffentliche Abfragen ──────────────────────────────────────────────────
    def get_active_levels(self, min_touches: int | None = None) -> List[SRLevel]:
        """Alle aktuell aktiven, bestätigten Level."""
        mt = min_touches if min_touches is not None else self.min_touches
        return [l for l in self.levels if l.active and l.touch_count >= mt]

    def get_level_state_at(
        self, bar: int, df: pd.DataFrame, min_touches: int | None = None
    ) -> List[SRLevel]:
        """
        Welche Level waren bei Bar `bar` bekannt und bestätigt?
        Look-Ahead-frei: Level ist erst ab born_bar + swing_n bekannt.
        """
        mt = min_touches if min_touches is not None else self.min_touches
        result = []
        for lvl in self.levels:
            # Level erst bekannt wenn swing_n Bars nach dem Swing vergangen sind
            if lvl.born_bar + self.swing_n > bar:
                continue
            # Nur Touches bis einschließlich dieses Bars
            touches_so_far = [(b, k) for b, k in lvl.touches if b <= bar]
            # Gebrochen? Nur wenn Break VOR diesem Bar (b < bar) — damit
            # break-at-bar-i noch für Break-Signale sichtbar bleibt.
            if any(k == "break" and b < bar for b, k in touches_so_far):
                continue
            # Ausreichend bestätigt?
            valid_touches = sum(1 for _, k in touches_so_far if k != "break")
            if valid_touches < mt:
                continue
            last_tb = max((b for b, _ in touches_so_far), default=-999)
            result.append(SRLevel(
                price=lvl.price, level_type=lvl.level_type,
                born_bar=lvl.born_bar, born_ts=lvl.born_ts,
                touches=touches_so_far, active=True,
                last_touch_bar=last_tb,
            ))
        return result

    def build_full(self, df: pd.DataFrame) -> "LevelTracker":
        """Convenience: gesamten DataFrame einmalig durchlaufen."""
        for i in range(len(df)):
            self.process_bar(i, df)
        return self
