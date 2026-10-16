# ============================================================
#  Bitcoin Live-Bot – MA100 Strategie auf Binance Testnet
#  Kauft wenn Preis > MA100, verkauft wenn Preis < MA100
# ============================================================

import os
import time
import signal
import logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException

# ── EINSTELLUNGEN ────────────────────────────────────────────
# Binance Testnet API-Keys
# Erstelle kostenlose Testnet-Keys unter: https://testnet.binance.vision/
API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

SYMBOL        = "BTCUSDT"       # Handelspaar
QUANTITY      = 0.001           # BTC-Menge pro Trade (ca. 60–70 $ zum Stand 2024)
PRUEF_INTERVALL = 300           # Sekunden zwischen Prüfungen (5 Minuten)
MA_PERIODE    = 100             # Anzahl Tageskerzen für den MA
LOG_DATEI     = "trades.log"    # Logdatei für alle Trades

# ── LOGGING EINRICHTEN ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DATEI, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── BOT-KLASSE ───────────────────────────────────────────────
class MA100Bot:
    def __init__(self):
        log.info("=" * 60)
        log.info("  Bitcoin MA100 Live-Bot gestartet (Binance Testnet)")
        log.info("=" * 60)

        # Verbindung zum Testnet herstellen
        self.client = Client(API_KEY, API_SECRET, testnet=True)
        self.laeuft = True
        self.im_markt = False   # True = BTC gehalten, False = USDT gehalten

        # Graceful Shutdown bei Ctrl+C
        signal.signal(signal.SIGINT,  self._stopp_signal)
        signal.signal(signal.SIGTERM, self._stopp_signal)

        # Verbindung testen
        self._verbindung_testen()
        # Aktuellen Portfolio-Status laden
        self._status_laden()

    # ── Verbindung testen ────────────────────────────────────
    def _verbindung_testen(self):
        try:
            info = self.client.get_account()
            log.info("Verbindung zum Binance Testnet erfolgreich.")
        except BinanceAPIException as e:
            log.error(f"Verbindung fehlgeschlagen: {e}")
            log.error("Prüfe BINANCE_API_KEY und BINANCE_API_SECRET in deiner .env bzw. Umgebung")
            raise SystemExit(1)

    # ── Aktuellen Zustand laden (BTC oder USDT im Portfolio?) ─
    def _status_laden(self):
        try:
            btc_guthaben = self._guthaben("BTC")
            usdt_guthaben = self._guthaben("USDT")
            self.im_markt = btc_guthaben >= 0.0001   # Mindest-BTC-Schwelle
            log.info(f"Portfolio: {btc_guthaben:.6f} BTC | {usdt_guthaben:.2f} USDT")
            log.info(f"Aktueller Status: {'IM MARKT (BTC)' if self.im_markt else 'AUSSER MARKT (USDT)'}")
        except BinanceAPIException as e:
            log.warning(f"Status konnte nicht geladen werden: {e}")

    # ── Guthaben abfragen ────────────────────────────────────
    def _guthaben(self, asset: str) -> float:
        balance = self.client.get_asset_balance(asset=asset)
        return float(balance["free"]) if balance else 0.0

    # ── Letzten 100 Tageskerzen + MA100 berechnen ────────────
    def _ma100_berechnen(self) -> tuple[float, float]:
        """Gibt (aktueller_preis, ma100) zurück."""
        kerzen = self.client.get_klines(
            symbol=SYMBOL,
            interval=Client.KLINE_INTERVAL_1DAY,
            limit=MA_PERIODE + 1,   # +1 damit wir 100 vollständige Kerzen haben
        )
        # Kerzen-Format: [open_time, open, high, low, close, volume, ...]
        schlusskurse = [float(k[4]) for k in kerzen]

        aktueller_preis = schlusskurse[-1]
        ma100 = sum(schlusskurse[-MA_PERIODE:]) / MA_PERIODE

        return aktueller_preis, ma100

    # ── Kauforder ausführen ──────────────────────────────────
    def _kaufen(self, preis: float, ma100: float):
        usdt_vorher = self._guthaben("USDT")
        try:
            order = self.client.order_market_buy(
                symbol=SYMBOL,
                quantity=QUANTITY,
            )
            self.im_markt = True
            log.info(
                f"KAUF  | Preis: {preis:>10.2f} $ | MA100: {ma100:>10.2f} $ "
                f"| Menge: {QUANTITY} BTC | USDT vorher: {usdt_vorher:.2f}"
            )
            log.info(f"Order-ID: {order['orderId']} | Status: {order['status']}")
        except BinanceAPIException as e:
            log.error(f"Kauf fehlgeschlagen: {e}")

    # ── Verkauforder ausführen ───────────────────────────────
    def _verkaufen(self, preis: float, ma100: float):
        btc_vorher = self._guthaben("BTC")
        # Tatsächliches BTC-Guthaben verwenden (nicht hartcodierte QUANTITY)
        menge = round(btc_vorher, 6)
        if menge < 0.0001:
            log.warning("Kein BTC-Guthaben zum Verkaufen vorhanden.")
            self.im_markt = False
            return
        try:
            order = self.client.order_market_sell(
                symbol=SYMBOL,
                quantity=menge,
            )
            self.im_markt = False
            log.info(
                f"VERKAUF | Preis: {preis:>10.2f} $ | MA100: {ma100:>10.2f} $ "
                f"| Menge: {menge} BTC"
            )
            log.info(f"Order-ID: {order['orderId']} | Status: {order['status']}")
        except BinanceAPIException as e:
            log.error(f"Verkauf fehlgeschlagen: {e}")

    # ── Hauptschleife ────────────────────────────────────────
    def starten(self):
        log.info(f"Strategie: Kauf wenn Preis > MA{MA_PERIODE}, Verkauf wenn Preis < MA{MA_PERIODE}")
        log.info(f"Prüfintervall: alle {PRUEF_INTERVALL // 60} Minuten")
        log.info("Druecke Ctrl+C zum sauberen Beenden.\n")

        while self.laeuft:
            try:
                preis, ma100 = self._ma100_berechnen()
                abstand_pct = (preis - ma100) / ma100 * 100
                signal_text = "ÜBER MA100" if preis > ma100 else "UNTER MA100"

                log.info(
                    f"Preis: {preis:>10.2f} $  |  MA100: {ma100:>10.2f} $  "
                    f"|  Abstand: {abstand_pct:>+.2f}%  |  Signal: {signal_text}"
                )

                # ── Handelssignale auswerten ─────────────────
                if preis > ma100 and not self.im_markt:
                    log.info(">>> KAUFSIGNAL ausgeloest <<<")
                    self._kaufen(preis, ma100)

                elif preis < ma100 and self.im_markt:
                    log.info(">>> VERKAUFSSIGNAL ausgeloest <<<")
                    self._verkaufen(preis, ma100)

                else:
                    status = "HALTEN (BTC)" if self.im_markt else "WARTEN (USDT)"
                    log.info(f"Keine Aktion: {status}")

            except BinanceAPIException as e:
                log.error(f"API-Fehler: {e}")
            except Exception as e:
                log.error(f"Unerwarteter Fehler: {e}")

            # Warten – unterbruchsfähig durch kleinen Sleep-Loop
            if self.laeuft:
                log.info(f"Naechste Prüfung in {PRUEF_INTERVALL // 60} Minuten...\n")
                for _ in range(PRUEF_INTERVALL):
                    if not self.laeuft:
                        break
                    time.sleep(1)

        self._beenden()

    # ── Sauberes Beenden ─────────────────────────────────────
    def _stopp_signal(self, signum, frame):
        log.info("\nCtrl+C erkannt – Bot wird sauber gestoppt...")
        self.laeuft = False

    def _beenden(self):
        log.info("=" * 60)
        log.info("  Bot gestoppt. Alle Trades wurden in trades.log gespeichert.")
        log.info("=" * 60)


# ── HAUPTPROGRAMM ────────────────────────────────────────────
if __name__ == "__main__":
    bot = MA100Bot()
    bot.starten()
