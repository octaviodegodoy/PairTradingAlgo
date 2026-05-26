"""
observability.py
----------------
Structured logging and Telegram alert helpers.

Structured logging
~~~~~~~~~~~~~~~~~~
``setup_logging()`` configures the root logger with two handlers:
  - StreamHandler  : human-readable output to stdout (level=INFO by default)
  - TimedRotatingFileHandler : structured JSON lines written to
    ``logs/trading_YYYY-MM-DD.log``, rotated daily and kept for 30 days.

Telegram alerts
~~~~~~~~~~~~~~~
To enable Telegram notifications set the following environment variables
(or add them to a ``.env`` file loaded before startup):

    TELEGRAM_BOT_TOKEN  — the token from @BotFather
    TELEGRAM_CHAT_ID    — your personal or group chat ID

When either variable is absent the ``TelegramAlerter`` class silently
no-ops so the live loop continues without interruption.

Usage example
~~~~~~~~~~~~~
    from observability import setup_logging, TelegramAlerter

    setup_logging()
    alerter = TelegramAlerter()

    alerter.send("Trade opened: WIN*/WDO* z=2.1")
    alerter.send_trade_open("WINM26", "WDOM26", z_score=2.13, hedge_ratio=1.42)
    alerter.send_trade_close("WINM26", "WDOM26", pnl=350.0, reason="z-reversion")
    alerter.send_stop_triggered(reason="max_loss", pnl=-420.0)
"""

import json
import logging
import logging.handlers
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path


# ── Structured JSON formatter ──────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# ── Public setup function ──────────────────────────────────────────────────────

def setup_logging(
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    backup_count: int = 30,
) -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    log_dir       : directory where log files are written (created if absent)
    console_level : minimum level for stdout handler
    file_level    : minimum level for the rotating file handler
    backup_count  : number of daily log files to keep
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    # Avoid adding duplicate handlers when called multiple times
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)

    # ── Console handler (human-readable) ──────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    )
    root.addHandler(console_handler)

    # ── Rotating file handler (JSON) ──────────────────────────────────────────
    log_file = os.path.join(log_dir, "trading.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=backup_count,
        encoding="utf-8",
        utc=True,
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(_JsonFormatter())
    # Suffix for rotated files: trading.log.2025-01-15
    file_handler.suffix = "%Y-%m-%d"
    root.addHandler(file_handler)


# ── Telegram alerter ──────────────────────────────────────────────────────────

class TelegramAlerter:
    """
    Send Telegram messages via the Bot API.

    If ``TELEGRAM_BOT_TOKEN`` or ``TELEGRAM_CHAT_ID`` are not set the instance
    silently no-ops — trading will not be interrupted by a missing alert.
    """

    _API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self._timeout = timeout
        self._enabled = bool(self._token and self._chat_id)
        self._logger = logging.getLogger(__name__)
        if not self._enabled:
            self._logger.debug(
                "TelegramAlerter: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
                "alerts are disabled."
            )

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------

    def send(self, text: str) -> bool:
        """
        Send a plain-text message.  Returns True on success, False on any error.
        Never raises — failures are logged at WARNING level only.
        """
        if not self._enabled:
            return False
        url = self._API_URL.format(token=self._token)
        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout):
                return True
        except (urllib.error.URLError, OSError) as exc:
            self._logger.warning("Telegram send failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Semantic helpers
    # ------------------------------------------------------------------

    def send_trade_open(
        self,
        symbol_y: str,
        symbol_x: str,
        z_score: float,
        hedge_ratio: float,
    ) -> bool:
        """Alert when a new spread grid is opened."""
        msg = (
            f"🟢 <b>Trade OPEN</b>\n"
            f"Pair: <code>{symbol_y} / {symbol_x}</code>\n"
            f"Z-score: <b>{z_score:.4f}</b>\n"
            f"Hedge ratio: {hedge_ratio:.4f}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        return self.send(msg)

    def send_trade_close(
        self,
        symbol_y: str,
        symbol_x: str,
        pnl: float,
        reason: str = "",
    ) -> bool:
        """Alert when positions are closed."""
        sign = "🟢" if pnl >= 0 else "🔴"
        msg = (
            f"{sign} <b>Trade CLOSE</b>\n"
            f"Pair: <code>{symbol_y} / {symbol_x}</code>\n"
            f"P&L: <b>{pnl:+.2f}</b>\n"
            f"Reason: {reason or 'n/a'}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        return self.send(msg)

    def send_stop_triggered(self, reason: str, pnl: float) -> bool:
        """Alert when a hard stop (max-loss or time) is triggered."""
        msg = (
            f"🛑 <b>STOP TRIGGERED</b>\n"
            f"Reason: {reason}\n"
            f"Session P&L: <b>{pnl:+.2f}</b>\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        return self.send(msg)

    def send_error(self, message: str) -> bool:
        """Alert on unexpected errors."""
        msg = (
            f"⚠️ <b>ERROR</b>\n"
            f"{message}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        return self.send(msg)
