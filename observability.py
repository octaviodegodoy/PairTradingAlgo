"""
observability.py
----------------
Structured logging helpers.

Structured logging
~~~~~~~~~~~~~~~~~~
``setup_logging()`` configures the root logger with two handlers:
  - StreamHandler  : human-readable output to stdout (level=INFO by default)
  - TimedRotatingFileHandler : structured JSON lines written to
    ``logs/trading_YYYY-MM-DD.log``, rotated daily and kept for 30 days.

Usage example
~~~~~~~~~~~~~
    from observability import setup_logging

    setup_logging()
"""

import json
import logging
import logging.handlers
import os
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
    log_dir: str = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    backup_count: int = 30,
) -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    log_dir       : directory where log files are written (created if absent).
                    Defaults to a 'logs' folder next to this file.
    console_level : minimum level for stdout handler
    file_level    : minimum level for the rotating file handler
    backup_count  : number of daily log files to keep
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
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

