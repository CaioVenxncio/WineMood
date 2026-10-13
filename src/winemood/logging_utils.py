"""Structured (key=value) logging on top of the standard library."""

from __future__ import annotations

import logging
from typing import Any

_RESERVED = set(vars(logging.makeLogRecord({})))


class KeyValueFormatter(logging.Formatter):
    """Render records as ``ts level logger msg key=value ...``.

    Extra fields passed via ``logger.info("msg", extra={...})`` are appended.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')} "
            f"level={record.levelname} logger={record.name} msg={record.getMessage()!r}"
        )
        extras = {k: v for k, v in vars(record).items() if k not in _RESERVED}
        extras.pop("message", None)
        extras.pop("asctime", None)
        if extras:
            base += " " + " ".join(f"{k}={v!r}" for k, v in sorted(extras.items()))
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install the key=value formatter on the ``winemood`` logger (idempotent)."""
    logger = logging.getLogger("winemood")
    logger.setLevel(level)
    if not any(getattr(h, "_winemood", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(KeyValueFormatter())
        handler._winemood = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("winemood") else f"winemood.{name}")


def log_kv(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    logger.log(level, msg, extra=fields)
