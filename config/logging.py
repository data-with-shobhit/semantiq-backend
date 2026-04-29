"""Structlog configuration — call configure_logging() once at process startup."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog

_LOG_DIR = Path("logs")


def configure_logging(log_level: str = "INFO", log_format: str = "console") -> None:
    try:
        level = getattr(logging, log_level.upper(), logging.INFO)

        _LOG_DIR.mkdir(exist_ok=True)

        shared_processors: list[structlog.types.Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
        ]

        if log_format == "json":
            console_renderer = structlog.processors.JSONRenderer()
        else:
            console_renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Console handler
        console_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                console_renderer,
            ],
        )
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(console_formatter)

        # File handler — always JSON for easy parsing
        file_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_DIR / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)

        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(console_handler)
        root.addHandler(file_handler)
        root.setLevel(level)

        for name in ("httpx", "httpcore", "asyncio", "urllib3", "multipart"):
            logging.getLogger(name).setLevel(logging.WARNING)

    except Exception as exc:
        logging.basicConfig(level=logging.INFO)
        logging.error("logging.configure_failed: %s", exc)
        raise
    finally:
        logging.getLogger(__name__).debug("logging.configure_exit")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    try:
        return structlog.get_logger(name)
    except Exception as exc:
        logging.error("logging.get_logger_failed: %s", exc)
        raise
