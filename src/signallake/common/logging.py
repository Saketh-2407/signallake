"""structlog setup: pretty console output on a TTY, JSON lines otherwise."""

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json_logs: bool | None = None) -> None:
    """Configure structlog (and stdlib logging). Safe to call more than once."""
    if json_logs is None:
        json_logs = not sys.stderr.isatty()

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level.upper())

    renderer = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
