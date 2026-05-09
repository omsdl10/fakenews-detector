"""
Structured logging using structlog.
Outputs JSON in production, colored console in development.
"""

import logging
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from backend.core.config import get_settings


def _add_app_context(
    logger: Any, method_name: str, event_dict: EventDict
) -> EventDict:
    """Inject app-level fields into every log record."""
    settings = get_settings()
    event_dict["app"] = settings.APP_NAME
    event_dict["version"] = settings.APP_VERSION
    return event_dict


def _drop_color_message_key(
    logger: Any, method_name: str, event_dict: EventDict
) -> EventDict:
    """Remove uvicorn's color_message to avoid log pollution."""
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging() -> None:
    settings = get_settings()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _add_app_context,
        _drop_color_message_key,
    ]

    if settings.LOG_JSON:
        # Production: machine-readable JSON
        processors: list[Processor] = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: human-friendly console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so uvicorn/sqlalchemy logs flow through
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    return structlog.get_logger(name)
