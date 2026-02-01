"""Structured JSON logging for the trading bot."""

import json
import logging
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_location: bool = False,
        extra_fields: Optional[dict] = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_location = include_location
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data: dict[str, Any] = {}

        # Add timestamp
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcfromtimestamp(record.created).isoformat() + "Z"

        # Add log level
        if self.include_level:
            log_data["level"] = record.levelname

        # Add logger name
        if self.include_logger:
            log_data["logger"] = record.name

        # Add source location
        if self.include_location:
            log_data["location"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add message
        log_data["message"] = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            ):
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        # Add static extra fields
        for key, value in self.extra_fields.items():
            if key not in log_data:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class TradingContextFilter(logging.Filter):
    """Add trading context to log records."""

    _context = threading.local()

    @classmethod
    def set_context(cls, **kwargs) -> None:
        """Set context values for current thread."""
        if not hasattr(cls._context, "data"):
            cls._context.data = {}
        cls._context.data.update(kwargs)

    @classmethod
    def clear_context(cls) -> None:
        """Clear context for current thread."""
        cls._context.data = {}

    @classmethod
    def get_context(cls) -> dict:
        """Get current context."""
        if not hasattr(cls._context, "data"):
            cls._context.data = {}
        return cls._context.data.copy()

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        context = self.get_context()
        for key, value in context.items():
            setattr(record, key, value)
        return True


class TradingLogger:
    """
    Configured logger for the trading bot.

    Provides structured JSON logging with context support.
    """

    def __init__(
        self,
        name: str = "trading_bot",
        level: str = "INFO",
        log_file: Optional[str] = None,
        json_format: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        console_output: bool = True,
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []  # Clear existing handlers

        # Add context filter
        self.context_filter = TradingContextFilter()
        self.logger.addFilter(self.context_filter)

        # Create formatters
        if json_format:
            formatter = JSONFormatter(include_location=True)
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def set_context(self, **kwargs) -> None:
        """Set logging context for current thread."""
        TradingContextFilter.set_context(**kwargs)

    def clear_context(self) -> None:
        """Clear logging context for current thread."""
        TradingContextFilter.clear_context()

    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up logging for the trading bot.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        json_format: Use JSON formatting
        console_output: Output to console

    Returns:
        Configured logger
    """
    trading_logger = TradingLogger(
        name="trading_bot",
        level=level,
        log_file=log_file,
        json_format=json_format,
        console_output=console_output,
    )
    return trading_logger.get_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger with the trading_bot parent.

    Args:
        name: Logger name (will be prefixed with trading_bot.)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"trading_bot.{name}")


class LogContext:
    """Context manager for scoped logging context."""

    def __init__(self, **kwargs):
        self.context = kwargs
        self.previous_context: dict = {}

    def __enter__(self):
        self.previous_context = TradingContextFilter.get_context()
        TradingContextFilter.set_context(**self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        TradingContextFilter.clear_context()
        if self.previous_context:
            TradingContextFilter.set_context(**self.previous_context)
        return False


# Trade-specific logging helpers
def log_trade(
    logger: logging.Logger,
    action: str,
    symbol: str,
    quantity: int,
    price: float,
    order_id: Optional[str] = None,
    **kwargs,
) -> None:
    """Log a trade event with structured data."""
    logger.info(
        f"{action} {quantity} shares of {symbol} at ${price:.2f}",
        extra={
            "event_type": "trade",
            "action": action,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            **kwargs,
        },
    )


def log_signal(
    logger: logging.Logger,
    signal_type: str,
    symbol: str,
    strength: float,
    indicators: Optional[dict] = None,
    **kwargs,
) -> None:
    """Log a trading signal with structured data."""
    logger.info(
        f"{signal_type.upper()} signal for {symbol} (strength: {strength:.2f})",
        extra={
            "event_type": "signal",
            "signal_type": signal_type,
            "symbol": symbol,
            "strength": strength,
            "indicators": indicators or {},
            **kwargs,
        },
    )


def log_risk_event(
    logger: logging.Logger,
    event_type: str,
    message: str,
    **kwargs,
) -> None:
    """Log a risk management event."""
    logger.warning(
        f"Risk event: {message}",
        extra={
            "event_type": "risk",
            "risk_type": event_type,
            **kwargs,
        },
    )


def log_performance(
    logger: logging.Logger,
    metric_type: str,
    value: float,
    **kwargs,
) -> None:
    """Log a performance metric."""
    logger.info(
        f"Performance: {metric_type} = {value:.4f}",
        extra={
            "event_type": "performance",
            "metric_type": metric_type,
            "value": value,
            **kwargs,
        },
    )
