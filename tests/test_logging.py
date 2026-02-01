"""Tests for structured JSON logging."""

import json
import logging
import tempfile
from pathlib import Path

import pytest

from src.utils.logging import (
    JSONFormatter,
    TradingContextFilter,
    TradingLogger,
    LogContext,
    setup_logging,
    get_logger,
    log_trade,
    log_signal,
    log_risk_event,
    log_performance,
)


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_basic_format(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_format_with_args(self):
        """Test formatting with message arguments."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Value: %d",
            args=(42,),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["message"] == "Value: 42"

    def test_format_with_extra_fields(self):
        """Test formatting with static extra fields."""
        formatter = JSONFormatter(extra_fields={"app": "trading_bot", "version": "1.0"})
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["app"] == "trading_bot"
        assert data["version"] == "1.0"

    def test_format_with_location(self):
        """Test formatting with source location."""
        formatter = JSONFormatter(include_location=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_function"

        result = formatter.format(record)
        data = json.loads(result)

        assert "location" in data
        assert data["location"]["file"] == "/path/to/test.py"
        assert data["location"]["line"] == 10
        assert data["location"]["function"] == "test_function"

    def test_format_without_optional_fields(self):
        """Test formatting without optional fields."""
        formatter = JSONFormatter(
            include_timestamp=False,
            include_level=False,
            include_logger=False,
        )
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" not in data
        assert "level" not in data
        assert "logger" not in data
        assert data["message"] == "Test"

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "Test error" in data["exception"]


class TestTradingContextFilter:
    """Tests for logging context filter."""

    def setup_method(self):
        """Clear context before each test."""
        TradingContextFilter.clear_context()

    def test_set_and_get_context(self):
        """Test setting and getting context."""
        TradingContextFilter.set_context(symbol="AAPL", order_id="123")
        context = TradingContextFilter.get_context()

        assert context["symbol"] == "AAPL"
        assert context["order_id"] == "123"

    def test_clear_context(self):
        """Test clearing context."""
        TradingContextFilter.set_context(symbol="AAPL")
        TradingContextFilter.clear_context()
        context = TradingContextFilter.get_context()

        assert context == {}

    def test_filter_adds_context_to_record(self):
        """Test that filter adds context to log records."""
        TradingContextFilter.set_context(symbol="MSFT", quantity=100)
        filter_instance = TradingContextFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = filter_instance.filter(record)

        assert result is True
        assert record.symbol == "MSFT"
        assert record.quantity == 100

    def test_update_context(self):
        """Test updating existing context."""
        TradingContextFilter.set_context(symbol="AAPL", quantity=50)
        TradingContextFilter.set_context(quantity=100, price=150.0)
        context = TradingContextFilter.get_context()

        assert context["symbol"] == "AAPL"
        assert context["quantity"] == 100
        assert context["price"] == 150.0


class TestLogContext:
    """Tests for LogContext context manager."""

    def setup_method(self):
        """Clear context before each test."""
        TradingContextFilter.clear_context()

    def test_context_manager_sets_context(self):
        """Test that context manager sets context."""
        with LogContext(symbol="GOOGL", action="buy"):
            context = TradingContextFilter.get_context()
            assert context["symbol"] == "GOOGL"
            assert context["action"] == "buy"

    def test_context_manager_clears_on_exit(self):
        """Test that context is cleared on exit."""
        with LogContext(symbol="GOOGL"):
            pass

        context = TradingContextFilter.get_context()
        assert "symbol" not in context

    def test_context_manager_restores_previous(self):
        """Test that previous context is restored."""
        TradingContextFilter.set_context(symbol="AAPL")

        with LogContext(symbol="GOOGL"):
            assert TradingContextFilter.get_context()["symbol"] == "GOOGL"

        assert TradingContextFilter.get_context()["symbol"] == "AAPL"


class TestTradingLogger:
    """Tests for TradingLogger class."""

    def test_logger_creation(self):
        """Test logger creation."""
        trading_logger = TradingLogger(
            name="test_logger",
            level="DEBUG",
            console_output=False,
        )
        logger = trading_logger.get_logger()

        assert logger.name == "test_logger"
        assert logger.level == logging.DEBUG

    def test_logger_with_file_handler(self):
        """Test logger with file handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            trading_logger = TradingLogger(
                name="file_test",
                log_file=str(log_file),
                console_output=False,
            )
            logger = trading_logger.get_logger()

            logger.info("Test message")

            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

    def test_logger_json_format(self):
        """Test logger with JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            trading_logger = TradingLogger(
                name="json_test",
                log_file=str(log_file),
                json_format=True,
                console_output=False,
            )
            logger = trading_logger.get_logger()

            logger.info("JSON test")

            content = log_file.read_text()
            data = json.loads(content.strip())
            assert data["message"] == "JSON test"

    def test_set_context(self):
        """Test setting context through TradingLogger."""
        trading_logger = TradingLogger(
            name="context_test",
            console_output=False,
        )

        trading_logger.set_context(symbol="TSLA")
        context = TradingContextFilter.get_context()

        assert context["symbol"] == "TSLA"

        trading_logger.clear_context()


class TestLoggingHelpers:
    """Tests for logging helper functions."""

    def setup_method(self):
        """Set up test logger."""
        self.logger = logging.getLogger("test_helpers")
        self.logger.setLevel(logging.DEBUG)
        self.handler = logging.handlers.MemoryHandler(capacity=100)
        self.handler.setFormatter(JSONFormatter())
        self.logger.addHandler(self.handler)

    def teardown_method(self):
        """Clean up handler."""
        self.handler.close()
        self.logger.removeHandler(self.handler)

    def test_log_trade(self):
        """Test log_trade helper."""
        log_trade(
            self.logger,
            action="BUY",
            symbol="AAPL",
            quantity=100,
            price=150.50,
            order_id="ord_123",
        )

        self.handler.flush()
        assert len(self.handler.buffer) == 1

        record = self.handler.buffer[0]
        assert record.event_type == "trade"
        assert record.action == "BUY"
        assert record.symbol == "AAPL"
        assert record.quantity == 100
        assert record.price == 150.50
        assert record.order_id == "ord_123"

    def test_log_signal(self):
        """Test log_signal helper."""
        log_signal(
            self.logger,
            signal_type="buy",
            symbol="MSFT",
            strength=0.85,
            indicators={"rsi": 25, "macd": 0.5},
        )

        self.handler.flush()
        record = self.handler.buffer[0]

        assert record.event_type == "signal"
        assert record.signal_type == "buy"
        assert record.symbol == "MSFT"
        assert record.strength == 0.85
        assert record.indicators["rsi"] == 25

    def test_log_risk_event(self):
        """Test log_risk_event helper."""
        log_risk_event(
            self.logger,
            event_type="circuit_breaker",
            message="Daily loss limit reached",
            current_loss=0.025,
            limit=0.02,
        )

        self.handler.flush()
        record = self.handler.buffer[0]

        assert record.event_type == "risk"
        assert record.risk_type == "circuit_breaker"
        assert record.current_loss == 0.025

    def test_log_performance(self):
        """Test log_performance helper."""
        log_performance(
            self.logger,
            metric_type="sharpe_ratio",
            value=1.85,
            period="30d",
        )

        self.handler.flush()
        record = self.handler.buffer[0]

        assert record.event_type == "performance"
        assert record.metric_type == "sharpe_ratio"
        assert record.value == 1.85
        assert record.period == "30d"


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        logger = setup_logging(
            level="DEBUG",
            json_format=False,
            console_output=False,
        )

        assert logger.name == "trading_bot"
        assert logger.level == logging.DEBUG

    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("broker")

        assert logger.name == "trading_bot.broker"


# Add MemoryHandler to logging.handlers for tests
import logging.handlers
