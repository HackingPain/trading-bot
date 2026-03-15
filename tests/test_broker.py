"""Tests for broker module."""

import pytest
from unittest.mock import MagicMock, patch

from src.execution.broker import (
    Account,
    AlpacaBroker,
    AlpacaConfig,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)


@pytest.fixture
def alpaca_config():
    """Create test Alpaca configuration."""
    return AlpacaConfig(
        api_key="test_key",
        secret_key="test_secret",
        base_url="https://paper-api.alpaca.markets",
        is_paper=True,
    )


@pytest.fixture
def broker(alpaca_config):
    """Create an AlpacaBroker instance for testing."""
    return AlpacaBroker(alpaca_config)


class TestAlpacaConfig:
    """Tests for AlpacaConfig."""

    def test_from_settings_paper_mode(self):
        """Test config creation from settings in paper mode."""
        settings = {
            "trading": {"paper_mode": True},
            "api": {
                "alpaca": {
                    "key": "my_key",
                    "secret": "my_secret",
                }
            },
        }

        config = AlpacaConfig.from_settings(settings)

        assert config.api_key == "my_key"
        assert config.secret_key == "my_secret"
        assert config.is_paper is True
        assert "paper" in config.base_url

    def test_from_settings_live_mode_forces_paper_url(self):
        """Test that paper mode forces paper URL."""
        settings = {
            "trading": {"paper_mode": True},
            "api": {
                "alpaca": {
                    "key": "my_key",
                    "secret": "my_secret",
                    "base_url": "https://api.alpaca.markets",  # Live URL
                }
            },
        }

        config = AlpacaConfig.from_settings(settings)

        # Should force paper URL when paper_mode is True
        assert "paper" in config.base_url

    def test_from_settings_uses_env_vars(self, monkeypatch):
        """Test config uses environment variables as fallback."""
        monkeypatch.setenv("ALPACA_API_KEY", "env_key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "env_secret")

        settings = {"trading": {"paper_mode": True}, "api": {"alpaca": {}}}

        config = AlpacaConfig.from_settings(settings)

        assert config.api_key == "env_key"
        assert config.secret_key == "env_secret"


class TestOrder:
    """Tests for Order dataclass."""

    def test_market_order_creation(self):
        """Test creating a market order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )

        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 10
        assert order.order_type == OrderType.MARKET
        assert order.time_in_force == TimeInForce.DAY

    def test_limit_order_creation(self):
        """Test creating a limit order."""
        order = Order(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.LIMIT,
            limit_price=150.00,
        )

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.00

    def test_stop_order_creation(self):
        """Test creating a stop order."""
        order = Order(
            symbol="GOOGL",
            side=OrderSide.SELL,
            quantity=3,
            order_type=OrderType.STOP,
            stop_price=100.00,
        )

        assert order.order_type == OrderType.STOP
        assert order.stop_price == 100.00


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_successful_result(self):
        """Test successful order result."""
        result = OrderResult(
            success=True,
            order_id="abc123",
            status=OrderStatus.FILLED,
            filled_quantity=10,
            filled_price=150.50,
            message="Order filled",
        )

        assert result.success is True
        assert result.order_id == "abc123"
        assert result.status == OrderStatus.FILLED

    def test_failed_result(self):
        """Test failed order result."""
        result = OrderResult(
            success=False,
            status=OrderStatus.REJECTED,
            message="Insufficient buying power",
        )

        assert result.success is False
        assert result.order_id is None
        assert result.status == OrderStatus.REJECTED


class TestAccount:
    """Tests for Account dataclass."""

    def test_account_creation(self):
        """Test Account dataclass creation."""
        account = Account(
            account_id="123",
            equity=50000.0,
            cash=30000.0,
            buying_power=60000.0,
            positions_value=20000.0,
            is_paper=True,
        )

        assert account.equity == 50000.0
        assert account.is_paper is True
        assert account.trading_blocked is False


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test Position dataclass creation."""
        position = Position(
            symbol="AAPL",
            quantity=10,
            avg_entry_price=150.0,
            current_price=155.0,
            market_value=1550.0,
            cost_basis=1500.0,
            unrealized_pnl=50.0,
            unrealized_pnl_pct=3.33,
        )

        assert position.symbol == "AAPL"
        assert position.unrealized_pnl == 50.0
        assert position.side == "long"


class TestAlpacaBroker:
    """Tests for AlpacaBroker (with mocked client)."""

    def test_broker_initialization(self, broker):
        """Test broker initializes correctly."""
        assert broker.config.is_paper is True
        assert broker._initialized is False

    def test_broker_requires_credentials(self):
        """Test broker raises error without credentials."""
        config = AlpacaConfig(api_key="", secret_key="", is_paper=True)
        broker = AlpacaBroker(config)

        with pytest.raises(ValueError, match="credentials not configured"):
            broker._ensure_initialized()

    @patch("alpaca.trading.client.TradingClient")
    def test_get_account(self, mock_trading_client, broker):
        """Test getting account information."""
        # Setup mock
        mock_client = MagicMock()
        mock_trading_client.return_value = mock_client

        mock_account = MagicMock()
        mock_account.id = "123"
        mock_account.equity = "50000"
        mock_account.cash = "30000"
        mock_account.buying_power = "60000"
        mock_account.long_market_value = "20000"
        mock_account.short_market_value = "0"
        mock_account.daytrade_count = 0
        mock_account.pattern_day_trader = False
        mock_account.trading_blocked = False
        mock_account.currency = "USD"

        mock_client.get_account.return_value = mock_account

        # Test
        account = broker.get_account()

        assert account.equity == 50000.0
        assert account.cash == 30000.0
        assert account.is_paper is True

    @patch("alpaca.trading.client.TradingClient")
    def test_get_positions(self, mock_trading_client, broker):
        """Test getting positions."""
        mock_client = MagicMock()
        mock_trading_client.return_value = mock_client

        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "10"
        mock_pos.avg_entry_price = "150.0"
        mock_pos.current_price = "155.0"
        mock_pos.market_value = "1550.0"
        mock_pos.cost_basis = "1500.0"
        mock_pos.unrealized_pl = "50.0"
        mock_pos.unrealized_plpc = "0.0333"
        mock_pos.side = MagicMock(value="long")

        mock_client.get_all_positions.return_value = [mock_pos]

        positions = broker.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == 10.0

    @patch("alpaca.trading.client.TradingClient")
    def test_submit_order_success(self, mock_trading_client, broker):
        """Test successful order submission."""
        mock_client = MagicMock()
        mock_trading_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.id = "order123"
        mock_response.status = MagicMock(value="accepted")
        mock_response.filled_qty = "0"
        mock_response.filled_avg_price = None

        mock_client.submit_order.return_value = mock_response

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )

        result = broker.submit_order(order)

        assert result.success is True
        assert result.order_id == "order123"

    @patch("alpaca.trading.client.TradingClient")
    def test_submit_order_failure(self, mock_trading_client, broker):
        """Test failed order submission."""
        mock_client = MagicMock()
        mock_trading_client.return_value = mock_client

        mock_client.submit_order.side_effect = Exception("Insufficient buying power")

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10000,
            order_type=OrderType.MARKET,
        )

        result = broker.submit_order(order)

        assert result.success is False
        assert "Insufficient buying power" in result.message

    @patch("alpaca.trading.client.TradingClient")
    def test_cancel_order(self, mock_trading_client, broker):
        """Test order cancellation."""
        mock_client = MagicMock()
        mock_trading_client.return_value = mock_client

        result = broker.cancel_order("order123")

        assert result is True
        mock_client.cancel_order_by_id.assert_called_once_with("order123")

    @patch("alpaca.trading.client.TradingClient")
    def test_is_market_open(self, mock_trading_client, broker):
        """Test market status check."""
        mock_client = MagicMock()
        mock_trading_client.return_value = mock_client

        mock_clock = MagicMock()
        mock_clock.is_open = True

        mock_client.get_clock.return_value = mock_clock

        assert broker.is_market_open() is True

    @patch("alpaca.trading.client.TradingClient")
    def test_close_position(self, mock_trading_client, broker):
        """Test closing a position."""
        mock_client = MagicMock()
        mock_trading_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.id = "close123"
        mock_response.status = MagicMock(value="pending")

        mock_client.close_position.return_value = mock_response

        result = broker.close_position("AAPL")

        assert result.success is True
        mock_client.close_position.assert_called_once_with("AAPL")


class TestOrderSideAndType:
    """Tests for order enums."""

    def test_order_side_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"

    def test_time_in_force_values(self):
        """Test TimeInForce enum values."""
        assert TimeInForce.DAY.value == "day"
        assert TimeInForce.GTC.value == "gtc"
