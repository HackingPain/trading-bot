"""Broker abstraction and Alpaca implementation."""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..utils.retry import retry_with_backoff, NETWORK_EXCEPTIONS

logger = logging.getLogger(__name__)

# Alpaca-specific retryable exceptions
BROKER_RETRYABLE_EXCEPTIONS = NETWORK_EXCEPTIONS + (
    ConnectionResetError,
    TimeoutError,
)


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, Enum):
    """Time in force for orders."""
    DAY = "day"
    GTC = "gtc"  # Good til cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


class OrderStatus(str, Enum):
    """Order status."""
    NEW = "new"
    PENDING = "pending"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Account:
    """Broker account information."""
    account_id: str
    equity: float
    cash: float
    buying_power: float
    positions_value: float
    is_paper: bool = True
    day_trade_count: int = 0
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    currency: str = "USD"


@dataclass
class Position:
    """Broker position."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str = "long"


@dataclass
class Order:
    """Order to be submitted."""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_percent: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: Optional[str] = None
    extended_hours: bool = False


@dataclass
class OrderResult:
    """Result of order submission."""
    success: bool
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0
    filled_price: Optional[float] = None
    message: str = ""
    raw_response: dict = field(default_factory=dict)


class Broker(ABC):
    """
    Abstract base class for broker implementations.

    Defines the interface for account management, position tracking,
    and order execution.
    """

    @abstractmethod
    def get_account(self) -> Account:
        """Get current account information."""
        pass

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all current positions."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult:
        """Submit an order for execution."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[dict]:
        """Get order status by ID."""
        pass

    @abstractmethod
    def get_open_orders(self) -> list[dict]:
        """Get all open orders."""
        pass

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> OrderResult:
        """Close entire position for a symbol."""
        pass

    @abstractmethod
    def close_all_positions(self) -> list[OrderResult]:
        """Close all open positions."""
        pass


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    is_paper: bool = True

    @classmethod
    def from_settings(cls, settings: dict) -> "AlpacaConfig":
        """Create config from settings dictionary."""
        api_settings = settings.get("api", {}).get("alpaca", {})
        trading_settings = settings.get("trading", {})

        # Get from settings or environment variables
        api_key = api_settings.get("key") or os.getenv("ALPACA_API_KEY", "")
        secret_key = api_settings.get("secret") or os.getenv("ALPACA_SECRET_KEY", "")

        is_paper = trading_settings.get("paper_mode", True)
        base_url = api_settings.get("base_url", "https://paper-api.alpaca.markets")

        # Force paper URL if paper mode is enabled
        if is_paper and "paper" not in base_url:
            base_url = "https://paper-api.alpaca.markets"

        return cls(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
            data_url=api_settings.get("data_url", "https://data.alpaca.markets"),
            is_paper=is_paper,
        )


class AlpacaBroker(Broker):
    """
    Alpaca broker implementation.

    Supports both paper and live trading through the Alpaca API.
    """

    def __init__(self, config: AlpacaConfig):
        self.config = config
        self._client = None
        self._trading_client = None
        self._initialized = False

        # Verify paper mode safety
        if not config.is_paper:
            logger.warning("⚠️  LIVE TRADING MODE ENABLED - Real money at risk!")

    @classmethod
    def from_settings(cls, settings: dict) -> "AlpacaBroker":
        """Create broker from settings dictionary."""
        config = AlpacaConfig.from_settings(settings)
        return cls(config)

    def _ensure_initialized(self) -> None:
        """Lazy initialization of Alpaca client."""
        if self._initialized:
            return

        if not self.config.api_key or not self.config.secret_key:
            raise ValueError(
                "Alpaca API credentials not configured. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import (
                GetOrdersRequest,
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                TrailingStopOrderRequest,
            )
            from alpaca.trading.enums import (
                OrderSide as AlpacaOrderSide,
                TimeInForce as AlpacaTimeInForce,
                QueryOrderStatus,
            )

            self._trading_client = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.is_paper,
            )
            self._initialized = True
            logger.info(f"Alpaca client initialized (paper={self.config.is_paper})")

        except ImportError:
            raise ImportError("alpaca-py package not installed. Run: pip install alpaca-py")

    @retry_with_backoff(max_attempts=3, base_delay=1.0, retryable_exceptions=BROKER_RETRYABLE_EXCEPTIONS)
    def get_account(self) -> Account:
        """Get current account information from Alpaca."""
        self._ensure_initialized()

        account = self._trading_client.get_account()

        return Account(
            account_id=account.id,
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            positions_value=float(account.long_market_value) + float(account.short_market_value),
            is_paper=self.config.is_paper,
            day_trade_count=account.daytrade_count,
            pattern_day_trader=account.pattern_day_trader,
            trading_blocked=account.trading_blocked,
            currency=account.currency,
        )

    @retry_with_backoff(max_attempts=3, base_delay=1.0, retryable_exceptions=BROKER_RETRYABLE_EXCEPTIONS)
    def get_positions(self) -> list[Position]:
        """Get all current positions from Alpaca."""
        self._ensure_initialized()

        positions = self._trading_client.get_all_positions()
        result = []

        for pos in positions:
            result.append(Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                cost_basis=float(pos.cost_basis),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
                side=pos.side.value,
            ))

        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        self._ensure_initialized()

        try:
            pos = self._trading_client.get_open_position(symbol)

            return Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                cost_basis=float(pos.cost_basis),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
                side=pos.side.value,
            )

        except Exception as e:
            if "position does not exist" in str(e).lower():
                return None
            logger.error(f"Failed to get position for {symbol}: {e}")
            raise

    def submit_order(self, order: Order) -> OrderResult:
        """Submit an order to Alpaca."""
        self._ensure_initialized()

        # Log order before submission
        logger.info(
            f"Submitting order: {order.side.value} {order.quantity} {order.symbol} "
            f"@ {order.order_type.value} (paper={self.config.is_paper})"
        )

        try:
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                TrailingStopOrderRequest,
            )
            from alpaca.trading.enums import (
                OrderSide as AlpacaOrderSide,
                TimeInForce as AlpacaTimeInForce,
            )

            # Map order side
            side = AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL

            # Map time in force
            tif_map = {
                TimeInForce.DAY: AlpacaTimeInForce.DAY,
                TimeInForce.GTC: AlpacaTimeInForce.GTC,
                TimeInForce.IOC: AlpacaTimeInForce.IOC,
                TimeInForce.FOK: AlpacaTimeInForce.FOK,
            }
            tif = tif_map.get(order.time_in_force, AlpacaTimeInForce.DAY)

            # Create appropriate order request
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.LIMIT:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.STOP:
                request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    stop_price=order.stop_price,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.STOP_LIMIT:
                from alpaca.trading.requests import StopLimitOrderRequest
                request = StopLimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    stop_price=order.stop_price,
                    limit_price=order.limit_price,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.TRAILING_STOP:
                request = TrailingStopOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    trail_percent=order.trail_percent,
                    client_order_id=order.client_order_id,
                )
            else:
                return OrderResult(
                    success=False,
                    message=f"Unsupported order type: {order.order_type}",
                )

            # Submit order
            response = self._trading_client.submit_order(request)

            logger.info(f"Order submitted: {response.id} status={response.status}")

            return OrderResult(
                success=True,
                order_id=str(response.id),
                status=OrderStatus(response.status.value),
                filled_quantity=float(response.filled_qty or 0),
                filled_price=float(response.filled_avg_price) if response.filled_avg_price else None,
                message="Order submitted successfully",
                raw_response={"id": str(response.id), "status": response.status.value},
            )

        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        self._ensure_initialized()

        try:
            self._trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[dict]:
        """Get order status by ID."""
        self._ensure_initialized()

        try:
            order = self._trading_client.get_order_by_id(order_id)
            return {
                "id": str(order.id),
                "symbol": order.symbol,
                "side": order.side.value,
                "type": order.type.value,
                "qty": float(order.qty),
                "filled_qty": float(order.filled_qty or 0),
                "status": order.status.value,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            }

        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def get_open_orders(self) -> list[dict]:
        """Get all open orders."""
        self._ensure_initialized()

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self._trading_client.get_orders(request)

            return [
                {
                    "id": str(order.id),
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "type": order.type.value,
                    "qty": float(order.qty),
                    "filled_qty": float(order.filled_qty or 0),
                    "status": order.status.value,
                }
                for order in orders
            ]

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    @retry_with_backoff(max_attempts=3, base_delay=0.5, retryable_exceptions=BROKER_RETRYABLE_EXCEPTIONS)
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        self._ensure_initialized()

        try:
            clock = self._trading_client.get_clock()
            return clock.is_open

        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False

    def close_position(self, symbol: str) -> OrderResult:
        """Close entire position for a symbol."""
        self._ensure_initialized()

        logger.info(f"Closing position: {symbol}")

        try:
            response = self._trading_client.close_position(symbol)

            return OrderResult(
                success=True,
                order_id=str(response.id),
                status=OrderStatus(response.status.value),
                message=f"Position close order submitted for {symbol}",
            )

        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return OrderResult(
                success=False,
                message=str(e),
            )

    def close_all_positions(self) -> list[OrderResult]:
        """Close all open positions."""
        self._ensure_initialized()

        logger.warning("Closing ALL positions")

        try:
            responses = self._trading_client.close_all_positions(cancel_orders=True)
            results = []

            for response in responses:
                if hasattr(response, "id"):
                    results.append(OrderResult(
                        success=True,
                        order_id=str(response.id),
                        message="Position close order submitted",
                    ))
                else:
                    results.append(OrderResult(
                        success=False,
                        message=str(response),
                    ))

            return results

        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return [OrderResult(success=False, message=str(e))]

    def get_clock(self) -> dict:
        """Get market clock information."""
        self._ensure_initialized()

        try:
            clock = self._trading_client.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": clock.next_open.isoformat() if clock.next_open else None,
                "next_close": clock.next_close.isoformat() if clock.next_close else None,
            }

        except Exception as e:
            logger.error(f"Failed to get clock: {e}")
            return {"is_open": False}
