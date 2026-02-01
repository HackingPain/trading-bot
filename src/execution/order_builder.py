"""
Order builder utilities for easier order creation.

Provides fluent API for building various order types.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from .broker import Order, OrderSide, OrderType, TimeInForce

logger = logging.getLogger(__name__)


class OrderBuilder:
    """
    Fluent builder for creating orders.

    Example usage:
        order = (OrderBuilder()
            .symbol("AAPL")
            .buy(100)
            .limit(150.00)
            .good_til_cancelled()
            .build())
    """

    def __init__(self):
        self._symbol: Optional[str] = None
        self._side: Optional[OrderSide] = None
        self._quantity: int = 0
        self._order_type: OrderType = OrderType.MARKET
        self._limit_price: Optional[float] = None
        self._stop_price: Optional[float] = None
        self._trail_percent: Optional[float] = None
        self._time_in_force: TimeInForce = TimeInForce.DAY
        self._client_order_id: Optional[str] = None
        self._extended_hours: bool = False

    def symbol(self, symbol: str) -> "OrderBuilder":
        """Set the symbol."""
        self._symbol = symbol.upper()
        return self

    def buy(self, quantity: int) -> "OrderBuilder":
        """Set as buy order with quantity."""
        self._side = OrderSide.BUY
        self._quantity = quantity
        return self

    def sell(self, quantity: int) -> "OrderBuilder":
        """Set as sell order with quantity."""
        self._side = OrderSide.SELL
        self._quantity = quantity
        return self

    def quantity(self, qty: int) -> "OrderBuilder":
        """Set order quantity."""
        self._quantity = qty
        return self

    def market(self) -> "OrderBuilder":
        """Set as market order."""
        self._order_type = OrderType.MARKET
        return self

    def limit(self, price: float) -> "OrderBuilder":
        """Set as limit order with price."""
        self._order_type = OrderType.LIMIT
        self._limit_price = price
        return self

    def stop(self, price: float) -> "OrderBuilder":
        """Set as stop order with trigger price."""
        self._order_type = OrderType.STOP
        self._stop_price = price
        return self

    def stop_limit(self, stop_price: float, limit_price: float) -> "OrderBuilder":
        """Set as stop-limit order."""
        self._order_type = OrderType.STOP_LIMIT
        self._stop_price = stop_price
        self._limit_price = limit_price
        return self

    def trailing_stop(self, trail_percent: float) -> "OrderBuilder":
        """Set as trailing stop order with trail percentage."""
        self._order_type = OrderType.TRAILING_STOP
        self._trail_percent = trail_percent
        return self

    def day(self) -> "OrderBuilder":
        """Set time in force to DAY."""
        self._time_in_force = TimeInForce.DAY
        return self

    def good_til_cancelled(self) -> "OrderBuilder":
        """Set time in force to GTC."""
        self._time_in_force = TimeInForce.GTC
        return self

    def gtc(self) -> "OrderBuilder":
        """Alias for good_til_cancelled."""
        return self.good_til_cancelled()

    def immediate_or_cancel(self) -> "OrderBuilder":
        """Set time in force to IOC."""
        self._time_in_force = TimeInForce.IOC
        return self

    def ioc(self) -> "OrderBuilder":
        """Alias for immediate_or_cancel."""
        return self.immediate_or_cancel()

    def fill_or_kill(self) -> "OrderBuilder":
        """Set time in force to FOK."""
        self._time_in_force = TimeInForce.FOK
        return self

    def fok(self) -> "OrderBuilder":
        """Alias for fill_or_kill."""
        return self.fill_or_kill()

    def extended_hours(self, enabled: bool = True) -> "OrderBuilder":
        """Enable or disable extended hours trading."""
        self._extended_hours = enabled
        return self

    def client_order_id(self, order_id: str) -> "OrderBuilder":
        """Set custom client order ID."""
        self._client_order_id = order_id
        return self

    def with_generated_id(self) -> "OrderBuilder":
        """Generate a unique client order ID."""
        self._client_order_id = str(uuid.uuid4())
        return self

    def build(self) -> Order:
        """Build and return the Order."""
        if not self._symbol:
            raise ValueError("Symbol is required")
        if not self._side:
            raise ValueError("Order side (buy/sell) is required")
        if self._quantity <= 0:
            raise ValueError("Quantity must be positive")

        # Validate order type requirements
        if self._order_type == OrderType.LIMIT and self._limit_price is None:
            raise ValueError("Limit price is required for limit orders")
        if self._order_type == OrderType.STOP and self._stop_price is None:
            raise ValueError("Stop price is required for stop orders")
        if self._order_type == OrderType.STOP_LIMIT:
            if self._stop_price is None or self._limit_price is None:
                raise ValueError("Both stop and limit prices are required for stop-limit orders")
        if self._order_type == OrderType.TRAILING_STOP and self._trail_percent is None:
            raise ValueError("Trail percent is required for trailing stop orders")

        return Order(
            symbol=self._symbol,
            side=self._side,
            quantity=self._quantity,
            order_type=self._order_type,
            limit_price=self._limit_price,
            stop_price=self._stop_price,
            trail_percent=self._trail_percent,
            time_in_force=self._time_in_force,
            client_order_id=self._client_order_id,
            extended_hours=self._extended_hours,
        )


# Convenience functions for common order types

def market_buy(symbol: str, quantity: int) -> Order:
    """Create a market buy order."""
    return (OrderBuilder()
        .symbol(symbol)
        .buy(quantity)
        .market()
        .build())


def market_sell(symbol: str, quantity: int) -> Order:
    """Create a market sell order."""
    return (OrderBuilder()
        .symbol(symbol)
        .sell(quantity)
        .market()
        .build())


def limit_buy(symbol: str, quantity: int, price: float, tif: TimeInForce = TimeInForce.DAY) -> Order:
    """Create a limit buy order."""
    builder = OrderBuilder().symbol(symbol).buy(quantity).limit(price)
    if tif == TimeInForce.GTC:
        builder.gtc()
    return builder.build()


def limit_sell(symbol: str, quantity: int, price: float, tif: TimeInForce = TimeInForce.DAY) -> Order:
    """Create a limit sell order."""
    builder = OrderBuilder().symbol(symbol).sell(quantity).limit(price)
    if tif == TimeInForce.GTC:
        builder.gtc()
    return builder.build()


def stop_loss(symbol: str, quantity: int, stop_price: float) -> Order:
    """Create a stop loss (stop market) order."""
    return (OrderBuilder()
        .symbol(symbol)
        .sell(quantity)
        .stop(stop_price)
        .gtc()
        .build())


def stop_limit_sell(symbol: str, quantity: int, stop_price: float, limit_price: float) -> Order:
    """Create a stop-limit sell order."""
    return (OrderBuilder()
        .symbol(symbol)
        .sell(quantity)
        .stop_limit(stop_price, limit_price)
        .gtc()
        .build())


def trailing_stop_sell(symbol: str, quantity: int, trail_percent: float) -> Order:
    """Create a trailing stop sell order."""
    return (OrderBuilder()
        .symbol(symbol)
        .sell(quantity)
        .trailing_stop(trail_percent)
        .gtc()
        .build())


def bracket_orders(
    symbol: str,
    quantity: int,
    entry_price: float,
    take_profit_price: float,
    stop_loss_price: float,
) -> tuple[Order, Order, Order]:
    """
    Create a bracket of orders: entry, take profit, and stop loss.

    Returns:
        Tuple of (entry_order, take_profit_order, stop_loss_order)

    Note: These should be submitted with OCO (one-cancels-other) logic,
    which requires broker-specific implementation.
    """
    entry = limit_buy(symbol, quantity, entry_price, TimeInForce.GTC)
    take_profit = limit_sell(symbol, quantity, take_profit_price, TimeInForce.GTC)
    stop_loss_order = stop_loss(symbol, quantity, stop_loss_price)

    return entry, take_profit, stop_loss_order


@dataclass
class BracketOrderConfig:
    """Configuration for bracket orders."""
    symbol: str
    quantity: int
    side: OrderSide
    entry_type: OrderType = OrderType.MARKET
    entry_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    trailing_stop_percent: Optional[float] = None


class BracketOrderBuilder:
    """
    Builder for creating bracket orders (entry + profit target + stop loss).

    Note: Full bracket order support requires broker OCO functionality.
    This builder creates the individual orders that would compose a bracket.
    """

    def __init__(self, symbol: str):
        self._config = BracketOrderConfig(
            symbol=symbol.upper(),
            quantity=0,
            side=OrderSide.BUY,
        )

    def buy(self, quantity: int) -> "BracketOrderBuilder":
        """Set as buy order."""
        self._config.side = OrderSide.BUY
        self._config.quantity = quantity
        return self

    def sell(self, quantity: int) -> "BracketOrderBuilder":
        """Set as sell order."""
        self._config.side = OrderSide.SELL
        self._config.quantity = quantity
        return self

    def market_entry(self) -> "BracketOrderBuilder":
        """Use market order for entry."""
        self._config.entry_type = OrderType.MARKET
        return self

    def limit_entry(self, price: float) -> "BracketOrderBuilder":
        """Use limit order for entry."""
        self._config.entry_type = OrderType.LIMIT
        self._config.entry_price = price
        return self

    def take_profit(self, price: float) -> "BracketOrderBuilder":
        """Set take profit price."""
        self._config.take_profit_price = price
        return self

    def stop_loss(self, price: float) -> "BracketOrderBuilder":
        """Set stop loss price."""
        self._config.stop_loss_price = price
        return self

    def trailing_stop(self, percent: float) -> "BracketOrderBuilder":
        """Use trailing stop instead of fixed stop."""
        self._config.trailing_stop_percent = percent
        return self

    def build(self) -> dict[str, Order]:
        """
        Build bracket orders.

        Returns:
            Dict with keys 'entry', 'take_profit', 'stop_loss'
        """
        if self._config.quantity <= 0:
            raise ValueError("Quantity must be positive")

        orders = {}

        # Entry order
        entry_builder = OrderBuilder().symbol(self._config.symbol)
        if self._config.side == OrderSide.BUY:
            entry_builder.buy(self._config.quantity)
        else:
            entry_builder.sell(self._config.quantity)

        if self._config.entry_type == OrderType.LIMIT:
            if self._config.entry_price is None:
                raise ValueError("Entry price required for limit entry")
            entry_builder.limit(self._config.entry_price)
        else:
            entry_builder.market()

        orders["entry"] = entry_builder.build()

        # Exit orders (opposite side)
        exit_side = OrderSide.SELL if self._config.side == OrderSide.BUY else OrderSide.BUY

        # Take profit
        if self._config.take_profit_price:
            tp_builder = (OrderBuilder()
                .symbol(self._config.symbol)
                .quantity(self._config.quantity)
                .limit(self._config.take_profit_price)
                .gtc())

            if exit_side == OrderSide.SELL:
                tp_builder.sell(self._config.quantity)
            else:
                tp_builder.buy(self._config.quantity)

            orders["take_profit"] = tp_builder.build()

        # Stop loss
        if self._config.trailing_stop_percent:
            sl_builder = (OrderBuilder()
                .symbol(self._config.symbol)
                .quantity(self._config.quantity)
                .trailing_stop(self._config.trailing_stop_percent)
                .gtc())

            if exit_side == OrderSide.SELL:
                sl_builder.sell(self._config.quantity)
            else:
                sl_builder.buy(self._config.quantity)

            orders["stop_loss"] = sl_builder.build()

        elif self._config.stop_loss_price:
            sl_builder = (OrderBuilder()
                .symbol(self._config.symbol)
                .quantity(self._config.quantity)
                .stop(self._config.stop_loss_price)
                .gtc())

            if exit_side == OrderSide.SELL:
                sl_builder.sell(self._config.quantity)
            else:
                sl_builder.buy(self._config.quantity)

            orders["stop_loss"] = sl_builder.build()

        return orders
