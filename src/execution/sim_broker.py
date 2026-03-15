"""Simulated broker for paper trading and backtesting.

Provides a fully in-memory broker implementation that requires no external
services.  Useful for:
  - Paper trading against live or replayed data feeds
  - Unit / integration testing of strategies
  - Backtesting with realistic commission and slippage modelling
"""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Any, Optional

from .broker import (
    Broker,
    Account,
    Position,
    Order,
    OrderResult,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulatedBrokerConfig:
    """Configuration for the simulated broker."""

    initial_capital: float = 100_000.0
    commission_per_share: float = 0.0
    commission_minimum: float = 0.0
    commission_per_order: float = 0.0
    slippage_pct: float = 0.0005  # 0.05 % default slippage
    market_open: time = time(9, 30)  # ET
    market_close: time = time(16, 0)
    enforce_market_hours: bool = False
    account_id: str = "SIM-001"
    currency: str = "USD"


# ---------------------------------------------------------------------------
# Internal order record
# ---------------------------------------------------------------------------

@dataclass
class _InternalOrder:
    """Full lifecycle record kept for every order."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float]
    stop_price: Optional[float]
    trail_percent: Optional[float]
    time_in_force: TimeInForce
    client_order_id: Optional[str]
    extended_hours: bool

    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0
    filled_price: Optional[float] = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "qty": self.quantity,
            "filled_qty": self.filled_quantity,
            "filled_avg_price": self.filled_price,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "time_in_force": self.time_in_force.value,
            "client_order_id": self.client_order_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# Internal position record
# ---------------------------------------------------------------------------

@dataclass
class _InternalPosition:
    """Mutable position state tracked by the broker."""

    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    cost_basis: float = 0.0


# ---------------------------------------------------------------------------
# SimulatedBroker
# ---------------------------------------------------------------------------

class SimulatedBroker(Broker):
    """
    Paper-trading broker that executes orders entirely in memory.

    Price data must be fed externally via :meth:`set_last_price` so the broker
    can evaluate market and limit order fills.

    Example::

        broker = SimulatedBroker(SimulatedBrokerConfig(initial_capital=50_000))
        broker.set_last_price("AAPL", 185.50)

        result = broker.submit_order(Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        ))
    """

    def __init__(self, config: Optional[SimulatedBrokerConfig] = None) -> None:
        self._config = config or SimulatedBrokerConfig()
        self._cash: float = self._config.initial_capital
        self._positions: dict[str, _InternalPosition] = {}
        self._orders: dict[str, _InternalOrder] = {}
        self._last_prices: dict[str, float] = {}
        self._day_trade_count: int = 0

        # Track same-day round trips for PDT detection:
        # symbol -> list of entry timestamps within the current day
        self._day_trade_tracker: dict[str, list[str]] = defaultdict(list)

        self._force_market_open: Optional[bool] = None  # override for testing

        logger.info(
            "SimulatedBroker initialised: capital=%.2f, commission=%.4f/share, "
            "slippage=%.4f%%",
            self._config.initial_capital,
            self._config.commission_per_share,
            self._config.slippage_pct * 100,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: dict) -> "SimulatedBroker":
        """Create a :class:`SimulatedBroker` from a settings dictionary.

        Expected structure::

            {
              "trading": {
                "initial_capital": 100000,
                "commission_per_share": 0.005,
                "commission_minimum": 1.0,
                "commission_per_order": 0.0,
                "slippage_pct": 0.0005,
                "enforce_market_hours": false,
              }
            }
        """
        ts = settings.get("trading", {})
        sim = ts.get("simulation", ts)  # allow nested "simulation" key

        config = SimulatedBrokerConfig(
            initial_capital=float(sim.get("initial_capital", 100_000)),
            commission_per_share=float(sim.get("commission_per_share", 0.0)),
            commission_minimum=float(sim.get("commission_minimum", 0.0)),
            commission_per_order=float(sim.get("commission_per_order", 0.0)),
            slippage_pct=float(sim.get("slippage_pct", 0.0005)),
            enforce_market_hours=bool(sim.get("enforce_market_hours", False)),
            account_id=str(sim.get("account_id", "SIM-001")),
            currency=str(sim.get("currency", "USD")),
        )

        # Parse market hours if provided
        for key, attr in [("market_open", "market_open"), ("market_close", "market_close")]:
            raw = sim.get(key)
            if raw and isinstance(raw, str):
                parts = raw.split(":")
                setattr(config, attr, time(int(parts[0]), int(parts[1])))

        return cls(config)

    # ------------------------------------------------------------------
    # Price feed
    # ------------------------------------------------------------------

    def set_last_price(self, symbol: str, price: float) -> None:
        """Update the last-traded price for *symbol*.

        This must be called before submitting market orders or for limit
        order evaluation to work correctly.
        """
        self._last_prices[symbol] = price
        # Try to fill pending limit orders when price updates
        self._check_pending_orders(symbol, price)

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Return the last known price for *symbol*, or ``None``."""
        return self._last_prices.get(symbol)

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    def set_market_open_override(self, is_open: Optional[bool]) -> None:
        """Override the market-open check.  Pass ``None`` to revert to clock-based."""
        self._force_market_open = is_open

    def is_market_open(self) -> bool:
        """Check whether the (simulated) market is open."""
        if self._force_market_open is not None:
            return self._force_market_open

        now = datetime.now(timezone.utc)
        # Rough ET offset (does not handle DST perfectly, but adequate for sim)
        et_hour = (now.hour - 5) % 24
        et_time = time(et_hour, now.minute, now.second)

        # Weekday check (Mon=0 .. Sun=6)
        if now.weekday() >= 5:
            return False

        return self._config.market_open <= et_time < self._config.market_close

    # ------------------------------------------------------------------
    # Account & positions
    # ------------------------------------------------------------------

    def get_account(self) -> Account:
        positions_value = self._total_positions_value()
        equity = self._cash + positions_value

        return Account(
            account_id=self._config.account_id,
            equity=equity,
            cash=self._cash,
            buying_power=self._cash,  # simplified: no margin
            positions_value=positions_value,
            is_paper=True,
            day_trade_count=self._day_trade_count,
            pattern_day_trader=self._day_trade_count >= 4,
            trading_blocked=False,
            currency=self._config.currency,
        )

    def get_positions(self) -> list[Position]:
        result: list[Position] = []
        for sym, pos in self._positions.items():
            if pos.quantity == 0:
                continue
            result.append(self._to_position(pos))
        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        pos = self._positions.get(symbol)
        if pos is None or pos.quantity == 0:
            return None
        return self._to_position(pos)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> OrderResult:
        # Enforce market hours if configured
        if self._config.enforce_market_hours and not order.extended_hours:
            if not self.is_market_open():
                return OrderResult(
                    success=False,
                    status=OrderStatus.REJECTED,
                    message="Market is closed and extended_hours is not enabled",
                )

        # Basic validation
        if order.quantity <= 0:
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message="Order quantity must be positive",
            )

        order_id = order.client_order_id or str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        internal = _InternalOrder(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            trail_percent=order.trail_percent,
            time_in_force=order.time_in_force,
            client_order_id=order.client_order_id,
            extended_hours=order.extended_hours,
            status=OrderStatus.ACCEPTED,
            created_at=now_iso,
            updated_at=now_iso,
        )

        self._orders[order_id] = internal

        # Market orders fill immediately
        if order.order_type == OrderType.MARKET:
            return self._fill_market_order(internal)

        # Limit orders: check if immediately fillable
        if order.order_type == OrderType.LIMIT:
            last = self._last_prices.get(order.symbol)
            if last is not None and self._limit_price_crossed(internal, last):
                return self._fill_limit_order(internal, last)

        # Stop orders: check if immediately triggered
        if order.order_type == OrderType.STOP:
            last = self._last_prices.get(order.symbol)
            if last is not None and self._stop_price_triggered(internal, last):
                return self._fill_market_order(internal)

        # Otherwise the order stays open (ACCEPTED) for later evaluation
        logger.info(
            "Order %s accepted (pending): %s %d %s @ %s",
            order_id, order.side.value, order.quantity, order.symbol,
            order.order_type.value,
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            status=OrderStatus.ACCEPTED,
            message="Order accepted, pending fill",
        )

    def cancel_order(self, order_id: str) -> bool:
        internal = self._orders.get(order_id)
        if internal is None:
            logger.warning("Cancel failed: order %s not found", order_id)
            return False

        if internal.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            logger.warning(
                "Cancel failed: order %s already in terminal state %s",
                order_id, internal.status.value,
            )
            return False

        internal.status = OrderStatus.CANCELLED
        internal.updated_at = datetime.now(timezone.utc).isoformat()
        logger.info("Order %s cancelled", order_id)
        return True

    def get_order(self, order_id: str) -> Optional[dict]:
        internal = self._orders.get(order_id)
        if internal is None:
            return None
        return internal.to_dict()

    def get_open_orders(self) -> list[dict]:
        open_statuses = {OrderStatus.NEW, OrderStatus.PENDING, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}
        return [
            o.to_dict()
            for o in self._orders.values()
            if o.status in open_statuses
        ]

    # ------------------------------------------------------------------
    # Close positions
    # ------------------------------------------------------------------

    def close_position(self, symbol: str) -> OrderResult:
        pos = self._positions.get(symbol)
        if pos is None or pos.quantity == 0:
            return OrderResult(success=False, message=f"No open position for {symbol}")

        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
        qty = abs(int(pos.quantity))

        order = Order(
            symbol=symbol,
            side=side,
            quantity=qty,
            order_type=OrderType.MARKET,
        )
        return self.submit_order(order)

    def close_all_positions(self) -> list[OrderResult]:
        results: list[OrderResult] = []
        symbols = [s for s, p in self._positions.items() if p.quantity != 0]
        for symbol in symbols:
            results.append(self.close_position(symbol))
        return results

    # ------------------------------------------------------------------
    # Stats helpers (useful for strategy introspection)
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        return self._cash + self._total_positions_value()

    @property
    def cash(self) -> float:
        return self._cash

    def reset(self) -> None:
        """Reset broker to initial state."""
        self._cash = self._config.initial_capital
        self._positions.clear()
        self._orders.clear()
        self._last_prices.clear()
        self._day_trade_count = 0
        self._day_trade_tracker.clear()
        logger.info("SimulatedBroker reset to initial state")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _total_positions_value(self) -> float:
        total = 0.0
        for sym, pos in self._positions.items():
            price = self._last_prices.get(sym, pos.avg_entry_price)
            total += pos.quantity * price
        return total

    def _to_position(self, pos: _InternalPosition) -> Position:
        current_price = self._last_prices.get(pos.symbol, pos.avg_entry_price)
        market_value = pos.quantity * current_price
        unrealized_pnl = market_value - pos.cost_basis
        pnl_pct = (unrealized_pnl / pos.cost_basis * 100) if pos.cost_basis != 0 else 0.0

        return Position(
            symbol=pos.symbol,
            quantity=pos.quantity,
            avg_entry_price=pos.avg_entry_price,
            current_price=current_price,
            market_value=market_value,
            cost_basis=pos.cost_basis,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=pnl_pct,
            side="long" if pos.quantity > 0 else "short",
        )

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage: adverse direction relative to the trade side."""
        if side == OrderSide.BUY:
            return price * (1 + self._config.slippage_pct)
        else:
            return price * (1 - self._config.slippage_pct)

    def _compute_commission(self, quantity: int) -> float:
        commission = max(
            self._config.commission_per_share * quantity,
            self._config.commission_minimum,
        )
        commission += self._config.commission_per_order
        return commission

    def _fill_market_order(self, internal: _InternalOrder) -> OrderResult:
        last = self._last_prices.get(internal.symbol)
        if last is None:
            internal.status = OrderStatus.REJECTED
            internal.updated_at = datetime.now(timezone.utc).isoformat()
            return OrderResult(
                success=False,
                order_id=internal.order_id,
                status=OrderStatus.REJECTED,
                message=f"No price available for {internal.symbol}",
            )

        fill_price = self._apply_slippage(last, internal.side)
        return self._execute_fill(internal, fill_price)

    def _fill_limit_order(self, internal: _InternalOrder, market_price: float) -> OrderResult:
        # Limit orders fill at the limit price (or better)
        fill_price = internal.limit_price if internal.limit_price is not None else market_price
        return self._execute_fill(internal, fill_price)

    def _limit_price_crossed(self, internal: _InternalOrder, market_price: float) -> bool:
        if internal.limit_price is None:
            return False
        if internal.side == OrderSide.BUY:
            return market_price <= internal.limit_price
        else:
            return market_price >= internal.limit_price

    def _stop_price_triggered(self, internal: _InternalOrder, market_price: float) -> bool:
        if internal.stop_price is None:
            return False
        if internal.side == OrderSide.BUY:
            return market_price >= internal.stop_price
        else:
            return market_price <= internal.stop_price

    def _execute_fill(self, internal: _InternalOrder, fill_price: float) -> OrderResult:
        quantity = internal.quantity
        commission = self._compute_commission(quantity)
        cost = fill_price * quantity

        # Check buying power for buys
        if internal.side == OrderSide.BUY:
            total_cost = cost + commission
            if total_cost > self._cash:
                internal.status = OrderStatus.REJECTED
                internal.updated_at = datetime.now(timezone.utc).isoformat()
                return OrderResult(
                    success=False,
                    order_id=internal.order_id,
                    status=OrderStatus.REJECTED,
                    message=(
                        f"Insufficient buying power: need {total_cost:.2f}, "
                        f"have {self._cash:.2f}"
                    ),
                )

        # Track day trades (simplified: any round trip is counted)
        existing = self._positions.get(internal.symbol)
        if existing and existing.quantity != 0:
            is_closing = (
                (existing.quantity > 0 and internal.side == OrderSide.SELL) or
                (existing.quantity < 0 and internal.side == OrderSide.BUY)
            )
            if is_closing:
                self._day_trade_count += 1
                logger.debug(
                    "Day trade detected for %s (count=%d)",
                    internal.symbol, self._day_trade_count,
                )

        # Update position
        pos = self._positions.get(internal.symbol)
        if pos is None:
            pos = _InternalPosition(symbol=internal.symbol)
            self._positions[internal.symbol] = pos

        if internal.side == OrderSide.BUY:
            new_quantity = pos.quantity + quantity
            if new_quantity != 0 and pos.quantity >= 0:
                # Adding to long: update average
                pos.cost_basis += cost
                pos.avg_entry_price = pos.cost_basis / new_quantity if new_quantity else 0
            elif new_quantity != 0:
                # Flipping short to long or partial cover
                pos.cost_basis = abs(new_quantity) * fill_price
                pos.avg_entry_price = fill_price
            else:
                pos.cost_basis = 0.0
                pos.avg_entry_price = 0.0
            pos.quantity = new_quantity
            self._cash -= cost + commission
        else:  # SELL
            new_quantity = pos.quantity - quantity
            if new_quantity != 0 and pos.quantity <= 0:
                # Adding to short
                pos.cost_basis += cost
                pos.avg_entry_price = pos.cost_basis / abs(new_quantity) if new_quantity else 0
            elif new_quantity != 0 and pos.quantity > 0:
                # Partial close of long - keep original avg entry, reduce cost basis proportionally
                proportion_remaining = abs(new_quantity) / pos.quantity
                pos.cost_basis = pos.cost_basis * proportion_remaining
                # avg_entry_price stays the same for remaining shares
            else:
                pos.cost_basis = 0.0
                pos.avg_entry_price = 0.0
            pos.quantity = new_quantity
            self._cash += cost - commission

        # Remove flat positions to keep dict tidy
        if pos.quantity == 0:
            del self._positions[internal.symbol]

        # Update order record
        internal.status = OrderStatus.FILLED
        internal.filled_quantity = quantity
        internal.filled_price = fill_price
        internal.updated_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Order %s filled: %s %d %s @ %.4f (commission=%.4f)",
            internal.order_id, internal.side.value, quantity,
            internal.symbol, fill_price, commission,
        )

        return OrderResult(
            success=True,
            order_id=internal.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=fill_price,
            message="Order filled",
            raw_response=internal.to_dict(),
        )

    def _check_pending_orders(self, symbol: str, price: float) -> None:
        """Evaluate all pending orders for *symbol* against the new *price*."""
        open_statuses = {OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}
        for internal in list(self._orders.values()):
            if internal.symbol != symbol or internal.status not in open_statuses:
                continue

            if internal.order_type == OrderType.LIMIT:
                if self._limit_price_crossed(internal, price):
                    self._fill_limit_order(internal, price)

            elif internal.order_type == OrderType.STOP:
                if self._stop_price_triggered(internal, price):
                    self._fill_market_order(internal)

            elif internal.order_type == OrderType.STOP_LIMIT:
                if internal.stop_price is not None and self._stop_price_triggered(internal, price):
                    # Stop triggered -- now treat as limit
                    if self._limit_price_crossed(internal, price):
                        self._fill_limit_order(internal, price)
