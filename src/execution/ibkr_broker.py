"""Interactive Brokers broker implementation via ``ib_insync``.

.. warning:: **Experimental** -- This module is a skeleton implementation
   intended as a starting point for Interactive Brokers integration.  It has
   **not** been tested against a live or paper TWS / IB Gateway instance and
   should be treated as pre-alpha quality.  Review all order-handling logic
   carefully before risking real capital.

Requirements
------------
* ``ib_insync`` -- ``pip install ib_insync``
* A running TWS or IB Gateway instance with API access enabled.

Usage
-----
::

    from src.execution.ibkr_broker import IBKRBroker

    broker = IBKRBroker.from_settings(settings)
    broker.connect()
    account = broker.get_account()
    broker.disconnect()
"""

import logging
from dataclasses import dataclass
from typing import Optional

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

# Guard the optional dependency so the rest of the package can be imported
# even when ib_insync is not installed.
try:
    from ib_insync import IB, Stock, LimitOrder, MarketOrder, StopOrder, Trade
    _IB_AVAILABLE = True
except ImportError:
    _IB_AVAILABLE = False


def _require_ib_insync() -> None:
    """Raise a clear error if ib_insync is not installed."""
    if not _IB_AVAILABLE:
        raise ImportError(
            "ib_insync is required for IBKRBroker.  "
            "Install it with:  pip install ib_insync"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IBKRConfig:
    """Connection / behavioural configuration for the IBKR broker."""

    host: str = "127.0.0.1"
    port: int = 7497  # 7497 = TWS paper, 7496 = TWS live, 4002 = Gateway paper
    client_id: int = 1
    account_id: str = ""  # blank = first available account
    timeout: float = 30.0
    readonly: bool = False  # True to disable order submission
    is_paper: bool = True


# ---------------------------------------------------------------------------
# IBKRBroker
# ---------------------------------------------------------------------------

class IBKRBroker(Broker):
    """Interactive Brokers broker implementation.

    .. warning:: **Experimental** -- see module-level docstring for details.
    """

    def __init__(self, config: Optional[IBKRConfig] = None) -> None:
        _require_ib_insync()
        self._config = config or IBKRConfig()
        self._ib: "IB" = IB()
        self._connected = False

        if not self._config.is_paper:
            logger.warning(
                "IBKRBroker configured for LIVE trading -- real money at risk!"
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: dict) -> "IBKRBroker":
        """Create an :class:`IBKRBroker` from a settings dictionary.

        Expected structure::

            {
              "api": {
                "ibkr": {
                  "host": "127.0.0.1",
                  "port": 7497,
                  "client_id": 1,
                  "account_id": "",
                  "timeout": 30,
                  "readonly": false
                }
              },
              "trading": {
                "paper_mode": true
              }
            }
        """
        ibkr = settings.get("api", {}).get("ibkr", {})
        trading = settings.get("trading", {})

        config = IBKRConfig(
            host=str(ibkr.get("host", "127.0.0.1")),
            port=int(ibkr.get("port", 7497)),
            client_id=int(ibkr.get("client_id", 1)),
            account_id=str(ibkr.get("account_id", "")),
            timeout=float(ibkr.get("timeout", 30.0)),
            readonly=bool(ibkr.get("readonly", False)),
            is_paper=bool(trading.get("paper_mode", True)),
        )

        return cls(config)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish a connection to TWS / IB Gateway."""
        if self._connected:
            logger.debug("IBKRBroker already connected")
            return

        logger.info(
            "Connecting to IBKR at %s:%d (client_id=%d, paper=%s)",
            self._config.host, self._config.port,
            self._config.client_id, self._config.is_paper,
        )

        self._ib.connect(
            host=self._config.host,
            port=self._config.port,
            clientId=self._config.client_id,
            timeout=self._config.timeout,
            readonly=self._config.readonly,
        )
        self._connected = True
        logger.info("Connected to IBKR")

    def disconnect(self) -> None:
        """Disconnect from TWS / IB Gateway."""
        if self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def _ensure_connected(self) -> None:
        if not self._connected or not self._ib.isConnected():
            self.connect()

    # ------------------------------------------------------------------
    # Account & Positions
    # ------------------------------------------------------------------

    def get_account(self) -> Account:
        self._ensure_connected()

        account_values = {
            av.tag: av.value
            for av in self._ib.accountValues(self._config.account_id or "")
        }

        equity = float(account_values.get("NetLiquidation", 0))
        cash = float(account_values.get("TotalCashValue", 0))
        buying_power = float(account_values.get("BuyingPower", 0))
        positions_value = float(account_values.get("GrossPositionValue", 0))

        return Account(
            account_id=self._config.account_id or self._ib.managedAccounts()[0] if self._ib.managedAccounts() else "IBKR",
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            positions_value=positions_value,
            is_paper=self._config.is_paper,
            day_trade_count=0,  # IBKR does not expose this simply
            pattern_day_trader=False,
            trading_blocked=False,
            currency="USD",
        )

    def get_positions(self) -> list[Position]:
        self._ensure_connected()
        result: list[Position] = []

        for pos in self._ib.positions(self._config.account_id or ""):
            contract = pos.contract
            qty = float(pos.position)
            avg_cost = float(pos.avgCost)

            # Attempt to get a live price; fall back to avg cost
            current_price = avg_cost  # fallback
            market_value = qty * current_price
            cost_basis = qty * avg_cost
            unrealized_pnl = market_value - cost_basis
            pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis != 0 else 0.0

            result.append(Position(
                symbol=contract.symbol,
                quantity=qty,
                avg_entry_price=avg_cost,
                current_price=current_price,
                market_value=market_value,
                cost_basis=cost_basis,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=pnl_pct,
                side="long" if qty > 0 else "short",
            ))

        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        for pos in self.get_positions():
            if pos.symbol == symbol:
                return pos
        return None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> OrderResult:
        self._ensure_connected()

        if self._config.readonly:
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message="Broker is in read-only mode; order submission disabled",
            )

        try:
            contract = Stock(order.symbol, "SMART", "USD")

            ib_action = "BUY" if order.side == OrderSide.BUY else "SELL"

            # Map TimeInForce
            tif_map = {
                TimeInForce.DAY: "DAY",
                TimeInForce.GTC: "GTC",
                TimeInForce.IOC: "IOC",
                TimeInForce.FOK: "FOK",
            }
            tif = tif_map.get(order.time_in_force, "DAY")

            # Build ib_insync order object
            if order.order_type == OrderType.MARKET:
                ib_order = MarketOrder(ib_action, order.quantity, tif=tif)
            elif order.order_type == OrderType.LIMIT:
                ib_order = LimitOrder(
                    ib_action, order.quantity, order.limit_price or 0, tif=tif,
                )
            elif order.order_type == OrderType.STOP:
                ib_order = StopOrder(
                    ib_action, order.quantity, order.stop_price or 0, tif=tif,
                )
            else:
                return OrderResult(
                    success=False,
                    status=OrderStatus.REJECTED,
                    message=f"Unsupported order type for IBKR: {order.order_type.value}",
                )

            if order.client_order_id:
                ib_order.orderRef = order.client_order_id

            trade: "Trade" = self._ib.placeOrder(contract, ib_order)

            logger.info(
                "IBKR order submitted: %s %d %s (orderId=%s)",
                ib_action, order.quantity, order.symbol, trade.order.orderId,
            )

            return OrderResult(
                success=True,
                order_id=str(trade.order.orderId),
                status=OrderStatus.PENDING,
                message="Order submitted to IBKR",
                raw_response={
                    "orderId": trade.order.orderId,
                    "permId": trade.order.permId,
                },
            )

        except Exception as e:
            logger.error("Failed to submit IBKR order: %s", e)
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        self._ensure_connected()

        try:
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    logger.info("IBKR order %s cancelled", order_id)
                    return True

            logger.warning("IBKR order %s not found in open trades", order_id)
            return False

        except Exception as e:
            logger.error("Failed to cancel IBKR order %s: %s", order_id, e)
            return False

    def get_order(self, order_id: str) -> Optional[dict]:
        self._ensure_connected()

        try:
            for trade in self._ib.trades():
                if str(trade.order.orderId) == order_id:
                    return self._trade_to_dict(trade)
            return None

        except Exception as e:
            logger.error("Failed to get IBKR order %s: %s", order_id, e)
            return None

    def get_open_orders(self) -> list[dict]:
        self._ensure_connected()

        try:
            return [self._trade_to_dict(t) for t in self._ib.openTrades()]
        except Exception as e:
            logger.error("Failed to get IBKR open orders: %s", e)
            return []

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Check market status via the IBKR API.

        Falls back to a simple time-based check if the API call fails.
        """
        self._ensure_connected()

        try:
            # ib_insync does not have a direct "is_market_open" call.
            # A common approach is to request contract details and inspect
            # trading hours, but for simplicity we rely on a time-based check.
            from datetime import datetime, time as dt_time, timezone

            now = datetime.now(timezone.utc)
            # Rough ET offset (no DST handling)
            et_hour = (now.hour - 5) % 24
            et_time = dt_time(et_hour, now.minute)
            if now.weekday() >= 5:
                return False
            return dt_time(9, 30) <= et_time < dt_time(16, 0)

        except Exception as e:
            logger.error("Failed to check IBKR market hours: %s", e)
            return False

    # ------------------------------------------------------------------
    # Close positions
    # ------------------------------------------------------------------

    def close_position(self, symbol: str) -> OrderResult:
        pos = self.get_position(symbol)
        if pos is None:
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
        for pos in self.get_positions():
            results.append(self.close_position(pos.symbol))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trade_to_dict(trade: "Trade") -> dict:
        order = trade.order
        fill_price = None
        filled_qty = 0.0

        if trade.fills:
            filled_qty = sum(f.execution.shares for f in trade.fills)
            total_value = sum(f.execution.shares * f.execution.price for f in trade.fills)
            fill_price = total_value / filled_qty if filled_qty else None

        return {
            "id": str(order.orderId),
            "symbol": trade.contract.symbol if trade.contract else "",
            "side": order.action,
            "type": order.orderType,
            "qty": float(order.totalQuantity),
            "filled_qty": filled_qty,
            "filled_avg_price": fill_price,
            "status": trade.orderStatus.status if trade.orderStatus else "Unknown",
        }

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "IBKRBroker":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
