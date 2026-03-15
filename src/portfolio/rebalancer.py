"""
Portfolio rebalancing module.

Maintains target allocations and rebalances when drift exceeds threshold.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..execution.broker import Broker, Order, OrderSide, OrderType, TimeInForce

logger = logging.getLogger(__name__)


@dataclass
class TargetAllocation:
    """Target allocation for a symbol."""
    symbol: str
    target_weight: float  # 0.0 to 1.0 (e.g., 0.10 = 10%)
    min_weight: Optional[float] = None  # Minimum allowed weight
    max_weight: Optional[float] = None  # Maximum allowed weight

    def __post_init__(self):
        if self.min_weight is None:
            self.min_weight = max(0, self.target_weight - 0.05)
        if self.max_weight is None:
            self.max_weight = min(1, self.target_weight + 0.05)


@dataclass
class CurrentHolding:
    """Current holding information."""
    symbol: str
    shares: float
    market_value: float
    current_weight: float
    cost_basis: float
    unrealized_pnl: float


@dataclass
class RebalanceAction:
    """A single rebalancing action."""
    symbol: str
    action: str  # 'buy' or 'sell'
    shares: int
    estimated_value: float
    current_weight: float
    target_weight: float
    reason: str


@dataclass
class RebalanceResult:
    """Result of rebalancing operation."""
    success: bool
    actions_taken: list[RebalanceAction] = field(default_factory=list)
    orders_submitted: list[str] = field(default_factory=list)  # Order IDs
    errors: list[str] = field(default_factory=list)
    message: str = ""


class PortfolioRebalancer:
    """
    Manages portfolio rebalancing to maintain target allocations.

    Features:
    - Configurable target allocations
    - Drift threshold triggering
    - Tax-aware rebalancing (optional)
    - Minimum trade size filtering
    """

    def __init__(
        self,
        broker: Broker,
        allocations: list[TargetAllocation],
        drift_threshold: float = 0.05,
        min_trade_value: float = 100.0,
        use_limit_orders: bool = False,
        limit_price_buffer: float = 0.001,
    ):
        """
        Initialize rebalancer.

        Args:
            broker: Broker instance for executing trades
            allocations: List of target allocations
            drift_threshold: Rebalance when drift exceeds this (e.g., 0.05 = 5%)
            min_trade_value: Minimum trade value to execute
            use_limit_orders: Use limit orders instead of market
            limit_price_buffer: Buffer for limit prices (e.g., 0.001 = 0.1%)
        """
        self.broker = broker
        self.allocations = {a.symbol: a for a in allocations}
        self.drift_threshold = drift_threshold
        self.min_trade_value = min_trade_value
        self.use_limit_orders = use_limit_orders
        self.limit_price_buffer = limit_price_buffer

        # Validate allocations sum to 1 (or less for cash reserve)
        total_weight = sum(a.target_weight for a in allocations)
        if total_weight > 1.0:
            raise ValueError(f"Target allocations sum to {total_weight}, must be <= 1.0")

        self.cash_target = 1.0 - total_weight  # Implicit cash allocation

    @classmethod
    def from_settings(cls, broker: Broker, settings: dict) -> "PortfolioRebalancer":
        """Create rebalancer from settings dictionary."""
        rebalance_settings = settings.get("rebalancing", {})
        allocation_settings = rebalance_settings.get("allocations", {})

        allocations = [
            TargetAllocation(
                symbol=symbol.upper(),
                target_weight=weight,
            )
            for symbol, weight in allocation_settings.items()
        ]

        return cls(
            broker=broker,
            allocations=allocations,
            drift_threshold=rebalance_settings.get("drift_threshold", 0.05),
            min_trade_value=rebalance_settings.get("min_trade_value", 100.0),
            use_limit_orders=rebalance_settings.get("use_limit_orders", False),
        )

    def get_current_holdings(self) -> tuple[list[CurrentHolding], float]:
        """
        Get current portfolio holdings.

        Returns:
            Tuple of (list of holdings, total portfolio value including cash)
        """
        account = self.broker.get_account()
        positions = self.broker.get_positions()

        total_value = account.equity

        holdings = []
        for pos in positions:
            weight = pos.market_value / total_value if total_value > 0 else 0

            holdings.append(CurrentHolding(
                symbol=pos.symbol,
                shares=pos.quantity,
                market_value=pos.market_value,
                current_weight=weight,
                cost_basis=pos.cost_basis,
                unrealized_pnl=pos.unrealized_pnl,
            ))

        return holdings, total_value

    def calculate_drift(self) -> dict[str, float]:
        """
        Calculate drift from target allocations.

        Returns:
            Dict mapping symbol to drift (positive = overweight, negative = underweight)
        """
        holdings, total_value = self.get_current_holdings()
        current_weights = {h.symbol: h.current_weight for h in holdings}

        drift = {}
        for symbol, allocation in self.allocations.items():
            current = current_weights.get(symbol, 0.0)
            drift[symbol] = current - allocation.target_weight

        return drift

    def needs_rebalancing(self) -> tuple[bool, dict[str, float]]:
        """
        Check if portfolio needs rebalancing.

        Returns:
            Tuple of (needs_rebalance, drift_dict)
        """
        drift = self.calculate_drift()

        # Check if any position exceeds drift threshold
        max_drift = max(abs(d) for d in drift.values()) if drift else 0

        return max_drift > self.drift_threshold, drift

    def calculate_rebalance_actions(
        self,
        current_prices: dict[str, float],
    ) -> list[RebalanceAction]:
        """
        Calculate what trades are needed to rebalance.

        Args:
            current_prices: Dict mapping symbol to current price

        Returns:
            List of rebalance actions to take
        """
        holdings, total_value = self.get_current_holdings()
        current_weights = {h.symbol: h.current_weight for h in holdings}
        current_shares = {h.symbol: h.shares for h in holdings}

        actions = []

        for symbol, allocation in self.allocations.items():
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = allocation.target_weight

            # Calculate weight difference
            weight_diff = target_weight - current_weight

            # Skip if within tolerance
            if abs(weight_diff) < 0.01:  # 1% tolerance
                continue

            # Calculate value to trade
            target_value = total_value * target_weight
            current_value = total_value * current_weight
            value_diff = target_value - current_value

            # Skip if below minimum trade value
            if abs(value_diff) < self.min_trade_value:
                continue

            # Get current price
            price = current_prices.get(symbol)
            if not price or price <= 0:
                logger.warning(f"No price available for {symbol}, skipping")
                continue

            # Calculate shares to trade
            shares = int(abs(value_diff) / price)
            if shares <= 0:
                continue

            action = "buy" if weight_diff > 0 else "sell"

            # Don't sell more than we own
            if action == "sell":
                owned = current_shares.get(symbol, 0)
                shares = min(shares, int(owned))
                if shares <= 0:
                    continue

            actions.append(RebalanceAction(
                symbol=symbol,
                action=action,
                shares=shares,
                estimated_value=shares * price,
                current_weight=current_weight,
                target_weight=target_weight,
                reason=f"Drift: {weight_diff*100:+.1f}%",
            ))

        # Sort: sells first (to free up cash), then buys
        actions.sort(key=lambda a: (0 if a.action == "sell" else 1, -a.estimated_value))

        return actions

    def execute_rebalance(
        self,
        current_prices: dict[str, float],
        dry_run: bool = False,
    ) -> RebalanceResult:
        """
        Execute rebalancing trades.

        Args:
            current_prices: Dict mapping symbol to current price
            dry_run: If True, calculate but don't execute

        Returns:
            RebalanceResult with actions taken
        """
        actions = self.calculate_rebalance_actions(current_prices)

        if not actions:
            return RebalanceResult(
                success=True,
                message="Portfolio is balanced, no trades needed",
            )

        if dry_run:
            return RebalanceResult(
                success=True,
                actions_taken=actions,
                message=f"Dry run: {len(actions)} trades would be executed",
            )

        result = RebalanceResult(success=True, actions_taken=actions)

        for action in actions:
            try:
                price = current_prices.get(action.symbol)

                # Create order
                if self.use_limit_orders and price:
                    # Add buffer for limit price
                    if action.action == "buy":
                        limit_price = price * (1 + self.limit_price_buffer)
                    else:
                        limit_price = price * (1 - self.limit_price_buffer)

                    order = Order(
                        symbol=action.symbol,
                        side=OrderSide.BUY if action.action == "buy" else OrderSide.SELL,
                        quantity=action.shares,
                        order_type=OrderType.LIMIT,
                        limit_price=round(limit_price, 2),
                        time_in_force=TimeInForce.DAY,
                    )
                else:
                    order = Order(
                        symbol=action.symbol,
                        side=OrderSide.BUY if action.action == "buy" else OrderSide.SELL,
                        quantity=action.shares,
                        order_type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY,
                    )

                # Submit order
                order_result = self.broker.submit_order(order)

                if order_result.success:
                    result.orders_submitted.append(order_result.order_id)
                    logger.info(
                        f"Rebalance: {action.action.upper()} {action.shares} {action.symbol} "
                        f"(order: {order_result.order_id})"
                    )
                else:
                    result.errors.append(f"{action.symbol}: {order_result.message}")
                    logger.error(f"Rebalance order failed for {action.symbol}: {order_result.message}")

            except Exception as e:
                result.errors.append(f"{action.symbol}: {str(e)}")
                logger.error(f"Error executing rebalance for {action.symbol}: {e}")

        if result.errors:
            result.success = False
            result.message = f"Rebalance completed with {len(result.errors)} errors"
        else:
            result.message = f"Rebalance complete: {len(result.orders_submitted)} orders submitted"

        return result

    def get_portfolio_summary(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """
        Get a summary of current portfolio vs targets.

        Returns:
            Dict with portfolio analysis
        """
        holdings, total_value = self.get_current_holdings()
        current_weights = {h.symbol: h.current_weight for h in holdings}
        drift = self.calculate_drift()

        summary = {
            "total_value": total_value,
            "positions": [],
            "cash_weight": 1.0 - sum(current_weights.values()),
            "needs_rebalancing": False,
            "max_drift": 0.0,
        }

        max_drift = 0.0

        for symbol, allocation in self.allocations.items():
            current = current_weights.get(symbol, 0.0)
            symbol_drift = drift.get(symbol, 0.0)
            max_drift = max(max_drift, abs(symbol_drift))

            # Get current holding info
            holding = next((h for h in holdings if h.symbol == symbol), None)

            summary["positions"].append({
                "symbol": symbol,
                "target_weight": allocation.target_weight,
                "current_weight": current,
                "drift": symbol_drift,
                "shares": holding.shares if holding else 0,
                "market_value": holding.market_value if holding else 0,
                "unrealized_pnl": holding.unrealized_pnl if holding else 0,
            })

        summary["max_drift"] = max_drift
        summary["needs_rebalancing"] = max_drift > self.drift_threshold

        return summary

    def set_target_allocation(self, symbol: str, weight: float) -> None:
        """Update target allocation for a symbol."""
        if symbol in self.allocations:
            self.allocations[symbol].target_weight = weight
        else:
            self.allocations[symbol] = TargetAllocation(symbol=symbol, target_weight=weight)

        # Recalculate cash target
        total_weight = sum(a.target_weight for a in self.allocations.values())
        self.cash_target = max(0, 1.0 - total_weight)

    def remove_target(self, symbol: str) -> bool:
        """Remove a symbol from target allocations."""
        if symbol in self.allocations:
            del self.allocations[symbol]
            total_weight = sum(a.target_weight for a in self.allocations.values())
            self.cash_target = max(0, 1.0 - total_weight)
            return True
        return False
