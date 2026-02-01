"""Risk management system for the trading bot."""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RiskCheck(str, Enum):
    """Types of risk checks performed."""
    CIRCUIT_BREAKER = "circuit_breaker"
    PDT_RULE = "pdt_rule"
    POSITION_SIZE = "position_size"
    SECTOR_EXPOSURE = "sector_exposure"
    PER_TRADE_RISK = "per_trade_risk"
    ACCOUNT_MINIMUM = "account_minimum"
    MAX_OPEN_POSITIONS = "max_open_positions"


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    check_type: RiskCheck
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"<RiskCheck {status} {self.check_type.value}: {self.message}>"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_pct: float = 0.10  # 10% max per position
    max_portfolio_risk_pct: float = 0.30  # 30% max total exposure
    max_daily_loss_pct: float = 0.02  # 2% daily circuit breaker
    max_daily_trades: int = 3  # PDT rule safety
    stop_loss_pct: float = 0.05  # 5% stop loss required
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    min_account_balance: float = 1000  # Minimum to trade
    pdt_threshold: float = 25000  # Pattern day trader threshold
    max_open_positions: int = 10  # Maximum concurrent positions
    max_sector_exposure_pct: float = 0.40  # 40% max in one sector

    @classmethod
    def from_settings(cls, settings: dict) -> "RiskConfig":
        """Create config from settings dictionary."""
        risk_settings = settings.get("risk", {})
        return cls(
            max_position_pct=risk_settings.get("max_position_pct", 0.10),
            max_portfolio_risk_pct=risk_settings.get("max_portfolio_risk_pct", 0.30),
            max_daily_loss_pct=risk_settings.get("max_daily_loss_pct", 0.02),
            max_daily_trades=risk_settings.get("max_daily_trades", 3),
            stop_loss_pct=risk_settings.get("stop_loss_pct", 0.05),
            trailing_stop_pct=risk_settings.get("trailing_stop_pct", 0.03),
            min_account_balance=risk_settings.get("min_account_balance", 1000),
            max_open_positions=risk_settings.get("max_open_positions", 10),
            max_sector_exposure_pct=risk_settings.get("max_sector_exposure_pct", 0.40),
        )


@dataclass
class AccountInfo:
    """Current account information for risk calculations."""
    equity: float
    cash: float
    buying_power: float
    positions_value: float
    day_trade_count: int = 0
    daily_pnl: float = 0.0
    starting_equity: float = 0.0  # Start of day equity


@dataclass
class PositionRisk:
    """Risk information for a position."""
    symbol: str
    quantity: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    weight_pct: float  # % of portfolio
    sector: Optional[str] = None


class RiskManager:
    """
    Manages all risk checks and position sizing.

    Risk checks are performed in order:
    1. Circuit breaker (daily loss limit)
    2. PDT rule compliance
    3. Account minimum
    4. Maximum open positions
    5. Position size limits
    6. Sector exposure
    7. Per-trade risk validation
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self._circuit_breaker_triggered = False
        self._daily_trades: list[datetime] = []
        self._last_reset_date: Optional[date] = None

    @classmethod
    def from_settings(cls, settings: dict) -> "RiskManager":
        """Create risk manager from settings dictionary."""
        config = RiskConfig.from_settings(settings)
        return cls(config)

    def _reset_daily_counters(self) -> None:
        """Reset daily counters if it's a new day."""
        today = date.today()
        if self._last_reset_date != today:
            self._daily_trades = []
            self._circuit_breaker_triggered = False
            self._last_reset_date = today
            logger.info("Daily risk counters reset")

    def check_all(
        self,
        account: AccountInfo,
        positions: list[PositionRisk],
        proposed_trade_value: float,
        proposed_symbol: str,
    ) -> list[RiskCheckResult]:
        """
        Run all risk checks for a proposed trade.

        Returns list of RiskCheckResult, one per check type.
        """
        self._reset_daily_counters()
        results = []

        # 1. Circuit breaker
        results.append(self.check_circuit_breaker(account))

        # 2. PDT rule
        results.append(self.check_pdt_rule(account))

        # 3. Account minimum
        results.append(self.check_account_minimum(account))

        # 4. Max open positions
        results.append(self.check_max_positions(positions))

        # 5. Position size
        results.append(self.check_position_size(account, proposed_trade_value))

        # 6. Sector exposure (simplified - would need sector data)
        results.append(self.check_sector_exposure(positions, proposed_symbol))

        # 7. Per-trade risk
        results.append(self.check_per_trade_risk(proposed_trade_value, account.equity))

        return results

    def can_trade(
        self,
        account: AccountInfo,
        positions: list[PositionRisk],
        proposed_trade_value: float,
        proposed_symbol: str,
    ) -> tuple[bool, list[RiskCheckResult]]:
        """
        Check if a trade is allowed based on all risk rules.

        Returns (can_trade, list of check results)
        """
        results = self.check_all(account, positions, proposed_trade_value, proposed_symbol)
        can_trade = all(r.passed for r in results)

        if not can_trade:
            failed = [r for r in results if not r.passed]
            logger.warning(f"Trade blocked by risk checks: {[r.check_type.value for r in failed]}")

        return can_trade, results

    def check_circuit_breaker(self, account: AccountInfo) -> RiskCheckResult:
        """
        Check if daily loss limit has been breached.

        Circuit breaker triggers if daily loss exceeds max_daily_loss_pct.
        """
        if self._circuit_breaker_triggered:
            return RiskCheckResult(
                check_type=RiskCheck.CIRCUIT_BREAKER,
                passed=False,
                message="Circuit breaker already triggered for today",
                details={"triggered": True},
            )

        if account.starting_equity <= 0:
            return RiskCheckResult(
                check_type=RiskCheck.CIRCUIT_BREAKER,
                passed=True,
                message="No starting equity recorded",
            )

        daily_loss_pct = -account.daily_pnl / account.starting_equity
        max_loss_pct = self.config.max_daily_loss_pct

        if daily_loss_pct >= max_loss_pct:
            self._circuit_breaker_triggered = True
            return RiskCheckResult(
                check_type=RiskCheck.CIRCUIT_BREAKER,
                passed=False,
                message=f"Daily loss {daily_loss_pct*100:.2f}% exceeds limit {max_loss_pct*100:.2f}%",
                details={
                    "daily_loss_pct": daily_loss_pct,
                    "max_loss_pct": max_loss_pct,
                    "daily_pnl": account.daily_pnl,
                },
            )

        return RiskCheckResult(
            check_type=RiskCheck.CIRCUIT_BREAKER,
            passed=True,
            message=f"Daily loss {daily_loss_pct*100:.2f}% within limit",
            details={"daily_loss_pct": daily_loss_pct, "max_loss_pct": max_loss_pct},
        )

    def check_pdt_rule(self, account: AccountInfo) -> RiskCheckResult:
        """
        Check Pattern Day Trader rule compliance.

        If account < $25k, limit day trades to max_daily_trades.
        """
        if account.equity >= self.config.pdt_threshold:
            return RiskCheckResult(
                check_type=RiskCheck.PDT_RULE,
                passed=True,
                message=f"Account ${account.equity:,.2f} above PDT threshold",
                details={"equity": account.equity, "threshold": self.config.pdt_threshold},
            )

        day_trades_today = len([t for t in self._daily_trades if t.date() == date.today()])

        if day_trades_today >= self.config.max_daily_trades:
            return RiskCheckResult(
                check_type=RiskCheck.PDT_RULE,
                passed=False,
                message=f"Day trade limit reached ({day_trades_today}/{self.config.max_daily_trades})",
                details={
                    "day_trades": day_trades_today,
                    "max_trades": self.config.max_daily_trades,
                    "equity": account.equity,
                },
            )

        return RiskCheckResult(
            check_type=RiskCheck.PDT_RULE,
            passed=True,
            message=f"Day trades: {day_trades_today}/{self.config.max_daily_trades}",
            details={"day_trades": day_trades_today, "max_trades": self.config.max_daily_trades},
        )

    def check_account_minimum(self, account: AccountInfo) -> RiskCheckResult:
        """Check if account meets minimum balance requirement."""
        if account.equity < self.config.min_account_balance:
            return RiskCheckResult(
                check_type=RiskCheck.ACCOUNT_MINIMUM,
                passed=False,
                message=f"Account ${account.equity:,.2f} below minimum ${self.config.min_account_balance:,.2f}",
                details={
                    "equity": account.equity,
                    "minimum": self.config.min_account_balance,
                },
            )

        return RiskCheckResult(
            check_type=RiskCheck.ACCOUNT_MINIMUM,
            passed=True,
            message=f"Account ${account.equity:,.2f} meets minimum",
        )

    def check_max_positions(self, positions: list[PositionRisk]) -> RiskCheckResult:
        """Check if maximum open positions limit is reached."""
        num_positions = len(positions)

        if num_positions >= self.config.max_open_positions:
            return RiskCheckResult(
                check_type=RiskCheck.MAX_OPEN_POSITIONS,
                passed=False,
                message=f"Max positions reached ({num_positions}/{self.config.max_open_positions})",
                details={
                    "current": num_positions,
                    "max": self.config.max_open_positions,
                },
            )

        return RiskCheckResult(
            check_type=RiskCheck.MAX_OPEN_POSITIONS,
            passed=True,
            message=f"Positions: {num_positions}/{self.config.max_open_positions}",
        )

    def check_position_size(
        self,
        account: AccountInfo,
        proposed_trade_value: float,
    ) -> RiskCheckResult:
        """Check if proposed position size is within limits."""
        if account.equity <= 0:
            return RiskCheckResult(
                check_type=RiskCheck.POSITION_SIZE,
                passed=False,
                message="Invalid account equity",
            )

        position_pct = proposed_trade_value / account.equity
        max_pct = self.config.max_position_pct

        if position_pct > max_pct:
            return RiskCheckResult(
                check_type=RiskCheck.POSITION_SIZE,
                passed=False,
                message=f"Position {position_pct*100:.1f}% exceeds max {max_pct*100:.1f}%",
                details={
                    "proposed_pct": position_pct,
                    "max_pct": max_pct,
                    "proposed_value": proposed_trade_value,
                    "equity": account.equity,
                },
            )

        return RiskCheckResult(
            check_type=RiskCheck.POSITION_SIZE,
            passed=True,
            message=f"Position size {position_pct*100:.1f}% within limit",
        )

    def check_sector_exposure(
        self,
        positions: list[PositionRisk],
        proposed_symbol: str,
    ) -> RiskCheckResult:
        """
        Check sector concentration risk.

        Note: This is simplified - production would need sector data.
        """
        # Simplified: just check if we already hold the same symbol
        held_symbols = [p.symbol for p in positions]

        if proposed_symbol in held_symbols:
            return RiskCheckResult(
                check_type=RiskCheck.SECTOR_EXPOSURE,
                passed=False,
                message=f"Already holding position in {proposed_symbol}",
                details={"symbol": proposed_symbol, "held_symbols": held_symbols},
            )

        return RiskCheckResult(
            check_type=RiskCheck.SECTOR_EXPOSURE,
            passed=True,
            message="Sector exposure within limits",
        )

    def check_per_trade_risk(
        self,
        trade_value: float,
        equity: float,
    ) -> RiskCheckResult:
        """
        Validate per-trade risk with stop loss.

        Ensures maximum loss per trade is acceptable.
        """
        max_loss = trade_value * self.config.stop_loss_pct
        max_loss_pct_of_equity = max_loss / equity if equity > 0 else float("inf")

        # Per-trade risk should not exceed 2% of equity
        max_per_trade_risk = 0.02

        if max_loss_pct_of_equity > max_per_trade_risk:
            return RiskCheckResult(
                check_type=RiskCheck.PER_TRADE_RISK,
                passed=False,
                message=f"Per-trade risk {max_loss_pct_of_equity*100:.2f}% exceeds {max_per_trade_risk*100:.1f}%",
                details={
                    "max_loss": max_loss,
                    "max_loss_pct": max_loss_pct_of_equity,
                    "limit": max_per_trade_risk,
                },
            )

        return RiskCheckResult(
            check_type=RiskCheck.PER_TRADE_RISK,
            passed=True,
            message=f"Per-trade risk {max_loss_pct_of_equity*100:.2f}% acceptable",
        )

    def calculate_position_size(
        self,
        account: AccountInfo,
        price: float,
        signal_strength: float = 1.0,
    ) -> int:
        """
        Calculate appropriate position size based on risk rules.

        Args:
            account: Current account info
            price: Current stock price
            signal_strength: Signal strength 0-1 (scales position)

        Returns:
            Number of shares to buy
        """
        # Maximum position value
        max_value = account.equity * self.config.max_position_pct

        # Scale by signal strength
        target_value = max_value * signal_strength

        # Ensure we don't exceed buying power
        target_value = min(target_value, account.buying_power * 0.95)  # Keep 5% buffer

        # Calculate shares
        shares = int(target_value / price)

        # Validate final position value
        final_value = shares * price
        if final_value / account.equity > self.config.max_position_pct:
            shares = int((account.equity * self.config.max_position_pct) / price)

        return max(0, shares)

    def calculate_stop_loss_price(self, entry_price: float) -> float:
        """Calculate stop loss price based on config."""
        return entry_price * (1 - self.config.stop_loss_pct)

    def calculate_trailing_stop_price(self, highest_price: float) -> float:
        """Calculate trailing stop price based on highest price."""
        return highest_price * (1 - self.config.trailing_stop_pct)

    def record_trade(self) -> None:
        """Record a trade for PDT tracking."""
        self._daily_trades.append(datetime.now())

    def trigger_circuit_breaker(self, reason: str) -> None:
        """Manually trigger circuit breaker."""
        self._circuit_breaker_triggered = True
        logger.warning(f"Circuit breaker triggered: {reason}")

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (use with caution)."""
        self._circuit_breaker_triggered = False
        logger.info("Circuit breaker reset")

    @property
    def is_circuit_breaker_triggered(self) -> bool:
        """Check if circuit breaker is currently triggered."""
        return self._circuit_breaker_triggered

    def get_risk_summary(
        self,
        account: AccountInfo,
        positions: list[PositionRisk],
    ) -> dict[str, Any]:
        """Get a summary of current risk metrics."""
        total_positions_value = sum(p.market_value for p in positions)
        total_exposure_pct = total_positions_value / account.equity if account.equity > 0 else 0

        return {
            "equity": account.equity,
            "cash": account.cash,
            "positions_value": total_positions_value,
            "exposure_pct": total_exposure_pct * 100,
            "max_exposure_pct": self.config.max_portfolio_risk_pct * 100,
            "daily_pnl": account.daily_pnl,
            "daily_pnl_pct": (account.daily_pnl / account.starting_equity * 100) if account.starting_equity > 0 else 0,
            "circuit_breaker_triggered": self._circuit_breaker_triggered,
            "day_trades_today": len([t for t in self._daily_trades if t.date() == date.today()]),
            "max_day_trades": self.config.max_daily_trades,
            "open_positions": len(positions),
            "max_positions": self.config.max_open_positions,
        }
