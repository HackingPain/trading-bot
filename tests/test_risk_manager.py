"""Tests for the risk manager module."""

import pytest
from datetime import date, datetime

from src.risk.risk_manager import (
    AccountInfo,
    PositionRisk,
    RiskCheck,
    RiskConfig,
    RiskManager,
)


@pytest.fixture
def risk_config():
    """Create a test risk configuration."""
    return RiskConfig(
        max_position_pct=0.10,
        max_portfolio_risk_pct=0.30,
        max_daily_loss_pct=0.02,
        max_daily_trades=3,
        stop_loss_pct=0.05,
        trailing_stop_pct=0.03,
        min_account_balance=1000,
        pdt_threshold=25000,
        max_open_positions=5,
    )


@pytest.fixture
def risk_manager(risk_config):
    """Create a risk manager for testing."""
    return RiskManager(risk_config)


@pytest.fixture
def account_info():
    """Create test account information."""
    return AccountInfo(
        equity=50000,
        cash=30000,
        buying_power=60000,
        positions_value=20000,
        day_trade_count=0,
        daily_pnl=0,
        starting_equity=50000,
    )


@pytest.fixture
def small_account():
    """Create a small account (below PDT threshold)."""
    return AccountInfo(
        equity=20000,
        cash=15000,
        buying_power=20000,
        positions_value=5000,
        day_trade_count=0,
        daily_pnl=0,
        starting_equity=20000,
    )


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_breaker_passes_no_loss(self, risk_manager, account_info):
        """Circuit breaker should pass when there's no loss."""
        result = risk_manager.check_circuit_breaker(account_info)
        assert result.passed is True
        assert result.check_type == RiskCheck.CIRCUIT_BREAKER

    def test_circuit_breaker_passes_small_loss(self, risk_manager, account_info):
        """Circuit breaker should pass for small losses."""
        account_info.daily_pnl = -500  # 1% loss
        result = risk_manager.check_circuit_breaker(account_info)
        assert result.passed is True

    def test_circuit_breaker_fails_large_loss(self, risk_manager, account_info):
        """Circuit breaker should fail for large losses."""
        account_info.daily_pnl = -1500  # 3% loss (exceeds 2% limit)
        result = risk_manager.check_circuit_breaker(account_info)
        assert result.passed is False
        assert "exceeds limit" in result.message.lower()

    def test_circuit_breaker_stays_triggered(self, risk_manager, account_info):
        """Once triggered, circuit breaker should stay triggered."""
        # Trigger it
        account_info.daily_pnl = -1500
        risk_manager.check_circuit_breaker(account_info)

        # Even with recovery, should still fail
        account_info.daily_pnl = 0
        result = risk_manager.check_circuit_breaker(account_info)
        assert result.passed is False
        assert "already triggered" in result.message.lower()

    def test_circuit_breaker_can_be_reset(self, risk_manager, account_info):
        """Circuit breaker can be manually reset."""
        account_info.daily_pnl = -1500
        risk_manager.check_circuit_breaker(account_info)

        risk_manager.reset_circuit_breaker()

        account_info.daily_pnl = 0
        result = risk_manager.check_circuit_breaker(account_info)
        assert result.passed is True


class TestPDTRule:
    """Tests for Pattern Day Trader rule compliance."""

    def test_pdt_passes_large_account(self, risk_manager, account_info):
        """PDT check should pass for accounts above $25k."""
        result = risk_manager.check_pdt_rule(account_info)
        assert result.passed is True
        assert "above PDT threshold" in result.message

    def test_pdt_passes_small_account_few_trades(self, risk_manager, small_account):
        """PDT should pass for small accounts with few day trades."""
        result = risk_manager.check_pdt_rule(small_account)
        assert result.passed is True

    def test_pdt_fails_after_limit(self, risk_manager, small_account):
        """PDT should fail when trade limit is reached."""
        # Record 3 trades
        for _ in range(3):
            risk_manager.record_trade()

        result = risk_manager.check_pdt_rule(small_account)
        assert result.passed is False
        assert "limit reached" in result.message.lower()


class TestPositionSize:
    """Tests for position size validation."""

    def test_position_size_passes_within_limit(self, risk_manager, account_info):
        """Position size should pass when within limit."""
        trade_value = 4000  # 8% of 50k (limit is 10%)
        result = risk_manager.check_position_size(account_info, trade_value)
        assert result.passed is True

    def test_position_size_fails_exceeds_limit(self, risk_manager, account_info):
        """Position size should fail when exceeding limit."""
        trade_value = 6000  # 12% of 50k (exceeds 10% limit)
        result = risk_manager.check_position_size(account_info, trade_value)
        assert result.passed is False
        assert "exceeds" in result.message.lower()


class TestMaxPositions:
    """Tests for maximum positions check."""

    def test_max_positions_passes_under_limit(self, risk_manager):
        """Max positions should pass when under limit."""
        positions = [
            PositionRisk("AAPL", 10, 1500, 1500, 0, 3),
            PositionRisk("MSFT", 5, 1000, 1000, 0, 2),
        ]
        result = risk_manager.check_max_positions(positions)
        assert result.passed is True

    def test_max_positions_fails_at_limit(self, risk_manager):
        """Max positions should fail when at limit."""
        positions = [
            PositionRisk(f"SYM{i}", 10, 1000, 1000, 0, 2)
            for i in range(5)  # 5 is the limit
        ]
        result = risk_manager.check_max_positions(positions)
        assert result.passed is False


class TestAccountMinimum:
    """Tests for account minimum check."""

    def test_account_minimum_passes(self, risk_manager, account_info):
        """Account minimum should pass when above minimum."""
        result = risk_manager.check_account_minimum(account_info)
        assert result.passed is True

    def test_account_minimum_fails(self, risk_manager):
        """Account minimum should fail when below minimum."""
        low_account = AccountInfo(
            equity=500,
            cash=500,
            buying_power=500,
            positions_value=0,
            starting_equity=500,
        )
        result = risk_manager.check_account_minimum(low_account)
        assert result.passed is False
        assert "below minimum" in result.message.lower()


class TestPositionSizing:
    """Tests for position sizing calculation."""

    def test_calculate_position_size_basic(self, risk_manager, account_info):
        """Test basic position size calculation."""
        shares = risk_manager.calculate_position_size(
            account=account_info,
            price=100,
            signal_strength=1.0,
        )
        # Max position is 10% of 50k = 5000, at $100/share = 50 shares
        assert shares == 50

    def test_calculate_position_size_reduced_strength(self, risk_manager, account_info):
        """Test position size with reduced signal strength."""
        shares = risk_manager.calculate_position_size(
            account=account_info,
            price=100,
            signal_strength=0.5,
        )
        # 50% of max = 25 shares
        assert shares == 25

    def test_calculate_position_size_respects_buying_power(self, risk_manager):
        """Position size should respect buying power."""
        limited_account = AccountInfo(
            equity=50000,
            cash=1000,
            buying_power=2000,  # Limited buying power
            positions_value=49000,
            starting_equity=50000,
        )
        shares = risk_manager.calculate_position_size(
            account=limited_account,
            price=100,
            signal_strength=1.0,
        )
        # Should be limited by buying power: 2000 * 0.95 / 100 = 19 shares
        assert shares <= 19


class TestCanTrade:
    """Tests for the can_trade combined check."""

    def test_can_trade_all_pass(self, risk_manager, account_info):
        """Can trade should return True when all checks pass."""
        positions = []
        can_trade, results = risk_manager.can_trade(
            account=account_info,
            positions=positions,
            proposed_trade_value=4000,
            proposed_symbol="AAPL",
        )
        assert can_trade is True
        assert all(r.passed for r in results)

    def test_can_trade_fails_on_any_check(self, risk_manager, account_info):
        """Can trade should return False if any check fails."""
        positions = []
        # Exceed position size limit
        can_trade, results = risk_manager.can_trade(
            account=account_info,
            positions=positions,
            proposed_trade_value=10000,  # 20% of equity
            proposed_symbol="AAPL",
        )
        assert can_trade is False
        failed = [r for r in results if not r.passed]
        assert len(failed) > 0


class TestRiskSummary:
    """Tests for risk summary generation."""

    def test_get_risk_summary(self, risk_manager, account_info):
        """Test risk summary generation."""
        positions = [
            PositionRisk("AAPL", 10, 1500, 1400, 100, 3),
        ]
        summary = risk_manager.get_risk_summary(account_info, positions)

        assert "equity" in summary
        assert "cash" in summary
        assert "positions_value" in summary
        assert "exposure_pct" in summary
        assert "circuit_breaker_triggered" in summary
        assert summary["open_positions"] == 1
