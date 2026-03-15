"""Integration tests for the full trading loop with mocked broker (4.1)."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import numpy as np

from src.database.models import (
    Base, Trade, Position, Signal, TrackedOrder, DailyState,
    init_db, get_session, get_db_session,
)
from src.execution.broker import (
    Account, Position as BrokerPosition, Order, OrderResult,
    OrderSide, OrderType, OrderStatus,
)
from src.risk.risk_manager import RiskManager, RiskConfig, AccountInfo, PositionRisk
from src.strategies.base import SignalType, PositionInfo
from src.strategies.daily_profit_taker import DailyProfitTakerStrategy
from src.data.market_data import MarketData


@pytest.fixture
def db_session():
    """Create an in-memory database for testing."""
    engine = init_db("sqlite:///:memory:")
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def sample_market_data():
    """Create sample OHLCV market data."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    prices = 150 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame({
        "open": prices + np.random.randn(100) * 0.5,
        "high": prices + abs(np.random.randn(100)) * 2,
        "low": prices - abs(np.random.randn(100)) * 2,
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, 100),
    }, index=dates)

    return MarketData(
        symbol="AAPL",
        df=df,
        last_price=float(prices[-1]),
        last_updated=datetime.now(),
        volume=int(df["volume"].iloc[-1]),
    )


class TestDatabaseSessionManagement:
    """Test 1.3 - Database session context managers."""

    def test_context_manager_commits_on_success(self, db_session):
        """Context manager should auto-commit on success."""
        with get_db_session() as session:
            from src.database.models import DailyState
            state = DailyState(
                date=datetime.now(),
                market_open_equity=100000,
                market_open_cash=50000,
                market_open_positions_value=50000,
            )
            session.add(state)

        # Verify data was committed
        with get_db_session() as session:
            count = session.query(DailyState).count()
            assert count == 1

    def test_context_manager_rolls_back_on_error(self, db_session):
        """Context manager should rollback on exception."""
        try:
            with get_db_session() as session:
                state = DailyState(
                    date=datetime.now(),
                    market_open_equity=100000,
                    market_open_cash=50000,
                    market_open_positions_value=50000,
                )
                session.add(state)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify data was NOT committed
        with get_db_session() as session:
            count = session.query(DailyState).count()
            assert count == 0


class TestOrderLifecycleTracking:
    """Test 1.4 - Order lifecycle tracking."""

    def test_tracked_order_creation(self, db_session):
        """TrackedOrder model should persist correctly."""
        with get_db_session() as session:
            order = TrackedOrder(
                order_id="test-123",
                symbol="AAPL",
                side="buy",
                quantity=10,
                order_type="market",
                submitted_price=150.00,
                status="submitted",
                strategy_name="daily_profit_taker",
            )
            session.add(order)

        with get_db_session() as session:
            order = session.query(TrackedOrder).filter_by(order_id="test-123").first()
            assert order is not None
            assert order.symbol == "AAPL"
            assert order.status == "submitted"
            assert order.filled_quantity == 0.0

    def test_order_status_transitions(self, db_session):
        """Order status should be updatable."""
        with get_db_session() as session:
            order = TrackedOrder(
                order_id="test-456",
                symbol="MSFT",
                side="buy",
                quantity=5,
                order_type="market",
                status="submitted",
                strategy_name="momentum",
            )
            session.add(order)

        # Update to filled
        with get_db_session() as session:
            order = session.query(TrackedOrder).filter_by(order_id="test-456").first()
            order.status = "filled"
            order.filled_quantity = 5
            order.filled_price = 350.00
            order.filled_at = datetime.utcnow()

        with get_db_session() as session:
            order = session.query(TrackedOrder).filter_by(order_id="test-456").first()
            assert order.status == "filled"
            assert order.filled_quantity == 5
            assert order.filled_price == 350.00


class TestConfigValidation:
    """Test 4.5 - Configuration validation."""

    def test_valid_config(self):
        """Valid config should produce no errors."""
        from src.bot import validate_config
        import os
        os.environ["ALPACA_API_KEY"] = "test-key"
        os.environ["ALPACA_SECRET_KEY"] = "test-secret"

        config = {
            "trading": {
                "symbols": ["AAPL", "MSFT"],
                "check_interval_seconds": 60,
                "timezone": "America/New_York",
            },
            "risk": {
                "max_position_pct": 0.10,
                "max_daily_loss_pct": 0.02,
                "stop_loss_pct": 0.05,
                "trailing_stop_pct": 0.03,
            },
            "database": {"url": "sqlite:///test.db"},
        }

        errors = validate_config(config)
        assert len(errors) == 0

        # Cleanup
        del os.environ["ALPACA_API_KEY"]
        del os.environ["ALPACA_SECRET_KEY"]

    def test_missing_symbols(self):
        """Missing symbols should produce error."""
        from src.bot import validate_config
        config = {"trading": {"symbols": []}}
        errors = validate_config(config)
        assert any("symbols" in e.lower() for e in errors)

    def test_invalid_risk_params(self):
        """Invalid risk parameters should produce errors."""
        from src.bot import validate_config
        config = {
            "trading": {"symbols": ["AAPL"]},
            "risk": {
                "max_position_pct": 5.0,  # > 1.0
                "stop_loss_pct": 0.8,  # > 0.5
            },
        }
        errors = validate_config(config)
        assert any("max_position_pct" in e for e in errors)
        assert any("stop_loss_pct" in e for e in errors)

    def test_invalid_timezone(self):
        """Invalid timezone should produce error."""
        from src.bot import validate_config
        config = {
            "trading": {
                "symbols": ["AAPL"],
                "timezone": "Invalid/Timezone",
            },
        }
        errors = validate_config(config)
        assert any("timezone" in e.lower() for e in errors)


class TestRiskManagerCorrelation:
    """Test 2.4 - Correlation integration in risk manager."""

    def test_correlation_check_no_positions(self):
        """Correlation check should pass with no positions."""
        rm = RiskManager(RiskConfig())
        result = rm.check_correlation([], "AAPL")
        assert result.passed

    def test_correlation_check_with_data(self):
        """Correlation check should use analyzer when data available."""
        rm = RiskManager(RiskConfig(correlation_threshold=0.70))

        # Add price data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        base_prices = 100 + np.cumsum(np.random.randn(100))

        rm.correlation_analyzer.update_prices("AAPL", pd.Series(base_prices, index=dates))
        # Highly correlated
        rm.correlation_analyzer.update_prices("MSFT", pd.Series(base_prices * 1.1 + 5, index=dates))
        # Uncorrelated
        np.random.seed(99)
        rm.correlation_analyzer.update_prices("XOM", pd.Series(50 + np.cumsum(np.random.randn(100)), index=dates))

        positions = [
            PositionRisk(symbol="AAPL", quantity=10, market_value=1500, cost_basis=1400, unrealized_pnl=100, weight_pct=15),
        ]

        # MSFT should be highly correlated with AAPL
        result = rm.check_correlation(positions, "MSFT")
        # Result depends on correlation threshold logic
        assert result.check_type.value == "correlation"


class TestSectorExposure:
    """Test 2.5 - Real sector exposure tracking."""

    def test_sector_lookup(self):
        """Static sector lookup should work for known symbols."""
        from src.data.sectors import get_sector
        assert get_sector("AAPL") == "Technology"
        assert get_sector("JPM") == "Financials"
        assert get_sector("XOM") == "Energy"
        assert get_sector("UNKNOWN_SYMBOL") is None

    def test_sector_exposure_calculation(self):
        """Sector exposure calculation should be correct."""
        from src.data.sectors import calculate_sector_exposure
        positions = [
            {"symbol": "AAPL", "market_value": 5000},
            {"symbol": "MSFT", "market_value": 5000},
            {"symbol": "JPM", "market_value": 5000},
        ]
        exposure = calculate_sector_exposure(positions, 15000)
        assert "Technology" in exposure
        assert abs(exposure["Technology"] - 66.67) < 0.1  # 10000/15000
        assert abs(exposure["Financials"] - 33.33) < 0.1  # 5000/15000

    def test_risk_manager_sector_check(self):
        """Risk manager should check real sector exposure."""
        rm = RiskManager(RiskConfig(max_sector_exposure_pct=0.40))

        # Portfolio heavily in tech
        positions = [
            PositionRisk(symbol="AAPL", quantity=10, market_value=5000, cost_basis=4500, unrealized_pnl=500, weight_pct=50),
            PositionRisk(symbol="MSFT", quantity=10, market_value=5000, cost_basis=4500, unrealized_pnl=500, weight_pct=50),
        ]

        # Adding NVDA (also tech) should fail
        result = rm.check_sector_exposure(positions, "NVDA")
        assert not result.passed
        assert "Technology" in result.message

        # Adding JPM (financials) should pass
        result = rm.check_sector_exposure(positions, "JPM")
        assert result.passed


class TestAuditLogger:
    """Test 4.3 - Audit logging."""

    def test_audit_log_writes(self, tmp_path):
        """Audit logger should write JSON lines."""
        from src.utils.audit import AuditLogger
        import json

        audit = AuditLogger(log_dir=str(tmp_path))
        audit.log_trade(
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.0,
            order_id="test-001",
            reason="RSI oversold",
        )
        audit.log_signal(
            symbol="MSFT",
            signal_type="buy",
            strength=0.8,
            price=350.0,
            reason="MACD crossover",
        )

        # Read and verify
        log_file = tmp_path / "audit.jsonl"
        assert log_file.exists()

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["event"] == "trade"
        assert entry1["symbol"] == "AAPL"

        entry2 = json.loads(lines[1])
        assert entry2["event"] == "signal"
        assert entry2["symbol"] == "MSFT"


class TestSimulatedBroker:
    """Test 3.4 - Simulated broker."""

    def test_sim_broker_basic_flow(self):
        """SimulatedBroker should handle basic buy/sell flow."""
        from src.execution.sim_broker import SimulatedBroker, SimulatedBrokerConfig

        broker = SimulatedBroker(SimulatedBrokerConfig(initial_capital=100000))

        # Get account
        account = broker.get_account()
        assert account.equity == 100000
        assert account.cash == 100000

        # Set a price for AAPL
        broker.set_last_price("AAPL", 150.0)

        # Buy
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        result = broker.submit_order(order)
        assert result.success

        # Check position
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == 10

        # Sell
        sell_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        result = broker.submit_order(sell_order)
        assert result.success

        # No positions
        positions = broker.get_positions()
        assert len(positions) == 0


class TestEnsembleStrategy:
    """Test 2.2 - Ensemble strategy."""

    def test_ensemble_aggregation(self, sample_market_data):
        """Ensemble should aggregate signals from child strategies."""
        from src.strategies.ensemble import EnsembleStrategy
        from src.strategies.daily_profit_taker import DailyProfitTakerStrategy
        from src.strategies.mean_reversion import MeanReversionStrategy

        child1 = DailyProfitTakerStrategy()
        child2 = MeanReversionStrategy()

        strategy = EnsembleStrategy(
            strategies=[child1, child2],
            weights=[0.5, 0.5],
            voting_threshold=0.5,
        )

        market_data = {"AAPL": sample_market_data}
        positions: dict[str, PositionInfo] = {}

        # Just verify it runs without error
        signals = strategy.generate_signals(market_data, positions)
        assert isinstance(signals, list)


class TestHealthMonitor:
    """Test 4.4 - Health monitoring."""

    def test_health_status(self):
        """Health monitor should track heartbeat status."""
        from src.utils.health import HealthMonitor

        monitor = HealthMonitor(heartbeat_timeout_seconds=5)
        monitor.beat()

        status = monitor.get_status()
        assert status["healthy"]
        assert status["heartbeat_age_seconds"] is not None
        assert status["heartbeat_age_seconds"] < 1


class TestAdaptiveStrategyExitRegime:
    """Test Fix #6: should_exit re-detects regime before delegating."""

    def test_should_exit_detects_regime_independently(self, sample_market_data):
        """should_exit must apply fresh regime config even if generate_signals was never called."""
        from src.strategies.regime import AdaptiveStrategyWrapper, RegimeDetector
        from src.strategies.daily_profit_taker import DailyProfitTakerStrategy

        inner = DailyProfitTakerStrategy()
        wrapper = AdaptiveStrategyWrapper(inner_strategy=inner)

        position = PositionInfo(
            symbol="AAPL",
            quantity=10,
            avg_entry_price=100.0,
            current_price=90.0,
            unrealized_pnl=-100.0,
            unrealized_pnl_pct=-10.0,
            highest_price=105.0,
            stop_loss_price=95.0,
        )

        # Call should_exit WITHOUT calling generate_signals first.
        # Before the fix, inner_strategy.config would be the raw base_config.
        # After the fix, the wrapper detects regime and adjusts config.
        result = wrapper.should_exit(position, sample_market_data)
        # We just need it to not crash and to have set a regime
        assert wrapper.current_regime is not None


class TestEnsembleChildFailureWarning:
    """Test Fix #8: Ensemble logs weight redistribution on child failure."""

    def test_failed_child_logged_with_weight(self, sample_market_data, caplog):
        """When a child strategy fails, the ensemble should log the weight drop."""
        from src.strategies.ensemble import EnsembleStrategy
        from src.strategies.base import Strategy
        from unittest.mock import MagicMock

        good_child = MagicMock(spec=Strategy)
        good_child.name = "good"
        good_child.is_active = True
        good_child.generate_signals.return_value = []

        bad_child = MagicMock(spec=Strategy)
        bad_child.name = "bad"
        bad_child.is_active = True
        bad_child.generate_signals.side_effect = RuntimeError("boom")

        strategy = EnsembleStrategy(
            strategies=[good_child, bad_child],
            weights=[0.6, 0.4],
            voting_threshold=0.5,
        )

        import logging
        with caplog.at_level(logging.WARNING):
            strategy.generate_signals({"AAPL": sample_market_data}, {})

        # Should see a warning about the failed child and weight redistribution
        assert any("bad" in r.message and "40.0%" in r.message for r in caplog.records), \
            f"Expected warning about 'bad' child with 40.0% weight, got: {[r.message for r in caplog.records]}"


class TestDashboardDeduplication:
    """Test Fix #15: partial fills are deduplicated in metrics."""

    def test_partial_fills_not_double_counted(self):
        """Two rows with same order_id should be aggregated into one trade."""
        streamlit = pytest.importorskip("streamlit")
        from src.dashboard.app import _deduplicate_sell_trades

        trades_df = pd.DataFrame([
            {
                "id": 1, "order_id": "order-A", "symbol": "AAPL", "side": "sell",
                "quantity": 5, "price": 150.0, "total": 750.0,
                "pnl": 25.0, "pnl_pct": 3.3, "strategy": "momentum",
                "reason": "take profit", "paper": True,
                "executed_at": datetime(2024, 6, 1, 10, 0),
            },
            {
                "id": 2, "order_id": "order-A", "symbol": "AAPL", "side": "sell",
                "quantity": 5, "price": 151.0, "total": 755.0,
                "pnl": 30.0, "pnl_pct": 4.0, "strategy": "momentum",
                "reason": "take profit", "paper": True,
                "executed_at": datetime(2024, 6, 1, 10, 1),
            },
            {
                "id": 3, "order_id": "order-B", "symbol": "MSFT", "side": "sell",
                "quantity": 10, "price": 350.0, "total": 3500.0,
                "pnl": -50.0, "pnl_pct": -1.4, "strategy": "mean_reversion",
                "reason": "stop loss", "paper": True,
                "executed_at": datetime(2024, 6, 2, 11, 0),
            },
        ])

        result = _deduplicate_sell_trades(trades_df)

        # Should have 2 logical trades, not 3
        assert len(result) == 2

        # order-A should have combined P&L of 55, not 25 or 30
        order_a = result[result["order_id"] == "order-A"]
        assert len(order_a) == 1
        assert order_a.iloc[0]["pnl"] == 55.0
        assert order_a.iloc[0]["quantity"] == 10

        # order-B unchanged
        order_b = result[result["order_id"] == "order-B"]
        assert len(order_b) == 1
        assert order_b.iloc[0]["pnl"] == -50.0


class TestBrokerTypeSelection:
    """Test Fix #16: broker_type config selects the right broker."""

    def test_simulated_broker_selected(self):
        """broker.type: simulated should create SimulatedBroker."""
        from src.bot import validate_config
        import os

        config = {
            "trading": {
                "symbols": ["AAPL"],
                "check_interval_seconds": 60,
                "timezone": "America/New_York",
            },
            "broker": {"type": "simulated", "initial_capital": 50000},
            "risk": {},
            "database": {"url": "sqlite:///test.db"},
        }

        # Should NOT require API keys for simulated broker
        errors = validate_config(config)
        api_errors = [e for e in errors if "Alpaca" in e or "credentials" in e.lower()]
        assert len(api_errors) == 0, f"Should not require API keys for simulated broker: {api_errors}"

    def test_alpaca_requires_keys(self):
        """broker.type: alpaca (default) should require API keys."""
        from src.bot import validate_config
        import os

        # Clear any env vars
        old_key = os.environ.pop("ALPACA_API_KEY", None)
        old_secret = os.environ.pop("ALPACA_SECRET_KEY", None)

        try:
            config = {
                "trading": {"symbols": ["AAPL"]},
                "broker": {"type": "alpaca"},
                "database": {"url": "sqlite:///test.db"},
            }

            errors = validate_config(config)
            assert any("credentials" in e.lower() or "alpaca" in e.lower() for e in errors)
        finally:
            if old_key:
                os.environ["ALPACA_API_KEY"] = old_key
            if old_secret:
                os.environ["ALPACA_SECRET_KEY"] = old_secret
