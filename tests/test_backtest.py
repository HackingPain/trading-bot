"""Tests for backtesting engine and metrics."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np

from src.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestPosition,
    BacktestResult,
)
from src.backtest.metrics import (
    PerformanceMetrics,
    Trade,
    calculate_metrics,
    calculate_daily_metrics,
    _max_consecutive,
)
from src.strategies.base import Strategy, Signal, SignalType, PositionInfo, ExitSignal, ExitReason
from src.data.market_data import MarketData


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_equity_curve():
    """Create sample equity curve for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    # Simulate equity with some volatility and overall growth
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)
    equity = 100000 * np.cumprod(1 + returns)
    return pd.Series(equity, index=dates, name="equity")


@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    return [
        Trade(
            symbol="AAPL",
            entry_date=datetime(2024, 1, 5),
            exit_date=datetime(2024, 1, 10),
            entry_price=150.0,
            exit_price=155.0,
            quantity=10,
            side="long",
            pnl=50.0,
            pnl_pct=3.33,
        ),
        Trade(
            symbol="MSFT",
            entry_date=datetime(2024, 1, 8),
            exit_date=datetime(2024, 1, 15),
            entry_price=350.0,
            exit_price=340.0,
            quantity=5,
            side="long",
            pnl=-50.0,
            pnl_pct=-2.86,
        ),
        Trade(
            symbol="GOOGL",
            entry_date=datetime(2024, 1, 12),
            exit_date=datetime(2024, 1, 20),
            entry_price=140.0,
            exit_price=150.0,
            quantity=20,
            side="long",
            pnl=200.0,
            pnl_pct=7.14,
        ),
        Trade(
            symbol="AAPL",
            entry_date=datetime(2024, 1, 22),
            exit_date=datetime(2024, 1, 25),
            entry_price=160.0,
            exit_price=158.0,
            quantity=10,
            side="long",
            pnl=-20.0,
            pnl_pct=-1.25,
        ),
    ]


@pytest.fixture
def mock_strategy():
    """Create mock strategy for testing."""
    strategy = Mock(spec=Strategy)
    strategy.name = "test_strategy"
    strategy.generate_signals = Mock(return_value=[])
    strategy.should_exit = Mock(return_value=None)
    return strategy


@pytest.fixture
def mock_data_provider():
    """Create mock data provider for testing."""
    provider = Mock()
    return provider


# =============================================================================
# PerformanceMetrics Tests
# =============================================================================

class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = PerformanceMetrics()

        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = PerformanceMetrics(
            total_return=1000.0,
            total_return_pct=10.0,
            sharpe_ratio=1.5,
            win_rate=60.0,
            total_trades=20,
        )

        result = metrics.to_dict()

        assert result["total_return"] == 1000.0
        assert result["total_return_pct"] == 10.0
        assert result["sharpe_ratio"] == 1.5
        assert result["win_rate"] == 60.0
        assert result["total_trades"] == 20

    def test_summary_generation(self):
        """Test summary text generation."""
        metrics = PerformanceMetrics(
            total_return=5000.0,
            total_return_pct=5.0,
            sharpe_ratio=1.2,
            max_drawdown_pct=8.5,
            total_trades=50,
            win_rate=55.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            trading_days=252,
        )

        summary = metrics.summary()

        assert "Performance Summary" in summary
        assert "$5,000.00" in summary
        assert "5.00%" in summary
        assert "1.20" in summary  # Sharpe
        assert "55.0%" in summary  # Win rate


# =============================================================================
# calculate_metrics Tests
# =============================================================================

class TestCalculateMetrics:
    """Tests for metrics calculation function."""

    def test_basic_metrics(self, sample_equity_curve, sample_trades):
        """Test basic metrics calculation."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades)

        assert metrics.total_trades == 4
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 2
        assert metrics.win_rate == 50.0

    def test_return_calculation(self, sample_equity_curve, sample_trades):
        """Test return calculations."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades)

        initial = sample_equity_curve.iloc[0]
        final = sample_equity_curve.iloc[-1]
        expected_return = final - initial

        assert abs(metrics.total_return - expected_return) < 1.0

    def test_pnl_statistics(self, sample_equity_curve, sample_trades):
        """Test P&L statistics."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades)

        assert metrics.gross_profit == 250.0  # 50 + 200
        assert metrics.gross_loss == 70.0  # 50 + 20
        assert metrics.net_profit == 180.0

    def test_profit_factor(self, sample_equity_curve, sample_trades):
        """Test profit factor calculation."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades)

        expected_pf = 250.0 / 70.0
        assert abs(metrics.profit_factor - expected_pf) < 0.01

    def test_average_trade_stats(self, sample_equity_curve, sample_trades):
        """Test average trade statistics."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades)

        assert metrics.avg_winning_trade == 125.0  # (50 + 200) / 2
        assert metrics.avg_losing_trade == -35.0  # (-50 + -20) / 2
        assert metrics.largest_winning_trade == 200.0
        assert metrics.largest_losing_trade == -50.0

    def test_drawdown_calculation(self, sample_equity_curve, sample_trades):
        """Test drawdown metrics."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades)

        assert metrics.max_drawdown >= 0
        assert metrics.max_drawdown_pct >= 0
        assert metrics.max_drawdown_pct <= 100

    def test_volatility_calculation(self, sample_equity_curve, sample_trades):
        """Test volatility metrics."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades)

        assert metrics.daily_volatility > 0
        assert metrics.annual_volatility > 0
        # Annual volatility should be roughly sqrt(252) times daily
        assert metrics.annual_volatility > metrics.daily_volatility

    def test_sharpe_ratio_calculation(self, sample_equity_curve, sample_trades):
        """Test Sharpe ratio calculation."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades, risk_free_rate=0.05)

        # Sharpe should be a reasonable value
        assert -5 < metrics.sharpe_ratio < 5

    def test_empty_equity_curve(self):
        """Test with empty equity curve."""
        metrics = calculate_metrics(pd.Series(dtype=float), [])

        assert metrics.total_return == 0.0
        assert metrics.total_trades == 0

    def test_empty_trades(self, sample_equity_curve):
        """Test with no trades."""
        metrics = calculate_metrics(sample_equity_curve, [])

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        # Should still calculate equity-based metrics
        assert metrics.total_return != 0.0

    def test_holding_period(self, sample_equity_curve, sample_trades):
        """Test average holding period calculation."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades)

        # Average of 5, 7, 8, 3 days
        expected_avg = (5 + 7 + 8 + 3) / 4
        assert abs(metrics.avg_holding_period_days - expected_avg) < 0.1

    def test_consecutive_wins_losses(self, sample_equity_curve, sample_trades):
        """Test consecutive wins/losses calculation."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades)

        # Based on sample_trades: win, loss, win, loss
        assert metrics.max_consecutive_wins >= 1
        assert metrics.max_consecutive_losses >= 1


class TestMaxConsecutive:
    """Tests for _max_consecutive helper."""

    def test_all_positive(self):
        """Test with all positive values."""
        values = [1, 2, 3, 4, 5]
        assert _max_consecutive(values, lambda x: x > 0) == 5

    def test_alternating(self):
        """Test with alternating values."""
        values = [1, -1, 1, -1, 1]
        assert _max_consecutive(values, lambda x: x > 0) == 1
        assert _max_consecutive(values, lambda x: x < 0) == 1

    def test_streak_in_middle(self):
        """Test with streak in middle."""
        values = [-1, 1, 1, 1, -1]
        assert _max_consecutive(values, lambda x: x > 0) == 3

    def test_empty_list(self):
        """Test with empty list."""
        assert _max_consecutive([], lambda x: x > 0) == 0


class TestCalculateDailyMetrics:
    """Tests for daily metrics calculation."""

    def test_daily_metrics_columns(self, sample_equity_curve):
        """Test that daily metrics has expected columns."""
        df = calculate_daily_metrics(sample_equity_curve)

        expected_columns = [
            "equity",
            "daily_return",
            "cumulative_return",
            "rolling_max",
            "drawdown",
            "rolling_sharpe_30d",
            "rolling_volatility_30d",
        ]

        for col in expected_columns:
            assert col in df.columns

    def test_cumulative_return(self, sample_equity_curve):
        """Test cumulative return calculation."""
        df = calculate_daily_metrics(sample_equity_curve)

        # First day should be 0% return
        assert df["cumulative_return"].iloc[0] == 0.0

        # Last day should match total return
        expected = ((sample_equity_curve.iloc[-1] / sample_equity_curve.iloc[0]) - 1) * 100
        assert abs(df["cumulative_return"].iloc[-1] - expected) < 0.01

    def test_drawdown_never_positive(self, sample_equity_curve):
        """Test that drawdown is never positive."""
        df = calculate_daily_metrics(sample_equity_curve)
        assert (df["drawdown"] <= 0).all()


# =============================================================================
# BacktestConfig Tests
# =============================================================================

class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.initial_capital == 100000.0
        assert config.commission_per_trade == 0.0
        assert config.commission_pct == 0.0
        assert config.slippage_pct == 0.001
        assert config.max_position_pct == 0.10
        assert config.stop_loss_pct == 0.05
        assert config.take_profit_pct == 0.02

    def test_custom_values(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_capital=50000.0,
            commission_per_trade=1.0,
            slippage_pct=0.002,
            max_position_pct=0.05,
        )

        assert config.initial_capital == 50000.0
        assert config.commission_per_trade == 1.0
        assert config.slippage_pct == 0.002
        assert config.max_position_pct == 0.05


# =============================================================================
# BacktestPosition Tests
# =============================================================================

class TestBacktestPosition:
    """Tests for BacktestPosition dataclass."""

    def test_to_position_info(self):
        """Test conversion to PositionInfo."""
        position = BacktestPosition(
            symbol="AAPL",
            quantity=10,
            entry_price=150.0,
            entry_date=datetime(2024, 1, 1),
            stop_loss=142.5,
            take_profit=153.0,
            trailing_stop=145.0,
            highest_price=152.0,
            cost_basis=1500.0,
        )

        info = position.to_position_info(current_price=155.0)

        assert info.symbol == "AAPL"
        assert info.quantity == 10
        assert info.avg_entry_price == 150.0
        assert info.current_price == 155.0
        assert info.unrealized_pnl == 50.0  # (155-150) * 10
        assert abs(info.unrealized_pnl_pct - 3.33) < 0.1
        assert info.stop_loss_price == 142.5
        assert info.take_profit_price == 153.0

    def test_to_position_info_with_loss(self):
        """Test conversion with losing position."""
        position = BacktestPosition(
            symbol="MSFT",
            quantity=5,
            entry_price=400.0,
            entry_date=datetime(2024, 1, 1),
            stop_loss=380.0,
            take_profit=420.0,
            trailing_stop=390.0,
            highest_price=400.0,
            cost_basis=2000.0,
        )

        info = position.to_position_info(current_price=385.0)

        assert info.unrealized_pnl == -75.0  # (385-400) * 5
        assert info.unrealized_pnl_pct < 0


# =============================================================================
# BacktestResult Tests
# =============================================================================

class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_summary_generation(self, sample_equity_curve, sample_trades):
        """Test summary generation."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades)
        config = BacktestConfig()

        result = BacktestResult(
            config=config,
            metrics=metrics,
            equity_curve=sample_equity_curve,
            daily_metrics=pd.DataFrame(),
            trades=sample_trades,
            signals=[],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 10),
        )

        summary = result.summary()

        assert "Backtest Results" in summary
        assert "2024-01-01" in summary
        assert "Initial Capital" in summary

    def test_to_dict(self, sample_equity_curve, sample_trades):
        """Test serialization to dict."""
        metrics = calculate_metrics(sample_equity_curve, sample_trades)
        config = BacktestConfig()

        result = BacktestResult(
            config=config,
            metrics=metrics,
            equity_curve=sample_equity_curve,
            daily_metrics=pd.DataFrame(),
            trades=sample_trades,
            signals=[],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 4, 10),
        )

        data = result.to_dict()

        assert "config" in data
        assert "metrics" in data
        assert "trades" in data
        assert data["config"]["initial_capital"] == 100000.0
        assert len(data["trades"]) == 4


# =============================================================================
# BacktestEngine Tests
# =============================================================================

class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_initialization(self, mock_strategy, mock_data_provider):
        """Test engine initialization."""
        config = BacktestConfig(initial_capital=50000.0)
        engine = BacktestEngine(mock_strategy, mock_data_provider, config)

        assert engine.config.initial_capital == 50000.0
        assert engine.strategy == mock_strategy

    def test_initialization_default_config(self, mock_strategy, mock_data_provider):
        """Test engine with default config."""
        engine = BacktestEngine(mock_strategy, mock_data_provider)

        assert engine.config.initial_capital == 100000.0

    def test_calculate_equity(self, mock_strategy, mock_data_provider):
        """Test equity calculation."""
        engine = BacktestEngine(mock_strategy, mock_data_provider)
        engine._cash = 50000.0
        engine._positions = {
            "AAPL": BacktestPosition(
                symbol="AAPL",
                quantity=100,
                entry_price=150.0,
                entry_date=datetime(2024, 1, 1),
                stop_loss=140.0,
                take_profit=160.0,
                trailing_stop=145.0,
                highest_price=155.0,
                cost_basis=15000.0,
            )
        }

        market_data = {
            "AAPL": Mock(last_price=160.0)
        }

        equity = engine._calculate_equity(market_data)

        # Cash + Position value (100 shares * $160)
        assert equity == 50000.0 + 16000.0

    def test_get_common_dates(self, mock_strategy, mock_data_provider):
        """Test common dates extraction."""
        engine = BacktestEngine(mock_strategy, mock_data_provider)

        dates1 = pd.date_range("2024-01-01", periods=10)
        dates2 = pd.date_range("2024-01-03", periods=10)

        historical_data = {
            "AAPL": pd.DataFrame(index=dates1),
            "MSFT": pd.DataFrame(index=dates2),
        }

        common = engine._get_common_dates(historical_data)

        # Should only include dates present in both
        assert len(common) == 8  # Jan 3-10

    def test_empty_result_on_no_data(self, mock_strategy, mock_data_provider):
        """Test empty result when no data available."""
        mock_data_provider.get_historical_data = Mock(return_value=pd.DataFrame())

        engine = BacktestEngine(mock_strategy, mock_data_provider)

        result = engine.run(
            symbols=["AAPL"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 1),
        )

        assert result.metrics.total_trades == 0
        assert result.equity_curve.empty


class TestBacktestEngineIntegration:
    """Integration tests for BacktestEngine with mock data."""

    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        np.random.seed(42)

        data = {
            "AAPL": pd.DataFrame({
                "open": 150 + np.cumsum(np.random.randn(60) * 2),
                "high": 152 + np.cumsum(np.random.randn(60) * 2),
                "low": 148 + np.cumsum(np.random.randn(60) * 2),
                "close": 150 + np.cumsum(np.random.randn(60) * 2),
                "volume": np.random.randint(1000000, 5000000, 60),
            }, index=dates),
        }

        # Ensure high > low > 0
        for symbol in data:
            data[symbol]["high"] = data[symbol][["open", "high", "close"]].max(axis=1) + 1
            data[symbol]["low"] = data[symbol][["open", "low", "close"]].min(axis=1) - 1
            data[symbol] = data[symbol].clip(lower=1)  # No negative prices

        return data

    def test_full_backtest_run(self, sample_historical_data):
        """Test complete backtest run."""
        # Create a simple strategy that generates signals
        strategy = Mock(spec=Strategy)
        strategy.name = "test"

        signal_count = [0]

        def generate_signals(market_data, positions):
            signal_count[0] += 1
            # Generate buy signal every 10 days if not holding
            if signal_count[0] % 10 == 5 and "AAPL" not in positions:
                return [Signal(
                    symbol="AAPL",
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    price=market_data["AAPL"].last_price,
                    reason="test signal",
                    strategy_name="test",
                )]
            return []

        strategy.generate_signals = generate_signals
        strategy.should_exit = Mock(return_value=None)

        # Mock data provider
        data_provider = Mock()
        data_provider.get_historical_data = Mock(
            side_effect=lambda symbol, **kwargs: sample_historical_data.get(symbol, pd.DataFrame())
        )

        engine = BacktestEngine(
            strategy,
            data_provider,
            BacktestConfig(
                initial_capital=100000.0,
                max_position_pct=0.10,
                stop_loss_pct=0.05,
                take_profit_pct=0.03,
            )
        )

        result = engine.run(
            symbols=["AAPL"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 29),
        )

        # Should have equity history
        assert len(result.equity_curve) > 0

        # Should have generated some signals
        assert len(result.signals) > 0
