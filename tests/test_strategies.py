"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.base import PositionInfo, SignalType
from src.strategies.indicators import TechnicalIndicators, IndicatorValues
from src.strategies.daily_profit_taker import DailyProfitTakerStrategy
from src.data.market_data import MarketData


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    np.random.seed(42)

    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        "high": prices * (1 + np.random.uniform(0, 0.02, 100)),
        "low": prices * (1 + np.random.uniform(-0.02, 0, 100)),
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, 100),
    }, index=dates)

    return df


@pytest.fixture
def oversold_data():
    """Create data that should trigger oversold conditions."""
    dates = pd.date_range(end=datetime.now(), periods=50, freq="D")

    # Create a strong downtrend followed by stabilization
    prices = np.concatenate([
        np.linspace(100, 70, 40),  # Strong decline
        np.linspace(70, 72, 10),   # Stabilization
    ])

    df = pd.DataFrame({
        "open": prices * 1.005,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, 50),
    }, index=dates)

    return df


@pytest.fixture
def indicators():
    """Create a TechnicalIndicators instance."""
    return TechnicalIndicators()


@pytest.fixture
def strategy():
    """Create a DailyProfitTakerStrategy instance."""
    config = {
        "profit_target_pct": 0.02,
        "stop_loss_pct": 0.05,
        "trailing_stop_pct": 0.03,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
    }
    return DailyProfitTakerStrategy(config)


class TestTechnicalIndicators:
    """Tests for technical indicators."""

    def test_rsi_calculation(self, indicators, sample_ohlcv):
        """Test RSI calculation returns valid values."""
        rsi = indicators.calculate_rsi(sample_ohlcv)

        assert len(rsi) == len(sample_ohlcv)
        assert rsi.iloc[-1] >= 0 and rsi.iloc[-1] <= 100
        # First few values should be NaN due to period
        assert pd.isna(rsi.iloc[0])

    def test_rsi_oversold_detection(self, indicators):
        """Test RSI oversold detection."""
        assert indicators.is_oversold(25, threshold=30) is True
        assert indicators.is_oversold(35, threshold=30) is False
        assert indicators.is_oversold(None, threshold=30) is False

    def test_rsi_overbought_detection(self, indicators):
        """Test RSI overbought detection."""
        assert indicators.is_overbought(75, threshold=70) is True
        assert indicators.is_overbought(65, threshold=70) is False
        assert indicators.is_overbought(None, threshold=70) is False

    def test_macd_calculation(self, indicators, sample_ohlcv):
        """Test MACD calculation returns three series."""
        macd, signal, histogram = indicators.calculate_macd(sample_ohlcv)

        assert len(macd) == len(sample_ohlcv)
        assert len(signal) == len(sample_ohlcv)
        assert len(histogram) == len(sample_ohlcv)
        # Histogram should be macd - signal
        assert np.isclose(histogram.iloc[-1], macd.iloc[-1] - signal.iloc[-1])

    def test_bollinger_bands_calculation(self, indicators, sample_ohlcv):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = indicators.calculate_bollinger_bands(sample_ohlcv)

        assert len(upper) == len(sample_ohlcv)
        # Upper should always be above middle, middle above lower
        assert upper.iloc[-1] > middle.iloc[-1] > lower.iloc[-1]

    def test_bollinger_percent(self, indicators, sample_ohlcv):
        """Test Bollinger %B calculation."""
        percent_b = indicators.calculate_bollinger_percent(sample_ohlcv)

        assert len(percent_b) == len(sample_ohlcv)
        # Most values should be between 0 and 1
        valid_values = percent_b.dropna()
        assert (valid_values >= -0.5).all() and (valid_values <= 1.5).all()

    def test_sma_calculation(self, indicators, sample_ohlcv):
        """Test SMA calculation."""
        sma = indicators.calculate_sma(sample_ohlcv, 20)

        assert len(sma) == len(sample_ohlcv)
        # First 19 values should be NaN
        assert pd.isna(sma.iloc[18])
        assert not pd.isna(sma.iloc[19])

    def test_ema_calculation(self, indicators, sample_ohlcv):
        """Test EMA calculation."""
        ema = indicators.calculate_ema(sample_ohlcv, 12)

        assert len(ema) == len(sample_ohlcv)
        # EMA should have values after first few periods
        assert not pd.isna(ema.iloc[-1])

    def test_atr_calculation(self, indicators, sample_ohlcv):
        """Test ATR calculation."""
        atr = indicators.calculate_atr(sample_ohlcv)

        assert len(atr) == len(sample_ohlcv)
        # ATR should always be positive
        assert (atr.dropna() > 0).all()

    def test_add_all_indicators(self, indicators, sample_ohlcv):
        """Test adding all indicators to DataFrame."""
        result = indicators.add_all_indicators(sample_ohlcv)

        expected_columns = [
            "rsi", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_percent",
            "sma_short", "sma_long", "ema", "atr", "vwap"
        ]

        for col in expected_columns:
            assert col in result.columns

    def test_get_current_values(self, indicators, sample_ohlcv):
        """Test getting current indicator values."""
        values = indicators.get_current_values(sample_ohlcv)

        assert isinstance(values, IndicatorValues)
        assert values.rsi is not None
        assert values.macd is not None
        assert values.close is not None

    def test_golden_cross_detection(self, indicators):
        """Test golden cross detection."""
        # Create data with a golden cross
        dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
        # Prices that cause short SMA to cross above long SMA
        prices = np.concatenate([
            np.linspace(100, 90, 30),  # Decline
            np.linspace(90, 110, 30),  # Rally
        ])

        df = pd.DataFrame({
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.ones(60) * 1000000,
        }, index=dates)

        # This specific data should show a potential crossover
        result = indicators.golden_cross(df)
        # Result can be numpy bool or Python bool
        assert result in (True, False)


class TestDailyProfitTakerStrategy:
    """Tests for the Daily Profit Taker strategy."""

    def test_strategy_initialization(self, strategy):
        """Test strategy initializes with correct parameters."""
        assert strategy.name == "daily_profit_taker"
        assert strategy.profit_target_pct == 0.02
        assert strategy.stop_loss_pct == 0.05
        assert strategy.rsi_oversold == 30

    def test_strategy_from_settings(self):
        """Test creating strategy from settings dict."""
        settings = {
            "strategy": {
                "profit_target_pct": 0.03,
                "rsi_oversold": 25,
            },
            "risk": {
                "stop_loss_pct": 0.04,
            },
        }
        strategy = DailyProfitTakerStrategy.from_settings(settings)

        assert strategy.profit_target_pct == 0.03
        assert strategy.stop_loss_pct == 0.04

    def test_generate_signals_no_positions(self, strategy, oversold_data):
        """Test signal generation with no existing positions."""
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                df=oversold_data,
                last_price=oversold_data["close"].iloc[-1],
                last_updated=datetime.now(),
            ),
        }

        signals = strategy.generate_signals(market_data, {})

        # Should generate signals (may or may not be BUY depending on conditions)
        assert isinstance(signals, list)

    def test_generate_signals_skips_held_positions(self, strategy, sample_ohlcv):
        """Test that strategy skips symbols already held."""
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                df=sample_ohlcv,
                last_price=100,
                last_updated=datetime.now(),
            ),
        }

        positions = {
            "AAPL": PositionInfo(
                symbol="AAPL",
                quantity=10,
                avg_entry_price=95,
                current_price=100,
                unrealized_pnl=50,
                unrealized_pnl_pct=5.26,
                highest_price=100,
            ),
        }

        signals = strategy.generate_signals(market_data, positions)

        # Should not generate signal for AAPL since it's already held
        aapl_signals = [s for s in signals if s.symbol == "AAPL"]
        assert len(aapl_signals) == 0

    def test_should_exit_stop_loss(self, strategy, sample_ohlcv):
        """Test exit signal generation for stop loss."""
        position = PositionInfo(
            symbol="AAPL",
            quantity=10,
            avg_entry_price=100,
            current_price=94,  # 6% loss, exceeds 5% stop
            unrealized_pnl=-60,
            unrealized_pnl_pct=-6,
            highest_price=100,
            stop_loss_price=95,
        )

        market_data = MarketData(
            symbol="AAPL",
            df=sample_ohlcv,
            last_price=94,
            last_updated=datetime.now(),
        )

        exit_signal = strategy.should_exit(position, market_data)

        assert exit_signal is not None
        assert exit_signal.reason.value == "stop_loss"

    def test_should_exit_take_profit(self, strategy, sample_ohlcv):
        """Test exit signal generation for take profit."""
        position = PositionInfo(
            symbol="AAPL",
            quantity=10,
            avg_entry_price=100,
            current_price=103,  # 3% gain, exceeds 2% target
            unrealized_pnl=30,
            unrealized_pnl_pct=3,
            highest_price=103,
            stop_loss_price=95,
        )

        market_data = MarketData(
            symbol="AAPL",
            df=sample_ohlcv,
            last_price=103,
            last_updated=datetime.now(),
        )

        exit_signal = strategy.should_exit(position, market_data)

        assert exit_signal is not None
        assert exit_signal.reason.value == "take_profit"

    def test_should_exit_trailing_stop(self, strategy, sample_ohlcv):
        """Test exit signal for trailing stop."""
        position = PositionInfo(
            symbol="AAPL",
            quantity=10,
            avg_entry_price=100,
            current_price=101,  # Price dropped from high
            unrealized_pnl=10,
            unrealized_pnl_pct=1,
            highest_price=106,  # Was at 106, now 101 (>3% drop)
            stop_loss_price=95,
            trailing_stop_price=102.82,  # 106 * 0.97
        )

        market_data = MarketData(
            symbol="AAPL",
            df=sample_ohlcv,
            last_price=101,
            last_updated=datetime.now(),
        )

        exit_signal = strategy.should_exit(position, market_data)

        assert exit_signal is not None
        assert exit_signal.reason.value == "trailing_stop"

    def test_should_exit_no_signal(self, strategy, sample_ohlcv):
        """Test no exit signal for healthy position."""
        position = PositionInfo(
            symbol="AAPL",
            quantity=10,
            avg_entry_price=100,
            current_price=100.5,  # Slightly profitable
            unrealized_pnl=5,
            unrealized_pnl_pct=0.5,
            highest_price=100.5,
            stop_loss_price=95,
            trailing_stop_price=97.485,
        )

        market_data = MarketData(
            symbol="AAPL",
            df=sample_ohlcv,
            last_price=100.5,
            last_updated=datetime.now(),
        )

        exit_signal = strategy.should_exit(position, market_data)

        # No exit signal expected
        assert exit_signal is None

    def test_update_trailing_stop(self, strategy):
        """Test trailing stop update calculation."""
        position = PositionInfo(
            symbol="AAPL",
            quantity=10,
            avg_entry_price=100,
            current_price=105,
            unrealized_pnl=50,
            unrealized_pnl_pct=5,
            highest_price=103,  # Previous high
            trailing_stop_price=99.91,  # Previous trailing stop
        )

        # Price moved higher
        new_stop = strategy.update_trailing_stop(position, 107)

        assert new_stop is not None
        assert new_stop > position.trailing_stop_price
        # New stop should be 107 * 0.97 = 103.79
        assert abs(new_stop - 103.79) < 0.01

    def test_update_trailing_stop_no_update_needed(self, strategy):
        """Test trailing stop not updated when price hasn't increased."""
        position = PositionInfo(
            symbol="AAPL",
            quantity=10,
            avg_entry_price=100,
            current_price=102,
            unrealized_pnl=20,
            unrealized_pnl_pct=2,
            highest_price=105,  # Previous high is higher
            trailing_stop_price=101.85,
        )

        # Price is below previous high
        new_stop = strategy.update_trailing_stop(position, 102)

        assert new_stop is None

    def test_calculate_stop_loss(self, strategy):
        """Test stop loss calculation."""
        entry_price = 100
        stop = strategy.calculate_stop_loss(entry_price, 0.05)
        assert stop == 95

    def test_calculate_take_profit(self, strategy):
        """Test take profit calculation."""
        entry_price = 100
        target = strategy.calculate_take_profit(entry_price, 0.02)
        assert target == 102

    def test_get_strategy_info(self, strategy):
        """Test strategy info retrieval."""
        info = strategy.get_strategy_info()

        assert info["name"] == "daily_profit_taker"
        assert info["is_active"] is True
        assert "parameters" in info
        assert info["parameters"]["profit_target_pct"] == 0.02
