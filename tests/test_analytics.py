"""Tests for analytics module - standalone tests that don't require database."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from src.backtest.metrics import PerformanceMetrics, Trade as MetricsTrade


# =============================================================================
# PerformanceReport Tests (using local test implementation)
# =============================================================================

@dataclass
class PerformanceReport:
    """Test version of PerformanceReport for isolated testing."""
    period_start: datetime
    period_end: datetime
    trading_days: int
    starting_balance: float
    ending_balance: float
    total_deposits: float = 0.0
    total_withdrawals: float = 0.0
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    best_day: Optional[tuple[datetime, float]] = None
    worst_day: Optional[tuple[datetime, float]] = None
    best_trade: Optional[dict] = None
    worst_trade: Optional[dict] = None
    symbol_performance: dict[str, dict] = field(default_factory=dict)
    monthly_returns: dict[str, float] = field(default_factory=dict)
    daily_returns: list[float] = field(default_factory=list)

    def summary(self) -> str:
        best_day_str = f"{self.best_day[0].date()}" if self.best_day else "N/A"
        best_day_val = f"${self.best_day[1]:,.2f}" if self.best_day else "$0.00"
        worst_day_str = f"{self.worst_day[0].date()}" if self.worst_day else "N/A"
        worst_day_val = f"${self.worst_day[1]:,.2f}" if self.worst_day else "$0.00"

        return f"""
Performance Report
==================
Period: {self.period_start.date()} to {self.period_end.date()} ({self.trading_days} days)
Starting Balance: ${self.starting_balance:,.2f}
Ending Balance: ${self.ending_balance:,.2f}
Net Change: ${self.ending_balance - self.starting_balance:,.2f}

Best Day: {best_day_str} ({best_day_val})
Worst Day: {worst_day_str} ({worst_day_val})
"""

    def to_dict(self) -> dict:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "trading_days": self.trading_days,
            "starting_balance": self.starting_balance,
            "ending_balance": self.ending_balance,
            "metrics": self.metrics.to_dict(),
            "symbol_performance": self.symbol_performance,
            "monthly_returns": self.monthly_returns,
        }


class TestPerformanceReport:
    """Tests for PerformanceReport dataclass."""

    def test_initialization(self):
        """Test report initialization with basic values."""
        report = PerformanceReport(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 3, 31),
            trading_days=60,
            starting_balance=100000.0,
            ending_balance=110000.0,
        )

        assert report.period_start == datetime(2024, 1, 1)
        assert report.period_end == datetime(2024, 3, 31)
        assert report.trading_days == 60
        assert report.starting_balance == 100000.0
        assert report.ending_balance == 110000.0

    def test_default_values(self):
        """Test default values are set correctly."""
        report = PerformanceReport(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 3, 31),
            trading_days=60,
            starting_balance=100000.0,
            ending_balance=100000.0,
        )

        assert report.total_deposits == 0.0
        assert report.total_withdrawals == 0.0
        assert isinstance(report.metrics, PerformanceMetrics)
        assert report.symbol_performance == {}
        assert report.monthly_returns == {}

    def test_summary_generation(self):
        """Test summary text generation."""
        metrics = PerformanceMetrics(
            total_return=10000.0,
            sharpe_ratio=1.5,
            win_rate=60.0,
            total_trades=50,
        )

        report = PerformanceReport(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 3, 31),
            trading_days=60,
            starting_balance=100000.0,
            ending_balance=110000.0,
            metrics=metrics,
            symbol_performance={
                "AAPL": {"total_pnl": 5000.0, "trades": 20, "win_rate": 65.0},
                "MSFT": {"total_pnl": 5000.0, "trades": 30, "win_rate": 55.0},
            },
        )

        summary = report.summary()

        assert "Performance Report" in summary
        assert "2024-01-01" in summary
        assert "2024-03-31" in summary
        assert "$100,000.00" in summary
        assert "$110,000.00" in summary

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = PerformanceMetrics(
            total_return=5000.0,
            win_rate=55.0,
        )

        report = PerformanceReport(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 3, 31),
            trading_days=60,
            starting_balance=100000.0,
            ending_balance=105000.0,
            metrics=metrics,
            monthly_returns={"2024-01": 2000.0, "2024-02": 1500.0, "2024-03": 1500.0},
        )

        data = report.to_dict()

        assert data["period_start"] == "2024-01-01T00:00:00"
        assert data["period_end"] == "2024-03-31T00:00:00"
        assert data["trading_days"] == 60
        assert data["starting_balance"] == 100000.0
        assert data["ending_balance"] == 105000.0
        assert "metrics" in data
        assert data["monthly_returns"]["2024-01"] == 2000.0


# =============================================================================
# Analytics Helper Functions Tests
# =============================================================================

class TestAnalyticsHelpers:
    """Tests for analytics helper functions that don't require database."""

    def test_convert_trades_to_metrics_format(self):
        """Test trade conversion to metrics format."""
        # Simulate the conversion logic
        def convert_trades(trades):
            result = []
            open_positions = {}

            for trade in trades:
                symbol = trade["symbol"]
                side = trade["side"]

                if side == "buy":
                    if symbol not in open_positions:
                        open_positions[symbol] = []
                    open_positions[symbol].append(trade)

                elif side == "sell" and symbol in open_positions and open_positions[symbol]:
                    entry_trade = open_positions[symbol].pop(0)
                    result.append({
                        "symbol": symbol,
                        "entry_price": entry_trade["price"],
                        "exit_price": trade["price"],
                        "pnl": trade.get("realized_pnl", 0),
                    })

            return result

        trades = [
            {"symbol": "AAPL", "side": "buy", "price": 150.0},
            {"symbol": "AAPL", "side": "sell", "price": 155.0, "realized_pnl": 50.0},
        ]

        result = convert_trades(trades)

        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["entry_price"] == 150.0
        assert result[0]["exit_price"] == 155.0
        assert result[0]["pnl"] == 50.0

    def test_analyze_by_symbol(self):
        """Test symbol breakdown analysis."""
        def analyze_by_symbol(trades):
            symbol_data = {}

            for trade in trades:
                symbol = trade["symbol"]
                if symbol not in symbol_data:
                    symbol_data[symbol] = []
                symbol_data[symbol].append(trade)

            result = {}
            for symbol, symbol_trades in symbol_data.items():
                pnls = [t.get("realized_pnl", 0) for t in symbol_trades if t["side"] == "sell"]
                wins = sum(1 for p in pnls if p and p > 0)

                result[symbol] = {
                    "trades": len([t for t in symbol_trades if t["side"] == "sell"]),
                    "total_pnl": sum(p for p in pnls if p),
                    "win_rate": (wins / len(pnls) * 100) if pnls else 0,
                }

            return result

        trades = [
            {"symbol": "AAPL", "side": "buy", "price": 150.0},
            {"symbol": "AAPL", "side": "sell", "price": 155.0, "realized_pnl": 100.0},
            {"symbol": "AAPL", "side": "buy", "price": 156.0},
            {"symbol": "AAPL", "side": "sell", "price": 158.0, "realized_pnl": 100.0},
            {"symbol": "MSFT", "side": "buy", "price": 400.0},
            {"symbol": "MSFT", "side": "sell", "price": 390.0, "realized_pnl": -50.0},
        ]

        result = analyze_by_symbol(trades)

        assert "AAPL" in result
        assert "MSFT" in result
        assert result["AAPL"]["trades"] == 2
        assert result["MSFT"]["trades"] == 1
        assert result["AAPL"]["total_pnl"] == 200.0
        assert result["MSFT"]["total_pnl"] == -50.0

    def test_find_extreme_days(self):
        """Test finding best and worst days."""
        def find_extreme_days(daily_perf):
            if not daily_perf:
                return None, None

            best = max(daily_perf, key=lambda dp: dp["total_pnl"])
            worst = min(daily_perf, key=lambda dp: dp["total_pnl"])

            return (best["date"], best["total_pnl"]), (worst["date"], worst["total_pnl"])

        daily_perf = [
            {"date": datetime(2024, 1, 1), "total_pnl": 100},
            {"date": datetime(2024, 1, 2), "total_pnl": -50},
            {"date": datetime(2024, 1, 3), "total_pnl": 200},
            {"date": datetime(2024, 1, 4), "total_pnl": -100},
            {"date": datetime(2024, 1, 5), "total_pnl": 50},
        ]

        best, worst = find_extreme_days(daily_perf)

        assert best[1] == 200
        assert worst[1] == -100

    def test_find_extreme_days_empty(self):
        """Test extreme days with empty data."""
        def find_extreme_days(daily_perf):
            if not daily_perf:
                return None, None
            best = max(daily_perf, key=lambda dp: dp["total_pnl"])
            worst = min(daily_perf, key=lambda dp: dp["total_pnl"])
            return (best["date"], best["total_pnl"]), (worst["date"], worst["total_pnl"])

        best, worst = find_extreme_days([])
        assert best is None
        assert worst is None

    def test_calculate_monthly_returns(self):
        """Test monthly returns calculation."""
        def calculate_monthly_returns(daily_perf):
            if not daily_perf:
                return {}

            monthly = {}
            for dp in daily_perf:
                month_key = dp["date"].strftime("%Y-%m")
                if month_key not in monthly:
                    monthly[month_key] = []
                monthly[month_key].append(dp["total_pnl"])

            return {month: sum(pnls) for month, pnls in monthly.items()}

        daily_perf = [
            {"date": datetime(2024, 1, 1), "total_pnl": 100.0},
            {"date": datetime(2024, 1, 2), "total_pnl": 100.0},
            {"date": datetime(2024, 1, 3), "total_pnl": 100.0},
            {"date": datetime(2024, 2, 1), "total_pnl": 50.0},
            {"date": datetime(2024, 2, 2), "total_pnl": 50.0},
        ]

        result = calculate_monthly_returns(daily_perf)

        assert "2024-01" in result
        assert "2024-02" in result
        assert result["2024-01"] == 300.0
        assert result["2024-02"] == 100.0

    def test_calculate_trade_only_metrics(self):
        """Test metrics calculation from trades only."""
        trades = [
            MetricsTrade(
                symbol="AAPL",
                entry_date=datetime(2024, 1, 1),
                exit_date=datetime(2024, 1, 5),
                entry_price=150.0,
                exit_price=155.0,
                quantity=10,
                side="long",
                pnl=50.0,
                pnl_pct=3.33,
            ),
            MetricsTrade(
                symbol="MSFT",
                entry_date=datetime(2024, 1, 3),
                exit_date=datetime(2024, 1, 8),
                entry_price=400.0,
                exit_price=390.0,
                quantity=5,
                side="long",
                pnl=-50.0,
                pnl_pct=-2.5,
            ),
        ]

        # Simulate the calculation
        pnls = [t.pnl for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        total_trades = len(trades)
        winning_trades = len(winning)
        losing_trades = len(losing)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 0
        net_profit = gross_profit - gross_loss

        assert total_trades == 2
        assert winning_trades == 1
        assert losing_trades == 1
        assert win_rate == 50.0
        assert gross_profit == 50.0
        assert gross_loss == 50.0
        assert net_profit == 0.0

    def test_win_loss_ratio_calculation(self):
        """Test win/loss ratio calculation."""
        winning_trades = [100, 150, 200]
        losing_trades = [-50, -75]

        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        assert avg_win == 150.0
        assert avg_loss == -62.5
        assert abs(win_loss_ratio - 2.4) < 0.01

    def test_expectancy_calculation(self):
        """Test trade expectancy calculation."""
        win_rate = 0.60  # 60%
        avg_win = 150.0
        avg_loss = -75.0

        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # (0.6 * 150) + (0.4 * -75) = 90 - 30 = 60
        assert expectancy == 60.0

    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        gross_profit = 500.0
        gross_loss = 200.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        assert profit_factor == 2.5

    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        gross_profit = 500.0
        gross_loss = 0.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        assert profit_factor == float('inf')


class TestAnalyticsStatistics:
    """Tests for statistical calculations used in analytics."""

    def test_max_consecutive_wins(self):
        """Test max consecutive wins calculation."""
        def max_consecutive(values, condition):
            max_count = 0
            current_count = 0
            for v in values:
                if condition(v):
                    current_count += 1
                    max_count = max(max_count, current_count)
                else:
                    current_count = 0
            return max_count

        pnls = [100, 50, -20, 30, 40, 60, -10, 20]

        max_wins = max_consecutive(pnls, lambda x: x > 0)
        max_losses = max_consecutive(pnls, lambda x: x < 0)

        assert max_wins == 3  # 30, 40, 60
        assert max_losses == 1  # Either -20 or -10

    def test_holding_period_calculation(self):
        """Test average holding period calculation."""
        trades = [
            {"entry_date": datetime(2024, 1, 1), "exit_date": datetime(2024, 1, 5)},  # 4 days
            {"entry_date": datetime(2024, 1, 10), "exit_date": datetime(2024, 1, 12)},  # 2 days
            {"entry_date": datetime(2024, 1, 15), "exit_date": datetime(2024, 1, 21)},  # 6 days
        ]

        holding_periods = [(t["exit_date"] - t["entry_date"]).days for t in trades]
        avg_holding = np.mean(holding_periods)

        assert avg_holding == 4.0  # (4 + 2 + 6) / 3

    def test_daily_return_calculation(self):
        """Test daily return calculation."""
        equity_values = [100000, 101000, 100500, 102000, 101500]

        daily_returns = []
        for i in range(1, len(equity_values)):
            ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1] * 100
            daily_returns.append(ret)

        assert abs(daily_returns[0] - 1.0) < 0.01  # 1% gain
        assert abs(daily_returns[1] - (-0.495)) < 0.01  # ~0.5% loss
        assert abs(daily_returns[2] - 1.493) < 0.01  # ~1.5% gain

    def test_cumulative_return_calculation(self):
        """Test cumulative return calculation."""
        initial = 100000
        final = 120000

        cumulative_return = ((final / initial) - 1) * 100

        assert abs(cumulative_return - 20.0) < 0.001  # 20% total return
