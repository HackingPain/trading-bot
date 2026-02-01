"""Performance metrics calculation for backtesting and live trading."""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a trading period."""

    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    avg_drawdown: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L statistics
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0

    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    largest_winning_trade: float = 0.0
    largest_losing_trade: float = 0.0
    avg_trade: float = 0.0

    # Ratios
    avg_win_loss_ratio: float = 0.0
    expectancy: float = 0.0

    # Time analysis
    avg_holding_period_days: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Volatility
    daily_volatility: float = 0.0
    annual_volatility: float = 0.0

    # Period info
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_winning_trade": self.avg_winning_trade,
            "avg_losing_trade": self.avg_losing_trade,
            "expectancy": self.expectancy,
            "avg_holding_period_days": self.avg_holding_period_days,
            "annual_volatility": self.annual_volatility,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "trading_days": self.trading_days,
        }

    def summary(self) -> str:
        """Generate a text summary of metrics."""
        return f"""
Performance Summary
==================
Period: {self.start_date.date() if self.start_date else 'N/A'} to {self.end_date.date() if self.end_date else 'N/A'} ({self.trading_days} days)

Returns
-------
Total Return:       ${self.total_return:,.2f} ({self.total_return_pct:.2f}%)
Annualized Return:  {self.annualized_return:.2f}%

Risk Metrics
------------
Sharpe Ratio:       {self.sharpe_ratio:.2f}
Sortino Ratio:      {self.sortino_ratio:.2f}
Max Drawdown:       {self.max_drawdown_pct:.2f}%
Annual Volatility:  {self.annual_volatility:.2f}%

Trade Statistics
----------------
Total Trades:       {self.total_trades}
Win Rate:           {self.win_rate:.1f}%
Profit Factor:      {self.profit_factor:.2f}
Avg Win/Loss Ratio: {self.avg_win_loss_ratio:.2f}
Expectancy:         ${self.expectancy:.2f}

P&L Breakdown
-------------
Gross Profit:       ${self.gross_profit:,.2f}
Gross Loss:         ${self.gross_loss:,.2f}
Net Profit:         ${self.net_profit:,.2f}
Avg Winning Trade:  ${self.avg_winning_trade:.2f}
Avg Losing Trade:   ${self.avg_losing_trade:.2f}
"""


@dataclass
class Trade:
    """Represents a completed trade for metrics calculation."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    commission: float = 0.0


def calculate_metrics(
    equity_curve: pd.Series,
    trades: list[Trade],
    risk_free_rate: float = 0.05,
    trading_days_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: Series of portfolio values indexed by date
        trades: List of completed Trade objects
        risk_free_rate: Annual risk-free rate (default 5%)
        trading_days_per_year: Trading days per year (default 252)

    Returns:
        PerformanceMetrics object with all calculated metrics
    """
    metrics = PerformanceMetrics()

    if equity_curve.empty:
        return metrics

    # Period info
    metrics.start_date = equity_curve.index[0].to_pydatetime() if hasattr(equity_curve.index[0], 'to_pydatetime') else equity_curve.index[0]
    metrics.end_date = equity_curve.index[-1].to_pydatetime() if hasattr(equity_curve.index[-1], 'to_pydatetime') else equity_curve.index[-1]
    metrics.trading_days = len(equity_curve)

    # Calculate returns
    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]

    metrics.total_return = final_value - initial_value
    metrics.total_return_pct = (metrics.total_return / initial_value) * 100 if initial_value > 0 else 0

    # Annualized return
    years = metrics.trading_days / trading_days_per_year
    if years > 0 and initial_value > 0:
        metrics.annualized_return = ((final_value / initial_value) ** (1 / years) - 1) * 100

    # Daily returns for volatility and Sharpe
    daily_returns = equity_curve.pct_change().dropna()

    if len(daily_returns) > 1:
        # Volatility
        metrics.daily_volatility = daily_returns.std() * 100
        metrics.annual_volatility = metrics.daily_volatility * np.sqrt(trading_days_per_year)

        # Sharpe Ratio
        daily_rf = risk_free_rate / trading_days_per_year
        excess_returns = daily_returns - daily_rf
        if excess_returns.std() > 0:
            metrics.sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days_per_year)

        # Sortino Ratio (only downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics.sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(trading_days_per_year)

    # Drawdown analysis
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve - rolling_max
    drawdown_pct = (drawdown / rolling_max) * 100

    metrics.max_drawdown = abs(drawdown.min())
    metrics.max_drawdown_pct = abs(drawdown_pct.min())
    metrics.avg_drawdown = abs(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else 0

    # Max drawdown duration
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        drawdown_periods = (~in_drawdown).cumsum()
        drawdown_groups = in_drawdown.groupby(drawdown_periods)
        max_duration = max(group.sum() for _, group in drawdown_groups if group.sum() > 0)
        metrics.max_drawdown_duration_days = int(max_duration)

    # Calmar Ratio
    if metrics.max_drawdown_pct > 0:
        metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown_pct

    # Trade statistics
    if trades:
        metrics.total_trades = len(trades)

        pnls = [t.pnl for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100 if metrics.total_trades > 0 else 0

        # P&L statistics
        metrics.gross_profit = sum(winning) if winning else 0
        metrics.gross_loss = abs(sum(losing)) if losing else 0
        metrics.net_profit = metrics.gross_profit - metrics.gross_loss

        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float('inf')

        metrics.avg_winning_trade = np.mean(winning) if winning else 0
        metrics.avg_losing_trade = np.mean(losing) if losing else 0
        metrics.avg_trade = np.mean(pnls)

        metrics.largest_winning_trade = max(winning) if winning else 0
        metrics.largest_losing_trade = min(losing) if losing else 0

        # Win/Loss ratio
        if metrics.avg_losing_trade != 0:
            metrics.avg_win_loss_ratio = abs(metrics.avg_winning_trade / metrics.avg_losing_trade)

        # Expectancy
        win_rate_decimal = metrics.win_rate / 100
        if metrics.avg_losing_trade != 0:
            metrics.expectancy = (win_rate_decimal * metrics.avg_winning_trade) + ((1 - win_rate_decimal) * metrics.avg_losing_trade)

        # Holding period
        holding_periods = [(t.exit_date - t.entry_date).days for t in trades]
        metrics.avg_holding_period_days = np.mean(holding_periods) if holding_periods else 0

        # Consecutive wins/losses
        metrics.max_consecutive_wins = _max_consecutive(pnls, lambda x: x > 0)
        metrics.max_consecutive_losses = _max_consecutive(pnls, lambda x: x < 0)

    return metrics


def _max_consecutive(values: list, condition) -> int:
    """Calculate maximum consecutive occurrences matching condition."""
    max_count = 0
    current_count = 0
    for v in values:
        if condition(v):
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count


def calculate_daily_metrics(
    equity_curve: pd.Series,
) -> pd.DataFrame:
    """
    Calculate rolling metrics for each day.

    Returns DataFrame with daily metrics for charting.
    """
    df = pd.DataFrame(index=equity_curve.index)
    df["equity"] = equity_curve
    df["daily_return"] = equity_curve.pct_change() * 100
    df["cumulative_return"] = ((equity_curve / equity_curve.iloc[0]) - 1) * 100

    # Rolling max and drawdown
    df["rolling_max"] = equity_curve.cummax()
    df["drawdown"] = ((equity_curve - df["rolling_max"]) / df["rolling_max"]) * 100

    # Rolling Sharpe (30-day)
    daily_returns = equity_curve.pct_change()
    df["rolling_sharpe_30d"] = (
        daily_returns.rolling(30).mean() / daily_returns.rolling(30).std()
    ) * np.sqrt(252)

    # Rolling volatility (30-day annualized)
    df["rolling_volatility_30d"] = daily_returns.rolling(30).std() * np.sqrt(252) * 100

    return df
