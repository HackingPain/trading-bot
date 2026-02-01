"""Performance analytics for live trading."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..database.models import Trade, DailyPerformance, Position, get_session
from ..backtest.metrics import PerformanceMetrics, Trade as MetricsTrade, calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    # Time period
    period_start: datetime
    period_end: datetime
    trading_days: int

    # Account summary
    starting_balance: float
    ending_balance: float
    total_deposits: float = 0.0
    total_withdrawals: float = 0.0

    # Performance metrics
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # Additional analysis
    best_day: Optional[tuple[datetime, float]] = None
    worst_day: Optional[tuple[datetime, float]] = None
    best_trade: Optional[dict] = None
    worst_trade: Optional[dict] = None

    # Symbol breakdown
    symbol_performance: dict[str, dict] = field(default_factory=dict)

    # Time analysis
    monthly_returns: dict[str, float] = field(default_factory=dict)
    daily_returns: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Generate text summary."""
        return f"""
Performance Report
==================
Period: {self.period_start.date()} to {self.period_end.date()} ({self.trading_days} days)
Starting Balance: ${self.starting_balance:,.2f}
Ending Balance: ${self.ending_balance:,.2f}
Net Change: ${self.ending_balance - self.starting_balance:,.2f} ({((self.ending_balance/self.starting_balance)-1)*100:.2f}%)

{self.metrics.summary()}

Best Day: {self.best_day[0].date() if self.best_day else 'N/A'} (${self.best_day[1] if self.best_day else 0:,.2f})
Worst Day: {self.worst_day[0].date() if self.worst_day else 'N/A'} (${self.worst_day[1] if self.worst_day else 0:,.2f})

Symbol Performance
------------------
"""
        + "\n".join(
            f"  {sym}: {perf.get('total_pnl', 0):+.2f} ({perf.get('trades', 0)} trades, "
            f"{perf.get('win_rate', 0):.1f}% win rate)"
            for sym, perf in self.symbol_performance.items()
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
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


class PerformanceAnalyzer:
    """
    Analyzes trading performance from database records.

    Provides comprehensive analytics for live trading including:
    - Overall performance metrics
    - Per-symbol breakdown
    - Time-based analysis
    - Risk metrics
    """

    def __init__(self):
        self._session = None

    def _get_session(self):
        """Get database session."""
        if self._session is None:
            self._session = get_session()
        return self._session

    def get_trades(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> list[Trade]:
        """Fetch trades from database with optional filters."""
        session = self._get_session()
        query = session.query(Trade)

        if start_date:
            query = query.filter(Trade.executed_at >= start_date)
        if end_date:
            query = query.filter(Trade.executed_at <= end_date)
        if symbol:
            query = query.filter(Trade.symbol == symbol)

        return query.order_by(Trade.executed_at).all()

    def get_daily_performance(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[DailyPerformance]:
        """Fetch daily performance records."""
        session = self._get_session()
        query = session.query(DailyPerformance)

        if start_date:
            query = query.filter(DailyPerformance.date >= start_date)
        if end_date:
            query = query.filter(DailyPerformance.date <= end_date)

        return query.order_by(DailyPerformance.date).all()

    def analyze_period(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report for a period.

        Args:
            start_date: Start of analysis period (default: all time)
            end_date: End of analysis period (default: now)

        Returns:
            PerformanceReport with all metrics
        """
        end_date = end_date or datetime.now()

        # Get trades for period
        trades = self.get_trades(start_date, end_date)
        daily_perf = self.get_daily_performance(start_date, end_date)

        if not trades and not daily_perf:
            logger.warning("No data available for analysis period")
            return PerformanceReport(
                period_start=start_date or datetime.now(),
                period_end=end_date,
                trading_days=0,
                starting_balance=0,
                ending_balance=0,
            )

        # Determine actual period from data
        if daily_perf:
            actual_start = daily_perf[0].date
            actual_end = daily_perf[-1].date
            starting_balance = daily_perf[0].starting_balance
            ending_balance = daily_perf[-1].ending_balance
        else:
            actual_start = trades[0].executed_at if trades else end_date
            actual_end = trades[-1].executed_at if trades else end_date
            starting_balance = 0
            ending_balance = 0

        # Build equity curve from daily performance
        equity_curve = pd.Series(
            {dp.date: dp.ending_balance for dp in daily_perf},
            name="equity"
        )

        # Convert trades to metrics format
        metrics_trades = self._convert_trades_to_metrics_format(trades)

        # Calculate core metrics
        if not equity_curve.empty:
            metrics = calculate_metrics(equity_curve, metrics_trades)
        else:
            metrics = self._calculate_trade_only_metrics(metrics_trades)

        # Calculate symbol breakdown
        symbol_performance = self._analyze_by_symbol(trades)

        # Find best/worst days
        best_day, worst_day = self._find_extreme_days(daily_perf)

        # Find best/worst trades
        best_trade, worst_trade = self._find_extreme_trades(trades)

        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns(daily_perf)

        # Daily returns
        daily_returns = [dp.total_pnl_pct for dp in daily_perf]

        return PerformanceReport(
            period_start=actual_start,
            period_end=actual_end,
            trading_days=len(daily_perf),
            starting_balance=starting_balance,
            ending_balance=ending_balance,
            metrics=metrics,
            best_day=best_day,
            worst_day=worst_day,
            best_trade=best_trade,
            worst_trade=worst_trade,
            symbol_performance=symbol_performance,
            monthly_returns=monthly_returns,
            daily_returns=daily_returns,
        )

    def _convert_trades_to_metrics_format(self, trades: list[Trade]) -> list[MetricsTrade]:
        """Convert database trades to metrics Trade format."""
        # Group by symbol to pair buy/sell trades
        result = []

        # Track open positions
        open_positions: dict[str, list] = {}

        for trade in trades:
            symbol = trade.symbol
            side = trade.side.value if hasattr(trade.side, 'value') else trade.side

            if side == "buy":
                # Open or add to position
                if symbol not in open_positions:
                    open_positions[symbol] = []
                open_positions[symbol].append(trade)

            elif side == "sell" and symbol in open_positions and open_positions[symbol]:
                # Close position
                entry_trade = open_positions[symbol].pop(0)

                metrics_trade = MetricsTrade(
                    symbol=symbol,
                    entry_date=entry_trade.executed_at,
                    exit_date=trade.executed_at,
                    entry_price=entry_trade.price,
                    exit_price=trade.price,
                    quantity=trade.quantity,
                    side="long",
                    pnl=trade.realized_pnl or 0,
                    pnl_pct=trade.realized_pnl_pct or 0,
                    commission=trade.commission,
                )
                result.append(metrics_trade)

        return result

    def _calculate_trade_only_metrics(self, trades: list[MetricsTrade]) -> PerformanceMetrics:
        """Calculate metrics from trades only (no equity curve)."""
        metrics = PerformanceMetrics()

        if not trades:
            return metrics

        pnls = [t.pnl for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        metrics.total_trades = len(trades)
        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100 if metrics.total_trades > 0 else 0

        metrics.gross_profit = sum(winning) if winning else 0
        metrics.gross_loss = abs(sum(losing)) if losing else 0
        metrics.net_profit = metrics.gross_profit - metrics.gross_loss
        metrics.total_return = metrics.net_profit

        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float('inf')

        metrics.avg_winning_trade = np.mean(winning) if winning else 0
        metrics.avg_losing_trade = np.mean(losing) if losing else 0
        metrics.avg_trade = np.mean(pnls)

        metrics.largest_winning_trade = max(winning) if winning else 0
        metrics.largest_losing_trade = min(losing) if losing else 0

        if metrics.avg_losing_trade != 0:
            metrics.avg_win_loss_ratio = abs(metrics.avg_winning_trade / metrics.avg_losing_trade)

        win_rate_decimal = metrics.win_rate / 100
        if metrics.avg_losing_trade != 0:
            metrics.expectancy = (win_rate_decimal * metrics.avg_winning_trade) + ((1 - win_rate_decimal) * metrics.avg_losing_trade)

        return metrics

    def _analyze_by_symbol(self, trades: list[Trade]) -> dict[str, dict]:
        """Break down performance by symbol."""
        symbol_data: dict[str, list] = {}

        for trade in trades:
            symbol = trade.symbol
            if symbol not in symbol_data:
                symbol_data[symbol] = []
            symbol_data[symbol].append(trade)

        result = {}
        for symbol, symbol_trades in symbol_data.items():
            pnls = [t.realized_pnl or 0 for t in symbol_trades if t.side.value == "sell"]
            wins = sum(1 for p in pnls if p > 0)

            result[symbol] = {
                "trades": len([t for t in symbol_trades if t.side.value == "sell"]),
                "total_pnl": sum(pnls),
                "win_rate": (wins / len(pnls) * 100) if pnls else 0,
                "avg_pnl": np.mean(pnls) if pnls else 0,
                "best_trade": max(pnls) if pnls else 0,
                "worst_trade": min(pnls) if pnls else 0,
            }

        return result

    def _find_extreme_days(
        self,
        daily_perf: list[DailyPerformance],
    ) -> tuple[Optional[tuple], Optional[tuple]]:
        """Find best and worst trading days."""
        if not daily_perf:
            return None, None

        best = max(daily_perf, key=lambda dp: dp.total_pnl)
        worst = min(daily_perf, key=lambda dp: dp.total_pnl)

        return (best.date, best.total_pnl), (worst.date, worst.total_pnl)

    def _find_extreme_trades(
        self,
        trades: list[Trade],
    ) -> tuple[Optional[dict], Optional[dict]]:
        """Find best and worst individual trades."""
        sell_trades = [t for t in trades if t.side.value == "sell" and t.realized_pnl is not None]

        if not sell_trades:
            return None, None

        best = max(sell_trades, key=lambda t: t.realized_pnl or 0)
        worst = min(sell_trades, key=lambda t: t.realized_pnl or 0)

        return (
            {"symbol": best.symbol, "pnl": best.realized_pnl, "date": best.executed_at},
            {"symbol": worst.symbol, "pnl": worst.realized_pnl, "date": worst.executed_at},
        )

    def _calculate_monthly_returns(
        self,
        daily_perf: list[DailyPerformance],
    ) -> dict[str, float]:
        """Calculate returns by month."""
        if not daily_perf:
            return {}

        monthly: dict[str, list] = {}
        for dp in daily_perf:
            month_key = dp.date.strftime("%Y-%m")
            if month_key not in monthly:
                monthly[month_key] = []
            monthly[month_key].append(dp.total_pnl)

        return {month: sum(pnls) for month, pnls in monthly.items()}

    def get_current_positions_summary(self) -> dict[str, Any]:
        """Get summary of current open positions."""
        session = self._get_session()
        positions = session.query(Position).all()

        total_value = sum(p.market_value for p in positions)
        total_pnl = sum(p.unrealized_pnl for p in positions)

        return {
            "count": len(positions),
            "total_value": total_value,
            "total_unrealized_pnl": total_pnl,
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "entry_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                }
                for p in positions
            ],
        }

    def get_quick_stats(self) -> dict[str, Any]:
        """Get quick statistics for dashboard."""
        session = self._get_session()

        # Today's stats
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = session.query(Trade).filter(Trade.executed_at >= today).all()
        today_pnl = sum(t.realized_pnl or 0 for t in today_trades if t.side.value == "sell")

        # All time stats
        all_trades = session.query(Trade).all()
        total_trades = len([t for t in all_trades if t.side.value == "sell"])
        total_pnl = sum(t.realized_pnl or 0 for t in all_trades if t.side.value == "sell")

        winning = sum(1 for t in all_trades if t.side.value == "sell" and (t.realized_pnl or 0) > 0)
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

        return {
            "today_pnl": today_pnl,
            "today_trades": len([t for t in today_trades if t.side.value == "sell"]),
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "win_rate": win_rate,
        }

    def close(self):
        """Close database session."""
        if self._session:
            self._session.close()
            self._session = None
