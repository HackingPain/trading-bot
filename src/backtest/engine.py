"""Backtesting engine for simulating trading strategies on historical data."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from ..data.market_data import MarketData, MarketDataProvider
from ..strategies.base import Strategy, Signal, SignalType, PositionInfo, ExitSignal
from .metrics import PerformanceMetrics, Trade, calculate_metrics, calculate_daily_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    commission_per_trade: float = 0.0  # Flat fee per trade
    commission_pct: float = 0.0  # Percentage of trade value
    slippage_pct: float = 0.001  # 0.1% slippage estimate
    max_position_pct: float = 0.10  # Max 10% per position
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.02  # 2% take profit
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    risk_free_rate: float = 0.05  # For Sharpe calculation


@dataclass
class BacktestPosition:
    """Represents an open position during backtesting."""
    symbol: str
    quantity: float
    entry_price: float
    entry_date: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: float
    highest_price: float
    cost_basis: float

    def to_position_info(self, current_price: float) -> PositionInfo:
        """Convert to PositionInfo for strategy evaluation."""
        pnl = (current_price - self.entry_price) * self.quantity
        pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100

        return PositionInfo(
            symbol=self.symbol,
            quantity=self.quantity,
            avg_entry_price=self.entry_price,
            current_price=current_price,
            unrealized_pnl=pnl,
            unrealized_pnl_pct=pnl_pct,
            highest_price=self.highest_price,
            stop_loss_price=self.stop_loss,
            trailing_stop_price=self.trailing_stop,
            take_profit_price=self.take_profit,
            opened_at=self.entry_date,
        )


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    config: BacktestConfig
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    daily_metrics: pd.DataFrame
    trades: list[Trade]
    signals: list[Signal]
    start_date: datetime
    end_date: datetime

    def summary(self) -> str:
        """Generate summary report."""
        return f"""
Backtest Results
================
Period: {self.start_date.date()} to {self.end_date.date()}
Initial Capital: ${self.config.initial_capital:,.2f}
Final Value: ${self.equity_curve.iloc[-1]:,.2f}

{self.metrics.summary()}
"""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config": {
                "initial_capital": self.config.initial_capital,
                "commission_pct": self.config.commission_pct,
                "slippage_pct": self.config.slippage_pct,
                "max_position_pct": self.config.max_position_pct,
            },
            "metrics": self.metrics.to_dict(),
            "equity_curve": self.equity_curve.to_dict(),
            "trades": [
                {
                    "symbol": t.symbol,
                    "entry_date": t.entry_date.isoformat(),
                    "exit_date": t.exit_date.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                }
                for t in self.trades
            ],
            "total_signals": len(self.signals),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }


class BacktestEngine:
    """
    Backtesting engine for simulating trading strategies.

    Simulates trading on historical data with realistic constraints
    including commissions, slippage, and position sizing.
    """

    def __init__(
        self,
        strategy: Strategy,
        data_provider: MarketDataProvider,
        config: Optional[BacktestConfig] = None,
    ):
        self.strategy = strategy
        self.data_provider = data_provider
        self.config = config or BacktestConfig()

        # State during backtest
        self._cash: float = 0
        self._positions: dict[str, BacktestPosition] = {}
        self._trades: list[Trade] = []
        self._signals: list[Signal] = []
        self._equity_history: list[tuple[datetime, float]] = []

    def run(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            interval: Data interval (1d, 1h, etc.)

        Returns:
            BacktestResult with performance metrics and trade history
        """
        logger.info(
            f"Starting backtest: {start_date.date()} to {end_date.date()}, "
            f"symbols={symbols}, strategy={self.strategy.name}"
        )

        # Reset state
        self._cash = self.config.initial_capital
        self._positions = {}
        self._trades = []
        self._signals = []
        self._equity_history = []

        # Fetch historical data for all symbols
        historical_data = self._fetch_historical_data(symbols, start_date, end_date, interval)

        if not historical_data:
            logger.error("No historical data available")
            return self._create_empty_result(start_date, end_date)

        # Get common date range across all symbols
        dates = self._get_common_dates(historical_data)
        logger.info(f"Backtesting over {len(dates)} trading days")

        # Simulate each trading day
        for date in dates:
            self._simulate_day(date, historical_data, symbols)

        # Calculate final metrics
        equity_curve = pd.Series(
            {date: equity for date, equity in self._equity_history},
            name="equity"
        )

        metrics = calculate_metrics(
            equity_curve,
            self._trades,
            risk_free_rate=self.config.risk_free_rate,
        )

        daily_metrics = calculate_daily_metrics(equity_curve)

        result = BacktestResult(
            config=self.config,
            metrics=metrics,
            equity_curve=equity_curve,
            daily_metrics=daily_metrics,
            trades=self._trades,
            signals=self._signals,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(f"Backtest complete: {len(self._trades)} trades, "
                   f"return={metrics.total_return_pct:.2f}%")

        return result

    def _fetch_historical_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols."""
        data = {}

        # Calculate period needed
        days = (end_date - start_date).days + 100  # Extra days for indicators

        for symbol in symbols:
            try:
                df = self.data_provider.get_historical_data(
                    symbol,
                    period=f"{days}d",
                    interval=interval,
                )

                if not df.empty:
                    # Filter to date range
                    df = df[df.index >= start_date]
                    df = df[df.index <= end_date]
                    data[symbol] = df
                    logger.debug(f"Loaded {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")

        return data

    def _get_common_dates(self, historical_data: dict[str, pd.DataFrame]) -> list:
        """Get trading dates common to all symbols."""
        if not historical_data:
            return []

        # Start with dates from first symbol
        common_dates = set(historical_data[list(historical_data.keys())[0]].index)

        # Intersect with all other symbols
        for df in historical_data.values():
            common_dates &= set(df.index)

        return sorted(common_dates)

    def _simulate_day(
        self,
        date: datetime,
        historical_data: dict[str, pd.DataFrame],
        symbols: list[str],
    ) -> None:
        """Simulate trading for a single day."""
        # Build market data dict for this day
        market_data = {}
        for symbol in symbols:
            if symbol in historical_data:
                df = historical_data[symbol]
                if date in df.index:
                    # Create MarketData with data up to this date
                    df_to_date = df[df.index <= date]
                    current_bar = df.loc[date]

                    market_data[symbol] = MarketData(
                        symbol=symbol,
                        df=df_to_date,
                        last_price=float(current_bar["close"]),
                        last_updated=date if isinstance(date, datetime) else datetime.combine(date.date(), datetime.min.time()),
                        volume=int(current_bar["volume"]),
                    )

        if not market_data:
            return

        # Update position prices and check stops
        self._update_positions(date, market_data)

        # Check exit signals for existing positions
        self._check_exit_signals(date, market_data)

        # Get current positions as PositionInfo
        positions_info = {
            symbol: pos.to_position_info(market_data[symbol].last_price)
            for symbol, pos in self._positions.items()
            if symbol in market_data
        }

        # Generate entry signals
        signals = self.strategy.generate_signals(market_data, positions_info)
        self._signals.extend(signals)

        # Process buy signals
        for signal in signals:
            if signal.signal_type == SignalType.BUY:
                self._execute_entry(signal, market_data, date)

        # Record equity for this day
        equity = self._calculate_equity(market_data)
        self._equity_history.append((date, equity))

    def _update_positions(self, date: datetime, market_data: dict[str, MarketData]) -> None:
        """Update position prices and trailing stops."""
        for symbol, position in list(self._positions.items()):
            if symbol not in market_data:
                continue

            current_price = market_data[symbol].last_price

            # Update highest price
            if current_price > position.highest_price:
                position.highest_price = current_price

                # Update trailing stop
                if self.config.use_trailing_stop:
                    new_trailing = current_price * (1 - self.config.trailing_stop_pct)
                    position.trailing_stop = max(position.trailing_stop, new_trailing)

    def _check_exit_signals(self, date: datetime, market_data: dict[str, MarketData]) -> None:
        """Check and execute exit signals (stops, take profit, strategy signals)."""
        for symbol in list(self._positions.keys()):
            if symbol not in market_data:
                continue

            position = self._positions[symbol]
            current_price = market_data[symbol].last_price
            position_info = position.to_position_info(current_price)

            exit_reason = None
            exit_price = current_price

            # Check stop loss
            if current_price <= position.stop_loss:
                exit_reason = "stop_loss"
                exit_price = position.stop_loss  # Assume filled at stop

            # Check trailing stop
            elif current_price <= position.trailing_stop:
                exit_reason = "trailing_stop"
                exit_price = position.trailing_stop

            # Check take profit
            elif current_price >= position.take_profit:
                exit_reason = "take_profit"
                exit_price = position.take_profit

            # Check strategy exit signal
            else:
                exit_signal = self.strategy.should_exit(position_info, market_data[symbol])
                if exit_signal:
                    exit_reason = exit_signal.reason.value
                    exit_price = current_price

            if exit_reason:
                self._execute_exit(position, exit_price, exit_reason, date)

    def _execute_entry(
        self,
        signal: Signal,
        market_data: dict[str, MarketData],
        date: datetime,
    ) -> None:
        """Execute a buy signal."""
        symbol = signal.symbol

        # Skip if already have position
        if symbol in self._positions:
            return

        # Skip if no market data
        if symbol not in market_data:
            return

        price = market_data[symbol].last_price

        # Apply slippage
        fill_price = price * (1 + self.config.slippage_pct)

        # Calculate position size
        equity = self._calculate_equity(market_data)
        max_value = equity * self.config.max_position_pct * signal.strength
        quantity = int(max_value / fill_price)

        if quantity <= 0:
            return

        # Calculate trade cost
        trade_value = quantity * fill_price
        commission = self.config.commission_per_trade + (trade_value * self.config.commission_pct)
        total_cost = trade_value + commission

        # Check if we have enough cash
        if total_cost > self._cash:
            # Reduce position size
            available = self._cash - self.config.commission_per_trade
            quantity = int(available / (fill_price * (1 + self.config.commission_pct)))
            if quantity <= 0:
                return
            trade_value = quantity * fill_price
            commission = self.config.commission_per_trade + (trade_value * self.config.commission_pct)
            total_cost = trade_value + commission

        # Execute buy
        self._cash -= total_cost

        # Calculate stops
        stop_loss = fill_price * (1 - self.config.stop_loss_pct)
        take_profit = fill_price * (1 + self.config.take_profit_pct)
        trailing_stop = fill_price * (1 - self.config.trailing_stop_pct)

        position = BacktestPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=fill_price,
            entry_date=date,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            highest_price=fill_price,
            cost_basis=total_cost,
        )

        self._positions[symbol] = position
        logger.debug(f"Opened position: {quantity} {symbol} @ ${fill_price:.2f}")

    def _execute_exit(
        self,
        position: BacktestPosition,
        exit_price: float,
        reason: str,
        date: datetime,
    ) -> None:
        """Execute position exit."""
        symbol = position.symbol

        # Apply slippage
        fill_price = exit_price * (1 - self.config.slippage_pct)

        # Calculate proceeds
        trade_value = position.quantity * fill_price
        commission = self.config.commission_per_trade + (trade_value * self.config.commission_pct)
        proceeds = trade_value - commission

        # Calculate P&L
        pnl = proceeds - position.cost_basis
        pnl_pct = (pnl / position.cost_basis) * 100

        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=fill_price,
            quantity=position.quantity,
            side="long",
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
        )
        self._trades.append(trade)

        # Update cash
        self._cash += proceeds

        # Remove position
        del self._positions[symbol]

        logger.debug(f"Closed position: {symbol} @ ${fill_price:.2f}, PnL=${pnl:.2f} ({reason})")

    def _calculate_equity(self, market_data: dict[str, MarketData]) -> float:
        """Calculate total equity (cash + positions value)."""
        positions_value = sum(
            pos.quantity * market_data[symbol].last_price
            for symbol, pos in self._positions.items()
            if symbol in market_data
        )
        return self._cash + positions_value

    def _create_empty_result(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Create empty result when backtest cannot run."""
        return BacktestResult(
            config=self.config,
            metrics=PerformanceMetrics(),
            equity_curve=pd.Series(dtype=float),
            daily_metrics=pd.DataFrame(),
            trades=[],
            signals=[],
            start_date=start_date,
            end_date=end_date,
        )


def run_backtest(
    strategy: Strategy,
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000.0,
    **config_kwargs,
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        strategy: Trading strategy to test
        symbols: List of symbols to trade
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        **config_kwargs: Additional BacktestConfig parameters

    Returns:
        BacktestResult
    """
    from ..data.market_data import MarketDataConfig, MarketDataProvider

    config = BacktestConfig(initial_capital=initial_capital, **config_kwargs)
    data_provider = MarketDataProvider(MarketDataConfig())
    engine = BacktestEngine(strategy, data_provider, config)

    return engine.run(symbols, start_date, end_date)
