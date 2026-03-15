"""
Pairs Trading (Statistical Arbitrage) Strategy

A market-neutral strategy that:
- Identifies correlated stock pairs
- Trades when pairs diverge from historical relationship
- Profits from mean reversion of the spread
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..data.market_data import MarketData
from .base import ExitReason, ExitSignal, PositionInfo, Signal, SignalType, Strategy

logger = logging.getLogger(__name__)


@dataclass
class PairInfo:
    """Information about a trading pair."""
    symbol1: str
    symbol2: str
    correlation: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    current_zscore: float
    half_life: float  # Mean reversion half-life


class PairsTradingStrategy(Strategy):
    """
    Statistical arbitrage strategy trading correlated pairs.

    Entry conditions:
    - Pair has high historical correlation (>0.7)
    - Current spread z-score exceeds entry threshold
    - Spread shows mean-reverting behavior

    Exit conditions:
    - Spread z-score returns to near zero
    - Stop loss on spread widening
    - Maximum holding period exceeded

    Note: This strategy requires going long one stock and short another.
    Short selling requires margin account and may not be available in paper trading.
    For simplicity, this implementation trades only the long side of pairs.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(name="pairs_trading", config=config)

        # Pair selection parameters
        self.min_correlation = self.get_config_value("min_correlation", 0.70)
        self.lookback_period = self.get_config_value("lookback_period", 60)
        self.min_half_life = self.get_config_value("min_half_life", 5)
        self.max_half_life = self.get_config_value("max_half_life", 30)

        # Entry/exit parameters
        self.entry_zscore = self.get_config_value("entry_zscore", 2.0)
        self.exit_zscore = self.get_config_value("exit_zscore", 0.5)
        self.stop_loss_zscore = self.get_config_value("stop_loss_zscore", 3.5)

        # Risk parameters
        self.max_holding_days = self.get_config_value("max_holding_days", 20)
        self.stop_loss_pct = self.get_config_value("stop_loss_pct", 0.05)

        # Predefined pairs (sector-based pairs that typically correlate)
        self.pair_definitions = self.get_config_value("pairs", [
            ("AAPL", "MSFT"),      # Tech giants
            ("JPM", "BAC"),        # Banks
            ("XOM", "CVX"),        # Oil majors
            ("KO", "PEP"),         # Beverages
            ("HD", "LOW"),         # Home improvement
            ("V", "MA"),           # Payment networks
            ("UNH", "CVS"),        # Healthcare
            ("DIS", "NFLX"),       # Entertainment
        ])

        # Cache for pair analysis
        self._pair_cache: dict[tuple[str, str], PairInfo] = {}

    @classmethod
    def from_settings(cls, settings: dict) -> "PairsTradingStrategy":
        """Create strategy from settings dictionary."""
        strategy_config = settings.get("strategy", {})
        risk_config = settings.get("risk", {})

        config = {
            **strategy_config,
            "stop_loss_pct": risk_config.get("stop_loss_pct", 0.05),
        }

        return cls(config=config)

    def _calculate_correlation(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
    ) -> float:
        """Calculate rolling correlation between two price series."""
        returns1 = prices1.pct_change().dropna()
        returns2 = prices2.pct_change().dropna()

        # Align the series
        aligned = pd.concat([returns1, returns2], axis=1).dropna()
        if len(aligned) < 20:
            return 0.0

        return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    def _calculate_hedge_ratio(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
    ) -> float:
        """
        Calculate hedge ratio using OLS regression.

        Hedge ratio tells us how many shares of stock2 to trade per share of stock1.
        """
        # Use log prices for cointegration
        log_prices1 = np.log(prices1)
        log_prices2 = np.log(prices2)

        # Simple OLS: log(price1) = alpha + beta * log(price2)
        aligned = pd.concat([log_prices1, log_prices2], axis=1).dropna()
        if len(aligned) < 20:
            return 1.0

        x = aligned.iloc[:, 1].values
        y = aligned.iloc[:, 0].values

        # OLS
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

        return float(beta)

    def _calculate_spread(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        hedge_ratio: float,
    ) -> pd.Series:
        """Calculate the spread between two price series."""
        log_prices1 = np.log(prices1)
        log_prices2 = np.log(prices2)

        spread = log_prices1 - hedge_ratio * log_prices2
        return spread

    def _calculate_zscore(
        self,
        spread: pd.Series,
        lookback: int = None,
    ) -> float:
        """Calculate z-score of current spread."""
        if lookback is None:
            lookback = self.lookback_period

        if len(spread) < lookback:
            lookback = len(spread)

        recent = spread.tail(lookback)
        mean = recent.mean()
        std = recent.std()

        if std == 0:
            return 0.0

        current = spread.iloc[-1]
        return float((current - mean) / std)

    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate mean reversion half-life using Ornstein-Uhlenbeck process.

        Half-life = -log(2) / log(theta)
        where theta is the mean reversion speed from AR(1) regression.
        """
        if len(spread) < 30:
            return float("inf")

        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align series
        aligned = pd.concat([spread_diff, spread_lag], axis=1).dropna()
        if len(aligned) < 20:
            return float("inf")

        y = aligned.iloc[:, 0].values
        x = aligned.iloc[:, 1].values

        # OLS regression: spread_diff = theta * spread_lag + error
        x_mean = np.mean(x)
        theta = np.sum((x - x_mean) * y) / np.sum((x - x_mean) ** 2)

        if theta >= 0:
            return float("inf")  # Not mean reverting

        half_life = -np.log(2) / theta
        return float(half_life)

    def _analyze_pair(
        self,
        symbol1: str,
        symbol2: str,
        prices1: pd.Series,
        prices2: pd.Series,
    ) -> Optional[PairInfo]:
        """Analyze a pair for trading suitability."""
        # Calculate correlation
        correlation = self._calculate_correlation(prices1, prices2)
        if correlation < self.min_correlation:
            logger.debug(f"Pair {symbol1}/{symbol2} correlation too low: {correlation:.3f}")
            return None

        # Calculate hedge ratio
        hedge_ratio = self._calculate_hedge_ratio(prices1, prices2)

        # Calculate spread
        spread = self._calculate_spread(prices1, prices2, hedge_ratio)

        # Calculate half-life
        half_life = self._calculate_half_life(spread)
        if half_life < self.min_half_life or half_life > self.max_half_life:
            logger.debug(f"Pair {symbol1}/{symbol2} half-life unsuitable: {half_life:.1f}")
            return None

        # Calculate z-score
        zscore = self._calculate_zscore(spread)

        return PairInfo(
            symbol1=symbol1,
            symbol2=symbol2,
            correlation=correlation,
            hedge_ratio=hedge_ratio,
            spread_mean=float(spread.mean()),
            spread_std=float(spread.std()),
            current_zscore=zscore,
            half_life=half_life,
        )

    def _get_tradeable_pairs(
        self,
        market_data: dict[str, MarketData],
    ) -> list[PairInfo]:
        """Get all pairs that are suitable for trading."""
        tradeable_pairs = []

        for symbol1, symbol2 in self.pair_definitions:
            if symbol1 not in market_data or symbol2 not in market_data:
                continue

            data1 = market_data[symbol1]
            data2 = market_data[symbol2]

            if data1.df.empty or data2.df.empty:
                continue

            # Get aligned price series
            prices1 = data1.df["close"]
            prices2 = data2.df["close"]

            pair_info = self._analyze_pair(symbol1, symbol2, prices1, prices2)
            if pair_info:
                tradeable_pairs.append(pair_info)
                self._pair_cache[(symbol1, symbol2)] = pair_info

        return tradeable_pairs

    def generate_signals(
        self,
        market_data: dict[str, MarketData],
        current_positions: dict[str, PositionInfo],
    ) -> list[Signal]:
        """Generate pairs trading signals."""
        signals = []

        # Analyze all pairs
        tradeable_pairs = self._get_tradeable_pairs(market_data)

        for pair_info in tradeable_pairs:
            # Skip if already in either position
            if pair_info.symbol1 in current_positions or pair_info.symbol2 in current_positions:
                continue

            zscore = pair_info.current_zscore

            # Check for entry signal
            # If zscore < -entry_threshold: spread is too low, buy symbol1 (undervalued)
            # If zscore > entry_threshold: spread is too high, buy symbol2 (undervalued)
            if abs(zscore) < self.entry_zscore:
                continue

            if zscore < -self.entry_zscore:
                # Spread too low - symbol1 is undervalued relative to symbol2
                target_symbol = pair_info.symbol1
                other_symbol = pair_info.symbol2
                reason = f"Pairs trade: {pair_info.symbol1}/{pair_info.symbol2} spread z-score={zscore:.2f} (buy undervalued)"
            else:
                # Spread too high - symbol2 is undervalued relative to symbol1
                target_symbol = pair_info.symbol2
                other_symbol = pair_info.symbol1
                reason = f"Pairs trade: {pair_info.symbol1}/{pair_info.symbol2} spread z-score={zscore:.2f} (buy undervalued)"

            if target_symbol not in market_data:
                continue

            data = market_data[target_symbol]

            # Calculate signal strength based on z-score
            strength = min(1.0, abs(zscore) / 3.0)

            signal = Signal(
                symbol=target_symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                price=data.last_price,
                reason=reason,
                strategy_name=self.name,
                suggested_stop_loss=data.last_price * (1 - self.stop_loss_pct),
                indicators={
                    "pair_symbol": other_symbol,
                    "correlation": pair_info.correlation,
                    "zscore": zscore,
                    "hedge_ratio": pair_info.hedge_ratio,
                    "half_life": pair_info.half_life,
                },
            )
            signals.append(signal)
            logger.info(f"Generated pairs trade signal: {signal}")

        return signals

    def should_exit(
        self,
        position: PositionInfo,
        market_data: MarketData,
    ) -> Optional[ExitSignal]:
        """Check if a pairs position should be closed."""
        # Check stop loss
        if position.stop_loss_price and position.current_price <= position.stop_loss_price:
            return ExitSignal(
                symbol=position.symbol,
                reason=ExitReason.STOP_LOSS,
                exit_price=position.current_price,
                description=f"Stop loss hit at ${position.current_price:.2f}",
                urgency="high",
            )

        # Find the pair this position belongs to
        pair_info = None
        for (s1, s2), info in self._pair_cache.items():
            if position.symbol in (s1, s2):
                pair_info = info
                break

        if pair_info is None:
            return None

        # Check if z-score has reverted
        if abs(pair_info.current_zscore) <= self.exit_zscore:
            return ExitSignal(
                symbol=position.symbol,
                reason=ExitReason.SIGNAL,
                exit_price=position.current_price,
                description=f"Spread reverted (z-score: {pair_info.current_zscore:.2f})",
                urgency="normal",
            )

        # Check if spread is widening too much (stop loss on spread)
        if abs(pair_info.current_zscore) > self.stop_loss_zscore:
            return ExitSignal(
                symbol=position.symbol,
                reason=ExitReason.STOP_LOSS,
                exit_price=position.current_price,
                description=f"Spread widening too much (z-score: {pair_info.current_zscore:.2f})",
                urgency="high",
            )

        # Check holding period
        if position.opened_at:
            from datetime import datetime, timedelta
            holding_days = (datetime.now() - position.opened_at).days
            if holding_days > self.max_holding_days:
                return ExitSignal(
                    symbol=position.symbol,
                    reason=ExitReason.SIGNAL,
                    exit_price=position.current_price,
                    description=f"Maximum holding period exceeded ({holding_days} days)",
                    urgency="normal",
                )

        return None

    def get_pair_analysis(self) -> list[dict]:
        """Get analysis of all cached pairs."""
        return [
            {
                "pair": f"{info.symbol1}/{info.symbol2}",
                "correlation": round(info.correlation, 3),
                "hedge_ratio": round(info.hedge_ratio, 3),
                "zscore": round(info.current_zscore, 2),
                "half_life": round(info.half_life, 1),
            }
            for info in self._pair_cache.values()
        ]

    def get_strategy_info(self) -> dict[str, Any]:
        """Return strategy configuration."""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "parameters": {
                "min_correlation": self.min_correlation,
                "lookback_period": self.lookback_period,
                "entry_zscore": self.entry_zscore,
                "exit_zscore": self.exit_zscore,
                "stop_loss_zscore": self.stop_loss_zscore,
                "max_holding_days": self.max_holding_days,
                "pairs": self.pair_definitions,
            },
            "pair_analysis": self.get_pair_analysis(),
        }
