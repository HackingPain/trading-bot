"""
Momentum Strategy

A trend-following strategy that:
- Enters positions when strong momentum is detected
- Uses EMA crossovers for trend direction
- Confirms with MACD and RSI momentum
- Uses wider stops to let trends run
"""

import logging
from typing import Any, Optional

import pandas as pd

from ..data.market_data import MarketData
from .base import ExitReason, ExitSignal, PositionInfo, Signal, SignalType, Strategy
from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class MomentumStrategy(Strategy):
    """
    Momentum/trend-following strategy.

    Entry conditions:
    - Fast EMA crosses above slow EMA (bullish trend)
    - MACD histogram is positive and increasing
    - RSI is in bullish range (40-70) showing momentum without being overbought
    - Price above VWAP (for intraday momentum)

    Exit conditions:
    - Fast EMA crosses below slow EMA (trend reversal)
    - MACD histogram turns negative
    - Trailing stop hit
    - RSI becomes overbought (>75) - partial or full exit
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(name="momentum", config=config)

        # Strategy parameters
        self.stop_loss_pct = self.get_config_value("stop_loss_pct", 0.06)
        self.trailing_stop_pct = self.get_config_value("trailing_stop_pct", 0.04)
        self.take_profit_pct = self.get_config_value("take_profit_pct", 0.10)
        self.use_trailing_stop = self.get_config_value("use_trailing_stop", True)

        # EMA parameters
        self.fast_ema_period = self.get_config_value("fast_ema_period", 9)
        self.slow_ema_period = self.get_config_value("slow_ema_period", 21)

        # RSI parameters for momentum
        self.rsi_momentum_low = self.get_config_value("rsi_momentum_low", 40)
        self.rsi_momentum_high = self.get_config_value("rsi_momentum_high", 70)
        self.rsi_overbought = self.get_config_value("rsi_overbought", 75)

        # MACD confirmation
        self.macd_histogram_threshold = self.get_config_value("macd_histogram_threshold", 0)

        # Minimum trend strength
        self.min_trend_strength = self.get_config_value("min_trend_strength", 0.5)

        # Initialize indicators
        indicator_config = config.copy() if config else {}
        indicator_config.update({
            "ema_fast_period": self.fast_ema_period,
            "ema_slow_period": self.slow_ema_period,
        })
        self.indicators = TechnicalIndicators.from_settings({"indicators": indicator_config})

    @classmethod
    def from_settings(cls, settings: dict) -> "MomentumStrategy":
        """Create strategy from settings dictionary."""
        strategy_config = settings.get("strategy", {})
        risk_config = settings.get("risk", {})

        config = {
            **strategy_config,
            "stop_loss_pct": risk_config.get("stop_loss_pct", 0.06),
            "trailing_stop_pct": risk_config.get("trailing_stop_pct", 0.04),
        }

        return cls(config=config)

    def _add_ema_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA indicators to dataframe."""
        df = df.copy()

        # Calculate fast and slow EMAs
        df["ema_fast"] = df["close"].ewm(span=self.fast_ema_period, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow_ema_period, adjust=False).mean()

        # Calculate EMA difference (trend strength indicator)
        df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"] * 100

        return df

    def _detect_ema_crossover(self, df: pd.DataFrame) -> tuple[bool, bool]:
        """
        Detect EMA crossovers.

        Returns:
            Tuple of (bullish_crossover, bearish_crossover)
        """
        if len(df) < 3:
            return False, False

        current_fast = df["ema_fast"].iloc[-1]
        current_slow = df["ema_slow"].iloc[-1]
        prev_fast = df["ema_fast"].iloc[-2]
        prev_slow = df["ema_slow"].iloc[-2]

        bullish = prev_fast <= prev_slow and current_fast > current_slow
        bearish = prev_fast >= prev_slow and current_fast < current_slow

        return bullish, bearish

    def _calculate_trend_strength(self, df: pd.DataFrame, indicator_values) -> float:
        """
        Calculate overall trend strength from 0 to 1.

        Considers:
        - EMA difference magnitude
        - MACD histogram direction
        - RSI momentum
        """
        strength = 0.0

        if len(df) < 2:
            return 0.0

        # EMA trend component (0-0.4)
        ema_diff = df["ema_diff"].iloc[-1] if "ema_diff" in df else 0
        if ema_diff > 0:
            # Normalize: 1% diff = 0.2 strength, max at 2%
            strength += min(0.4, abs(ema_diff) * 0.2)

        # MACD histogram component (0-0.35)
        macd_hist = indicator_values.macd_histogram
        if macd_hist is not None and macd_hist > 0:
            # Check if histogram is increasing
            if len(df) >= 2 and "macd_histogram" in df:
                prev_hist = df["macd_histogram"].iloc[-2]
                if macd_hist > prev_hist:
                    strength += 0.35
                else:
                    strength += 0.15

        # RSI momentum component (0-0.25)
        rsi = indicator_values.rsi
        if rsi is not None:
            if self.rsi_momentum_low <= rsi <= self.rsi_momentum_high:
                # Optimal momentum zone
                strength += 0.25
            elif 30 <= rsi < self.rsi_momentum_low:
                # Building momentum
                strength += 0.10

        return min(1.0, strength)

    def _evaluate_entry_conditions(
        self,
        symbol: str,
        current_price: float,
        indicator_values,
        df_with_indicators: pd.DataFrame,
    ) -> tuple[float, list[str]]:
        """
        Evaluate momentum entry conditions.

        Returns:
            Tuple of (strength 0-1, list of reasons)
        """
        reasons = []

        # Add EMA indicators
        df = self._add_ema_indicators(df_with_indicators)

        # Check for bullish EMA crossover or existing bullish trend
        bullish_cross, _ = self._detect_ema_crossover(df)
        ema_diff = df["ema_diff"].iloc[-1] if len(df) > 0 else 0

        if bullish_cross:
            reasons.append(f"EMA bullish crossover ({self.fast_ema_period}/{self.slow_ema_period})")
        elif ema_diff > 0.5:  # Strong existing trend
            reasons.append(f"Strong uptrend (EMA diff: {ema_diff:.2f}%)")
        else:
            return 0.0, []

        # Check MACD confirmation
        macd_hist = indicator_values.macd_histogram
        if macd_hist is not None and macd_hist > self.macd_histogram_threshold:
            reasons.append(f"MACD histogram positive ({macd_hist:.4f})")
        else:
            return 0.0, []

        # Check RSI is in momentum zone
        rsi = indicator_values.rsi
        if rsi is not None:
            if self.rsi_momentum_low <= rsi <= self.rsi_momentum_high:
                reasons.append(f"RSI in momentum zone ({rsi:.1f})")
            elif rsi < self.rsi_momentum_low:
                reasons.append(f"RSI building ({rsi:.1f})")
            elif rsi > self.rsi_overbought:
                # Too overbought, skip entry
                logger.debug(f"{symbol}: RSI overbought ({rsi:.1f}), skipping")
                return 0.0, []

        # Check VWAP if available
        vwap = indicator_values.vwap
        if vwap is not None and current_price > vwap:
            reasons.append(f"Price above VWAP (${vwap:.2f})")

        # Calculate overall strength
        strength = self._calculate_trend_strength(df, indicator_values)

        return strength, reasons

    def _evaluate_exit_conditions(
        self,
        position: PositionInfo,
        indicator_values,
        df_with_indicators: pd.DataFrame,
    ) -> Optional[tuple[ExitReason, str]]:
        """
        Evaluate momentum exit conditions.
        """
        current_price = position.current_price
        pnl_pct = position.unrealized_pnl_pct / 100

        # Check stop loss
        if position.stop_loss_price and current_price <= position.stop_loss_price:
            return (
                ExitReason.STOP_LOSS,
                f"Stop loss hit at ${current_price:.2f}",
            )

        # Check trailing stop
        if self.use_trailing_stop and position.trailing_stop_price:
            if current_price <= position.trailing_stop_price:
                return (
                    ExitReason.TRAILING_STOP,
                    f"Trailing stop hit at ${current_price:.2f}",
                )

        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            return (
                ExitReason.TAKE_PROFIT,
                f"Take profit reached: {pnl_pct*100:.2f}% gain",
            )

        # Add EMA indicators for trend analysis
        df = self._add_ema_indicators(df_with_indicators)

        # Check for bearish EMA crossover (trend reversal)
        _, bearish_cross = self._detect_ema_crossover(df)
        if bearish_cross:
            return (
                ExitReason.SIGNAL,
                f"EMA bearish crossover - trend reversing",
            )

        # Check MACD turning bearish
        macd_hist = indicator_values.macd_histogram
        if macd_hist is not None and macd_hist < -0.01:  # MACD negative
            # Only exit if we have profits to protect
            if pnl_pct > 0.02:
                return (
                    ExitReason.SIGNAL,
                    f"MACD histogram negative ({macd_hist:.4f}), protecting gains",
                )

        # Check extreme overbought
        rsi = indicator_values.rsi
        if rsi is not None and rsi > 80:
            if pnl_pct > 0.03:  # Only if profitable
                return (
                    ExitReason.SIGNAL,
                    f"RSI extremely overbought ({rsi:.1f}), taking profits",
                )

        return None

    def generate_signals(
        self,
        market_data: dict[str, MarketData],
        current_positions: dict[str, PositionInfo],
    ) -> list[Signal]:
        """Generate momentum entry signals."""
        signals = []

        for symbol, data in market_data.items():
            # Skip if already have a position
            if symbol in current_positions:
                continue

            if data.df.empty or len(data.df) < self.slow_ema_period + 5:
                logger.debug(f"Insufficient data for {symbol}")
                continue

            try:
                # Calculate indicators
                df_with_indicators = self.indicators.add_all_indicators(data.df)
                indicator_values = self.indicators.get_current_values(data.df)

                # Evaluate entry conditions
                strength, reasons = self._evaluate_entry_conditions(
                    symbol,
                    data.last_price,
                    indicator_values,
                    df_with_indicators,
                )

                # Generate signal if strength is sufficient
                if strength >= self.min_trend_strength:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=strength,
                        price=data.last_price,
                        reason="; ".join(reasons),
                        strategy_name=self.name,
                        suggested_stop_loss=self.calculate_stop_loss(
                            data.last_price, self.stop_loss_pct
                        ),
                        suggested_take_profit=self.calculate_take_profit(
                            data.last_price, self.take_profit_pct
                        ),
                        indicators={
                            "rsi": indicator_values.rsi,
                            "macd": indicator_values.macd,
                            "macd_histogram": indicator_values.macd_histogram,
                            "vwap": indicator_values.vwap,
                        },
                    )
                    signals.append(signal)
                    logger.info(f"Generated momentum BUY signal for {symbol}: {signal}")

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

        return signals

    def should_exit(
        self,
        position: PositionInfo,
        market_data: MarketData,
    ) -> Optional[ExitSignal]:
        """Check if a momentum position should be closed."""
        if market_data.df.empty:
            return None

        try:
            # Calculate indicators
            df_with_indicators = self.indicators.add_all_indicators(market_data.df)
            indicator_values = self.indicators.get_current_values(market_data.df)

            # Evaluate exit conditions
            exit_result = self._evaluate_exit_conditions(
                position,
                indicator_values,
                df_with_indicators,
            )

            if exit_result:
                reason, description = exit_result

                urgency = "normal"
                if reason in (ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP):
                    urgency = "high"

                return ExitSignal(
                    symbol=position.symbol,
                    reason=reason,
                    exit_price=position.current_price,
                    description=description,
                    urgency=urgency,
                )

        except Exception as e:
            logger.error(f"Error evaluating exit for {position.symbol}: {e}")

        return None

    def update_trailing_stop(
        self,
        position: PositionInfo,
        current_price: float,
    ) -> Optional[float]:
        """Calculate updated trailing stop if price has moved higher."""
        if not self.use_trailing_stop:
            return None

        if current_price > position.highest_price:
            new_trailing_stop = self.calculate_trailing_stop(
                current_price, self.trailing_stop_pct
            )

            if position.trailing_stop_price is None or new_trailing_stop > position.trailing_stop_price:
                return new_trailing_stop

        return None

    def get_strategy_info(self) -> dict[str, Any]:
        """Return strategy configuration."""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "parameters": {
                "stop_loss_pct": self.stop_loss_pct,
                "trailing_stop_pct": self.trailing_stop_pct,
                "take_profit_pct": self.take_profit_pct,
                "use_trailing_stop": self.use_trailing_stop,
                "fast_ema_period": self.fast_ema_period,
                "slow_ema_period": self.slow_ema_period,
                "rsi_momentum_low": self.rsi_momentum_low,
                "rsi_momentum_high": self.rsi_momentum_high,
                "min_trend_strength": self.min_trend_strength,
            },
        }
