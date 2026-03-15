"""
Daily Profit Taker Strategy

A swing trading strategy that:
- Enters positions based on RSI oversold + MACD bullish signals
- Takes profits at a target percentage
- Uses trailing stops to protect gains
- Exits on RSI overbought or MACD bearish signals
"""

import logging
from datetime import datetime
from typing import Any, Optional

from ..data.market_data import MarketData
from .base import ExitReason, ExitSignal, PositionInfo, Signal, SignalType, Strategy
from .indicators import IndicatorValues, TechnicalIndicators

logger = logging.getLogger(__name__)


class DailyProfitTakerStrategy(Strategy):
    """
    Daily profit-taking strategy combining RSI, MACD, and Bollinger Bands.

    Entry conditions (all must be true for strong signal):
    - RSI below oversold threshold (30)
    - MACD bullish crossover or positive histogram
    - Price near or below lower Bollinger Band

    Exit conditions (any triggers exit):
    - Stop loss hit
    - Trailing stop hit
    - Take profit target reached
    - RSI overbought (70) + MACD bearish
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(name="daily_profit_taker", config=config)

        # Strategy parameters with defaults
        self.profit_target_pct = self.get_config_value("profit_target_pct", 0.02)
        self.stop_loss_pct = self.get_config_value("stop_loss_pct", 0.05)
        self.trailing_stop_pct = self.get_config_value("trailing_stop_pct", 0.03)
        self.use_trailing_stop = self.get_config_value("use_trailing_stop", True)

        # Indicator thresholds
        self.rsi_oversold = self.get_config_value("rsi_oversold", 30)
        self.rsi_overbought = self.get_config_value("rsi_overbought", 70)
        self.macd_threshold = self.get_config_value("macd_signal_threshold", 0)

        # Initialize indicators calculator
        self.indicators = TechnicalIndicators.from_settings({"indicators": config} if config else {})

    @classmethod
    def from_settings(cls, settings: dict) -> "DailyProfitTakerStrategy":
        """Create strategy from settings dictionary."""
        strategy_config = settings.get("strategy", {})
        risk_config = settings.get("risk", {})

        # Merge strategy and risk settings
        config = {
            **strategy_config,
            "stop_loss_pct": risk_config.get("stop_loss_pct", 0.05),
            "trailing_stop_pct": risk_config.get("trailing_stop_pct", 0.03),
        }

        return cls(config=config)

    def _evaluate_entry_conditions(
        self,
        indicators: IndicatorValues,
        df_with_indicators,
    ) -> tuple[float, list[str]]:
        """
        Evaluate entry conditions and return signal strength and reasons.

        Returns:
            Tuple of (strength 0-1, list of reasons)
        """
        strength = 0.0
        reasons = []

        # Check RSI oversold
        if indicators.rsi is not None and indicators.rsi < self.rsi_oversold:
            strength += 0.35
            reasons.append(f"RSI oversold ({indicators.rsi:.1f})")

        # Check MACD bullish
        if self.indicators.macd_bullish_crossover(df_with_indicators):
            strength += 0.35
            reasons.append("MACD bullish crossover")
        elif indicators.macd_histogram is not None and indicators.macd_histogram > self.macd_threshold:
            strength += 0.2
            reasons.append(f"MACD histogram positive ({indicators.macd_histogram:.3f})")

        # Check Bollinger Bands
        if indicators.bb_percent is not None and indicators.bb_percent < 0.2:
            strength += 0.3
            reasons.append(f"Near lower Bollinger Band ({indicators.bb_percent:.2f})")

        return min(strength, 1.0), reasons

    def _evaluate_exit_conditions(
        self,
        position: PositionInfo,
        indicators: IndicatorValues,
        df_with_indicators,
    ) -> Optional[tuple[ExitReason, str]]:
        """
        Evaluate exit conditions for an existing position.

        Returns:
            Tuple of (ExitReason, description) if should exit, None otherwise
        """
        current_price = position.current_price
        entry_price = position.avg_entry_price
        pnl_pct = position.unrealized_pnl_pct / 100  # Convert to decimal

        # Check stop loss
        if position.stop_loss_price and current_price <= position.stop_loss_price:
            return (
                ExitReason.STOP_LOSS,
                f"Stop loss hit at ${current_price:.2f} (entry: ${entry_price:.2f})",
            )

        # Check trailing stop
        if self.use_trailing_stop and position.trailing_stop_price:
            if current_price <= position.trailing_stop_price:
                return (
                    ExitReason.TRAILING_STOP,
                    f"Trailing stop hit at ${current_price:.2f} (highest: ${position.highest_price:.2f})",
                )

        # Check take profit
        if pnl_pct >= self.profit_target_pct:
            return (
                ExitReason.TAKE_PROFIT,
                f"Take profit target reached: {pnl_pct*100:.2f}% gain",
            )

        # Check RSI overbought + MACD bearish
        if indicators.rsi is not None and indicators.rsi > self.rsi_overbought:
            if self.indicators.macd_bearish_crossover(df_with_indicators):
                return (
                    ExitReason.SIGNAL,
                    f"RSI overbought ({indicators.rsi:.1f}) + MACD bearish crossover",
                )

        return None

    def generate_signals(
        self,
        market_data: dict[str, MarketData],
        current_positions: dict[str, PositionInfo],
    ) -> list[Signal]:
        """
        Generate buy/sell signals based on current market conditions.

        Only generates BUY signals for symbols not already held.
        Exit signals are handled by should_exit().
        """
        signals = []

        for symbol, data in market_data.items():
            # Skip if we already have a position
            if symbol in current_positions:
                logger.debug(f"Skipping {symbol} - already have position")
                continue

            if data.df.empty:
                logger.warning(f"No data available for {symbol}")
                continue

            try:
                # Calculate indicators
                df_with_indicators = self.indicators.add_all_indicators(data.df)
                indicator_values = self.indicators.get_current_values(data.df)

                # Evaluate entry conditions
                strength, reasons = self._evaluate_entry_conditions(
                    indicator_values,
                    df_with_indicators,
                )

                # Generate signal if strength is sufficient
                if strength >= 0.5:  # Minimum threshold for a signal
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
                            data.last_price, self.profit_target_pct
                        ),
                        indicators={
                            "rsi": indicator_values.rsi,
                            "macd": indicator_values.macd,
                            "macd_signal": indicator_values.macd_signal,
                            "bb_percent": indicator_values.bb_percent,
                        },
                    )
                    signals.append(signal)
                    logger.info(f"Generated BUY signal for {symbol}: {signal}")

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

        return signals

    def should_exit(
        self,
        position: PositionInfo,
        market_data: MarketData,
    ) -> Optional[ExitSignal]:
        """
        Check if an existing position should be closed.

        Evaluates stop loss, trailing stop, take profit, and indicator-based exits.
        """
        if market_data.df.empty:
            logger.warning(f"No market data for {position.symbol}")
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

                # Determine urgency
                urgency = "normal"
                if reason in (ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP):
                    urgency = "high"
                elif reason == ExitReason.CIRCUIT_BREAKER:
                    urgency = "immediate"

                exit_signal = ExitSignal(
                    symbol=position.symbol,
                    reason=reason,
                    exit_price=position.current_price,
                    description=description,
                    urgency=urgency,
                )
                logger.info(f"Generated EXIT signal for {position.symbol}: {exit_signal}")
                return exit_signal

        except Exception as e:
            logger.error(f"Error evaluating exit for {position.symbol}: {e}")

        return None

    def update_trailing_stop(
        self,
        position: PositionInfo,
        current_price: float,
    ) -> Optional[float]:
        """
        Calculate updated trailing stop price if price has moved higher.

        Returns new trailing stop price if it should be updated, None otherwise.
        """
        if not self.use_trailing_stop:
            return None

        # Only update if price is higher than recorded highest
        if current_price > position.highest_price:
            new_trailing_stop = self.calculate_trailing_stop(
                current_price, self.trailing_stop_pct
            )

            # Only update if new stop is higher than current
            if position.trailing_stop_price is None or new_trailing_stop > position.trailing_stop_price:
                return new_trailing_stop

        return None

    def get_strategy_info(self) -> dict[str, Any]:
        """Return strategy configuration and status."""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "parameters": {
                "profit_target_pct": self.profit_target_pct,
                "stop_loss_pct": self.stop_loss_pct,
                "trailing_stop_pct": self.trailing_stop_pct,
                "use_trailing_stop": self.use_trailing_stop,
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
            },
        }
