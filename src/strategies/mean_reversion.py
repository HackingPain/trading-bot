"""
Mean Reversion Strategy

A strategy that trades on the assumption that prices tend to revert to their mean:
- Enters when price deviates significantly from moving average
- Exits when price returns to the mean or exceeds opposite deviation
- Uses Bollinger Bands and RSI for confirmation
"""

import logging
from typing import Any, Optional

from ..data.market_data import MarketData
from .base import ExitReason, ExitSignal, PositionInfo, Signal, SignalType, Strategy
from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy that profits from price returning to the mean.

    Entry conditions:
    - Price is below lower Bollinger Band (oversold) OR
    - Price is above upper Bollinger Band (overbought, for short if supported)
    - RSI confirms oversold/overbought condition
    - ATR indicates reasonable volatility (not too extreme)

    Exit conditions:
    - Price returns to middle Bollinger Band (SMA)
    - RSI returns to neutral zone (40-60)
    - Stop loss hit
    - Take profit reached
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(name="mean_reversion", config=config)

        # Strategy parameters
        self.stop_loss_pct = self.get_config_value("stop_loss_pct", 0.04)
        self.take_profit_pct = self.get_config_value("take_profit_pct", 0.025)

        # Mean reversion specific parameters
        self.bb_entry_threshold = self.get_config_value("bb_entry_threshold", 0.1)  # % beyond band
        self.bb_exit_threshold = self.get_config_value("bb_exit_threshold", 0.5)  # BB% for exit
        self.rsi_oversold = self.get_config_value("rsi_oversold", 25)
        self.rsi_overbought = self.get_config_value("rsi_overbought", 75)
        self.rsi_neutral_low = self.get_config_value("rsi_neutral_low", 40)
        self.rsi_neutral_high = self.get_config_value("rsi_neutral_high", 60)

        # Volatility filter
        self.min_atr_pct = self.get_config_value("min_atr_pct", 0.005)  # Min volatility
        self.max_atr_pct = self.get_config_value("max_atr_pct", 0.05)  # Max volatility

        # Initialize indicators
        self.indicators = TechnicalIndicators.from_settings({"indicators": config} if config else {})

    @classmethod
    def from_settings(cls, settings: dict) -> "MeanReversionStrategy":
        """Create strategy from settings dictionary."""
        strategy_config = settings.get("strategy", {})
        risk_config = settings.get("risk", {})

        config = {
            **strategy_config,
            "stop_loss_pct": risk_config.get("stop_loss_pct", 0.04),
        }

        return cls(config=config)

    def _calculate_atr_pct(self, current_price: float, atr: float) -> float:
        """Calculate ATR as percentage of current price."""
        if current_price <= 0:
            return 0
        return atr / current_price

    def _evaluate_entry_conditions(
        self,
        symbol: str,
        current_price: float,
        indicator_values,
        df_with_indicators,
    ) -> tuple[float, SignalType, list[str]]:
        """
        Evaluate mean reversion entry conditions.

        Returns:
            Tuple of (strength 0-1, signal type, list of reasons)
        """
        strength = 0.0
        signal_type = SignalType.HOLD
        reasons = []

        # Get indicator values
        bb_percent = indicator_values.bb_percent
        rsi = indicator_values.rsi
        atr = indicator_values.atr

        if bb_percent is None or rsi is None:
            return 0.0, SignalType.HOLD, []

        # Calculate ATR percentage for volatility filter
        atr_pct = self._calculate_atr_pct(current_price, atr) if atr else 0

        # Check volatility is in acceptable range
        if atr_pct < self.min_atr_pct:
            logger.debug(f"{symbol}: Volatility too low ({atr_pct:.4f})")
            return 0.0, SignalType.HOLD, []

        if atr_pct > self.max_atr_pct:
            logger.debug(f"{symbol}: Volatility too high ({atr_pct:.4f})")
            return 0.0, SignalType.HOLD, []

        # Check for oversold condition (BUY signal)
        if bb_percent < self.bb_entry_threshold:
            strength += 0.4
            reasons.append(f"Below lower BB ({bb_percent:.2f})")
            signal_type = SignalType.BUY

            if rsi < self.rsi_oversold:
                strength += 0.35
                reasons.append(f"RSI oversold ({rsi:.1f})")

            # Additional confirmation from price action
            if len(df_with_indicators) >= 3:
                recent_closes = df_with_indicators["close"].tail(3).values
                if recent_closes[-1] > recent_closes[-2]:  # Price starting to recover
                    strength += 0.25
                    reasons.append("Price showing recovery")

        # Note: Short selling would require additional broker support
        # For now, we only take long positions on oversold conditions

        return min(strength, 1.0), signal_type, reasons

    def _evaluate_exit_conditions(
        self,
        position: PositionInfo,
        indicator_values,
    ) -> Optional[tuple[ExitReason, str]]:
        """
        Evaluate mean reversion exit conditions.

        Returns:
            Tuple of (ExitReason, description) if should exit, None otherwise
        """
        current_price = position.current_price
        entry_price = position.avg_entry_price
        pnl_pct = position.unrealized_pnl_pct / 100

        # Check stop loss
        if position.stop_loss_price and current_price <= position.stop_loss_price:
            return (
                ExitReason.STOP_LOSS,
                f"Stop loss hit at ${current_price:.2f}",
            )

        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            return (
                ExitReason.TAKE_PROFIT,
                f"Take profit reached: {pnl_pct*100:.2f}% gain",
            )

        # Check mean reversion exit (price returned to middle band)
        bb_percent = indicator_values.bb_percent
        rsi = indicator_values.rsi

        if bb_percent is not None and bb_percent >= self.bb_exit_threshold:
            if rsi is None or (self.rsi_neutral_low <= rsi <= self.rsi_neutral_high):
                return (
                    ExitReason.SIGNAL,
                    f"Price reverted to mean (BB%: {bb_percent:.2f}, RSI: {rsi:.1f if rsi else 'N/A'})",
                )

        # Check for overbought - price went too far the other way
        if bb_percent is not None and bb_percent > 0.9:
            if rsi is not None and rsi > self.rsi_overbought:
                return (
                    ExitReason.SIGNAL,
                    f"Overbought - take profits (BB%: {bb_percent:.2f}, RSI: {rsi:.1f})",
                )

        return None

    def generate_signals(
        self,
        market_data: dict[str, MarketData],
        current_positions: dict[str, PositionInfo],
    ) -> list[Signal]:
        """Generate mean reversion entry signals."""
        signals = []

        for symbol, data in market_data.items():
            # Skip if already have a position
            if symbol in current_positions:
                continue

            if data.df.empty:
                logger.warning(f"No data available for {symbol}")
                continue

            try:
                # Calculate indicators
                df_with_indicators = self.indicators.add_all_indicators(data.df)
                indicator_values = self.indicators.get_current_values(data.df)

                # Evaluate entry conditions
                strength, signal_type, reasons = self._evaluate_entry_conditions(
                    symbol,
                    data.last_price,
                    indicator_values,
                    df_with_indicators,
                )

                # Generate signal if strength is sufficient
                if strength >= 0.5 and signal_type == SignalType.BUY:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
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
                            "bb_percent": indicator_values.bb_percent,
                            "bb_lower": indicator_values.bb_lower,
                            "bb_upper": indicator_values.bb_upper,
                            "atr": indicator_values.atr,
                        },
                    )
                    signals.append(signal)
                    logger.info(f"Generated {signal_type.value.upper()} signal for {symbol}: {signal}")

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

        return signals

    def should_exit(
        self,
        position: PositionInfo,
        market_data: MarketData,
    ) -> Optional[ExitSignal]:
        """Check if a mean reversion position should be closed."""
        if market_data.df.empty:
            return None

        try:
            # Calculate indicators
            df_with_indicators = self.indicators.add_all_indicators(market_data.df)
            indicator_values = self.indicators.get_current_values(market_data.df)

            # Evaluate exit conditions
            exit_result = self._evaluate_exit_conditions(position, indicator_values)

            if exit_result:
                reason, description = exit_result

                urgency = "high" if reason == ExitReason.STOP_LOSS else "normal"

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

    def get_strategy_info(self) -> dict[str, Any]:
        """Return strategy configuration."""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "parameters": {
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_pct": self.take_profit_pct,
                "bb_entry_threshold": self.bb_entry_threshold,
                "bb_exit_threshold": self.bb_exit_threshold,
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
                "min_atr_pct": self.min_atr_pct,
                "max_atr_pct": self.max_atr_pct,
            },
        }
