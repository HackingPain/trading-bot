"""
Breakout Trading Strategy

A strategy that trades breakouts from consolidation patterns:
- Identifies support and resistance levels
- Enters when price breaks above resistance with volume confirmation
- Uses ATR-based stops and targets
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..data.market_data import MarketData
from .base import ExitReason, ExitSignal, PositionInfo, Signal, SignalType, Strategy
from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class BreakoutStrategy(Strategy):
    """
    Breakout strategy that trades price breakouts from consolidation zones.

    Entry conditions:
    - Price breaks above recent resistance (N-period high)
    - Volume is above average (confirms breakout)
    - RSI not extremely overbought (avoid false breakouts)
    - ATR shows adequate volatility

    Exit conditions:
    - Price falls back below breakout level (failed breakout)
    - Stop loss hit (ATR-based)
    - Take profit reached
    - Trailing stop hit
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(name="breakout", config=config)

        # Breakout detection parameters
        self.lookback_period = self.get_config_value("lookback_period", 20)
        self.volume_threshold = self.get_config_value("volume_threshold", 1.5)  # 1.5x avg volume
        self.breakout_margin_pct = self.get_config_value("breakout_margin_pct", 0.002)  # 0.2% above high

        # Risk parameters
        self.atr_stop_multiplier = self.get_config_value("atr_stop_multiplier", 2.0)
        self.atr_target_multiplier = self.get_config_value("atr_target_multiplier", 3.0)
        self.trailing_stop_pct = self.get_config_value("trailing_stop_pct", 0.04)
        self.use_trailing_stop = self.get_config_value("use_trailing_stop", True)

        # Filter parameters
        self.max_rsi = self.get_config_value("max_rsi", 75)  # Avoid extremely overbought
        self.min_atr_pct = self.get_config_value("min_atr_pct", 0.01)  # Minimum volatility

        # Initialize indicators
        self.indicators = TechnicalIndicators.from_settings({"indicators": config} if config else {})

    @classmethod
    def from_settings(cls, settings: dict) -> "BreakoutStrategy":
        """Create strategy from settings dictionary."""
        strategy_config = settings.get("strategy", {})
        risk_config = settings.get("risk", {})

        config = {
            **strategy_config,
            "trailing_stop_pct": risk_config.get("trailing_stop_pct", 0.04),
        }

        return cls(config=config)

    def _calculate_resistance(self, df: pd.DataFrame) -> float:
        """Calculate resistance level (recent high)."""
        return df["high"].tail(self.lookback_period).max()

    def _calculate_support(self, df: pd.DataFrame) -> float:
        """Calculate support level (recent low)."""
        return df["low"].tail(self.lookback_period).min()

    def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """Calculate current volume relative to average."""
        if "volume" not in df.columns or len(df) < self.lookback_period:
            return 1.0

        avg_volume = df["volume"].tail(self.lookback_period).mean()
        current_volume = df["volume"].iloc[-1]

        if avg_volume <= 0:
            return 1.0

        return current_volume / avg_volume

    def _is_breakout(
        self,
        current_price: float,
        resistance: float,
        prev_close: float,
    ) -> bool:
        """Check if current price is breaking above resistance."""
        breakout_level = resistance * (1 + self.breakout_margin_pct)

        # Price must close above breakout level
        # And previous close must be below resistance (confirming breakout)
        return current_price > breakout_level and prev_close <= resistance

    def _is_breakdown(
        self,
        current_price: float,
        support: float,
        prev_close: float,
    ) -> bool:
        """Check if current price is breaking below support."""
        breakdown_level = support * (1 - self.breakout_margin_pct)
        return current_price < breakdown_level and prev_close >= support

    def _evaluate_entry_conditions(
        self,
        symbol: str,
        current_price: float,
        df: pd.DataFrame,
        indicator_values,
    ) -> tuple[float, SignalType, list[str]]:
        """
        Evaluate breakout entry conditions.

        Returns:
            Tuple of (strength 0-1, signal type, list of reasons)
        """
        if len(df) < self.lookback_period + 5:
            return 0.0, SignalType.HOLD, []

        strength = 0.0
        signal_type = SignalType.HOLD
        reasons = []

        # Get levels
        resistance = self._calculate_resistance(df.iloc[:-1])  # Exclude current bar
        prev_close = df["close"].iloc[-2]

        # Check for breakout
        if not self._is_breakout(current_price, resistance, prev_close):
            return 0.0, SignalType.HOLD, []

        strength += 0.3
        reasons.append(f"Breakout above ${resistance:.2f}")
        signal_type = SignalType.BUY

        # Volume confirmation
        volume_ratio = self._calculate_volume_ratio(df)
        if volume_ratio >= self.volume_threshold:
            strength += 0.35
            reasons.append(f"Volume {volume_ratio:.1f}x average")
        elif volume_ratio >= 1.0:
            strength += 0.15
            reasons.append(f"Volume {volume_ratio:.1f}x (moderate)")
        else:
            # Low volume breakout - skeptical
            strength -= 0.1
            reasons.append(f"Low volume warning ({volume_ratio:.1f}x)")

        # RSI filter - avoid extremely overbought
        rsi = indicator_values.rsi
        if rsi is not None:
            if rsi > self.max_rsi:
                logger.debug(f"{symbol}: RSI too high ({rsi:.1f}), skipping breakout")
                return 0.0, SignalType.HOLD, []
            elif rsi > 60:
                strength += 0.15
                reasons.append(f"RSI confirms momentum ({rsi:.1f})")
            else:
                strength += 0.1

        # ATR volatility check
        atr = indicator_values.atr
        if atr is not None and current_price > 0:
            atr_pct = atr / current_price
            if atr_pct >= self.min_atr_pct:
                strength += 0.1
                reasons.append(f"Good volatility (ATR {atr_pct*100:.2f}%)")
            else:
                strength -= 0.1

        return min(max(strength, 0), 1.0), signal_type, reasons

    def _calculate_atr_stop(
        self,
        entry_price: float,
        atr: float,
    ) -> float:
        """Calculate stop loss based on ATR."""
        return entry_price - (atr * self.atr_stop_multiplier)

    def _calculate_atr_target(
        self,
        entry_price: float,
        atr: float,
    ) -> float:
        """Calculate take profit based on ATR."""
        return entry_price + (atr * self.atr_target_multiplier)

    def _evaluate_exit_conditions(
        self,
        position: PositionInfo,
        df: pd.DataFrame,
        indicator_values,
    ) -> Optional[tuple[ExitReason, str]]:
        """Evaluate breakout exit conditions."""
        current_price = position.current_price
        entry_price = position.avg_entry_price
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
        if position.take_profit_price and current_price >= position.take_profit_price:
            return (
                ExitReason.TAKE_PROFIT,
                f"Take profit reached at ${current_price:.2f}",
            )

        # Check for failed breakout (price falls back below entry)
        if len(df) >= 3:
            # If we've been in trade for a few bars and price falls below entry
            if current_price < entry_price * 0.98:  # 2% below entry
                return (
                    ExitReason.SIGNAL,
                    f"Failed breakout - price below entry (${current_price:.2f} < ${entry_price:.2f})",
                )

        return None

    def generate_signals(
        self,
        market_data: dict[str, MarketData],
        current_positions: dict[str, PositionInfo],
    ) -> list[Signal]:
        """Generate breakout entry signals."""
        signals = []

        for symbol, data in market_data.items():
            # Skip if already have a position
            if symbol in current_positions:
                continue

            if data.df.empty or len(data.df) < self.lookback_period + 5:
                logger.debug(f"Insufficient data for {symbol}")
                continue

            try:
                # Calculate indicators
                df_with_indicators = self.indicators.add_all_indicators(data.df)
                indicator_values = self.indicators.get_current_values(data.df)

                # Evaluate entry conditions
                strength, signal_type, reasons = self._evaluate_entry_conditions(
                    symbol,
                    data.last_price,
                    df_with_indicators,
                    indicator_values,
                )

                # Generate signal if strength is sufficient
                if strength >= 0.5 and signal_type == SignalType.BUY:
                    # Calculate ATR-based stops
                    atr = indicator_values.atr or (data.last_price * 0.02)
                    stop_loss = self._calculate_atr_stop(data.last_price, atr)
                    take_profit = self._calculate_atr_target(data.last_price, atr)

                    signal = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=strength,
                        price=data.last_price,
                        reason="; ".join(reasons),
                        strategy_name=self.name,
                        suggested_stop_loss=stop_loss,
                        suggested_take_profit=take_profit,
                        indicators={
                            "rsi": indicator_values.rsi,
                            "atr": indicator_values.atr,
                            "resistance": self._calculate_resistance(data.df.iloc[:-1]),
                            "volume_ratio": self._calculate_volume_ratio(data.df),
                        },
                    )
                    signals.append(signal)
                    logger.info(f"Generated breakout BUY signal for {symbol}: {signal}")

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

        return signals

    def should_exit(
        self,
        position: PositionInfo,
        market_data: MarketData,
    ) -> Optional[ExitSignal]:
        """Check if a breakout position should be closed."""
        if market_data.df.empty:
            return None

        try:
            # Calculate indicators
            df_with_indicators = self.indicators.add_all_indicators(market_data.df)
            indicator_values = self.indicators.get_current_values(market_data.df)

            # Evaluate exit conditions
            exit_result = self._evaluate_exit_conditions(
                position,
                df_with_indicators,
                indicator_values,
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
                "lookback_period": self.lookback_period,
                "volume_threshold": self.volume_threshold,
                "breakout_margin_pct": self.breakout_margin_pct,
                "atr_stop_multiplier": self.atr_stop_multiplier,
                "atr_target_multiplier": self.atr_target_multiplier,
                "trailing_stop_pct": self.trailing_stop_pct,
                "max_rsi": self.max_rsi,
            },
        }
