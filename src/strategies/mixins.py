"""Composable strategy mixins for common trading patterns.

These mixins extract duplicated indicator calculation and signal logic
from individual strategies into reusable, composable building blocks.

Usage:
    class MyStrategy(Strategy, TrendMixin, MomentumMixin):
        def generate_signals(self, market_data, current_positions):
            trend = self.detect_ema_crossover(df)
            momentum = self.evaluate_rsi_momentum(indicators)
            ...
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .base import ExitReason, ExitSignal, PositionInfo
from .indicators import IndicatorValues, TechnicalIndicators

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses returned by mixin methods
# ---------------------------------------------------------------------------

@dataclass
class TrendSignal:
    """Result of a trend analysis."""
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0.0 to 1.0
    description: str


@dataclass
class MomentumSignal:
    """Result of a momentum analysis."""
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0.0 to 1.0
    description: str


@dataclass
class VolatilityState:
    """Current volatility regime information."""
    regime: str  # "low", "normal", "high", "extreme"
    atr_value: float
    bb_width: float  # Bollinger Band width as fraction of price
    stop_distance: float  # Suggested ATR-based stop distance


# ---------------------------------------------------------------------------
# TrendMixin
# ---------------------------------------------------------------------------

class TrendMixin:
    """Mixin providing EMA/SMA crossover detection and trend analysis.

    Expects the consuming class to have ``self.indicators`` as a
    :class:`TechnicalIndicators` instance.
    """

    # -- EMA crossover -------------------------------------------------------

    def detect_ema_crossover(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
    ) -> TrendSignal:
        """Detect EMA crossover between *fast_period* and *slow_period*.

        Returns a :class:`TrendSignal` with direction, normalised strength
        (based on the percentage spread between the two EMAs), and a
        human-readable description.
        """
        indicators: TechnicalIndicators = self.indicators  # type: ignore[attr-defined]

        ema_fast = indicators.calculate_ema(df, period=fast_period)
        ema_slow = indicators.calculate_ema(df, period=slow_period)

        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return TrendSignal("neutral", 0.0, "Insufficient data for EMA crossover")

        prev_diff = ema_fast.iloc[-2] - ema_slow.iloc[-2]
        curr_diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]
        spread_pct = abs(curr_diff) / ema_slow.iloc[-1] if ema_slow.iloc[-1] != 0 else 0.0

        # Normalise spread into 0-1 strength (cap at 2% spread -> 1.0)
        strength = min(spread_pct / 0.02, 1.0)

        if prev_diff <= 0 < curr_diff:
            return TrendSignal("bullish", strength, f"Bullish EMA crossover (EMA{fast_period} > EMA{slow_period})")
        if prev_diff >= 0 > curr_diff:
            return TrendSignal("bearish", strength, f"Bearish EMA crossover (EMA{fast_period} < EMA{slow_period})")
        if curr_diff > 0:
            return TrendSignal("bullish", strength * 0.6, f"EMA{fast_period} above EMA{slow_period}")
        if curr_diff < 0:
            return TrendSignal("bearish", strength * 0.6, f"EMA{fast_period} below EMA{slow_period}")

        return TrendSignal("neutral", 0.0, "EMAs converged")

    # -- Trend strength ------------------------------------------------------

    def calculate_trend_strength(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
    ) -> TrendSignal:
        """Measure trend strength using price vs SMA alignment and slope.

        Uses the short and long SMAs configured on ``self.indicators`` to
        determine whether the market is trending and how strongly.
        """
        indicators: TechnicalIndicators = self.indicators  # type: ignore[attr-defined]

        sma_short = indicators.calculate_sma(df, indicators.sma_short_period)
        sma_long = indicators.calculate_sma(df, indicators.sma_long_period)

        if len(sma_short) < lookback or len(sma_long) < lookback:
            return TrendSignal("neutral", 0.0, "Insufficient data for trend strength")

        close = df["close"].iloc[-1]
        sma_s = sma_short.iloc[-1]
        sma_l = sma_long.iloc[-1]

        # Factor 1: price position relative to both SMAs (0 or 1 each)
        price_above_short = 1.0 if close > sma_s else 0.0
        price_above_long = 1.0 if close > sma_l else 0.0
        short_above_long = 1.0 if sma_s > sma_l else 0.0

        # Factor 2: SMA slope (normalised over lookback)
        slope = (sma_short.iloc[-1] - sma_short.iloc[-lookback]) / sma_short.iloc[-lookback]
        slope_strength = min(abs(slope) / 0.05, 1.0)  # 5% move over lookback -> 1.0

        # Composite
        alignment = (price_above_short + price_above_long + short_above_long) / 3.0
        strength = (alignment * 0.6 + slope_strength * 0.4)

        if alignment >= 0.66 and slope > 0:
            direction = "bullish"
        elif alignment <= 0.33 and slope < 0:
            direction = "bearish"
        else:
            direction = "neutral"

        return TrendSignal(
            direction,
            round(min(strength, 1.0), 4),
            f"Trend alignment={alignment:.0%}, slope={slope:.2%}",
        )

    # -- Golden / Death cross ------------------------------------------------

    def detect_golden_cross(self, df: pd.DataFrame) -> TrendSignal:
        """Detect a golden cross (short SMA crosses above long SMA).

        Delegates to ``self.indicators.golden_cross`` but wraps the
        result with strength information.
        """
        indicators: TechnicalIndicators = self.indicators  # type: ignore[attr-defined]

        if indicators.golden_cross(df):
            sma_s = indicators.calculate_sma(df, indicators.sma_short_period).iloc[-1]
            sma_l = indicators.calculate_sma(df, indicators.sma_long_period).iloc[-1]
            spread = abs(sma_s - sma_l) / sma_l if sma_l else 0.0
            strength = min(spread / 0.01, 1.0)
            return TrendSignal("bullish", max(strength, 0.7), "Golden cross detected")

        return TrendSignal("neutral", 0.0, "No golden cross")

    def detect_death_cross(self, df: pd.DataFrame) -> TrendSignal:
        """Detect a death cross (short SMA crosses below long SMA)."""
        indicators: TechnicalIndicators = self.indicators  # type: ignore[attr-defined]

        if indicators.death_cross(df):
            sma_s = indicators.calculate_sma(df, indicators.sma_short_period).iloc[-1]
            sma_l = indicators.calculate_sma(df, indicators.sma_long_period).iloc[-1]
            spread = abs(sma_s - sma_l) / sma_l if sma_l else 0.0
            strength = min(spread / 0.01, 1.0)
            return TrendSignal("bearish", max(strength, 0.7), "Death cross detected")

        return TrendSignal("neutral", 0.0, "No death cross")


# ---------------------------------------------------------------------------
# MomentumMixin
# ---------------------------------------------------------------------------

class MomentumMixin:
    """Mixin providing RSI, MACD, and volume-based momentum evaluation.

    Expects ``self.indicators`` as a :class:`TechnicalIndicators` instance.
    """

    # -- RSI -----------------------------------------------------------------

    def evaluate_rsi_momentum(
        self,
        indicators: IndicatorValues,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> MomentumSignal:
        """Evaluate momentum from the current RSI value.

        Returns a directional signal:
        - RSI < *oversold* -> bullish (potential reversal)
        - RSI > *overbought* -> bearish (potential reversal)
        - Middle zone strength scales linearly from the midpoint.
        """
        rsi = indicators.rsi
        if rsi is None:
            return MomentumSignal("neutral", 0.0, "RSI unavailable")

        if rsi < oversold:
            # Deeper oversold -> stronger bullish signal
            depth = (oversold - rsi) / oversold  # 0..1
            return MomentumSignal("bullish", min(0.5 + depth * 0.5, 1.0), f"RSI oversold ({rsi:.1f})")
        if rsi > overbought:
            depth = (rsi - overbought) / (100.0 - overbought)
            return MomentumSignal("bearish", min(0.5 + depth * 0.5, 1.0), f"RSI overbought ({rsi:.1f})")

        # Neutral zone – slight directional bias
        midpoint = (oversold + overbought) / 2.0
        deviation = (rsi - midpoint) / (overbought - oversold)
        if deviation > 0:
            return MomentumSignal("bearish", abs(deviation) * 0.4, f"RSI neutral-high ({rsi:.1f})")
        return MomentumSignal("bullish", abs(deviation) * 0.4, f"RSI neutral-low ({rsi:.1f})")

    # -- MACD ----------------------------------------------------------------

    def evaluate_macd_momentum(
        self,
        df: pd.DataFrame,
        indicators: IndicatorValues,
    ) -> MomentumSignal:
        """Evaluate momentum from MACD line, signal, and histogram.

        Combines crossover detection with histogram magnitude.
        """
        ind: TechnicalIndicators = self.indicators  # type: ignore[attr-defined]

        if indicators.macd is None or indicators.macd_signal is None:
            return MomentumSignal("neutral", 0.0, "MACD unavailable")

        bullish_cross = ind.macd_bullish_crossover(df)
        bearish_cross = ind.macd_bearish_crossover(df)

        hist = indicators.macd_histogram or 0.0
        close = indicators.close or 1.0
        hist_pct = abs(hist) / close  # normalise by price
        hist_strength = min(hist_pct / 0.005, 1.0)  # 0.5% of price -> 1.0

        if bullish_cross:
            return MomentumSignal("bullish", max(hist_strength, 0.7), "MACD bullish crossover")
        if bearish_cross:
            return MomentumSignal("bearish", max(hist_strength, 0.7), "MACD bearish crossover")

        if hist > 0:
            return MomentumSignal("bullish", hist_strength * 0.5, f"MACD histogram positive ({hist:.4f})")
        if hist < 0:
            return MomentumSignal("bearish", hist_strength * 0.5, f"MACD histogram negative ({hist:.4f})")

        return MomentumSignal("neutral", 0.0, "MACD flat")

    # -- Volume momentum -----------------------------------------------------

    def evaluate_volume_momentum(
        self,
        df: pd.DataFrame,
        lookback: int = 20,
        surge_threshold: float = 2.0,
    ) -> MomentumSignal:
        """Evaluate momentum based on recent volume relative to average.

        A volume surge (current volume > *surge_threshold* * average) is
        interpreted in the direction of the latest price move.
        """
        if len(df) < lookback + 1:
            return MomentumSignal("neutral", 0.0, "Insufficient data for volume analysis")

        avg_volume = df["volume"].iloc[-(lookback + 1):-1].mean()
        current_volume = df["volume"].iloc[-1]

        if avg_volume == 0:
            return MomentumSignal("neutral", 0.0, "Zero average volume")

        volume_ratio = current_volume / avg_volume
        price_change = df["close"].iloc[-1] - df["close"].iloc[-2]

        if volume_ratio >= surge_threshold:
            strength = min((volume_ratio - 1.0) / (surge_threshold * 2 - 1.0), 1.0)
            direction = "bullish" if price_change > 0 else "bearish" if price_change < 0 else "neutral"
            return MomentumSignal(direction, strength, f"Volume surge {volume_ratio:.1f}x average")

        # Normal volume – weak directional signal
        mild_strength = min(volume_ratio / surge_threshold, 1.0) * 0.3
        if price_change > 0:
            return MomentumSignal("bullish", mild_strength, f"Volume {volume_ratio:.1f}x avg, price up")
        if price_change < 0:
            return MomentumSignal("bearish", mild_strength, f"Volume {volume_ratio:.1f}x avg, price down")

        return MomentumSignal("neutral", 0.0, "No volume momentum")


# ---------------------------------------------------------------------------
# VolatilityMixin
# ---------------------------------------------------------------------------

class VolatilityMixin:
    """Mixin providing ATR stops, Bollinger Band analysis, and volatility regime detection.

    Expects ``self.indicators`` as a :class:`TechnicalIndicators` instance.
    """

    # -- ATR-based stops -----------------------------------------------------

    def calculate_atr_stop(
        self,
        df: pd.DataFrame,
        multiplier: float = 2.0,
        is_long: bool = True,
    ) -> float:
        """Calculate a stop-loss price using ATR distance from the current close.

        Args:
            df: OHLCV DataFrame.
            multiplier: Number of ATRs away from close for the stop.
            is_long: True for long positions, False for short.

        Returns:
            The stop-loss price.
        """
        indicators: TechnicalIndicators = self.indicators  # type: ignore[attr-defined]
        atr = indicators.calculate_atr(df)

        if atr.empty or pd.isna(atr.iloc[-1]):
            # Fallback: 2% from close
            close = df["close"].iloc[-1]
            return close * (0.98 if is_long else 1.02)

        close = df["close"].iloc[-1]
        distance = atr.iloc[-1] * multiplier

        return close - distance if is_long else close + distance

    def calculate_atr_take_profit(
        self,
        df: pd.DataFrame,
        multiplier: float = 3.0,
        is_long: bool = True,
    ) -> float:
        """Calculate a take-profit price using ATR distance.

        Default *multiplier* of 3.0 gives a 1.5:1 reward-to-risk ratio when
        paired with the default stop multiplier of 2.0.
        """
        indicators: TechnicalIndicators = self.indicators  # type: ignore[attr-defined]
        atr = indicators.calculate_atr(df)

        if atr.empty or pd.isna(atr.iloc[-1]):
            close = df["close"].iloc[-1]
            return close * (1.03 if is_long else 0.97)

        close = df["close"].iloc[-1]
        distance = atr.iloc[-1] * multiplier

        return close + distance if is_long else close - distance

    # -- Bollinger Band analysis ---------------------------------------------

    def analyze_bollinger_bands(self, indicators: IndicatorValues) -> MomentumSignal:
        """Analyse the current price position within Bollinger Bands.

        Returns a :class:`MomentumSignal` (reused for convenience) describing
        potential mean-reversion or breakout signals.
        """
        bb_pct = indicators.bb_percent
        if bb_pct is None:
            return MomentumSignal("neutral", 0.0, "Bollinger data unavailable")

        if bb_pct <= 0.0:
            # Price at or below lower band – potential bullish reversal
            strength = min(abs(bb_pct) + 0.5, 1.0)
            return MomentumSignal("bullish", strength, f"Price below lower BB (%B={bb_pct:.2f})")
        if bb_pct >= 1.0:
            strength = min(bb_pct - 1.0 + 0.5, 1.0)
            return MomentumSignal("bearish", strength, f"Price above upper BB (%B={bb_pct:.2f})")
        if bb_pct < 0.2:
            return MomentumSignal("bullish", 0.4, f"Price near lower BB (%B={bb_pct:.2f})")
        if bb_pct > 0.8:
            return MomentumSignal("bearish", 0.4, f"Price near upper BB (%B={bb_pct:.2f})")

        return MomentumSignal("neutral", 0.1, f"Price within BB (%B={bb_pct:.2f})")

    # -- Volatility regime ---------------------------------------------------

    def detect_volatility_regime(
        self,
        df: pd.DataFrame,
        lookback: int = 50,
    ) -> VolatilityState:
        """Classify the current volatility regime.

        Compares the current ATR (as a percentage of close) to its own
        historical distribution over *lookback* periods.

        Regimes:
        - ``low``: ATR% below 25th percentile
        - ``normal``: 25th–75th percentile
        - ``high``: 75th–90th percentile
        - ``extreme``: above 90th percentile
        """
        indicators: TechnicalIndicators = self.indicators  # type: ignore[attr-defined]

        atr_series = indicators.calculate_atr(df)
        upper, _, lower = indicators.calculate_bollinger_bands(df)

        close = df["close"].iloc[-1]
        atr_val = atr_series.iloc[-1] if not atr_series.empty else 0.0

        # BB width as fraction of price
        bb_width = (upper.iloc[-1] - lower.iloc[-1]) / close if close != 0 else 0.0

        # ATR% distribution
        atr_pct = atr_series / df["close"]
        atr_pct_clean = atr_pct.dropna()

        if len(atr_pct_clean) < lookback:
            return VolatilityState("normal", atr_val, bb_width, atr_val * 2)

        recent_window = atr_pct_clean.iloc[-lookback:]
        current_atr_pct = atr_pct_clean.iloc[-1]
        p25 = recent_window.quantile(0.25)
        p75 = recent_window.quantile(0.75)
        p90 = recent_window.quantile(0.90)

        if current_atr_pct >= p90:
            regime = "extreme"
        elif current_atr_pct >= p75:
            regime = "high"
        elif current_atr_pct <= p25:
            regime = "low"
        else:
            regime = "normal"

        return VolatilityState(
            regime=regime,
            atr_value=float(atr_val),
            bb_width=float(bb_width),
            stop_distance=float(atr_val * 2),
        )


# ---------------------------------------------------------------------------
# RiskManagementMixin
# ---------------------------------------------------------------------------

class RiskManagementMixin:
    """Mixin providing common exit-condition evaluation.

    Centralises stop-loss, trailing-stop, and take-profit logic so that
    individual strategies do not have to reimplement it.

    Expects the consuming class to be a :class:`Strategy` subclass (for
    access to ``self.config``, ``self.name``).
    """

    def evaluate_stop_loss(
        self,
        position: PositionInfo,
    ) -> Optional[tuple[ExitReason, str]]:
        """Check whether the position has hit its stop-loss level.

        Returns ``(ExitReason.STOP_LOSS, description)`` or ``None``.
        """
        if position.stop_loss_price is None:
            return None

        if position.current_price <= position.stop_loss_price:
            loss_pct = (
                (position.current_price - position.avg_entry_price) / position.avg_entry_price
            ) * 100
            return (
                ExitReason.STOP_LOSS,
                f"Stop loss triggered at ${position.current_price:.2f} "
                f"(stop=${position.stop_loss_price:.2f}, loss={loss_pct:.1f}%)",
            )
        return None

    def evaluate_trailing_stop(
        self,
        position: PositionInfo,
        trailing_stop_pct: Optional[float] = None,
    ) -> Optional[tuple[ExitReason, str]]:
        """Check whether the position has hit its trailing stop.

        If *trailing_stop_pct* is provided it recalculates the trailing
        stop from ``position.highest_price``; otherwise it uses the
        pre-computed ``position.trailing_stop_price``.

        Returns ``(ExitReason.TRAILING_STOP, description)`` or ``None``.
        """
        trailing_price = position.trailing_stop_price

        if trailing_stop_pct is not None:
            trailing_price = position.highest_price * (1 - trailing_stop_pct)

        if trailing_price is None:
            return None

        if position.current_price <= trailing_price:
            drop_pct = (
                (position.highest_price - position.current_price) / position.highest_price
            ) * 100
            return (
                ExitReason.TRAILING_STOP,
                f"Trailing stop triggered at ${position.current_price:.2f} "
                f"(trail=${trailing_price:.2f}, drop from high={drop_pct:.1f}%)",
            )
        return None

    def evaluate_take_profit(
        self,
        position: PositionInfo,
    ) -> Optional[tuple[ExitReason, str]]:
        """Check whether the position has reached its take-profit target.

        Returns ``(ExitReason.TAKE_PROFIT, description)`` or ``None``.
        """
        if position.take_profit_price is None:
            return None

        if position.current_price >= position.take_profit_price:
            gain_pct = (
                (position.current_price - position.avg_entry_price) / position.avg_entry_price
            ) * 100
            return (
                ExitReason.TAKE_PROFIT,
                f"Take profit reached at ${position.current_price:.2f} "
                f"(target=${position.take_profit_price:.2f}, gain={gain_pct:.1f}%)",
            )
        return None

    def evaluate_all_exits(
        self,
        position: PositionInfo,
        trailing_stop_pct: Optional[float] = None,
    ) -> Optional[tuple[ExitReason, str]]:
        """Run all exit checks in priority order.

        Evaluation order: stop loss -> trailing stop -> take profit.
        Returns the first triggered exit or ``None``.
        """
        # 1. Hard stop loss (highest priority)
        result = self.evaluate_stop_loss(position)
        if result is not None:
            return result

        # 2. Trailing stop
        result = self.evaluate_trailing_stop(position, trailing_stop_pct)
        if result is not None:
            return result

        # 3. Take profit
        result = self.evaluate_take_profit(position)
        if result is not None:
            return result

        return None

    def build_exit_signal(
        self,
        position: PositionInfo,
        reason: ExitReason,
        description: str,
        urgency: str = "normal",
    ) -> ExitSignal:
        """Convenience builder for :class:`ExitSignal`.

        Strategies can call ``evaluate_all_exits`` and, if a result is
        returned, pass it straight through::

            exit = self.evaluate_all_exits(position)
            if exit:
                return self.build_exit_signal(position, exit[0], exit[1])
        """
        if reason == ExitReason.STOP_LOSS:
            urgency = "immediate"
        elif reason == ExitReason.TRAILING_STOP:
            urgency = "high"

        return ExitSignal(
            symbol=position.symbol,
            reason=reason,
            exit_price=position.current_price,
            description=description,
            urgency=urgency,
        )
