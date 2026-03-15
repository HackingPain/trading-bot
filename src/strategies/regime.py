"""Market regime detection and adaptive strategy parameters (3.3)."""

import logging
from copy import deepcopy
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .base import Strategy, Signal, SignalType, ExitSignal, PositionInfo
from ..data.market_data import MarketData
from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classification."""
    LOW_VOLATILITY = "low_volatility"
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"


@dataclass
class RegimeThresholds:
    """Configurable thresholds for regime boundary detection."""
    # Rolling volatility ratio (20-day vol / 60-day avg vol)
    low_vol_upper: float = 0.8       # Below this -> LOW_VOLATILITY
    high_vol_lower: float = 1.5      # Above this -> HIGH_VOLATILITY
    crisis_lower: float = 2.5        # Above this -> CRISIS

    # ATR expansion ratio (current ATR / 60-day avg ATR)
    atr_high_threshold: float = 1.5
    atr_crisis_threshold: float = 2.5

    # Bollinger Band width ratio (current width / 60-day avg width)
    bb_width_low: float = 0.7
    bb_width_high: float = 1.5
    bb_width_crisis: float = 2.5

    # Minimum data points required for regime detection
    min_periods: int = 60


class RegimeDetector:
    """
    Detects the current market regime based on rolling volatility,
    ATR expansion/contraction, and Bollinger Band width analysis.
    """

    def __init__(
        self,
        thresholds: Optional[RegimeThresholds] = None,
        indicators: Optional[TechnicalIndicators] = None,
        vol_short_window: int = 20,
        vol_long_window: int = 60,
    ):
        self.thresholds = thresholds or RegimeThresholds()
        self.indicators = indicators or TechnicalIndicators()
        self.vol_short_window = vol_short_window
        self.vol_long_window = vol_long_window
        self._previous_regime: Optional[MarketRegime] = None

    @classmethod
    def from_settings(cls, settings: dict) -> "RegimeDetector":
        """Create a RegimeDetector from a settings dictionary."""
        regime_settings = settings.get("regime", {})
        threshold_settings = regime_settings.get("thresholds", {})

        thresholds = RegimeThresholds(
            low_vol_upper=threshold_settings.get("low_vol_upper", 0.8),
            high_vol_lower=threshold_settings.get("high_vol_lower", 1.5),
            crisis_lower=threshold_settings.get("crisis_lower", 2.5),
            atr_high_threshold=threshold_settings.get("atr_high_threshold", 1.5),
            atr_crisis_threshold=threshold_settings.get("atr_crisis_threshold", 2.5),
            bb_width_low=threshold_settings.get("bb_width_low", 0.7),
            bb_width_high=threshold_settings.get("bb_width_high", 1.5),
            bb_width_crisis=threshold_settings.get("bb_width_crisis", 2.5),
            min_periods=threshold_settings.get("min_periods", 60),
        )

        indicators = TechnicalIndicators.from_settings(settings)

        return cls(
            thresholds=thresholds,
            indicators=indicators,
            vol_short_window=regime_settings.get("vol_short_window", 20),
            vol_long_window=regime_settings.get("vol_long_window", 60),
        )

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect the current market regime from OHLCV data.

        Uses a voting system across three metrics:
        1. Rolling volatility ratio (20-day vs 60-day average)
        2. ATR expansion/contraction
        3. Bollinger Band width

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume).

        Returns:
            The detected MarketRegime.
        """
        if len(df) < self.thresholds.min_periods:
            logger.debug(
                f"Insufficient data for regime detection "
                f"({len(df)} < {self.thresholds.min_periods}), defaulting to NORMAL"
            )
            return MarketRegime.NORMAL

        vol_regime = self._volatility_regime(df)
        atr_regime = self._atr_regime(df)
        bb_regime = self._bb_width_regime(df)

        regime = self._vote(vol_regime, atr_regime, bb_regime)

        if self._previous_regime is not None and regime != self._previous_regime:
            logger.info(
                f"Regime change detected: {self._previous_regime.value} -> {regime.value}"
            )
        self._previous_regime = regime

        return regime

    def _volatility_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Classify regime using rolling return volatility ratio."""
        returns = df["close"].pct_change().dropna()

        short_vol = returns.rolling(window=self.vol_short_window).std().iloc[-1]
        long_vol = returns.rolling(window=self.vol_long_window).std().mean()

        if long_vol == 0 or np.isnan(long_vol):
            return MarketRegime.NORMAL

        ratio = short_vol / long_vol

        if ratio >= self.thresholds.crisis_lower:
            return MarketRegime.CRISIS
        elif ratio >= self.thresholds.high_vol_lower:
            return MarketRegime.HIGH_VOLATILITY
        elif ratio <= self.thresholds.low_vol_upper:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.NORMAL

    def _atr_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Classify regime using ATR expansion/contraction."""
        atr = self.indicators.calculate_atr(df)

        if len(atr.dropna()) < self.vol_long_window:
            return MarketRegime.NORMAL

        current_atr = atr.iloc[-1]
        avg_atr = atr.rolling(window=self.vol_long_window).mean().iloc[-1]

        if avg_atr == 0 or np.isnan(avg_atr):
            return MarketRegime.NORMAL

        ratio = current_atr / avg_atr

        if ratio >= self.thresholds.atr_crisis_threshold:
            return MarketRegime.CRISIS
        elif ratio >= self.thresholds.atr_high_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif ratio <= self.thresholds.bb_width_low:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.NORMAL

    def _bb_width_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Classify regime using Bollinger Band width ratio."""
        upper, middle, lower = self.indicators.calculate_bollinger_bands(df)

        bb_width = (upper - lower) / middle
        bb_width = bb_width.dropna()

        if len(bb_width) < self.vol_long_window:
            return MarketRegime.NORMAL

        current_width = bb_width.iloc[-1]
        avg_width = bb_width.rolling(window=self.vol_long_window).mean().iloc[-1]

        if avg_width == 0 or np.isnan(avg_width):
            return MarketRegime.NORMAL

        ratio = current_width / avg_width

        if ratio >= self.thresholds.bb_width_crisis:
            return MarketRegime.CRISIS
        elif ratio >= self.thresholds.bb_width_high:
            return MarketRegime.HIGH_VOLATILITY
        elif ratio <= self.thresholds.bb_width_low:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.NORMAL

    @staticmethod
    def _vote(
        vol_regime: MarketRegime,
        atr_regime: MarketRegime,
        bb_regime: MarketRegime,
    ) -> MarketRegime:
        """
        Determine the overall regime by majority vote.

        CRISIS takes priority: if any two indicators signal CRISIS, it wins.
        Otherwise the most common regime is selected, with ties broken
        toward the higher-volatility regime for safety.
        """
        votes = [vol_regime, atr_regime, bb_regime]

        # CRISIS priority: if at least 2 votes are CRISIS, declare CRISIS
        crisis_count = votes.count(MarketRegime.CRISIS)
        if crisis_count >= 2:
            return MarketRegime.CRISIS

        # Count votes for each regime
        regime_counts: dict[MarketRegime, int] = {}
        for v in votes:
            regime_counts[v] = regime_counts.get(v, 0) + 1

        max_count = max(regime_counts.values())
        candidates = [r for r, c in regime_counts.items() if c == max_count]

        if len(candidates) == 1:
            return candidates[0]

        # Tie-breaking: prefer the higher-volatility regime for safety
        priority = [
            MarketRegime.CRISIS,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.NORMAL,
            MarketRegime.LOW_VOLATILITY,
        ]
        for regime in priority:
            if regime in candidates:
                return regime

        return MarketRegime.NORMAL  # fallback


@dataclass
class RegimeParameters:
    """Parameter adjustments for a specific market regime."""
    stop_loss_pct: float
    take_profit_pct: float
    position_size_multiplier: float  # Multiplier applied to base position size
    allow_new_entries: bool = True
    reduce_exposure_pct: Optional[float] = None  # Target exposure reduction


# Default parameter sets per regime
DEFAULT_REGIME_PARAMS: dict[MarketRegime, RegimeParameters] = {
    MarketRegime.LOW_VOLATILITY: RegimeParameters(
        stop_loss_pct=0.03,
        take_profit_pct=0.015,
        position_size_multiplier=1.25,
        allow_new_entries=True,
    ),
    MarketRegime.NORMAL: RegimeParameters(
        stop_loss_pct=0.05,
        take_profit_pct=0.03,
        position_size_multiplier=1.0,
        allow_new_entries=True,
    ),
    MarketRegime.HIGH_VOLATILITY: RegimeParameters(
        stop_loss_pct=0.06,
        take_profit_pct=0.04,
        position_size_multiplier=0.6,
        allow_new_entries=True,
    ),
    MarketRegime.CRISIS: RegimeParameters(
        stop_loss_pct=0.08,
        take_profit_pct=0.06,
        position_size_multiplier=0.25,
        allow_new_entries=False,
        reduce_exposure_pct=0.50,
    ),
}


class AdaptiveParameters:
    """
    Maps market regimes to adjusted trading parameters.

    Takes a base configuration dict and adjusts it according to the
    detected regime, returning a modified copy.
    """

    def __init__(
        self,
        regime_params: Optional[dict[MarketRegime, RegimeParameters]] = None,
    ):
        self.regime_params = regime_params or deepcopy(DEFAULT_REGIME_PARAMS)

    @classmethod
    def from_settings(cls, settings: dict) -> "AdaptiveParameters":
        """Create AdaptiveParameters from a settings dictionary."""
        regime_settings = settings.get("regime", {}).get("parameters", {})
        if not regime_settings:
            return cls()

        params: dict[MarketRegime, RegimeParameters] = {}
        for regime in MarketRegime:
            regime_key = regime.value
            if regime_key in regime_settings:
                rs = regime_settings[regime_key]
                params[regime] = RegimeParameters(
                    stop_loss_pct=rs.get(
                        "stop_loss_pct",
                        DEFAULT_REGIME_PARAMS[regime].stop_loss_pct,
                    ),
                    take_profit_pct=rs.get(
                        "take_profit_pct",
                        DEFAULT_REGIME_PARAMS[regime].take_profit_pct,
                    ),
                    position_size_multiplier=rs.get(
                        "position_size_multiplier",
                        DEFAULT_REGIME_PARAMS[regime].position_size_multiplier,
                    ),
                    allow_new_entries=rs.get(
                        "allow_new_entries",
                        DEFAULT_REGIME_PARAMS[regime].allow_new_entries,
                    ),
                    reduce_exposure_pct=rs.get(
                        "reduce_exposure_pct",
                        DEFAULT_REGIME_PARAMS[regime].reduce_exposure_pct,
                    ),
                )
            else:
                params[regime] = deepcopy(DEFAULT_REGIME_PARAMS[regime])

        return cls(regime_params=params)

    def get_params(self, regime: MarketRegime, base_config: dict) -> dict:
        """
        Return a modified copy of base_config with parameters adjusted
        for the given market regime.

        The following keys are set/overridden in the returned dict:
            - stop_loss_pct
            - take_profit_pct
            - position_size_multiplier
            - allow_new_entries
            - reduce_exposure_pct
            - current_regime

        Args:
            regime: The detected market regime.
            base_config: The strategy's base configuration dict.

        Returns:
            A new dict with regime-adjusted parameters merged in.
        """
        rp = self.regime_params.get(regime, DEFAULT_REGIME_PARAMS[MarketRegime.NORMAL])
        adjusted = deepcopy(base_config)

        adjusted["stop_loss_pct"] = rp.stop_loss_pct
        adjusted["take_profit_pct"] = rp.take_profit_pct
        adjusted["position_size_multiplier"] = rp.position_size_multiplier
        adjusted["allow_new_entries"] = rp.allow_new_entries
        adjusted["reduce_exposure_pct"] = rp.reduce_exposure_pct
        adjusted["current_regime"] = regime.value

        return adjusted


class AdaptiveStrategyWrapper(Strategy):
    """
    Wraps any existing Strategy to add regime-aware adaptive parameter
    adjustment.

    Before the inner strategy generates signals, the wrapper:
      1. Detects the current market regime from price data.
      2. Adjusts the inner strategy's config using AdaptiveParameters.
      3. Delegates signal generation to the inner strategy.
      4. Filters out new-entry signals when the regime forbids them.
      5. Logs regime transitions.
    """

    def __init__(
        self,
        inner_strategy: Strategy,
        detector: Optional[RegimeDetector] = None,
        adaptive_params: Optional[AdaptiveParameters] = None,
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            name=f"adaptive_{inner_strategy.name}",
            config=config or inner_strategy.config,
        )
        self.inner_strategy = inner_strategy
        self.detector = detector or RegimeDetector()
        self.adaptive_params = adaptive_params or AdaptiveParameters()
        self._current_regime: MarketRegime = MarketRegime.NORMAL
        self._base_config = deepcopy(inner_strategy.config)

    @classmethod
    def from_settings(
        cls,
        settings: dict,
        inner_strategy: Optional[Strategy] = None,
    ) -> "AdaptiveStrategyWrapper":
        """
        Create an AdaptiveStrategyWrapper from a settings dictionary.

        If *inner_strategy* is not provided, the caller is responsible
        for setting it afterwards via the ``inner_strategy`` attribute.

        Args:
            settings: Full application settings dict.
            inner_strategy: The strategy to wrap. If None, a placeholder
                            is used and must be replaced before use.

        Returns:
            Configured AdaptiveStrategyWrapper instance.
        """
        detector = RegimeDetector.from_settings(settings)
        adaptive_params = AdaptiveParameters.from_settings(settings)

        if inner_strategy is None:
            # Create a minimal placeholder; caller must replace before use
            from .momentum import MomentumStrategy
            strategy_settings = settings.get("strategy", {})
            inner_strategy = MomentumStrategy(
                name="placeholder",
                config=strategy_settings,
            )

        return cls(
            inner_strategy=inner_strategy,
            detector=detector,
            adaptive_params=adaptive_params,
            config=settings.get("strategy", {}),
        )

    @property
    def current_regime(self) -> MarketRegime:
        """The most recently detected market regime."""
        return self._current_regime

    def _detect_and_adapt(self, market_data: dict[str, MarketData]) -> dict:
        """
        Detect regime from available market data and return adjusted params.

        Uses the longest available DataFrame among the provided symbols
        for regime detection.
        """
        # Pick the symbol with the most data for regime detection
        best_df: Optional[pd.DataFrame] = None
        for md in market_data.values():
            if md.df is not None and (best_df is None or len(md.df) > len(best_df)):
                best_df = md.df

        if best_df is not None and not best_df.empty:
            new_regime = self.detector.detect(best_df)
        else:
            new_regime = MarketRegime.NORMAL

        if new_regime != self._current_regime:
            logger.info(
                f"[{self.name}] Regime transition: "
                f"{self._current_regime.value} -> {new_regime.value}"
            )
            self._current_regime = new_regime

        adjusted = self.adaptive_params.get_params(new_regime, self._base_config)
        return adjusted

    def generate_signals(
        self,
        market_data: dict[str, MarketData],
        current_positions: dict[str, PositionInfo],
    ) -> list[Signal]:
        """
        Detect regime, adjust parameters, and generate signals from
        the inner strategy.

        In CRISIS mode (or any regime where allow_new_entries is False),
        BUY signals are suppressed and only SELL/HOLD signals pass through.
        """
        adjusted_config = self._detect_and_adapt(market_data)

        # Apply adjusted config to the inner strategy
        self.inner_strategy.config = adjusted_config

        signals = self.inner_strategy.generate_signals(market_data, current_positions)

        # Filter out new entry signals if the regime forbids them
        if not adjusted_config.get("allow_new_entries", True):
            original_count = len(signals)
            signals = [
                s for s in signals
                if s.signal_type != SignalType.BUY
            ]
            filtered_count = original_count - len(signals)
            if filtered_count > 0:
                logger.info(
                    f"[{self.name}] Regime {self._current_regime.value}: "
                    f"blocked {filtered_count} new entry signal(s)"
                )

        # Adjust position sizes based on regime multiplier
        multiplier = adjusted_config.get("position_size_multiplier", 1.0)
        if multiplier != 1.0:
            for signal in signals:
                if signal.suggested_quantity is not None:
                    signal.suggested_quantity = max(
                        1, int(signal.suggested_quantity * multiplier)
                    )

        # Attach regime info to signal indicators
        for signal in signals:
            signal.indicators["regime"] = self._current_regime.value
            signal.indicators["regime_stop_loss_pct"] = adjusted_config.get("stop_loss_pct")
            signal.indicators["regime_take_profit_pct"] = adjusted_config.get("take_profit_pct")

        return signals

    def should_exit(
        self,
        position: PositionInfo,
        market_data: MarketData,
    ) -> Optional[ExitSignal]:
        """
        Detect current regime and adjust inner strategy config before
        delegating exit evaluation.

        This ensures exit decisions always use fresh regime parameters,
        even if generate_signals was not called this cycle or raised
        an exception (Fix #6).
        """
        # Re-detect regime from the current market data so config is fresh
        if market_data.df is not None and not market_data.df.empty:
            new_regime = self.detector.detect(market_data.df)
            if new_regime != self._current_regime:
                logger.info(
                    f"[{self.name}] Regime transition (exit check): "
                    f"{self._current_regime.value} -> {new_regime.value}"
                )
                self._current_regime = new_regime

            adjusted = self.adaptive_params.get_params(
                self._current_regime, self._base_config
            )
            self.inner_strategy.config = adjusted

        return self.inner_strategy.should_exit(position, market_data)

    def __repr__(self) -> str:
        return (
            f"<AdaptiveStrategyWrapper(inner={self.inner_strategy.name}, "
            f"regime={self._current_regime.value}, "
            f"active={self._is_active})>"
        )
