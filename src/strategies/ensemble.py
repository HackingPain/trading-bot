"""
Ensemble Strategy (2.2)

Aggregates signals from multiple child strategies using configurable
weights and a voting threshold. Entry signals require consensus among
child strategies; exit signals are conservative (any child exit triggers).
"""

import logging
from datetime import datetime
from typing import Any, Optional

from ..data.market_data import MarketData
from .base import ExitReason, ExitSignal, PositionInfo, Signal, SignalType, Strategy

logger = logging.getLogger(__name__)


class EnsembleStrategy(Strategy):
    """
    Multi-strategy ensemble that combines signals from N child strategies.

    Entry logic:
        Each child strategy produces signals independently. BUY signals are
        aggregated: if the sum of weights for child strategies that agree on
        a BUY exceeds the voting threshold, a combined BUY signal is emitted
        with a strength equal to the weighted average of the agreeing
        strategies' signal strengths.

    Exit logic (conservative):
        If *any* child strategy emits an exit signal for a position, the
        ensemble forwards it immediately.

    Configuration example (YAML):
        strategy:
          name: ensemble
          strategies: [daily_profit_taker, momentum, mean_reversion]
          weights: [0.4, 0.35, 0.25]
          voting_threshold: 0.6
    """

    def __init__(
        self,
        strategies: list[Strategy],
        weights: list[float] | None = None,
        voting_threshold: float = 0.6,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(name="ensemble", config=config)

        if not strategies:
            raise ValueError("EnsembleStrategy requires at least one child strategy")

        self.strategies = strategies

        # Default to equal weights when none supplied
        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            if len(weights) != len(strategies):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of strategies ({len(strategies)})"
                )
            # Normalise so they sum to 1.0
            total = sum(weights)
            if total <= 0:
                raise ValueError("Weights must sum to a positive number")
            self.weights = [w / total for w in weights]

        if not 0.0 < voting_threshold <= 1.0:
            raise ValueError("voting_threshold must be in (0, 1]")
        self.voting_threshold = voting_threshold

        child_names = [s.name for s in self.strategies]
        logger.info(
            "EnsembleStrategy created with children=%s, weights=%s, threshold=%.2f",
            child_names,
            [f"{w:.3f}" for w in self.weights],
            self.voting_threshold,
        )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        market_data: dict[str, MarketData],
        current_positions: dict[str, PositionInfo],
    ) -> list[Signal]:
        """
        Collect signals from every child strategy, then aggregate BUY
        signals per symbol using weighted voting.
        """
        if not self.is_active:
            return []

        # Gather all child signals keyed by symbol
        # Each entry: list of (weight_index, signal)
        child_signals: dict[str, list[tuple[int, Signal]]] = {}
        failed_indices: list[int] = []

        for idx, strategy in enumerate(self.strategies):
            if not strategy.is_active:
                continue
            try:
                signals = strategy.generate_signals(market_data, current_positions)
            except Exception:
                logger.exception(
                    "Child strategy '%s' raised an exception during generate_signals",
                    strategy.name,
                )
                failed_indices.append(idx)
                continue

            for signal in signals:
                child_signals.setdefault(signal.symbol, []).append((idx, signal))

        # Warn about effective weight redistribution from failures (Fix #8)
        if failed_indices:
            failed_weight = sum(self.weights[i] for i in failed_indices)
            failed_names = [self.strategies[i].name for i in failed_indices]
            active_weight = 1.0 - failed_weight
            logger.warning(
                "Ensemble: %d child strategy(ies) failed (%s), "
                "dropping %.1f%% of voting weight. "
                "Effective active weight: %.1f%%. "
                "Voting threshold (%.1f%%) may be easier/harder to reach.",
                len(failed_indices),
                ", ".join(failed_names),
                failed_weight * 100,
                active_weight * 100,
                self.voting_threshold * 100,
            )

        # Aggregate per symbol
        ensemble_signals: list[Signal] = []

        for symbol, entries in child_signals.items():
            buy_entries = [
                (idx, sig) for idx, sig in entries if sig.signal_type == SignalType.BUY
            ]
            sell_entries = [
                (idx, sig) for idx, sig in entries if sig.signal_type == SignalType.SELL
            ]

            # --- BUY aggregation ---
            if buy_entries:
                signal = self._aggregate_buy(symbol, buy_entries, market_data)
                if signal is not None:
                    ensemble_signals.append(signal)

            # --- SELL aggregation (same threshold logic) ---
            if sell_entries:
                signal = self._aggregate_sell(symbol, sell_entries, market_data)
                if signal is not None:
                    ensemble_signals.append(signal)

        return ensemble_signals

    def _aggregate_buy(
        self,
        symbol: str,
        entries: list[tuple[int, Signal]],
        market_data: dict[str, MarketData],
    ) -> Optional[Signal]:
        """Aggregate BUY signals; emit if weighted votes meet threshold."""
        weighted_vote = sum(self.weights[idx] for idx, _ in entries)

        if weighted_vote < self.voting_threshold:
            logger.debug(
                "BUY vote for %s: %.2f < threshold %.2f - skipped",
                symbol,
                weighted_vote,
                self.voting_threshold,
            )
            return None

        # Weighted average strength
        weighted_strength = sum(
            self.weights[idx] * sig.strength for idx, sig in entries
        )
        # Normalise by the total weight of agreeing strategies
        strength = weighted_strength / weighted_vote if weighted_vote > 0 else 0.0
        strength = max(0.0, min(1.0, strength))

        # Use the most recent price from child signals
        price = entries[-1][1].price

        # Collect contributing strategy names
        contributors = [self.strategies[idx].name for idx, _ in entries]

        # Merge indicator dicts from children
        merged_indicators: dict[str, Any] = {}
        for idx, sig in entries:
            for key, val in sig.indicators.items():
                merged_indicators[f"{self.strategies[idx].name}.{key}"] = val
        merged_indicators["ensemble_vote"] = round(weighted_vote, 4)
        merged_indicators["contributors"] = contributors

        # Aggregate suggested order parameters (use most conservative stop
        # loss / most conservative take profit among children that supply them)
        stop_losses = [
            sig.suggested_stop_loss
            for _, sig in entries
            if sig.suggested_stop_loss is not None
        ]
        take_profits = [
            sig.suggested_take_profit
            for _, sig in entries
            if sig.suggested_take_profit is not None
        ]

        return Signal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            strength=strength,
            price=price,
            reason=(
                f"Ensemble BUY: {len(entries)}/{len(self.strategies)} strategies agree "
                f"(weighted vote {weighted_vote:.2f} >= {self.voting_threshold:.2f}): "
                f"{', '.join(contributors)}"
            ),
            strategy_name=self.name,
            timestamp=datetime.now(),
            suggested_stop_loss=max(stop_losses) if stop_losses else None,
            suggested_take_profit=min(take_profits) if take_profits else None,
            indicators=merged_indicators,
        )

    def _aggregate_sell(
        self,
        symbol: str,
        entries: list[tuple[int, Signal]],
        market_data: dict[str, MarketData],
    ) -> Optional[Signal]:
        """Aggregate SELL signals with the same voting threshold."""
        weighted_vote = sum(self.weights[idx] for idx, _ in entries)

        if weighted_vote < self.voting_threshold:
            return None

        weighted_strength = sum(
            self.weights[idx] * sig.strength for idx, sig in entries
        )
        strength = weighted_strength / weighted_vote if weighted_vote > 0 else 0.0
        strength = max(0.0, min(1.0, strength))

        price = entries[-1][1].price
        contributors = [self.strategies[idx].name for idx, _ in entries]

        return Signal(
            symbol=symbol,
            signal_type=SignalType.SELL,
            strength=strength,
            price=price,
            reason=(
                f"Ensemble SELL: {len(entries)}/{len(self.strategies)} strategies agree "
                f"(weighted vote {weighted_vote:.2f} >= {self.voting_threshold:.2f}): "
                f"{', '.join(contributors)}"
            ),
            strategy_name=self.name,
            timestamp=datetime.now(),
        )

    # ------------------------------------------------------------------
    # Exit evaluation (conservative: any child exit triggers)
    # ------------------------------------------------------------------

    def should_exit(
        self,
        position: PositionInfo,
        market_data: MarketData,
    ) -> Optional[ExitSignal]:
        """
        Conservative exit policy: if *any* child strategy says exit,
        the ensemble forwards the exit signal immediately.

        When multiple children want to exit, the one with the highest
        urgency is chosen (immediate > high > normal).
        """
        if not self.is_active:
            return None

        urgency_rank = {"immediate": 3, "high": 2, "normal": 1}
        best_exit: Optional[ExitSignal] = None
        best_urgency = 0

        for strategy in self.strategies:
            if not strategy.is_active:
                continue
            try:
                exit_signal = strategy.should_exit(position, market_data)
            except Exception:
                logger.exception(
                    "Child strategy '%s' raised an exception during should_exit",
                    strategy.name,
                )
                continue

            if exit_signal is not None:
                rank = urgency_rank.get(exit_signal.urgency, 1)
                if best_exit is None or rank > best_urgency:
                    best_exit = exit_signal
                    best_urgency = rank

        if best_exit is not None:
            # Wrap the description to make it clear it came from the ensemble
            best_exit.description = (
                f"[ensemble] {best_exit.description}"
            )

        return best_exit

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: dict) -> "EnsembleStrategy":
        """
        Create an EnsembleStrategy from a settings dictionary.

        Expected structure::

            {
                "strategy": {
                    "name": "ensemble",
                    "strategies": ["daily_profit_taker", "momentum", "mean_reversion"],
                    "weights": [0.4, 0.35, 0.25],       # optional
                    "voting_threshold": 0.6,              # optional, default 0.6
                }
            }

        Child strategies are resolved via ``get_strategy`` from the
        strategy factory.
        """
        # Import here to avoid circular import at module level
        from .factory import get_strategy

        strategy_config = settings.get("strategy", {})
        child_names: list[str] = strategy_config.get("strategies", [])

        if not child_names:
            raise ValueError(
                "Ensemble strategy requires a non-empty 'strategies' list in config"
            )

        weights: list[float] | None = strategy_config.get("weights")
        voting_threshold: float = strategy_config.get("voting_threshold", 0.6)

        # Build child strategies.  Each child receives the full settings so
        # it can read its own section if needed (e.g., risk parameters).
        children: list[Strategy] = []
        for name in child_names:
            child_settings = dict(settings)
            child_settings["strategy"] = {"name": name}
            try:
                child = get_strategy(name, config=strategy_config.get(name))
                children.append(child)
            except ValueError:
                logger.error("Unknown child strategy '%s' - skipping", name)

        if not children:
            raise ValueError("No valid child strategies could be created")

        return cls(
            strategies=children,
            weights=weights,
            voting_threshold=voting_threshold,
            config=strategy_config,
        )
