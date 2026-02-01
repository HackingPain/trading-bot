"""
Strategy Factory

Provides a registry and factory for creating trading strategies.
"""

import logging
from typing import Any, Type

from .base import Strategy
from .daily_profit_taker import DailyProfitTakerStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy

logger = logging.getLogger(__name__)


# Strategy registry
STRATEGY_REGISTRY: dict[str, Type[Strategy]] = {
    "daily_profit_taker": DailyProfitTakerStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
}


def get_strategy(name: str, config: dict[str, Any] | None = None) -> Strategy:
    """
    Get a strategy instance by name.

    Args:
        name: Strategy name (e.g., 'daily_profit_taker', 'mean_reversion', 'momentum')
        config: Optional configuration dict

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy name is not found
    """
    name = name.lower().replace("-", "_").replace(" ", "_")

    if name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown strategy: '{name}'. Available strategies: {available}"
        )

    strategy_class = STRATEGY_REGISTRY[name]
    logger.info(f"Creating strategy: {name}")

    return strategy_class(config=config)


def get_strategy_from_settings(settings: dict) -> Strategy:
    """
    Create a strategy from settings dictionary.

    Args:
        settings: Full settings dict with 'strategy', 'risk' sections

    Returns:
        Configured strategy instance
    """
    strategy_config = settings.get("strategy", {})
    strategy_name = strategy_config.get("name", "daily_profit_taker")

    name = strategy_name.lower().replace("-", "_").replace(" ", "_")

    if name not in STRATEGY_REGISTRY:
        logger.warning(f"Unknown strategy '{name}', using daily_profit_taker")
        name = "daily_profit_taker"

    strategy_class = STRATEGY_REGISTRY[name]

    # Use from_settings if available
    if hasattr(strategy_class, "from_settings"):
        return strategy_class.from_settings(settings)
    else:
        return strategy_class(config=strategy_config)


def list_strategies() -> list[dict[str, Any]]:
    """
    List all available strategies with their descriptions.

    Returns:
        List of strategy info dicts
    """
    strategies = []

    for name, strategy_class in STRATEGY_REGISTRY.items():
        # Get docstring for description
        doc = strategy_class.__doc__ or ""
        description = doc.strip().split("\n")[0] if doc else "No description"

        strategies.append({
            "name": name,
            "class": strategy_class.__name__,
            "description": description,
        })

    return strategies


def register_strategy(name: str, strategy_class: Type[Strategy]) -> None:
    """
    Register a custom strategy.

    Args:
        name: Strategy name (will be normalized)
        strategy_class: Strategy class (must inherit from Strategy)

    Raises:
        TypeError: If strategy_class doesn't inherit from Strategy
    """
    if not issubclass(strategy_class, Strategy):
        raise TypeError(f"{strategy_class} must inherit from Strategy")

    name = name.lower().replace("-", "_").replace(" ", "_")
    STRATEGY_REGISTRY[name] = strategy_class
    logger.info(f"Registered custom strategy: {name}")
