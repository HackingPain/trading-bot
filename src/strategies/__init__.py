"""Trading strategies and indicators."""

from .base import Signal, SignalType, Strategy, ExitSignal, ExitReason, PositionInfo
from .indicators import TechnicalIndicators
from .daily_profit_taker import DailyProfitTakerStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .breakout import BreakoutStrategy
from .pairs_trading import PairsTradingStrategy
from .factory import (
    get_strategy,
    get_strategy_from_settings,
    list_strategies,
    register_strategy,
    STRATEGY_REGISTRY,
)

__all__ = [
    # Base classes
    "Signal",
    "SignalType",
    "Strategy",
    "ExitSignal",
    "ExitReason",
    "PositionInfo",
    "TechnicalIndicators",
    # Strategies
    "DailyProfitTakerStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "BreakoutStrategy",
    "PairsTradingStrategy",
    # Factory
    "get_strategy",
    "get_strategy_from_settings",
    "list_strategies",
    "register_strategy",
    "STRATEGY_REGISTRY",
]
