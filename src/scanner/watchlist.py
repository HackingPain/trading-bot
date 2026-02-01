"""
Watchlist scanner for identifying trading opportunities.

Screens stocks based on technical criteria and alerts on potential setups.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd

from ..data.market_data import MarketDataProvider
from ..strategies.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class ScanType(str, Enum):
    """Types of scans."""
    OVERSOLD = "oversold"
    OVERBOUGHT = "overbought"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    VOLUME_SPIKE = "volume_spike"
    MACD_BULLISH = "macd_bullish"
    MACD_BEARISH = "macd_bearish"
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    NEW_HIGH = "new_high"
    NEW_LOW = "new_low"
    BOLLINGER_SQUEEZE = "bollinger_squeeze"
    CUSTOM = "custom"


@dataclass
class ScanResult:
    """Result of a scan for a single symbol."""
    symbol: str
    scan_type: ScanType
    triggered: bool
    score: float  # 0.0 to 1.0
    price: float
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        status = "TRIGGERED" if self.triggered else "NO_MATCH"
        return f"<ScanResult {self.symbol} {self.scan_type.value}: {status} (score={self.score:.2f})>"


@dataclass
class ScanAlert:
    """Alert generated from a scan."""
    symbol: str
    scan_type: ScanType
    message: str
    price: float
    score: float
    indicators: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: str = "normal"  # low, normal, high


@dataclass
class ScanConfig:
    """Configuration for a scan."""
    scan_type: ScanType
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


class WatchlistScanner:
    """
    Scans watchlist symbols for trading opportunities.

    Features:
    - Multiple predefined scan types
    - Custom scan support
    - Configurable thresholds
    - Alert generation
    """

    def __init__(
        self,
        data_provider: MarketDataProvider,
        watchlist: list[str],
        scan_configs: Optional[list[ScanConfig]] = None,
    ):
        """
        Initialize scanner.

        Args:
            data_provider: Market data provider
            watchlist: List of symbols to scan
            scan_configs: Optional list of scan configurations
        """
        self.data_provider = data_provider
        self.watchlist = [s.upper() for s in watchlist]
        self.indicators = TechnicalIndicators()

        # Default scan configs
        if scan_configs is None:
            self.scan_configs = [
                ScanConfig(ScanType.OVERSOLD, params={"rsi_threshold": 30}),
                ScanConfig(ScanType.OVERBOUGHT, params={"rsi_threshold": 70}),
                ScanConfig(ScanType.VOLUME_SPIKE, params={"volume_multiplier": 2.0}),
                ScanConfig(ScanType.MACD_BULLISH),
                ScanConfig(ScanType.BREAKOUT, params={"lookback": 20}),
            ]
        else:
            self.scan_configs = scan_configs

        # Custom scan functions
        self._custom_scans: dict[str, Callable] = {}

        # Alert history
        self._alerts: list[ScanAlert] = []

    def add_symbol(self, symbol: str) -> None:
        """Add symbol to watchlist."""
        symbol = symbol.upper()
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)

    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from watchlist."""
        symbol = symbol.upper()
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            return True
        return False

    def register_custom_scan(
        self,
        name: str,
        scan_func: Callable[[pd.DataFrame, dict], tuple[bool, float, dict]],
    ) -> None:
        """
        Register a custom scan function.

        Args:
            name: Unique name for the scan
            scan_func: Function that takes (dataframe, params) and returns
                      (triggered: bool, score: float, details: dict)
        """
        self._custom_scans[name] = scan_func

    def _scan_oversold(self, df: pd.DataFrame, params: dict) -> tuple[bool, float, dict]:
        """Scan for RSI oversold condition."""
        threshold = params.get("rsi_threshold", 30)

        indicator_values = self.indicators.get_current_values(df)
        rsi = indicator_values.rsi

        if rsi is None:
            return False, 0.0, {}

        triggered = rsi < threshold
        # Score: lower RSI = higher score
        score = max(0, (threshold - rsi) / threshold) if triggered else 0

        return triggered, score, {"rsi": rsi, "threshold": threshold}

    def _scan_overbought(self, df: pd.DataFrame, params: dict) -> tuple[bool, float, dict]:
        """Scan for RSI overbought condition."""
        threshold = params.get("rsi_threshold", 70)

        indicator_values = self.indicators.get_current_values(df)
        rsi = indicator_values.rsi

        if rsi is None:
            return False, 0.0, {}

        triggered = rsi > threshold
        score = max(0, (rsi - threshold) / (100 - threshold)) if triggered else 0

        return triggered, score, {"rsi": rsi, "threshold": threshold}

    def _scan_volume_spike(self, df: pd.DataFrame, params: dict) -> tuple[bool, float, dict]:
        """Scan for volume spike."""
        multiplier = params.get("volume_multiplier", 2.0)
        lookback = params.get("lookback", 20)

        if "volume" not in df.columns or len(df) < lookback:
            return False, 0.0, {}

        avg_volume = df["volume"].tail(lookback).mean()
        current_volume = df["volume"].iloc[-1]

        if avg_volume <= 0:
            return False, 0.0, {}

        volume_ratio = current_volume / avg_volume
        triggered = volume_ratio >= multiplier

        # Score based on how much volume exceeds threshold
        score = min(1.0, (volume_ratio - 1) / (multiplier * 2)) if triggered else 0

        return triggered, score, {
            "current_volume": current_volume,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio,
        }

    def _scan_macd_bullish(self, df: pd.DataFrame, params: dict) -> tuple[bool, float, dict]:
        """Scan for MACD bullish crossover."""
        df_with_ind = self.indicators.add_all_indicators(df)
        triggered = self.indicators.macd_bullish_crossover(df_with_ind)

        indicator_values = self.indicators.get_current_values(df)
        macd = indicator_values.macd
        signal = indicator_values.macd_signal
        histogram = indicator_values.macd_histogram

        score = 0.7 if triggered else 0

        return triggered, score, {
            "macd": macd,
            "signal": signal,
            "histogram": histogram,
        }

    def _scan_macd_bearish(self, df: pd.DataFrame, params: dict) -> tuple[bool, float, dict]:
        """Scan for MACD bearish crossover."""
        df_with_ind = self.indicators.add_all_indicators(df)
        triggered = self.indicators.macd_bearish_crossover(df_with_ind)

        indicator_values = self.indicators.get_current_values(df)

        score = 0.7 if triggered else 0

        return triggered, score, {
            "macd": indicator_values.macd,
            "signal": indicator_values.macd_signal,
            "histogram": indicator_values.macd_histogram,
        }

    def _scan_breakout(self, df: pd.DataFrame, params: dict) -> tuple[bool, float, dict]:
        """Scan for price breakout above recent high."""
        lookback = params.get("lookback", 20)

        if len(df) < lookback + 1:
            return False, 0.0, {}

        # Exclude current bar for resistance calculation
        resistance = df["high"].iloc[-(lookback+1):-1].max()
        current_price = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        triggered = current_price > resistance and prev_close <= resistance

        if triggered:
            # Score based on how much above resistance
            breakout_pct = (current_price - resistance) / resistance
            score = min(1.0, breakout_pct * 20)  # 5% breakout = max score
        else:
            score = 0

        return triggered, score, {
            "resistance": resistance,
            "current_price": current_price,
            "breakout_pct": (current_price - resistance) / resistance * 100 if resistance else 0,
        }

    def _scan_breakdown(self, df: pd.DataFrame, params: dict) -> tuple[bool, float, dict]:
        """Scan for price breakdown below recent low."""
        lookback = params.get("lookback", 20)

        if len(df) < lookback + 1:
            return False, 0.0, {}

        support = df["low"].iloc[-(lookback+1):-1].min()
        current_price = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        triggered = current_price < support and prev_close >= support

        if triggered:
            breakdown_pct = (support - current_price) / support
            score = min(1.0, breakdown_pct * 20)
        else:
            score = 0

        return triggered, score, {
            "support": support,
            "current_price": current_price,
            "breakdown_pct": (support - current_price) / support * 100 if support else 0,
        }

    def _scan_new_high(self, df: pd.DataFrame, params: dict) -> tuple[bool, float, dict]:
        """Scan for new 52-week high."""
        lookback = params.get("lookback", 252)  # ~1 year of trading days

        if len(df) < lookback:
            lookback = len(df)

        if lookback < 20:
            return False, 0.0, {}

        historical_high = df["high"].tail(lookback).max()
        current_price = df["close"].iloc[-1]

        triggered = current_price >= historical_high

        score = 0.8 if triggered else 0

        return triggered, score, {
            "historical_high": historical_high,
            "current_price": current_price,
        }

    def _scan_new_low(self, df: pd.DataFrame, params: dict) -> tuple[bool, float, dict]:
        """Scan for new 52-week low."""
        lookback = params.get("lookback", 252)

        if len(df) < lookback:
            lookback = len(df)

        if lookback < 20:
            return False, 0.0, {}

        historical_low = df["low"].tail(lookback).min()
        current_price = df["close"].iloc[-1]

        triggered = current_price <= historical_low

        score = 0.8 if triggered else 0

        return triggered, score, {
            "historical_low": historical_low,
            "current_price": current_price,
        }

    def _scan_bollinger_squeeze(self, df: pd.DataFrame, params: dict) -> tuple[bool, float, dict]:
        """Scan for Bollinger Band squeeze (low volatility, potential breakout)."""
        squeeze_threshold = params.get("squeeze_threshold", 0.04)  # 4% band width

        indicator_values = self.indicators.get_current_values(df)

        if indicator_values.bb_upper is None or indicator_values.bb_lower is None:
            return False, 0.0, {}

        current_price = df["close"].iloc[-1]
        band_width = (indicator_values.bb_upper - indicator_values.bb_lower) / current_price

        triggered = band_width < squeeze_threshold

        if triggered:
            score = min(1.0, (squeeze_threshold - band_width) / squeeze_threshold)
        else:
            score = 0

        return triggered, score, {
            "band_width": band_width,
            "bb_upper": indicator_values.bb_upper,
            "bb_lower": indicator_values.bb_lower,
            "squeeze_threshold": squeeze_threshold,
        }

    def _get_scan_function(self, scan_type: ScanType) -> Optional[Callable]:
        """Get the scan function for a scan type."""
        scan_functions = {
            ScanType.OVERSOLD: self._scan_oversold,
            ScanType.OVERBOUGHT: self._scan_overbought,
            ScanType.VOLUME_SPIKE: self._scan_volume_spike,
            ScanType.MACD_BULLISH: self._scan_macd_bullish,
            ScanType.MACD_BEARISH: self._scan_macd_bearish,
            ScanType.BREAKOUT: self._scan_breakout,
            ScanType.BREAKDOWN: self._scan_breakdown,
            ScanType.NEW_HIGH: self._scan_new_high,
            ScanType.NEW_LOW: self._scan_new_low,
            ScanType.BOLLINGER_SQUEEZE: self._scan_bollinger_squeeze,
        }
        return scan_functions.get(scan_type)

    def scan_symbol(
        self,
        symbol: str,
        scan_type: ScanType,
        params: Optional[dict] = None,
    ) -> ScanResult:
        """
        Run a single scan on a symbol.

        Args:
            symbol: Symbol to scan
            scan_type: Type of scan to run
            params: Optional parameters for the scan

        Returns:
            ScanResult
        """
        params = params or {}

        # Get market data
        try:
            market_data = self.data_provider.get_bars(symbol, limit=300)
            if market_data.df.empty:
                return ScanResult(
                    symbol=symbol,
                    scan_type=scan_type,
                    triggered=False,
                    score=0,
                    price=0,
                    details={"error": "No data available"},
                )
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return ScanResult(
                symbol=symbol,
                scan_type=scan_type,
                triggered=False,
                score=0,
                price=0,
                details={"error": str(e)},
            )

        # Get scan function
        if scan_type == ScanType.CUSTOM:
            scan_name = params.get("scan_name")
            if scan_name and scan_name in self._custom_scans:
                scan_func = self._custom_scans[scan_name]
            else:
                return ScanResult(
                    symbol=symbol,
                    scan_type=scan_type,
                    triggered=False,
                    score=0,
                    price=market_data.last_price,
                    details={"error": f"Custom scan '{scan_name}' not found"},
                )
        else:
            scan_func = self._get_scan_function(scan_type)
            if scan_func is None:
                return ScanResult(
                    symbol=symbol,
                    scan_type=scan_type,
                    triggered=False,
                    score=0,
                    price=market_data.last_price,
                    details={"error": f"Scan type '{scan_type}' not implemented"},
                )

        # Run scan
        try:
            triggered, score, details = scan_func(market_data.df, params)

            return ScanResult(
                symbol=symbol,
                scan_type=scan_type,
                triggered=triggered,
                score=score,
                price=market_data.last_price,
                details=details,
            )
        except Exception as e:
            logger.error(f"Error running scan for {symbol}: {e}")
            return ScanResult(
                symbol=symbol,
                scan_type=scan_type,
                triggered=False,
                score=0,
                price=market_data.last_price,
                details={"error": str(e)},
            )

    def scan_watchlist(
        self,
        scan_type: Optional[ScanType] = None,
    ) -> list[ScanResult]:
        """
        Run scans on entire watchlist.

        Args:
            scan_type: Optional specific scan type (runs all enabled if None)

        Returns:
            List of ScanResults (only triggered ones)
        """
        results = []

        for symbol in self.watchlist:
            if scan_type:
                # Run specific scan
                config = next(
                    (c for c in self.scan_configs if c.scan_type == scan_type and c.enabled),
                    None,
                )
                if config:
                    result = self.scan_symbol(symbol, scan_type, config.params)
                    if result.triggered:
                        results.append(result)
            else:
                # Run all enabled scans
                for config in self.scan_configs:
                    if not config.enabled:
                        continue

                    result = self.scan_symbol(symbol, config.scan_type, config.params)
                    if result.triggered:
                        results.append(result)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results

    def generate_alerts(
        self,
        min_score: float = 0.5,
    ) -> list[ScanAlert]:
        """
        Scan watchlist and generate alerts for triggered conditions.

        Args:
            min_score: Minimum score to generate alert

        Returns:
            List of ScanAlerts
        """
        results = self.scan_watchlist()
        alerts = []

        for result in results:
            if result.score < min_score:
                continue

            priority = "high" if result.score >= 0.8 else "normal"

            alert = ScanAlert(
                symbol=result.symbol,
                scan_type=result.scan_type,
                message=f"{result.scan_type.value.upper()} detected for {result.symbol}",
                price=result.price,
                score=result.score,
                indicators=result.details,
                priority=priority,
            )
            alerts.append(alert)

        self._alerts.extend(alerts)
        return alerts

    def get_recent_alerts(self, limit: int = 50) -> list[ScanAlert]:
        """Get recent alerts."""
        return sorted(self._alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def clear_alerts(self) -> None:
        """Clear alert history."""
        self._alerts.clear()

    def get_scan_summary(self) -> dict[str, Any]:
        """Get summary of scanner status."""
        return {
            "watchlist_size": len(self.watchlist),
            "watchlist": self.watchlist,
            "enabled_scans": [c.scan_type.value for c in self.scan_configs if c.enabled],
            "total_alerts": len(self._alerts),
            "custom_scans": list(self._custom_scans.keys()),
        }
