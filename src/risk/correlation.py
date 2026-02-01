"""
Multi-symbol correlation analysis for risk management.

Prevents concentrated risk from highly correlated positions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of correlation check between symbols."""
    symbol1: str
    symbol2: str
    correlation: float
    is_high_correlation: bool
    message: str

    def __repr__(self) -> str:
        status = "HIGH" if self.is_high_correlation else "OK"
        return f"<Correlation {self.symbol1}-{self.symbol2}: {self.correlation:.3f} [{status}]>"


@dataclass
class CorrelationCheckResult:
    """Result of correlation check for a proposed trade."""
    passed: bool
    symbol: str
    high_correlations: list[CorrelationResult] = field(default_factory=list)
    message: str = ""
    average_correlation: float = 0.0

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "BLOCKED"
        return f"<CorrelationCheck {status} for {self.symbol}: {self.message}>"


class CorrelationAnalyzer:
    """
    Analyzes correlation between symbols to prevent concentrated portfolio risk.

    High correlation between positions means they tend to move together,
    increasing overall portfolio risk.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.70,
        lookback_periods: int = 60,
        min_periods: int = 20,
    ):
        """
        Initialize correlation analyzer.

        Args:
            correlation_threshold: Correlation above this is considered high (0-1)
            lookback_periods: Number of periods to use for correlation calculation
            min_periods: Minimum periods required for valid correlation
        """
        self.correlation_threshold = correlation_threshold
        self.lookback_periods = lookback_periods
        self.min_periods = min_periods

        # Cache for price data and correlation matrix
        self._price_cache: dict[str, pd.Series] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._matrix_symbols: list[str] = []

    def update_prices(self, symbol: str, prices: pd.Series) -> None:
        """
        Update price data for a symbol.

        Args:
            symbol: Stock symbol
            prices: Series of closing prices (index should be dates)
        """
        # Keep only the lookback period
        self._price_cache[symbol] = prices.tail(self.lookback_periods)
        # Invalidate correlation matrix when prices update
        self._correlation_matrix = None

    def update_prices_batch(self, price_data: dict[str, pd.Series]) -> None:
        """
        Update prices for multiple symbols.

        Args:
            price_data: Dict mapping symbol to price series
        """
        for symbol, prices in price_data.items():
            self._price_cache[symbol] = prices.tail(self.lookback_periods)
        self._correlation_matrix = None

    def _build_correlation_matrix(self) -> pd.DataFrame:
        """Build correlation matrix from cached price data."""
        if len(self._price_cache) < 2:
            return pd.DataFrame()

        # Build price dataframe
        price_df = pd.DataFrame(self._price_cache)

        # Calculate returns
        returns_df = price_df.pct_change().dropna()

        if len(returns_df) < self.min_periods:
            logger.warning(
                f"Insufficient data for correlation ({len(returns_df)} < {self.min_periods})"
            )
            return pd.DataFrame()

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        self._correlation_matrix = corr_matrix
        self._matrix_symbols = list(corr_matrix.columns)

        return corr_matrix

    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """
        Get correlation between two symbols.

        Returns:
            Correlation coefficient (-1 to 1) or None if not available
        """
        if symbol1 == symbol2:
            return 1.0

        if symbol1 not in self._price_cache or symbol2 not in self._price_cache:
            return None

        if self._correlation_matrix is None:
            self._build_correlation_matrix()

        if self._correlation_matrix is None or self._correlation_matrix.empty:
            return None

        if symbol1 in self._correlation_matrix.index and symbol2 in self._correlation_matrix.columns:
            return float(self._correlation_matrix.loc[symbol1, symbol2])

        return None

    def check_correlation(
        self,
        proposed_symbol: str,
        held_symbols: list[str],
    ) -> CorrelationCheckResult:
        """
        Check if a proposed trade would create high correlation risk.

        Args:
            proposed_symbol: Symbol of proposed trade
            held_symbols: List of symbols currently held

        Returns:
            CorrelationCheckResult indicating if trade should proceed
        """
        if not held_symbols:
            return CorrelationCheckResult(
                passed=True,
                symbol=proposed_symbol,
                message="No existing positions to check",
            )

        if proposed_symbol in held_symbols:
            return CorrelationCheckResult(
                passed=False,
                symbol=proposed_symbol,
                message=f"Already holding {proposed_symbol}",
            )

        # Check correlation with each held position
        high_correlations = []
        correlations = []

        for held_symbol in held_symbols:
            corr = self.get_correlation(proposed_symbol, held_symbol)

            if corr is not None:
                correlations.append(corr)
                is_high = abs(corr) >= self.correlation_threshold

                result = CorrelationResult(
                    symbol1=proposed_symbol,
                    symbol2=held_symbol,
                    correlation=corr,
                    is_high_correlation=is_high,
                    message=f"Correlation with {held_symbol}: {corr:.3f}",
                )

                if is_high:
                    high_correlations.append(result)
                    logger.warning(
                        f"High correlation detected: {proposed_symbol} <-> {held_symbol} = {corr:.3f}"
                    )

        avg_correlation = np.mean(correlations) if correlations else 0.0

        # Determine if trade should be blocked
        # Block if more than half of positions are highly correlated
        max_high_corr = len(held_symbols) // 2 + 1
        passed = len(high_correlations) < max_high_corr

        if not passed:
            message = (
                f"High correlation with {len(high_correlations)} positions "
                f"(threshold: {self.correlation_threshold})"
            )
        elif high_correlations:
            message = (
                f"Some correlation detected ({len(high_correlations)} high), "
                f"but within limits"
            )
        else:
            message = "Correlation check passed"

        return CorrelationCheckResult(
            passed=passed,
            symbol=proposed_symbol,
            high_correlations=high_correlations,
            message=message,
            average_correlation=avg_correlation,
        )

    def get_correlation_matrix(
        self,
        symbols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get the correlation matrix for specified symbols.

        Args:
            symbols: List of symbols (None for all cached)

        Returns:
            DataFrame with correlation matrix
        """
        if self._correlation_matrix is None:
            self._build_correlation_matrix()

        if self._correlation_matrix is None or self._correlation_matrix.empty:
            return pd.DataFrame()

        if symbols:
            valid_symbols = [s for s in symbols if s in self._correlation_matrix.index]
            return self._correlation_matrix.loc[valid_symbols, valid_symbols]

        return self._correlation_matrix

    def get_portfolio_correlation_risk(
        self,
        held_symbols: list[str],
    ) -> dict[str, Any]:
        """
        Calculate overall portfolio correlation risk.

        Returns:
            Dict with correlation risk metrics
        """
        if len(held_symbols) < 2:
            return {
                "average_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "high_correlation_pairs": [],
                "diversification_score": 1.0,
            }

        matrix = self.get_correlation_matrix(held_symbols)

        if matrix.empty:
            return {
                "average_correlation": None,
                "max_correlation": None,
                "min_correlation": None,
                "high_correlation_pairs": [],
                "diversification_score": None,
                "message": "Insufficient data for correlation analysis",
            }

        # Get upper triangle (excluding diagonal)
        correlations = []
        high_pairs = []

        for i, sym1 in enumerate(held_symbols):
            for sym2 in held_symbols[i+1:]:
                if sym1 in matrix.index and sym2 in matrix.columns:
                    corr = matrix.loc[sym1, sym2]
                    correlations.append(corr)

                    if abs(corr) >= self.correlation_threshold:
                        high_pairs.append({
                            "symbol1": sym1,
                            "symbol2": sym2,
                            "correlation": round(corr, 3),
                        })

        if not correlations:
            return {
                "average_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "high_correlation_pairs": [],
                "diversification_score": 1.0,
            }

        avg_corr = np.mean(correlations)

        # Diversification score: 1.0 = perfectly uncorrelated, 0.0 = perfectly correlated
        diversification_score = 1 - (avg_corr + 1) / 2  # Normalize from [-1,1] to [0,1]

        return {
            "average_correlation": round(float(avg_corr), 3),
            "max_correlation": round(float(max(correlations)), 3),
            "min_correlation": round(float(min(correlations)), 3),
            "high_correlation_pairs": high_pairs,
            "diversification_score": round(float(diversification_score), 3),
            "num_positions": len(held_symbols),
        }

    def suggest_diversified_symbols(
        self,
        held_symbols: list[str],
        candidate_symbols: list[str],
        max_suggestions: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Suggest symbols that would add diversification.

        Args:
            held_symbols: Currently held symbols
            candidate_symbols: Potential symbols to add
            max_suggestions: Maximum number of suggestions

        Returns:
            List of (symbol, average_correlation) tuples, sorted by lowest correlation
        """
        if not held_symbols:
            return [(s, 0.0) for s in candidate_symbols[:max_suggestions]]

        suggestions = []

        for candidate in candidate_symbols:
            if candidate in held_symbols:
                continue

            # Calculate average correlation with held positions
            correlations = []
            for held in held_symbols:
                corr = self.get_correlation(candidate, held)
                if corr is not None:
                    correlations.append(abs(corr))

            if correlations:
                avg_corr = np.mean(correlations)
                suggestions.append((candidate, avg_corr))

        # Sort by lowest correlation (most diversifying)
        suggestions.sort(key=lambda x: x[1])

        return suggestions[:max_suggestions]

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._price_cache.clear()
        self._correlation_matrix = None
        self._matrix_symbols = []
