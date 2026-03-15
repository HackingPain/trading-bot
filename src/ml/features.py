"""Feature engineering for ML signal generation.

Transforms raw OHLCV data into ML-ready feature matrices using
technical indicators, price returns, volatility measures, and lagged features.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..strategies.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Generates ML features from OHLCV DataFrames.

    Uses TechnicalIndicators for base indicator computation, then derives
    additional statistical and lagged features suitable for classification.

    Args:
        indicators: TechnicalIndicators instance (created with defaults if None).
        lag_periods: Number of lag periods to generate for key indicators.
    """

    # Key columns to generate lags for
    LAG_COLUMNS = ["rsi", "macd", "macd_histogram", "bb_percent", "atr_norm"]

    def __init__(
        self,
        indicators: Optional[TechnicalIndicators] = None,
        lag_periods: int = 5,
    ):
        self.indicators = indicators or TechnicalIndicators()
        self.lag_periods = lag_periods

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the full feature matrix from an OHLCV DataFrame.

        Expects columns: open, high, low, close, volume.
        Returns a DataFrame with all engineered features. Rows with NaN
        values (due to rolling windows) are retained; the caller should
        decide whether to drop them.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.

        Returns:
            DataFrame with feature columns (original OHLCV columns removed).
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to build_features")
            return pd.DataFrame()

        features = pd.DataFrame(index=df.index)

        # --- Technical indicator features ---
        features = self._add_indicator_features(df, features)

        # --- Return features ---
        features = self._add_return_features(df, features)

        # --- Volatility features ---
        features = self._add_volatility_features(df, features)

        # --- Volume features ---
        features = self._add_volume_features(df, features)

        # --- Price position features ---
        features = self._add_price_position_features(df, features)

        # --- Lagged features ---
        features = self._add_lagged_features(features)

        logger.info(
            f"Built {len(features.columns)} features from {len(df)} rows "
            f"({features.dropna().shape[0]} complete rows)"
        )
        return features

    def build_target(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        threshold: float = 0.02,
    ) -> pd.Series:
        """
        Build a binary classification target.

        The target is 1 if the maximum close price within the next
        `forward_days` trading days is at least `threshold` percent
        above the current close, and 0 otherwise.

        Args:
            df: OHLCV DataFrame (must contain 'close' column).
            forward_days: Number of days to look ahead.
            threshold: Minimum fractional price increase (0.02 = 2%).

        Returns:
            Series of 0/1 labels aligned to df.index. The last
            `forward_days` rows will be NaN (no future data).
        """
        close = df["close"]

        # Rolling max of future closes over the forward window
        future_max = close.shift(-1).rolling(window=forward_days, min_periods=1).max()
        # Shift back so each row sees its own forward window
        # We need a custom approach: for each row, look at next forward_days closes
        future_max = (
            close[::-1]
            .rolling(window=forward_days, min_periods=1)
            .max()[::-1]
            .shift(-1)
        )

        future_return = (future_max - close) / close
        target = (future_return >= threshold).astype(float)

        # Mark the tail as NaN (no complete forward window)
        target.iloc[-forward_days:] = np.nan

        logger.info(
            f"Built target: forward_days={forward_days}, threshold={threshold}, "
            f"positive_rate={target.dropna().mean():.3f}"
        )
        return target.rename("target")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_indicator_features(
        self, df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add features derived from technical indicators."""
        ind = self.indicators

        # RSI
        features["rsi"] = ind.calculate_rsi(df)

        # MACD components
        macd, macd_signal, macd_hist = ind.calculate_macd(df)
        features["macd"] = macd
        features["macd_signal"] = macd_signal
        features["macd_histogram"] = macd_hist

        # Bollinger %B
        features["bb_percent"] = ind.calculate_bollinger_percent(df)

        # ATR normalised by close price
        atr = ind.calculate_atr(df)
        features["atr"] = atr
        features["atr_norm"] = atr / df["close"]

        # EMA ratios (close / EMA)
        ema_12 = ind.calculate_ema(df, period=12)
        ema_26 = ind.calculate_ema(df, period=26)
        features["ema_12_ratio"] = df["close"] / ema_12
        features["ema_26_ratio"] = df["close"] / ema_26
        features["ema_cross_ratio"] = ema_12 / ema_26

        return features

    def _add_return_features(
        self, df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add price return features over multiple horizons."""
        close = df["close"]

        for period in [1, 5, 10]:
            features[f"return_{period}d"] = close.pct_change(periods=period)

        return features

    def _add_volatility_features(
        self, df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add rolling volatility features."""
        log_returns = np.log(df["close"] / df["close"].shift(1))

        for window in [5, 20]:
            features[f"volatility_{window}d"] = log_returns.rolling(
                window=window
            ).std() * np.sqrt(252)  # annualised

        return features

    def _add_volume_features(
        self, df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add volume-based features."""
        volume = df["volume"].astype(float)

        # Current volume relative to 20-day average
        vol_ma_20 = volume.rolling(window=20).mean()
        features["volume_ratio"] = volume / vol_ma_20

        # Volume trend (5-day vs 20-day average)
        vol_ma_5 = volume.rolling(window=5).mean()
        features["volume_trend"] = vol_ma_5 / vol_ma_20

        return features

    def _add_price_position_features(
        self, df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Add features measuring price position relative to moving averages."""
        close = df["close"]
        ind = self.indicators

        sma_20 = ind.calculate_sma(df, period=20)
        sma_50 = ind.calculate_sma(df, period=50)
        sma_200 = ind.calculate_sma(df, period=200)

        features["price_sma20_ratio"] = close / sma_20
        features["price_sma50_ratio"] = close / sma_50
        features["price_sma200_ratio"] = close / sma_200

        # Distance from 52-week (252 trading day) high/low
        high_252 = close.rolling(window=252, min_periods=20).max()
        low_252 = close.rolling(window=252, min_periods=20).min()
        features["pct_from_high_252"] = (close - high_252) / high_252
        features["pct_from_low_252"] = (close - low_252) / low_252

        return features

    def _add_lagged_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add lagged values of key indicator columns."""
        for col in self.LAG_COLUMNS:
            if col not in features.columns:
                continue
            for lag in range(1, self.lag_periods + 1):
                features[f"{col}_lag{lag}"] = features[col].shift(lag)

        return features
