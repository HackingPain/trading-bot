"""Technical indicators for trading strategies."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IndicatorValues:
    """Container for current indicator values."""
    # RSI
    rsi: Optional[float] = None

    # MACD
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_percent: Optional[float] = None  # Position within bands (-1 to 1+)

    # Moving Averages
    sma_short: Optional[float] = None
    sma_long: Optional[float] = None
    ema: Optional[float] = None

    # ATR (Average True Range)
    atr: Optional[float] = None

    # Current price for reference
    close: Optional[float] = None


class TechnicalIndicators:
    """
    Calculate technical indicators from OHLCV data.

    All methods are designed to work with pandas DataFrames
    with standard OHLCV columns (open, high, low, close, volume).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        sma_short: int = 20,
        sma_long: int = 50,
        ema_period: int = 12,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
    ):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.sma_short_period = sma_short
        self.sma_long_period = sma_long
        self.ema_period = ema_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period

    @classmethod
    def from_settings(cls, settings: dict) -> "TechnicalIndicators":
        """Create indicators from settings dictionary."""
        ind_settings = settings.get("indicators", {})
        strat_settings = settings.get("strategy", {})

        return cls(
            rsi_period=ind_settings.get("rsi_period", 14),
            macd_fast=ind_settings.get("macd_fast", 12),
            macd_slow=ind_settings.get("macd_slow", 26),
            macd_signal=ind_settings.get("macd_signal", 9),
            sma_short=ind_settings.get("sma_short", 20),
            sma_long=ind_settings.get("sma_long", 50),
            ema_period=ind_settings.get("ema_period", 12),
            bb_period=strat_settings.get("bollinger_period", 20),
            bb_std=strat_settings.get("bollinger_std", 2),
            atr_period=ind_settings.get("atr_period", 14),
        )

    def calculate_rsi(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        period = period or self.rsi_period
        close = df["close"]

        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Returns: (macd_line, signal_line, histogram)
        """
        fast = fast or self.macd_fast
        slow = slow or self.macd_slow
        signal = signal or self.macd_signal_period

        close = df["close"]

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None,
        std: Optional[float] = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Returns: (upper_band, middle_band, lower_band)
        """
        period = period or self.bb_period
        std = std or self.bb_std

        close = df["close"]

        middle = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()

        upper = middle + (rolling_std * std)
        lower = middle - (rolling_std * std)

        return upper, middle, lower

    def calculate_bollinger_percent(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Bollinger %B (position within bands).

        %B = (Price - Lower Band) / (Upper Band - Lower Band)
        Values: 0 = at lower band, 1 = at upper band, <0 or >1 = outside bands
        """
        upper, middle, lower = self.calculate_bollinger_bands(df)
        close = df["close"]

        percent_b = (close - lower) / (upper - lower)
        return percent_b

    def calculate_sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return df["close"].rolling(window=period).mean()

    def calculate_ema(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """Calculate Exponential Moving Average."""
        period = period or self.ema_period
        return df["close"].ewm(span=period, adjust=False).mean()

    def calculate_atr(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = EMA(TR, period)
        """
        period = period or self.atr_period

        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).

        VWAP = cumsum(typical_price * volume) / cumsum(volume)
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return vwap

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        OBV accumulates volume based on price direction.
        """
        close = df["close"]
        volume = df["volume"]

        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        obv = (volume * direction).cumsum()

        return pd.Series(obv, index=df.index)

    def calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Returns: (%K, %D)
        """
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()

        k = 100 * (df["close"] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()

        return k, d

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators as columns to the DataFrame.

        Returns a copy with added indicator columns.
        """
        result = df.copy()

        # RSI
        result["rsi"] = self.calculate_rsi(df)

        # MACD
        macd, signal, hist = self.calculate_macd(df)
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_histogram"] = hist

        # Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(df)
        result["bb_upper"] = upper
        result["bb_middle"] = middle
        result["bb_lower"] = lower
        result["bb_percent"] = self.calculate_bollinger_percent(df)

        # Moving Averages
        result["sma_short"] = self.calculate_sma(df, self.sma_short_period)
        result["sma_long"] = self.calculate_sma(df, self.sma_long_period)
        result["ema"] = self.calculate_ema(df)

        # ATR
        result["atr"] = self.calculate_atr(df)

        # VWAP
        result["vwap"] = self.calculate_vwap(df)

        return result

    def get_current_values(self, df: pd.DataFrame) -> IndicatorValues:
        """
        Get the most recent indicator values.

        Returns an IndicatorValues dataclass with current values.
        """
        if df.empty:
            return IndicatorValues()

        # Add all indicators
        df_with_indicators = self.add_all_indicators(df)
        last = df_with_indicators.iloc[-1]

        return IndicatorValues(
            rsi=last.get("rsi"),
            macd=last.get("macd"),
            macd_signal=last.get("macd_signal"),
            macd_histogram=last.get("macd_histogram"),
            bb_upper=last.get("bb_upper"),
            bb_middle=last.get("bb_middle"),
            bb_lower=last.get("bb_lower"),
            bb_percent=last.get("bb_percent"),
            sma_short=last.get("sma_short"),
            sma_long=last.get("sma_long"),
            ema=last.get("ema"),
            atr=last.get("atr"),
            close=last.get("close"),
        )

    def is_oversold(self, rsi: float, threshold: float = 30) -> bool:
        """Check if RSI indicates oversold condition."""
        return rsi is not None and rsi < threshold

    def is_overbought(self, rsi: float, threshold: float = 70) -> bool:
        """Check if RSI indicates overbought condition."""
        return rsi is not None and rsi > threshold

    def macd_bullish_crossover(self, df: pd.DataFrame) -> bool:
        """Check for MACD bullish crossover (MACD crosses above signal)."""
        macd, signal, _ = self.calculate_macd(df)
        if len(macd) < 2:
            return False

        prev_diff = macd.iloc[-2] - signal.iloc[-2]
        curr_diff = macd.iloc[-1] - signal.iloc[-1]

        return prev_diff < 0 and curr_diff > 0

    def macd_bearish_crossover(self, df: pd.DataFrame) -> bool:
        """Check for MACD bearish crossover (MACD crosses below signal)."""
        macd, signal, _ = self.calculate_macd(df)
        if len(macd) < 2:
            return False

        prev_diff = macd.iloc[-2] - signal.iloc[-2]
        curr_diff = macd.iloc[-1] - signal.iloc[-1]

        return prev_diff > 0 and curr_diff < 0

    def golden_cross(self, df: pd.DataFrame) -> bool:
        """Check for golden cross (short SMA crosses above long SMA)."""
        sma_short = self.calculate_sma(df, self.sma_short_period)
        sma_long = self.calculate_sma(df, self.sma_long_period)

        if len(sma_short) < 2:
            return False

        prev_diff = sma_short.iloc[-2] - sma_long.iloc[-2]
        curr_diff = sma_short.iloc[-1] - sma_long.iloc[-1]

        return prev_diff < 0 and curr_diff > 0

    def death_cross(self, df: pd.DataFrame) -> bool:
        """Check for death cross (short SMA crosses below long SMA)."""
        sma_short = self.calculate_sma(df, self.sma_short_period)
        sma_long = self.calculate_sma(df, self.sma_long_period)

        if len(sma_short) < 2:
            return False

        prev_diff = sma_short.iloc[-2] - sma_long.iloc[-2]
        curr_diff = sma_short.iloc[-1] - sma_long.iloc[-1]

        return prev_diff > 0 and curr_diff < 0

    def price_below_lower_bb(self, df: pd.DataFrame) -> bool:
        """Check if price is below lower Bollinger Band."""
        _, _, lower = self.calculate_bollinger_bands(df)
        return df["close"].iloc[-1] < lower.iloc[-1]

    def price_above_upper_bb(self, df: pd.DataFrame) -> bool:
        """Check if price is above upper Bollinger Band."""
        upper, _, _ = self.calculate_bollinger_bands(df)
        return df["close"].iloc[-1] > upper.iloc[-1]
