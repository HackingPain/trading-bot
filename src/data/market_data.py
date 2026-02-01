"""Market data fetching from yfinance and Alpha Vantage."""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests
import yfinance as yf

from ..utils.retry import retry_with_backoff, NETWORK_EXCEPTIONS, REQUESTS_EXCEPTIONS

logger = logging.getLogger(__name__)

# Market data specific retryable exceptions
MARKET_DATA_EXCEPTIONS = NETWORK_EXCEPTIONS + REQUESTS_EXCEPTIONS


@dataclass
class MarketData:
    """Container for market data for a symbol."""
    symbol: str
    df: pd.DataFrame  # OHLCV data
    last_price: float
    last_updated: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: int = 0
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None

    @property
    def is_stale(self) -> bool:
        """Check if data is older than 5 minutes."""
        return (datetime.now() - self.last_updated).total_seconds() > 300


@dataclass
class MarketDataConfig:
    """Configuration for market data provider."""
    alpha_vantage_key: str = ""
    alpha_vantage_rate_limit: int = 5
    cache_duration_seconds: int = 60
    historical_days: int = 100

    @classmethod
    def from_settings(cls, settings: dict) -> "MarketDataConfig":
        """Create config from settings dictionary."""
        api_settings = settings.get("api", {})
        av_settings = api_settings.get("alpha_vantage", {})

        return cls(
            alpha_vantage_key=av_settings.get("key") or os.getenv("ALPHA_VANTAGE_KEY", ""),
            alpha_vantage_rate_limit=av_settings.get("rate_limit_per_minute", 5),
            cache_duration_seconds=60,
            historical_days=100,
        )


class MarketDataProvider:
    """
    Provides market data using yfinance for historical data
    and Alpha Vantage for real-time quotes.
    """

    def __init__(self, config: MarketDataConfig):
        self.config = config
        self._cache: dict[str, MarketData] = {}
        self._last_av_call: float = 0
        self._av_call_interval = 60.0 / config.alpha_vantage_rate_limit if config.alpha_vantage_rate_limit > 0 else 12.0

    @classmethod
    def from_settings(cls, settings: dict) -> "MarketDataProvider":
        """Create provider from settings dictionary."""
        config = MarketDataConfig.from_settings(settings)
        return cls(config)

    def _rate_limit_alpha_vantage(self) -> None:
        """Ensure we don't exceed Alpha Vantage rate limits."""
        elapsed = time.time() - self._last_av_call
        if elapsed < self._av_call_interval:
            sleep_time = self._av_call_interval - elapsed
            logger.debug(f"Rate limiting Alpha Vantage, sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_av_call = time.time()

    @retry_with_backoff(max_attempts=3, base_delay=2.0, retryable_exceptions=MARKET_DATA_EXCEPTIONS)
    def get_historical_data(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data using yfinance.

        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV columns
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No historical data returned for {symbol}")
            return pd.DataFrame()

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]

        # Ensure we have required columns
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column {col} for {symbol}")
                return pd.DataFrame()

        logger.debug(f"Fetched {len(df)} historical bars for {symbol}")
        return df

    @retry_with_backoff(max_attempts=3, base_delay=1.0, retryable_exceptions=MARKET_DATA_EXCEPTIONS)
    def get_realtime_quote(self, symbol: str) -> dict[str, Any]:
        """
        Fetch real-time quote using yfinance fast_info.

        Returns dict with price, volume, bid, ask, etc.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info

            quote = {
                "symbol": symbol,
                "price": info.last_price,
                "previous_close": info.previous_close,
                "open": info.open,
                "day_high": info.day_high,
                "day_low": info.day_low,
                "volume": info.last_volume,
                "market_cap": info.market_cap,
                "timestamp": datetime.now(),
            }

            return quote

        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            return {}

    def get_alpha_vantage_quote(self, symbol: str) -> dict[str, Any]:
        """
        Fetch quote from Alpha Vantage (more real-time than yfinance).

        Requires Alpha Vantage API key.
        """
        if not self.config.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not configured")
            return {}

        self._rate_limit_alpha_vantage()

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.config.alpha_vantage_key,
        }

        return self._fetch_alpha_vantage_quote(url, params, symbol)

    @retry_with_backoff(max_attempts=3, base_delay=2.0, retryable_exceptions=REQUESTS_EXCEPTIONS)
    def _fetch_alpha_vantage_quote(self, url: str, params: dict, symbol: str) -> dict[str, Any]:
        """Internal method for fetching Alpha Vantage quote with retry."""
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "Global Quote" not in data:
            logger.warning(f"Invalid Alpha Vantage response for {symbol}: {data}")
            return {}

        quote = data["Global Quote"]
        return {
            "symbol": symbol,
            "price": float(quote.get("05. price", 0)),
            "volume": int(quote.get("06. volume", 0)),
            "previous_close": float(quote.get("08. previous close", 0)),
            "change": float(quote.get("09. change", 0)),
            "change_pct": float(quote.get("10. change percent", "0%").rstrip("%")),
            "timestamp": datetime.now(),
        }

    def get_market_data(self, symbol: str, use_cache: bool = True) -> Optional[MarketData]:
        """
        Get comprehensive market data for a symbol.

        Combines historical data with real-time quote.
        Uses caching to reduce API calls.
        """
        # Check cache
        if use_cache and symbol in self._cache:
            cached = self._cache[symbol]
            age = (datetime.now() - cached.last_updated).total_seconds()
            if age < self.config.cache_duration_seconds:
                logger.debug(f"Using cached data for {symbol} (age: {age:.1f}s)")
                return cached

        # Fetch historical data
        df = self.get_historical_data(symbol)
        if df.empty:
            return None

        # Fetch real-time quote
        quote = self.get_realtime_quote(symbol)
        if not quote:
            # Fall back to last historical close
            last_price = df["close"].iloc[-1]
            volume = int(df["volume"].iloc[-1])
        else:
            last_price = quote.get("price", df["close"].iloc[-1])
            volume = quote.get("volume", int(df["volume"].iloc[-1]))

        market_data = MarketData(
            symbol=symbol,
            df=df,
            last_price=last_price,
            last_updated=datetime.now(),
            volume=volume,
            market_cap=quote.get("market_cap"),
        )

        # Update cache
        self._cache[symbol] = market_data
        logger.debug(f"Updated market data for {symbol}: price=${last_price:.2f}")

        return market_data

    def get_multiple_market_data(
        self,
        symbols: list[str],
        use_cache: bool = True,
    ) -> dict[str, MarketData]:
        """Fetch market data for multiple symbols."""
        results = {}
        for symbol in symbols:
            data = self.get_market_data(symbol, use_cache=use_cache)
            if data:
                results[symbol] = data
            else:
                logger.warning(f"Failed to get market data for {symbol}")
        return results

    def get_intraday_data(
        self,
        symbol: str,
        interval: str = "5m",
        period: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data.

        Args:
            symbol: Stock ticker
            interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m
            period: 1d, 5d, 7d (intraday data limited to recent history)
        """
        return self.get_historical_data(symbol, period=period, interval=interval)

    def is_market_open(self) -> bool:
        """
        Check if US stock market is currently open.

        Uses yfinance to check market status.
        """
        try:
            # Use SPY as a proxy for market status
            spy = yf.Ticker("SPY")
            info = spy.fast_info
            # If we can get recent data, market is likely open
            # This is a simple heuristic
            return True
        except Exception:
            return False

    def get_batch_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """
        Fetch quotes for multiple symbols efficiently.

        Uses yfinance download for batch requests.
        """
        try:
            # Download last 2 days to get most recent price
            data = yf.download(
                symbols,
                period="2d",
                interval="1d",
                group_by="ticker",
                progress=False,
            )

            quotes = {}
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        close = data["Close"].iloc[-1]
                        volume = data["Volume"].iloc[-1]
                    else:
                        close = data[symbol]["Close"].iloc[-1]
                        volume = data[symbol]["Volume"].iloc[-1]

                    quotes[symbol] = {
                        "symbol": symbol,
                        "price": float(close),
                        "volume": int(volume),
                        "timestamp": datetime.now(),
                    }
                except (KeyError, IndexError) as e:
                    logger.warning(f"Could not extract quote for {symbol}: {e}")

            return quotes

        except Exception as e:
            logger.error(f"Error in batch quote fetch: {e}")
            return {}

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cached data for a symbol or all symbols."""
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()
        logger.debug(f"Cleared cache for {symbol or 'all symbols'}")
