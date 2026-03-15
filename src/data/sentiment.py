"""News sentiment analysis integration (3.2).

Fetches news sentiment from free APIs and provides sentiment scores
that can be used as filters in trading strategies.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment analysis result for a symbol."""
    symbol: str
    score: float  # -1.0 (very negative) to 1.0 (very positive)
    label: str  # "bearish", "neutral", "bullish"
    article_count: int = 0
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_bearish(self) -> bool:
        return self.score < -0.2

    @property
    def is_bullish(self) -> bool:
        return self.score > 0.2

    @property
    def is_neutral(self) -> bool:
        return -0.2 <= self.score <= 0.2


class SentimentAnalyzer:
    """
    Fetches and analyzes news sentiment for trading symbols.

    Supports multiple data sources:
    - Finnhub news sentiment (free tier)
    - Alpha Vantage news sentiment
    - Fallback: no sentiment (neutral)
    """

    def __init__(
        self,
        finnhub_key: str = "",
        alpha_vantage_key: str = "",
        cache_duration_minutes: int = 30,
    ):
        self.finnhub_key = finnhub_key or os.getenv("FINNHUB_API_KEY", "")
        self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_KEY", "")
        self._cache: dict[str, SentimentScore] = {}
        self._cache_duration = timedelta(minutes=cache_duration_minutes)
        self._last_api_call: float = 0
        self._min_call_interval: float = 1.0  # Rate limiting

    @classmethod
    def from_settings(cls, settings: dict) -> "SentimentAnalyzer":
        """Create from settings dict."""
        api = settings.get("api", {})
        finnhub = api.get("finnhub", {})
        av = api.get("alpha_vantage", {})
        sentiment = settings.get("sentiment", {})

        return cls(
            finnhub_key=finnhub.get("key", ""),
            alpha_vantage_key=av.get("key", ""),
            cache_duration_minutes=sentiment.get("cache_duration_minutes", 30),
        )

    def _rate_limit(self) -> bool:
        """Check rate limit. Returns True if call is allowed, False to skip (Fix #11)."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self._min_call_interval:
            return False  # Skip instead of blocking the trading thread
        self._last_api_call = time.time()
        return True

    def get_sentiment(self, symbol: str) -> SentimentScore:
        """
        Get sentiment score for a symbol.

        Uses cache to avoid excessive API calls.
        Falls back to neutral if no API keys configured.
        """
        # Check cache
        if symbol in self._cache:
            cached = self._cache[symbol]
            age = datetime.utcnow() - cached.timestamp
            if age < self._cache_duration:
                return cached

        # Try Finnhub first
        if self.finnhub_key:
            score = self._get_finnhub_sentiment(symbol)
            if score:
                self._cache[symbol] = score
                return score

        # Try Alpha Vantage
        if self.alpha_vantage_key:
            score = self._get_alpha_vantage_sentiment(symbol)
            if score:
                self._cache[symbol] = score
                return score

        # Fallback: neutral
        return SentimentScore(
            symbol=symbol,
            score=0.0,
            label="neutral",
            source="none",
        )

    def get_batch_sentiment(self, symbols: list[str]) -> dict[str, SentimentScore]:
        """Get sentiment for multiple symbols."""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_sentiment(symbol)
        return results

    def _get_finnhub_sentiment(self, symbol: str) -> Optional[SentimentScore]:
        """Fetch sentiment from Finnhub news API."""
        if not self._rate_limit():
            return None  # Rate limited, return None to try cache or fallback

        try:
            # Get recent company news
            today = datetime.utcnow().strftime("%Y-%m-%d")
            week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": symbol,
                "from": week_ago,
                "to": today,
                "token": self.finnhub_key,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json()

            if not articles:
                return SentimentScore(
                    symbol=symbol,
                    score=0.0,
                    label="neutral",
                    article_count=0,
                    source="finnhub",
                )

            # Finnhub returns sentiment in article metadata
            # Simple heuristic: classify based on headline keywords
            positive_words = {
                "upgrade", "beat", "surge", "rally", "gain", "profit",
                "growth", "strong", "record", "breakout", "bullish",
                "outperform", "buy", "positive", "above",
            }
            negative_words = {
                "downgrade", "miss", "plunge", "drop", "loss", "decline",
                "weak", "warning", "cut", "bearish", "sell", "below",
                "crash", "layoff", "lawsuit", "investigation",
            }

            positive_count = 0
            negative_count = 0
            total = min(len(articles), 20)  # Cap at 20 articles

            for article in articles[:total]:
                headline = (article.get("headline", "") + " " + article.get("summary", "")).lower()
                pos = sum(1 for w in positive_words if w in headline)
                neg = sum(1 for w in negative_words if w in headline)
                positive_count += pos
                negative_count += neg

            total_signals = positive_count + negative_count
            if total_signals == 0:
                score = 0.0
            else:
                score = (positive_count - negative_count) / total_signals

            # Clamp to [-1, 1]
            score = max(-1.0, min(1.0, score))

            if score > 0.2:
                label = "bullish"
            elif score < -0.2:
                label = "bearish"
            else:
                label = "neutral"

            return SentimentScore(
                symbol=symbol,
                score=score,
                label=label,
                article_count=total,
                source="finnhub",
            )

        except Exception as e:
            logger.error(f"Finnhub sentiment fetch failed for {symbol}: {e}")
            return None

    def _get_alpha_vantage_sentiment(self, symbol: str) -> Optional[SentimentScore]:
        """Fetch sentiment from Alpha Vantage news sentiment API."""
        if not self._rate_limit():
            return None

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": self.alpha_vantage_key,
                "limit": 20,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            feed = data.get("feed", [])
            if not feed:
                return SentimentScore(
                    symbol=symbol,
                    score=0.0,
                    label="neutral",
                    article_count=0,
                    source="alpha_vantage",
                )

            # Extract ticker-specific sentiment scores
            scores = []
            for article in feed:
                ticker_sentiments = article.get("ticker_sentiment", [])
                for ts in ticker_sentiments:
                    if ts.get("ticker") == symbol:
                        score = float(ts.get("ticker_sentiment_score", 0))
                        scores.append(score)

            if not scores:
                return SentimentScore(
                    symbol=symbol,
                    score=0.0,
                    label="neutral",
                    article_count=len(feed),
                    source="alpha_vantage",
                )

            avg_score = sum(scores) / len(scores)

            if avg_score > 0.15:
                label = "bullish"
            elif avg_score < -0.15:
                label = "bearish"
            else:
                label = "neutral"

            return SentimentScore(
                symbol=symbol,
                score=avg_score,
                label=label,
                article_count=len(scores),
                source="alpha_vantage",
            )

        except Exception as e:
            logger.error(f"Alpha Vantage sentiment fetch failed for {symbol}: {e}")
            return None

    def should_block_entry(
        self,
        symbol: str,
        side: str = "buy",
        threshold: float = -0.3,
    ) -> tuple[bool, str]:
        """
        Check if sentiment should block a trade entry.

        Returns:
            Tuple of (should_block, reason)
        """
        sentiment = self.get_sentiment(symbol)

        if side.lower() == "buy" and sentiment.score < threshold:
            return True, (
                f"Negative sentiment for {symbol}: "
                f"score={sentiment.score:.2f} ({sentiment.label}, "
                f"{sentiment.article_count} articles from {sentiment.source})"
            )

        return False, ""
