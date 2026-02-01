"""Retry utilities with exponential backoff."""

import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: tuple = field(default_factory=lambda: (Exception,))
    on_retry: Optional[Callable[[Exception, int], None]] = None


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
) -> float:
    """Calculate delay with exponential backoff and optional jitter."""
    delay = min(base_delay * (exponential_base ** attempt), max_delay)
    if jitter:
        delay = delay * (0.5 + random.random())  # 50-150% of calculated delay
    return delay


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add randomness to delays
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called on each retry with (exception, attempt)

    Example:
        @retry_with_backoff(max_attempts=3, base_delay=1.0)
        def fetch_data():
            return api.get_data()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        delay = calculate_delay(
                            attempt, base_delay, max_delay, exponential_base, jitter
                        )

                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after error: {e}. Waiting {delay:.2f}s"
                        )

                        if on_retry:
                            on_retry(e, attempt + 1)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            # Re-raise the last exception after all attempts exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected state in retry for {func.__name__}")

        return wrapper
    return decorator


class RetryableClient:
    """
    Mixin class for adding retry capabilities to API clients.

    Example:
        class MyAPIClient(RetryableClient):
            def __init__(self):
                super().__init__(RetryConfig(max_attempts=3))

            def fetch_data(self):
                return self._retry(self._raw_fetch_data)
    """

    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self._retry_config = retry_config or RetryConfig()

    def _retry(
        self,
        func: Callable[..., T],
        *args: Any,
        config: Optional[RetryConfig] = None,
        **kwargs: Any,
    ) -> T:
        """Execute a function with retry logic."""
        cfg = config or self._retry_config
        last_exception: Optional[Exception] = None

        for attempt in range(cfg.max_attempts):
            try:
                return func(*args, **kwargs)
            except cfg.retryable_exceptions as e:
                last_exception = e

                if attempt < cfg.max_attempts - 1:
                    delay = calculate_delay(
                        attempt,
                        cfg.base_delay,
                        cfg.max_delay,
                        cfg.exponential_base,
                        cfg.jitter,
                    )

                    logger.warning(
                        f"Retry {attempt + 1}/{cfg.max_attempts} for {func.__name__} "
                        f"after error: {e}. Waiting {delay:.2f}s"
                    )

                    if cfg.on_retry:
                        cfg.on_retry(e, attempt + 1)

                    time.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected state in retry")


# Common exception sets for different APIs
NETWORK_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

try:
    import requests
    REQUESTS_EXCEPTIONS = (
        requests.RequestException,
        requests.ConnectionError,
        requests.Timeout,
    )
except ImportError:
    REQUESTS_EXCEPTIONS = NETWORK_EXCEPTIONS
