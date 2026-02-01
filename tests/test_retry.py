"""Tests for retry utilities."""

import pytest
import time
from unittest.mock import Mock, patch

from src.utils.retry import (
    retry_with_backoff,
    calculate_delay,
    RetryConfig,
    RetryableClient,
)


class TestCalculateDelay:
    """Tests for delay calculation."""

    def test_exponential_backoff(self):
        """Test that delay increases exponentially."""
        delay_0 = calculate_delay(0, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=False)
        delay_1 = calculate_delay(1, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=False)
        delay_2 = calculate_delay(2, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=False)

        assert delay_0 == 1.0
        assert delay_1 == 2.0
        assert delay_2 == 4.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        delay = calculate_delay(10, base_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False)
        assert delay == 5.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delay."""
        delays = set()
        for _ in range(10):
            delay = calculate_delay(1, base_delay=1.0, max_delay=60.0, exponential_base=2.0, jitter=True)
            delays.add(round(delay, 4))

        # With jitter, we should get different values
        assert len(delays) > 1

    def test_jitter_range(self):
        """Test that jitter keeps delay within 50-150% of base."""
        for _ in range(100):
            delay = calculate_delay(1, base_delay=2.0, max_delay=60.0, exponential_base=2.0, jitter=True)
            # Base delay at attempt 1 is 4.0, jitter should be 2.0-6.0
            assert 2.0 <= delay <= 6.0


class TestRetryWithBackoff:
    """Tests for the retry decorator."""

    def test_success_on_first_attempt(self):
        """Test function succeeds without retry."""
        mock_func = Mock(return_value="success")

        @retry_with_backoff(max_attempts=3)
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_success_after_retry(self):
        """Test function succeeds after initial failures."""
        mock_func = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_raises_after_max_attempts(self):
        """Test exception is raised after all retries exhausted."""
        mock_func = Mock(side_effect=ValueError("persistent error"))

        @retry_with_backoff(max_attempts=3, base_delay=0.01, retryable_exceptions=(ValueError,))
        def test_func():
            return mock_func()

        with pytest.raises(ValueError, match="persistent error"):
            test_func()

        assert mock_func.call_count == 3

    def test_non_retryable_exception_not_retried(self):
        """Test that non-retryable exceptions are raised immediately."""
        mock_func = Mock(side_effect=TypeError("type error"))

        @retry_with_backoff(max_attempts=3, base_delay=0.01, retryable_exceptions=(ValueError,))
        def test_func():
            return mock_func()

        with pytest.raises(TypeError, match="type error"):
            test_func()

        # Should only be called once since TypeError is not retryable
        assert mock_func.call_count == 1

    def test_on_retry_callback_called(self):
        """Test that on_retry callback is called on each retry."""
        callback = Mock()
        exceptions = [Exception("fail1"), Exception("fail2")]
        mock_func = Mock(side_effect=exceptions + ["success"])

        @retry_with_backoff(max_attempts=3, base_delay=0.01, on_retry=callback)
        def test_func():
            return mock_func()

        test_func()

        # Callback should be called twice (for attempt 1 and 2 failures)
        assert callback.call_count == 2
        # Check that callback was called with exception and attempt number
        assert callback.call_args_list[0][0][1] == 1  # First retry, attempt 1
        assert callback.call_args_list[1][0][1] == 2  # Second retry, attempt 2

    def test_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""
        @retry_with_backoff()
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_works_with_arguments(self):
        """Test decorator works with function arguments."""
        @retry_with_backoff(max_attempts=2, base_delay=0.01)
        def add(a, b, c=0):
            return a + b + c

        assert add(1, 2) == 3
        assert add(1, 2, c=3) == 6

    def test_custom_exception_tuple(self):
        """Test with multiple retryable exception types."""
        call_count = 0

        @retry_with_backoff(
            max_attempts=4,
            base_delay=0.01,
            retryable_exceptions=(ValueError, ConnectionError)
        )
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("value error")
            elif call_count == 2:
                raise ConnectionError("connection error")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 3


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retryable_exceptions == (Exception,)
        assert config.on_retry is None

    def test_custom_values(self):
        """Test custom configuration values."""
        callback = Mock()
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
            retryable_exceptions=(ValueError, TypeError),
            on_retry=callback,
        )

        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retryable_exceptions == (ValueError, TypeError)
        assert config.on_retry is callback


class TestRetryableClient:
    """Tests for RetryableClient mixin."""

    def test_retry_method_success(self):
        """Test _retry method with successful function."""
        client = RetryableClient(RetryConfig(max_attempts=3, base_delay=0.01))
        mock_func = Mock(return_value="success")

        result = client._retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_method_with_retries(self):
        """Test _retry method with retries."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        client = RetryableClient(config)

        call_count = [0]
        def failing_then_success():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("fail")
            return "success"

        result = client._retry(failing_then_success)

        assert result == "success"
        assert call_count[0] == 2

    def test_retry_method_with_args(self):
        """Test _retry method passes arguments correctly."""
        client = RetryableClient(RetryConfig(max_attempts=2, base_delay=0.01))

        def add(a, b):
            return a + b

        result = client._retry(add, 3, 4)
        assert result == 7

    def test_retry_method_custom_config(self):
        """Test _retry method with custom config override."""
        client = RetryableClient(RetryConfig(max_attempts=1, base_delay=0.01))
        custom_config = RetryConfig(max_attempts=3, base_delay=0.01)

        call_count = [0]
        def failing_then_success():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("fail")
            return "success"

        result = client._retry(failing_then_success, config=custom_config)

        assert result == "success"
        assert call_count[0] == 3

    def test_default_config(self):
        """Test client with default config."""
        client = RetryableClient()

        assert client._retry_config.max_attempts == 3
        assert client._retry_config.base_delay == 1.0
