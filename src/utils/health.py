"""Health monitoring and heartbeat system (4.4)."""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Monitors bot health via heartbeat mechanism.

    If no heartbeat is received within the timeout period,
    triggers a critical alert callback.
    """

    def __init__(
        self,
        heartbeat_timeout_seconds: int = 300,
        on_stale: Optional[Callable[[float], None]] = None,
        check_interval_seconds: int = 30,
    ):
        self._timeout = heartbeat_timeout_seconds
        self._on_stale = on_stale
        self._check_interval = check_interval_seconds
        self._last_heartbeat: Optional[datetime] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()  # Fix #13: thread-safe heartbeat state
        self._alerted = False

    def beat(self) -> None:
        """Record a heartbeat."""
        with self._lock:
            self._last_heartbeat = datetime.utcnow()
            self._alerted = False

    def start(self) -> None:
        """Start the health monitor background thread."""
        if self._running:
            return

        self._running = True
        self._last_heartbeat = datetime.utcnow()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="health-monitor",
        )
        self._monitor_thread.start()
        logger.info(f"Health monitor started (timeout={self._timeout}s)")

    def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Health monitor stopped")

    def _monitor_loop(self) -> None:
        """Background loop checking heartbeat freshness."""
        while self._running:
            try:
                with self._lock:
                    last_hb = self._last_heartbeat
                    already_alerted = self._alerted

                if last_hb:
                    age = (datetime.utcnow() - last_hb).total_seconds()
                    if age > self._timeout and not already_alerted:
                        logger.critical(
                            f"HEARTBEAT STALE: No heartbeat for {age:.0f}s "
                            f"(timeout: {self._timeout}s)"
                        )
                        with self._lock:
                            self._alerted = True
                        if self._on_stale:
                            try:
                                self._on_stale(age)
                            except Exception as e:
                                logger.error(f"Stale heartbeat callback failed: {e}")
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

            time.sleep(self._check_interval)

    def get_status(self) -> dict[str, Any]:
        """Get current health status."""
        now = datetime.utcnow()
        age = None
        if self._last_heartbeat:
            age = (now - self._last_heartbeat).total_seconds()

        healthy = age is not None and age < self._timeout

        return {
            "healthy": healthy,
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            "heartbeat_age_seconds": age,
            "timeout_seconds": self._timeout,
            "monitoring": self._running,
        }
