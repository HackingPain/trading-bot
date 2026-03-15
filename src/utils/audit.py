"""Structured append-only audit log for all trading decisions (4.3)."""

import atexit
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Append-only audit log recording every signal, risk check, order, and fill.

    Writes JSON Lines (one JSON object per line) to an audit log file.
    Each entry includes a timestamp, event type, and event-specific data.
    Uses a persistent file handle with a lock for thread safety (Fix #7).
    """

    def __init__(self, log_dir: str = "logs", filename: str = "audit.jsonl"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / filename
        self._lock = threading.Lock()
        self._file = None
        self._open_file()
        atexit.register(self._close_file)

    def _open_file(self) -> None:
        """Open the audit log file for appending."""
        try:
            self._file = open(self._log_path, "a", encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to open audit log file: {e}")
            self._file = None

    def _close_file(self) -> None:
        """Close the audit log file."""
        if self._file and not self._file.closed:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass

    def _write(self, event_type: str, data: dict[str, Any]) -> None:
        """Write an audit entry to the log file (thread-safe)."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event_type,
            **data,
        }
        with self._lock:
            try:
                if self._file is None or self._file.closed:
                    self._open_file()
                if self._file:
                    self._file.write(json.dumps(entry, default=str) + "\n")
                    self._file.flush()
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a generic event."""
        self._write(event_type, data)

    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        price: float,
        reason: str,
        was_executed: bool = False,
    ) -> None:
        """Log a trading signal."""
        self._write("signal", {
            "symbol": symbol,
            "signal_type": signal_type,
            "strength": strength,
            "price": price,
            "reason": reason,
            "was_executed": was_executed,
        })

    def log_risk_check(
        self,
        symbol: str,
        action: str,
        checks: list[dict[str, Any]],
    ) -> None:
        """Log risk check results."""
        self._write("risk_check", {
            "symbol": symbol,
            "action": action,
            "checks": checks,
            "all_passed": all(c.get("passed", False) for c in checks),
        })

    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: str,
        reason: str,
        realized_pnl: Optional[float] = None,
    ) -> None:
        """Log a trade execution."""
        self._write("trade", {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            "reason": reason,
            "realized_pnl": realized_pnl,
        })

    def log_order_status_change(
        self,
        order_id: str,
        symbol: str,
        old_status: str,
        new_status: str,
        filled_qty: float = 0,
    ) -> None:
        """Log an order status transition."""
        self._write("order_status_change", {
            "order_id": order_id,
            "symbol": symbol,
            "old_status": old_status,
            "new_status": new_status,
            "filled_qty": filled_qty,
        })

    def log_order_cancelled(
        self,
        order_id: str,
        symbol: str,
        reason: str,
    ) -> None:
        """Log an order cancellation."""
        self._write("order_cancelled", {
            "order_id": order_id,
            "symbol": symbol,
            "reason": reason,
        })

    def log_reconciliation(
        self,
        symbol: str,
        action: str,
        broker_qty: float,
        db_qty: float,
    ) -> None:
        """Log a position reconciliation event."""
        self._write("reconciliation", {
            "symbol": symbol,
            "action": action,
            "broker_qty": broker_qty,
            "db_qty": db_qty,
        })
