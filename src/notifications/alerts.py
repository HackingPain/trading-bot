"""Notification services for Discord and Telegram alerts."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications."""
    TRADE_EXECUTED = "trade_executed"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"
    DAILY_SUMMARY = "daily_summary"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    CIRCUIT_BREAKER = "circuit_breaker"
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"


@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    discord_webhook_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    enabled: bool = True
    notify_on: list[str] | None = None

    @classmethod
    def from_settings(cls, settings: dict) -> "NotificationConfig":
        """Create config from settings dictionary."""
        notif_settings = settings.get("notifications", {})
        discord_settings = notif_settings.get("discord", {})
        telegram_settings = notif_settings.get("telegram", {})

        return cls(
            discord_webhook_url=discord_settings.get("webhook_url") or os.getenv("DISCORD_WEBHOOK_URL", ""),
            telegram_bot_token=telegram_settings.get("bot_token") or os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=telegram_settings.get("chat_id") or os.getenv("TELEGRAM_CHAT_ID", ""),
            enabled=notif_settings.get("enabled", True),
            notify_on=discord_settings.get("notify_on") or telegram_settings.get("notify_on"),
        )


class NotificationManager:
    """Manages sending notifications via Discord and Telegram."""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self._rate_limit_timestamps: dict[str, datetime] = {}
        self._min_interval_seconds = 1  # Minimum time between notifications

    @classmethod
    def from_settings(cls, settings: dict) -> "NotificationManager":
        """Create NotificationManager from settings dictionary."""
        config = NotificationConfig.from_settings(settings)
        return cls(config)

    def _should_notify(self, notification_type: NotificationType) -> bool:
        """Check if this notification type should be sent."""
        if not self.config.enabled:
            return False

        if self.config.notify_on is None:
            return True

        return notification_type.value in self.config.notify_on

    def _rate_limit_check(self, key: str) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        last_sent = self._rate_limit_timestamps.get(key)

        if last_sent is None:
            self._rate_limit_timestamps[key] = now
            return True

        elapsed = (now - last_sent).total_seconds()
        if elapsed >= self._min_interval_seconds:
            self._rate_limit_timestamps[key] = now
            return True

        return False

    def send(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        fields: dict[str, Any] | None = None,
        color: int | None = None,
    ) -> bool:
        """Send notification to all configured channels."""
        if not self._should_notify(notification_type):
            logger.debug(f"Notification type {notification_type} not in notify_on list, skipping")
            return False

        success = True

        if self.config.discord_webhook_url:
            if not self._send_discord(notification_type, title, message, fields, color):
                success = False

        if self.config.telegram_bot_token and self.config.telegram_chat_id:
            if not self._send_telegram(notification_type, title, message, fields):
                success = False

        return success

    def _get_color_for_type(self, notification_type: NotificationType) -> int:
        """Get Discord embed color for notification type."""
        colors = {
            NotificationType.TRADE_EXECUTED: 0x00FF00,  # Green
            NotificationType.STOP_LOSS_TRIGGERED: 0xFF0000,  # Red
            NotificationType.TAKE_PROFIT_TRIGGERED: 0x00FF00,  # Green
            NotificationType.DAILY_SUMMARY: 0x0000FF,  # Blue
            NotificationType.ERROR: 0xFF0000,  # Red
            NotificationType.WARNING: 0xFFA500,  # Orange
            NotificationType.INFO: 0x808080,  # Gray
            NotificationType.CIRCUIT_BREAKER: 0xFF0000,  # Red
            NotificationType.BOT_STARTED: 0x00FF00,  # Green
            NotificationType.BOT_STOPPED: 0xFFA500,  # Orange
        }
        return colors.get(notification_type, 0x808080)

    def _send_discord(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        fields: dict[str, Any] | None = None,
        color: int | None = None,
    ) -> bool:
        """Send notification to Discord webhook."""
        if not self._rate_limit_check(f"discord_{notification_type.value}"):
            logger.debug("Discord rate limit hit, skipping notification")
            return False

        if color is None:
            color = self._get_color_for_type(notification_type)

        embed = {
            "title": f"🤖 {title}",
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": f"Trading Bot | {notification_type.value}"},
        }

        if fields:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in fields.items()
            ]

        payload = {"embeds": [embed]}

        try:
            response = requests.post(
                self.config.discord_webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            logger.debug(f"Discord notification sent: {title}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def _send_telegram(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        fields: dict[str, Any] | None = None,
    ) -> bool:
        """Send notification to Telegram."""
        if not self._rate_limit_check(f"telegram_{notification_type.value}"):
            logger.debug("Telegram rate limit hit, skipping notification")
            return False

        # Format message for Telegram (HTML)
        text_parts = [f"<b>🤖 {title}</b>", "", message]

        if fields:
            text_parts.append("")
            for key, value in fields.items():
                text_parts.append(f"<b>{key}:</b> {value}")

        text_parts.append("")
        text_parts.append(f"<i>{notification_type.value}</i>")

        text = "\n".join(text_parts)

        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": text,
            "parse_mode": "HTML",
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.debug(f"Telegram notification sent: {title}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

    # Convenience methods for common notifications

    def notify_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        total: float,
        is_paper: bool = True,
    ) -> bool:
        """Send trade execution notification."""
        mode = "PAPER" if is_paper else "LIVE"
        emoji = "📈" if side.lower() == "buy" else "📉"

        return self.send(
            NotificationType.TRADE_EXECUTED,
            f"{emoji} Trade Executed ({mode})",
            f"**{side.upper()}** {quantity} shares of **{symbol}**",
            fields={
                "Symbol": symbol,
                "Side": side.upper(),
                "Quantity": quantity,
                "Price": f"${price:.2f}",
                "Total": f"${total:.2f}",
            },
        )

    def notify_stop_loss(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        loss: float,
        loss_pct: float,
    ) -> bool:
        """Send stop loss triggered notification."""
        return self.send(
            NotificationType.STOP_LOSS_TRIGGERED,
            "🛑 Stop Loss Triggered",
            f"Position in **{symbol}** closed at stop loss",
            fields={
                "Symbol": symbol,
                "Quantity": quantity,
                "Entry": f"${entry_price:.2f}",
                "Exit": f"${exit_price:.2f}",
                "Loss": f"-${abs(loss):.2f} ({loss_pct:.2f}%)",
            },
        )

    def notify_take_profit(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        profit: float,
        profit_pct: float,
    ) -> bool:
        """Send take profit triggered notification."""
        return self.send(
            NotificationType.TAKE_PROFIT_TRIGGERED,
            "🎯 Take Profit Hit",
            f"Position in **{symbol}** closed at take profit",
            fields={
                "Symbol": symbol,
                "Quantity": quantity,
                "Entry": f"${entry_price:.2f}",
                "Exit": f"${exit_price:.2f}",
                "Profit": f"+${profit:.2f} ({profit_pct:.2f}%)",
            },
        )

    def notify_daily_summary(
        self,
        date: str,
        starting_balance: float,
        ending_balance: float,
        total_pnl: float,
        total_pnl_pct: float,
        trades_count: int,
        win_rate: float,
    ) -> bool:
        """Send end-of-day summary notification."""
        emoji = "📈" if total_pnl >= 0 else "📉"
        pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"

        return self.send(
            NotificationType.DAILY_SUMMARY,
            f"{emoji} Daily Summary - {date}",
            f"Today's P&L: **{pnl_str}** ({total_pnl_pct:+.2f}%)",
            fields={
                "Starting": f"${starting_balance:,.2f}",
                "Ending": f"${ending_balance:,.2f}",
                "Trades": trades_count,
                "Win Rate": f"{win_rate:.1f}%",
            },
        )

    def notify_error(self, error_type: str, message: str, details: str | None = None) -> bool:
        """Send error notification."""
        fields = {"Error Type": error_type}
        if details:
            fields["Details"] = details[:200]  # Truncate long details

        return self.send(
            NotificationType.ERROR,
            "❌ Error Occurred",
            message,
            fields=fields,
        )

    def notify_circuit_breaker(
        self,
        reason: str,
        daily_loss: float,
        daily_loss_pct: float,
        limit_pct: float,
    ) -> bool:
        """Send circuit breaker notification."""
        return self.send(
            NotificationType.CIRCUIT_BREAKER,
            "🚨 Circuit Breaker Triggered",
            f"Trading halted: {reason}",
            fields={
                "Daily Loss": f"-${abs(daily_loss):.2f}",
                "Loss %": f"{daily_loss_pct:.2f}%",
                "Limit": f"{limit_pct:.2f}%",
            },
        )

    def notify_bot_started(self, mode: str, symbols: list[str]) -> bool:
        """Send bot started notification."""
        return self.send(
            NotificationType.BOT_STARTED,
            f"✅ Bot Started ({mode})",
            f"Monitoring {len(symbols)} symbols",
            fields={
                "Mode": mode,
                "Symbols": ", ".join(symbols[:5]) + ("..." if len(symbols) > 5 else ""),
            },
        )

    def notify_bot_stopped(self, reason: str = "Manual shutdown") -> bool:
        """Send bot stopped notification."""
        return self.send(
            NotificationType.BOT_STOPPED,
            "⏹️ Bot Stopped",
            reason,
        )

    def test_notifications(self) -> dict[str, bool]:
        """Send test notifications to verify configuration."""
        results = {}

        if self.config.discord_webhook_url:
            results["discord"] = self._send_discord(
                NotificationType.INFO,
                "Test Notification",
                "This is a test message from your trading bot.",
                {"Status": "Connected"},
            )
        else:
            results["discord"] = False
            logger.warning("Discord webhook URL not configured")

        if self.config.telegram_bot_token and self.config.telegram_chat_id:
            results["telegram"] = self._send_telegram(
                NotificationType.INFO,
                "Test Notification",
                "This is a test message from your trading bot.",
                {"Status": "Connected"},
            )
        else:
            results["telegram"] = False
            logger.warning("Telegram credentials not configured")

        return results
