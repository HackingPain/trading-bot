"""
Email notification service for the trading bot.

Supports SMTP-based email notifications with HTML formatting.
"""

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

from .alerts import NotificationType

logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    """Email notification configuration."""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    sender_email: str = ""
    recipient_emails: list[str] | None = None
    use_tls: bool = True
    enabled: bool = True

    @classmethod
    def from_settings(cls, settings: dict) -> "EmailConfig":
        """Create config from settings dictionary."""
        email_settings = settings.get("notifications", {}).get("email", {})

        recipient_emails = email_settings.get("recipients", [])
        if isinstance(recipient_emails, str):
            recipient_emails = [recipient_emails]

        return cls(
            smtp_host=email_settings.get("smtp_host") or os.getenv("SMTP_HOST", ""),
            smtp_port=email_settings.get("smtp_port") or int(os.getenv("SMTP_PORT", "587")),
            smtp_user=email_settings.get("smtp_user") or os.getenv("SMTP_USER", ""),
            smtp_password=email_settings.get("smtp_password") or os.getenv("SMTP_PASSWORD", ""),
            sender_email=email_settings.get("sender_email") or os.getenv("SMTP_SENDER", ""),
            recipient_emails=recipient_emails or os.getenv("EMAIL_RECIPIENTS", "").split(","),
            use_tls=email_settings.get("use_tls", True),
            enabled=email_settings.get("enabled", True),
        )

    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return bool(
            self.smtp_host
            and self.smtp_user
            and self.smtp_password
            and self.sender_email
            and self.recipient_emails
        )


class EmailNotifier:
    """
    Sends email notifications for trading events.

    Supports HTML-formatted emails with trade details.
    """

    def __init__(self, config: EmailConfig):
        self.config = config
        self._last_sent: dict[str, datetime] = {}
        self._min_interval_seconds = 60  # Rate limit per notification type

    @classmethod
    def from_settings(cls, settings: dict) -> "EmailNotifier":
        """Create EmailNotifier from settings dictionary."""
        config = EmailConfig.from_settings(settings)
        return cls(config)

    def _rate_limit_check(self, key: str) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        last_sent = self._last_sent.get(key)

        if last_sent is None:
            self._last_sent[key] = now
            return True

        elapsed = (now - last_sent).total_seconds()
        if elapsed >= self._min_interval_seconds:
            self._last_sent[key] = now
            return True

        return False

    def _get_color_for_type(self, notification_type: NotificationType) -> str:
        """Get color code for notification type."""
        colors = {
            NotificationType.TRADE_EXECUTED: "#28a745",  # Green
            NotificationType.STOP_LOSS_TRIGGERED: "#dc3545",  # Red
            NotificationType.TAKE_PROFIT_TRIGGERED: "#28a745",  # Green
            NotificationType.DAILY_SUMMARY: "#007bff",  # Blue
            NotificationType.ERROR: "#dc3545",  # Red
            NotificationType.WARNING: "#ffc107",  # Yellow/Orange
            NotificationType.INFO: "#6c757d",  # Gray
            NotificationType.CIRCUIT_BREAKER: "#dc3545",  # Red
            NotificationType.BOT_STARTED: "#28a745",  # Green
            NotificationType.BOT_STOPPED: "#ffc107",  # Orange
        }
        return colors.get(notification_type, "#6c757d")

    def _build_html_email(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        fields: dict[str, Any] | None = None,
    ) -> str:
        """Build HTML email content."""
        color = self._get_color_for_type(notification_type)

        fields_html = ""
        if fields:
            fields_rows = "".join(
                f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">{k}</td>'
                f'<td style="padding: 8px; border-bottom: 1px solid #eee;">{v}</td></tr>'
                for k, v in fields.items()
            )
            fields_html = f'''
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                {fields_rows}
            </table>
            '''

        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <!-- Header -->
                <div style="background: {color}; color: white; padding: 20px; text-align: center;">
                    <h1 style="margin: 0; font-size: 24px;">Trading Bot</h1>
                </div>

                <!-- Content -->
                <div style="padding: 25px;">
                    <h2 style="color: #333; margin-top: 0; margin-bottom: 15px;">{title}</h2>
                    <p style="color: #555; font-size: 16px; line-height: 1.5;">{message}</p>
                    {fields_html}
                </div>

                <!-- Footer -->
                <div style="background: #f8f9fa; padding: 15px; text-align: center; color: #666; font-size: 12px;">
                    <p style="margin: 0;">
                        {notification_type.value} | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </p>
                    <p style="margin: 5px 0 0 0;">
                        Trading Bot Notification System
                    </p>
                </div>
            </div>
        </body>
        </html>
        '''
        return html

    def _build_text_email(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        fields: dict[str, Any] | None = None,
    ) -> str:
        """Build plain text email content."""
        lines = [
            "=" * 50,
            "TRADING BOT NOTIFICATION",
            "=" * 50,
            "",
            title,
            "-" * len(title),
            "",
            message,
            "",
        ]

        if fields:
            for key, value in fields.items():
                lines.append(f"{key}: {value}")
            lines.append("")

        lines.extend([
            "-" * 50,
            f"Type: {notification_type.value}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
        ])

        return "\n".join(lines)

    def send(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        fields: dict[str, Any] | None = None,
        subject_prefix: str = "[Trading Bot]",
    ) -> bool:
        """
        Send email notification.

        Args:
            notification_type: Type of notification
            title: Email title/heading
            message: Main message content
            fields: Optional dict of additional fields to display
            subject_prefix: Prefix for email subject

        Returns:
            True if email sent successfully
        """
        if not self.config.enabled:
            logger.debug("Email notifications disabled")
            return False

        if not self.config.is_configured():
            logger.warning("Email not configured, skipping notification")
            return False

        # Rate limit check
        if not self._rate_limit_check(notification_type.value):
            logger.debug(f"Email rate limit hit for {notification_type.value}")
            return False

        # Filter out empty recipients
        recipients = [r.strip() for r in self.config.recipient_emails if r.strip()]
        if not recipients:
            logger.warning("No valid email recipients configured")
            return False

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"{subject_prefix} {title}"
            msg["From"] = self.config.sender_email
            msg["To"] = ", ".join(recipients)

            # Attach both plain text and HTML versions
            text_content = self._build_text_email(notification_type, title, message, fields)
            html_content = self._build_html_email(notification_type, title, message, fields)

            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # Send email
            if self.config.use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.config.smtp_user, self.config.smtp_password)
                    server.sendmail(
                        self.config.sender_email,
                        recipients,
                        msg.as_string(),
                    )
            else:
                with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                    server.login(self.config.smtp_user, self.config.smtp_password)
                    server.sendmail(
                        self.config.sender_email,
                        recipients,
                        msg.as_string(),
                    )

            logger.info(f"Email sent: {title} to {len(recipients)} recipients")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"Email authentication failed: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error sending email: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    # Convenience methods

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

        return self.send(
            NotificationType.TRADE_EXECUTED,
            f"Trade Executed: {side.upper()} {symbol} ({mode})",
            f"Successfully executed {side.upper()} order for {quantity} shares of {symbol}.",
            fields={
                "Symbol": symbol,
                "Side": side.upper(),
                "Quantity": quantity,
                "Price": f"${price:.2f}",
                "Total Value": f"${total:.2f}",
                "Mode": mode,
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
            f"Stop Loss Triggered: {symbol}",
            f"Position in {symbol} was automatically closed due to stop loss trigger.",
            fields={
                "Symbol": symbol,
                "Quantity": quantity,
                "Entry Price": f"${entry_price:.2f}",
                "Exit Price": f"${exit_price:.2f}",
                "Loss": f"-${abs(loss):.2f}",
                "Loss %": f"-{abs(loss_pct):.2f}%",
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
            f"Take Profit Hit: {symbol}",
            f"Position in {symbol} was closed after reaching the profit target.",
            fields={
                "Symbol": symbol,
                "Quantity": quantity,
                "Entry Price": f"${entry_price:.2f}",
                "Exit Price": f"${exit_price:.2f}",
                "Profit": f"+${profit:.2f}",
                "Profit %": f"+{profit_pct:.2f}%",
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
        winning_trades: int = 0,
        losing_trades: int = 0,
    ) -> bool:
        """Send end-of-day summary notification."""
        pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
        status = "Profitable Day" if total_pnl >= 0 else "Loss Day"

        return self.send(
            NotificationType.DAILY_SUMMARY,
            f"Daily Summary: {date} - {status}",
            f"Your trading bot completed today with a P&L of {pnl_str} ({total_pnl_pct:+.2f}%).",
            fields={
                "Date": date,
                "Starting Balance": f"${starting_balance:,.2f}",
                "Ending Balance": f"${ending_balance:,.2f}",
                "Daily P&L": pnl_str,
                "Daily P&L %": f"{total_pnl_pct:+.2f}%",
                "Total Trades": trades_count,
                "Winning Trades": winning_trades,
                "Losing Trades": losing_trades,
                "Win Rate": f"{win_rate:.1f}%",
            },
        )

    def notify_error(
        self,
        error_type: str,
        message: str,
        details: Optional[str] = None,
    ) -> bool:
        """Send error notification."""
        fields = {"Error Type": error_type}
        if details:
            fields["Details"] = details[:500]  # Truncate long details

        return self.send(
            NotificationType.ERROR,
            f"Error: {error_type}",
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
            "ALERT: Circuit Breaker Triggered",
            f"Trading has been halted due to: {reason}. No new trades will be executed until the next trading day or manual reset.",
            fields={
                "Reason": reason,
                "Daily Loss": f"-${abs(daily_loss):.2f}",
                "Daily Loss %": f"-{abs(daily_loss_pct):.2f}%",
                "Limit": f"{limit_pct:.2f}%",
                "Action": "Trading Halted",
            },
        )

    def notify_bot_started(self, mode: str, symbols: list[str]) -> bool:
        """Send bot started notification."""
        symbols_display = ", ".join(symbols[:10])
        if len(symbols) > 10:
            symbols_display += f" (+{len(symbols) - 10} more)"

        return self.send(
            NotificationType.BOT_STARTED,
            f"Trading Bot Started ({mode} Mode)",
            f"Your trading bot has started in {mode} mode and is now monitoring {len(symbols)} symbols.",
            fields={
                "Mode": mode,
                "Symbols": symbols_display,
                "Start Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    def notify_bot_stopped(self, reason: str = "Manual shutdown") -> bool:
        """Send bot stopped notification."""
        return self.send(
            NotificationType.BOT_STOPPED,
            "Trading Bot Stopped",
            f"Your trading bot has been stopped. Reason: {reason}",
            fields={
                "Reason": reason,
                "Stop Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    def test_email(self) -> bool:
        """Send a test email to verify configuration."""
        return self.send(
            NotificationType.INFO,
            "Test Email",
            "This is a test email from your trading bot. If you received this, your email notifications are configured correctly!",
            fields={
                "Status": "Configuration Verified",
                "SMTP Host": self.config.smtp_host,
                "Sender": self.config.sender_email,
            },
        )
