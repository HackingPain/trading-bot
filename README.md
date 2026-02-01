# Stock Trading Bot

A modular personal stock trading bot built in Python 3.10+ with paper trading support, risk management, and real-time monitoring.

## Features

- **Paper Trading**: Test strategies without risking real money
- **Multiple Data Sources**: yfinance for historical data, Alpha Vantage for real-time quotes
- **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA/EMA, ATR, and more
- **Risk Management**: Circuit breakers, position limits, PDT compliance, stop-losses
- **Broker Integration**: Alpaca API support (paper and live)
- **Notifications**: Discord and Telegram alerts
- **Dashboard**: Streamlit-based monitoring interface
- **Database**: SQLite for trade history and performance tracking

## Quick Start

### 1. Install Dependencies

```bash
cd /root/trading-bot
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit `config/settings.yaml` or set environment variables:

```bash
export ALPACA_API_KEY="your_alpaca_key"
export ALPACA_SECRET_KEY="your_alpaca_secret"
export ALPHA_VANTAGE_KEY="your_alpha_vantage_key"  # Optional
export DISCORD_WEBHOOK_URL="your_discord_webhook"  # Optional
export TELEGRAM_BOT_TOKEN="your_telegram_token"    # Optional
export TELEGRAM_CHAT_ID="your_chat_id"             # Optional
```

### 3. Run in Paper Mode

```bash
# Single cycle test
python -m src.bot --once

# Start trading bot
python -m src.bot

# Test notifications
python -m src.bot --test-notifications
```

### 4. Run Backtest

```bash
# Basic backtest
python -m src.cli.backtest -s AAPL MSFT GOOGL

# Custom date range and parameters
python -m src.cli.backtest -s AAPL --start 2023-01-01 --end 2024-01-01 -v
```

### 5. View Dashboard

```bash
streamlit run src/dashboard/app.py
```

Then open http://localhost:8501 in your browser.

## Docker Deployment

```bash
# Build and run all services
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

## Configuration

All settings are in `config/settings.yaml`:

```yaml
trading:
  paper_mode: true         # ALWAYS start with paper trading
  symbols:
    - AAPL
    - MSFT
    - GOOGL
  check_interval_seconds: 60

risk:
  max_position_pct: 0.10   # 10% max per position
  max_daily_loss_pct: 0.02 # 2% daily circuit breaker
  stop_loss_pct: 0.05      # 5% stop loss
  trailing_stop_pct: 0.03  # 3% trailing stop
  max_daily_trades: 3      # PDT rule safety

strategy:
  profit_target_pct: 0.02  # 2% take profit
  rsi_oversold: 30
  rsi_overbought: 70
```

## Project Structure

```
trading-bot/
├── src/
│   ├── bot.py                    # Main orchestrator
│   ├── data/
│   │   └── market_data.py        # Data fetching
│   ├── strategies/
│   │   ├── base.py               # Strategy interface
│   │   ├── daily_profit_taker.py # Main strategy
│   │   └── indicators.py         # Technical indicators
│   ├── risk/
│   │   └── risk_manager.py       # Risk checks
│   ├── execution/
│   │   └── broker.py             # Broker abstraction
│   ├── database/
│   │   └── models.py             # SQLAlchemy models
│   ├── notifications/
│   │   └── alerts.py             # Discord/Telegram
│   └── dashboard/
│       └── app.py                # Streamlit dashboard
├── config/
│   └── settings.yaml             # Configuration
├── tests/                        # Unit tests
├── logs/                         # Runtime logs
└── data/                         # SQLite database
```

## Trading Strategy

The default **Daily Profit Taker** strategy:

### Entry Conditions (BUY)
- RSI below oversold threshold (30)
- MACD bullish crossover or positive histogram
- Price near lower Bollinger Band

### Exit Conditions (SELL)
- Stop loss hit (5% below entry)
- Trailing stop hit (3% below highest price)
- Take profit target reached (2% gain)
- RSI overbought (70) + MACD bearish crossover

## Risk Management

Built-in safety features:

1. **Circuit Breaker**: Trading halts if daily loss exceeds 2%
2. **PDT Compliance**: Limits day trades for accounts under $25k
3. **Position Sizing**: Max 10% of portfolio per position
4. **Stop-Losses**: Required for every trade
5. **Paper Mode Flag**: Checked at multiple levels

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_risk_manager.py -v
```

## Live Trading

⚠️ **WARNING**: Live trading involves real financial risk.

To enable live trading:

1. Set `paper_mode: false` in settings.yaml
2. Update Alpaca URL to production: `https://api.alpaca.markets`
3. Set environment variable: `CONFIRM_LIVE_TRADING=yes`
4. Double-check all risk parameters

```bash
CONFIRM_LIVE_TRADING=yes python -m src.bot
```

## API Reference

### Adding a New Strategy

```python
from src.strategies.base import Strategy, Signal, ExitSignal

class MyStrategy(Strategy):
    def generate_signals(self, market_data, positions):
        signals = []
        # Your logic here
        return signals

    def should_exit(self, position, market_data):
        # Your exit logic here
        return None  # or ExitSignal
```

### Adding a New Broker

```python
from src.execution.broker import Broker

class MyBroker(Broker):
    def get_account(self):
        # Implementation
        pass

    def submit_order(self, order):
        # Implementation
        pass
    # ... other abstract methods
```

## Troubleshooting

### Common Issues

**"Alpaca API credentials not configured"**
- Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` environment variables

**"Circuit breaker triggered"**
- Daily loss limit reached; trading halts until next day
- Check logs for details

**"PDT rule violation"**
- Account under $25k with too many day trades
- Reduce `max_daily_trades` in settings

**Database errors**
- Delete `data/trading_bot.db` to reset
- Ensure `data/` directory exists

### Logs

Check logs for detailed information:
```bash
tail -f logs/trading_bot.log
```

## Disclaimer

This software is for educational purposes only. Trading stocks involves substantial risk of loss. Past performance is not indicative of future results. Always paper trade first and never invest more than you can afford to lose.

## License

MIT License - See LICENSE file for details.
