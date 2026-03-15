# Stock Trading Bot

An automated stock trading bot with paper trading support, multiple strategies, risk management, backtesting, ML signal confirmation, and a web dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-225%20passing-brightgreen.svg)

---

## Quick Start

### Option A: No API keys needed (simulated broker)

```bash
git clone https://github.com/HackingPain/trading-bot.git
cd trading-bot
pip install -r requirements.txt
```

Set `broker.type: simulated` in `config/settings.yaml`, then:

```bash
make dev
```

### Option B: With Alpaca paper trading

1. Create a free account at [Alpaca Markets](https://app.alpaca.markets/signup)
2. Go to Paper Trading, generate API keys
3. Copy `.env.example` to `.env` and add your keys
4. Run:

```bash
make dev
```

### Option C: Docker

```bash
cp .env.example .env
# Edit .env with your API keys
make build && make run
```

---

## Features

| Feature | Description |
|---------|-------------|
| **7 Strategies** | RSI/MACD, Mean Reversion, Momentum, Breakout, Pairs Trading, Ensemble, Adaptive |
| **Multi-Strategy Ensemble** | Combine multiple strategies with weighted voting |
| **ML Signal Layer** | Gradient boosting classifier for signal confirmation (optional) |
| **Regime Detection** | Auto-adjusts parameters for low/normal/high volatility and crisis markets |
| **Risk Management** | Circuit breaker, position limits, sector exposure, correlation checks, PDT protection |
| **Position Reconciliation** | Syncs local state with broker on startup and periodically |
| **Order Lifecycle Tracking** | Tracks orders from submission through fill/reject/cancel |
| **Walk-Forward Backtesting** | Rolling train/test windows with overfitting detection |
| **Monte Carlo Simulation** | Confidence intervals on strategy returns |
| **Sentiment Analysis** | News sentiment filtering via Finnhub or Alpha Vantage |
| **Audit Log** | Append-only JSON Lines log of every signal, risk check, and trade |
| **Multi-Broker** | Alpaca (paper + live), Simulated (offline), Interactive Brokers (experimental) |
| **Web Dashboard** | Streamlit UI with performance attribution, drawdown analysis, rolling Sharpe |
| **Notifications** | Discord, Telegram, Email alerts |
| **Docker Deployment** | Multi-stage Docker build with health checks |

---

## Strategies

| Strategy | Best For | How It Works |
|----------|----------|--------------|
| `daily_profit_taker` | Default, beginners | RSI oversold + MACD crossover + Bollinger Bands |
| `mean_reversion` | Sideways markets | Buys below lower Bollinger Band, sells at mean |
| `momentum` | Trending markets | EMA crossovers with MACD and volume confirmation |
| `breakout` | Volatile stocks | Buys resistance breaks with volume surge |
| `pairs_trading` | Market-neutral | Statistical arbitrage on correlated pairs |
| `ensemble` | Higher confidence | Weighted voting across multiple child strategies |
| `adaptive` | Changing conditions | Wraps any strategy with regime-aware parameter adjustment |

### Ensemble example

```yaml
strategy:
  name: ensemble
  strategies: [daily_profit_taker, momentum, mean_reversion]
  weights: [0.4, 0.35, 0.25]
  voting_threshold: 0.6
```

---

## Broker Configuration

```yaml
broker:
  type: simulated          # "alpaca", "simulated", or "ibkr"
  initial_capital: 100000  # For simulated broker
```

- **simulated** - No API keys needed. Fully in-memory paper trading with commission and slippage modeling.
- **alpaca** - Connects to Alpaca Markets API. Supports both paper and live trading.
- **ibkr** - Interactive Brokers via ib_insync (experimental).

---

## Risk Management

Multiple layers of protection run before every trade:

1. **Circuit Breaker** - Halts trading if equity drops 2% intraday (realized + unrealized)
2. **PDT Rule** - Limits day trades for accounts under $25k
3. **Account Minimum** - Requires minimum balance to trade
4. **Max Open Positions** - Caps concurrent positions
5. **Position Size Limits** - No single position exceeds 10% of equity
6. **Sector Exposure** - GICS sector concentration limits (200+ stocks mapped)
7. **Correlation Check** - Blocks entries that would push portfolio correlation too high
8. **Per-Trade Risk** - Validates max loss per trade with stop loss
9. **Consecutive Error Halt** - Bot stops itself after 10 straight failures

---

## Web Dashboard

```bash
make dashboard
```

Open http://localhost:8501. Six tabs:

- **Positions** - Open positions with P&L, stops, and targets
- **Trades** - Recent trade history with partial-fill deduplication
- **Performance** - Equity curve with daily P&L overlay
- **Signals** - All generated signals with indicator values
- **Analytics** - Win rate, profit factor, monthly returns, symbol breakdown
- **Attribution** - Per-strategy P&L, day-of-week analysis, drawdown, rolling Sharpe

---

## Backtesting

```bash
make backtest          # Single backtest run
make walk-forward      # Walk-forward analysis with overfitting detection
```

Walk-forward analysis trains on rolling windows and tests out-of-sample, reporting an overfitting ratio. Monte Carlo simulation shuffles trade order to produce confidence intervals.

---

## ML Signal Layer (Optional)

Train a gradient boosting classifier on 30+ features derived from technical indicators:

```bash
pip install scikit-learn joblib
make train-ml
```

Features include RSI, MACD, Bollinger %B, ATR, returns (1/5/10 day), rolling volatility, volume ratios, price position vs moving averages, and lagged indicators. The model outputs probability scores that can filter or scale signal strength.

---

## Development

```bash
make test              # Run 225 tests with coverage
make lint              # Ruff + mypy (hard failures)
make validate          # Validate config file
make dev-once          # Run a single trading cycle
make dev-sim           # Run with simulated broker (no API keys)
```

### Project Structure

```
trading-bot/
├── src/
│   ├── bot.py                 # Main orchestrator
│   ├── strategies/            # 7 trading strategies + mixins
│   ├── execution/             # Alpaca, Simulated, IBKR brokers
│   ├── risk/                  # Risk manager + correlation analyzer
│   ├── data/                  # Market data, sectors, sentiment
│   ├── ml/                    # ML feature engineering + model
│   ├── backtest/              # Engine, metrics, walk-forward, Monte Carlo
│   ├── dashboard/             # Streamlit UI
│   ├── database/              # SQLAlchemy models
│   ├── notifications/         # Discord, Telegram, Email
│   └── utils/                 # Audit log, health monitor, retry
├── tests/                     # 225 tests
├── config/settings.yaml       # All configuration
├── Dockerfile                 # Multi-stage build
├── docker-compose.yml         # Bot + dashboard services
└── Makefile                   # Development commands
```

---

## Configuration Reference

All settings are in `config/settings.yaml`. Key sections:

- `trading` - Symbols, interval, timezone, paper mode
- `broker` - Broker type and broker-specific settings
- `risk` - Position limits, stop loss, circuit breaker, sector/correlation thresholds
- `strategy` - Strategy selection and parameters
- `indicators` - Technical indicator periods
- `api` - Alpaca, Alpha Vantage, Finnhub credentials
- `notifications` - Discord/Telegram/Email configuration
- `logging` - Log level, file path, audit directory

See the file itself for all available options with inline documentation.

---

## Live Trading

The bot runs in paper trading mode by default. To trade with real money:

**1. Run paper mode first.** Watch it for at least a few weeks. Check the dashboard, review the audit log at `logs/audit.jsonl`, and make sure the strategies, risk limits, and position sizing behave the way you expect.

**2. Switch to live mode.** Two changes are required in `config/settings.yaml`:

```yaml
trading:
  paper_mode: false

broker:
  type: alpaca            # Must use a real broker, not simulated
```

**3. Set your live API keys.** In your `.env` file, replace the paper trading keys with your Alpaca live keys. The bot will automatically use the live API endpoint instead of the paper endpoint.

**4. Set the confirmation environment variable.** The bot refuses to start in live mode without this:

```bash
export CONFIRM_LIVE_TRADING=yes
```

**5. Start the bot.**

```bash
make dev
```

### What happens in live mode

- The bot places real orders through the Alpaca live API
- After every entry, broker-side GTC stop loss and take profit orders are placed automatically. These execute even if the bot crashes or loses connectivity.
- When the bot exits a position through its own logic (trailing stop, signal exit, etc.), it cancels the broker-side bracket orders first to prevent orphaned sells.
- The circuit breaker halts all trading if your account equity drops more than 2% intraday (configurable via `risk.max_daily_loss_pct`)
- If the bot encounters 10 consecutive errors, it stops itself and sends a critical notification
- All decisions are logged to `logs/audit.jsonl` for post-trade review

### Switching back to paper mode

Set `paper_mode: true` in the config. No other changes needed. The `CONFIRM_LIVE_TRADING` variable is ignored in paper mode.

---

## Disclaimer

**This software is for educational and research purposes only.**

- Always start with paper trading
- Never invest money you cannot afford to lose
- Past performance does not guarantee future results
- The authors are not responsible for any financial losses

---

## License

MIT License. Free to use, modify, and distribute.
