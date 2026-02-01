# 📈 Stock Trading Bot

An automated stock trading bot with paper trading support, multiple strategies, and a web dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Download & Install

**Option A: One-Command Install (Mac/Linux)**
```bash
git clone https://github.com/HackingPain/trading-bot.git && cd trading-bot && ./install.sh
```

**Option B: Windows**
1. Download this repository (Code → Download ZIP)
2. Extract the ZIP file
3. Open Command Prompt in the folder
4. Run: `python scripts/setup.py`

### Step 2: Get Your FREE API Keys

1. Go to **[Alpaca Markets](https://app.alpaca.markets/signup)** (completely free!)
2. Create an account (takes 2 minutes)
3. Navigate to **Paper Trading** → **API Keys**
4. Click **Generate New Keys**
5. Copy your API Key and Secret Key

### Step 3: Start the Bot

```bash
./start.sh
```

This opens an **interactive menu** where you can:
- ✅ Configure your API keys
- ✅ Start the trading bot
- ✅ Open the web dashboard
- ✅ Run backtests

---

## ✨ What Can It Do?

| Feature | Description |
|---------|-------------|
| 🤖 **Automated Trading** | Runs on autopilot, buys and sells based on signals |
| 📊 **Web Dashboard** | Beautiful interface to monitor your trades |
| 📈 **5 Strategies** | RSI/MACD, Mean Reversion, Momentum, Breakout, Pairs |
| 🛡️ **Risk Protection** | Stop-losses, position limits, daily loss limits |
| 📉 **Backtesting** | Test strategies on historical data before using real money |
| 🔔 **Notifications** | Get alerts via Discord, Telegram, or Email |
| 💼 **Paper Trading** | Practice with fake money first (this is the default!) |

---

## 🎮 How to Use

### Option 1: Interactive Menu (Recommended)
```bash
./start.sh
```
Just follow the on-screen prompts!

### Option 2: Direct Commands
```bash
./start.sh run          # Start trading bot
./start.sh dashboard    # Open web dashboard
./start.sh backtest     # Run backtest wizard
./start.sh test         # Run single test cycle
```

---

## 📊 Web Dashboard

See your trades, profits, and performance in real-time:

```bash
./start.sh dashboard
```

Then open **http://localhost:8501** in your browser.

The dashboard shows:
- 💰 Total P&L and win rate
- 📈 Performance charts
- 📋 Trade history
- 🎯 Active signals
- 📉 Drawdown analysis

---

## ⚙️ Configuration

### Basic Settings
Edit `config/settings.yaml` to customize:

```yaml
trading:
  paper_mode: true           # Keep TRUE until ready for real money!
  symbols:
    - AAPL
    - MSFT
    - GOOGL

strategy:
  name: daily_profit_taker   # Choose your strategy
  profit_target_pct: 0.02    # Take profits at 2% gain

risk:
  max_position_pct: 0.10     # Max 10% in any single stock
  max_daily_loss_pct: 0.02   # Stop if down 2% for the day
  stop_loss_pct: 0.05        # Exit if a trade loses 5%
```

### Available Strategies

| Strategy | Best For | Description |
|----------|----------|-------------|
| `daily_profit_taker` | Beginners | Uses RSI & MACD signals, takes quick profits |
| `mean_reversion` | Sideways markets | Buys oversold stocks, sells when they recover |
| `momentum` | Trending markets | Follows strong price trends |
| `breakout` | Volatile stocks | Buys when price breaks resistance |
| `pairs_trading` | Advanced | Trades correlated stock pairs |

---

## 🛡️ Safety Features

Your money is protected by multiple safety systems:

1. **📋 Paper Trading Mode** - Uses fake money by default
2. **🚨 Daily Loss Limit** - Stops trading if you lose too much in one day
3. **📊 Position Limits** - Never puts too much in one stock
4. **🛑 Stop-Losses** - Automatically exits losing trades
5. **⚖️ PDT Protection** - Follows Pattern Day Trader rules

---

## ❓ Troubleshooting

### "API keys not configured"
Run `./start.sh` and select option **6** to set up your API keys.

### "Python not found"
Install Python 3.10 or higher:
- **Mac**: `brew install python@3.11`
- **Ubuntu/Debian**: `sudo apt install python3.11`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

### "Module not found" errors
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Bot not trading
1. Check if the market is open (9:30 AM - 4:00 PM ET, Mon-Fri)
2. Verify your API keys are correct
3. Check `logs/trading_bot.log` for errors

---

## 📁 File Structure

```
trading-bot/
├── start.sh              # 👈 START HERE - Interactive launcher
├── install.sh            # One-click installer
├── config/
│   └── settings.yaml     # ⚙️ Your settings (edit this)
├── src/                  # Source code (don't touch unless developing)
├── data/                 # Trade database
└── logs/                 # Log files
```

---

## ⚠️ Important Disclaimer

**This software is for educational purposes only.**

- ✅ Always start with paper trading
- ✅ Never invest money you can't afford to lose
- ✅ Past performance doesn't guarantee future results
- ❌ The authors are not responsible for any financial losses

---

## 🆘 Need Help?

- 📖 [View Documentation](https://github.com/HackingPain/trading-bot/wiki)
- 🐛 [Report a Bug](https://github.com/HackingPain/trading-bot/issues)
- 💬 [Ask a Question](https://github.com/HackingPain/trading-bot/discussions)

---

## 📄 License

MIT License - Free to use, modify, and distribute.

---

<p align="center">
Made with ❤️ for algorithmic trading enthusiasts
</p>
