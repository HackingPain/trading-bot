#!/bin/bash
#
# Trading Bot Launcher
# Simple interface for running the trading bot
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo -e "${RED}Error: Virtual environment not found.${NC}"
    echo "Please run: ./install.sh"
    exit 1
fi

# Check if .env has API keys configured
check_api_keys() {
    if [ ! -f ".env" ]; then
        return 1
    fi

    source .env 2>/dev/null || true

    if [ -z "$ALPACA_API_KEY" ] || [ "$ALPACA_API_KEY" = "your_alpaca_api_key_here" ]; then
        return 1
    fi

    if [ -z "$ALPACA_SECRET_KEY" ] || [ "$ALPACA_SECRET_KEY" = "your_alpaca_secret_key_here" ]; then
        return 1
    fi

    return 0
}

# Setup wizard for API keys
setup_wizard() {
    clear
    echo -e "${BLUE}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                   📝 SETUP WIZARD                         ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "${YELLOW}You need Alpaca API keys to use this bot.${NC}"
    echo ""
    echo "1. Go to: https://app.alpaca.markets/signup"
    echo "2. Create a FREE account"
    echo "3. Go to Paper Trading → API Keys"
    echo "4. Click 'Generate New Keys'"
    echo "5. Copy your API Key and Secret Key"
    echo ""
    echo -e "${CYAN}Enter your API keys below (they will be saved securely):${NC}"
    echo ""

    read -p "Alpaca API Key: " api_key
    read -p "Alpaca Secret Key: " secret_key

    if [ -n "$api_key" ] && [ -n "$secret_key" ]; then
        # Update .env file
        sed -i "s/ALPACA_API_KEY=.*/ALPACA_API_KEY=$api_key/" .env
        sed -i "s/ALPACA_SECRET_KEY=.*/ALPACA_SECRET_KEY=$secret_key/" .env

        echo ""
        echo -e "${GREEN}✓ API keys saved successfully!${NC}"
        echo ""
        read -p "Press Enter to continue..."
    else
        echo ""
        echo -e "${RED}API keys not provided. You can set them later in .env${NC}"
        read -p "Press Enter to continue..."
    fi
}

# Show main menu
show_menu() {
    clear
    echo -e "${BLUE}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║              📈 STOCK TRADING BOT                         ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Check API key status
    if check_api_keys; then
        echo -e "  Status: ${GREEN}● Ready${NC} (API keys configured)"
    else
        echo -e "  Status: ${YELLOW}● Setup Required${NC} (API keys needed)"
    fi
    echo ""
    echo -e "${BOLD}  What would you like to do?${NC}"
    echo ""
    echo -e "  ${GREEN}1)${NC}  🚀 Start Trading Bot"
    echo -e "  ${GREEN}2)${NC}  📊 Open Dashboard"
    echo -e "  ${GREEN}3)${NC}  📈 Run Backtest"
    echo -e "  ${GREEN}4)${NC}  🧪 Test Single Cycle"
    echo -e "  ${GREEN}5)${NC}  ⚙️  Configure Settings"
    echo -e "  ${GREEN}6)${NC}  🔑 Setup API Keys"
    echo -e "  ${GREEN}7)${NC}  📖 View Documentation"
    echo -e "  ${GREEN}8)${NC}  ❌ Exit"
    echo ""
    read -p "  Enter choice [1-8]: " choice

    case $choice in
        1) run_bot ;;
        2) run_dashboard ;;
        3) run_backtest ;;
        4) run_test ;;
        5) configure_settings ;;
        6) setup_wizard; show_menu ;;
        7) show_docs ;;
        8) exit 0 ;;
        *) show_menu ;;
    esac
}

# Run the trading bot
run_bot() {
    if ! check_api_keys; then
        echo -e "${RED}Error: API keys not configured.${NC}"
        echo "Please run option 6 to set up your API keys first."
        read -p "Press Enter to continue..."
        show_menu
        return
    fi

    clear
    echo -e "${GREEN}${BOLD}Starting Trading Bot...${NC}"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""
    python -m src.bot

    echo ""
    read -p "Press Enter to return to menu..."
    show_menu
}

# Run dashboard
run_dashboard() {
    clear
    echo -e "${GREEN}${BOLD}Starting Dashboard...${NC}"
    echo ""
    echo "Opening dashboard at: http://localhost:8501"
    echo "Press Ctrl+C to stop"
    echo ""

    # Try to open browser
    if command -v xdg-open &> /dev/null; then
        (sleep 2 && xdg-open http://localhost:8501) &
    elif command -v open &> /dev/null; then
        (sleep 2 && open http://localhost:8501) &
    fi

    streamlit run src/dashboard/app.py

    show_menu
}

# Run backtest
run_backtest() {
    clear
    echo -e "${BLUE}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                   📈 BACKTEST                             ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo "Enter stock symbols to backtest (comma-separated)"
    echo "Example: AAPL, MSFT, GOOGL"
    echo ""
    read -p "Symbols: " symbols

    if [ -z "$symbols" ]; then
        symbols="AAPL,MSFT"
    fi

    # Convert commas to spaces and clean up
    symbols=$(echo "$symbols" | tr ',' ' ' | tr -s ' ')

    echo ""
    echo "Choose time period:"
    echo "  1) 1 Month"
    echo "  2) 3 Months"
    echo "  3) 6 Months"
    echo "  4) 1 Year"
    echo ""
    read -p "Period [1-4]: " period

    case $period in
        1) days=30 ;;
        2) days=90 ;;
        3) days=180 ;;
        4) days=365 ;;
        *) days=90 ;;
    esac

    start_date=$(date -d "-$days days" +%Y-%m-%d 2>/dev/null || date -v-${days}d +%Y-%m-%d)

    echo ""
    echo -e "${GREEN}Running backtest...${NC}"
    echo ""

    python -m src.cli.backtest -s $symbols --start "$start_date" -v

    echo ""
    read -p "Press Enter to return to menu..."
    show_menu
}

# Run single test cycle
run_test() {
    if ! check_api_keys; then
        echo -e "${RED}Error: API keys not configured.${NC}"
        read -p "Press Enter to continue..."
        show_menu
        return
    fi

    clear
    echo -e "${GREEN}${BOLD}Running Single Trading Cycle...${NC}"
    echo ""
    python -m src.bot --once

    echo ""
    read -p "Press Enter to return to menu..."
    show_menu
}

# Configure settings
configure_settings() {
    clear
    echo -e "${BLUE}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                   ⚙️  SETTINGS                            ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo "  1) Edit Trading Settings (config/settings.yaml)"
    echo "  2) Edit API Keys (.env)"
    echo "  3) View Current Settings"
    echo "  4) Reset to Defaults"
    echo "  5) Back to Main Menu"
    echo ""
    read -p "  Choice [1-5]: " choice

    case $choice in
        1)
            if command -v nano &> /dev/null; then
                nano config/settings.yaml
            elif command -v vim &> /dev/null; then
                vim config/settings.yaml
            else
                echo "No text editor found. Please edit config/settings.yaml manually."
                read -p "Press Enter to continue..."
            fi
            configure_settings
            ;;
        2)
            if command -v nano &> /dev/null; then
                nano .env
            elif command -v vim &> /dev/null; then
                vim .env
            else
                echo "No text editor found. Please edit .env manually."
                read -p "Press Enter to continue..."
            fi
            configure_settings
            ;;
        3)
            clear
            echo -e "${BOLD}Current Settings:${NC}"
            echo ""
            echo "--- config/settings.yaml ---"
            cat config/settings.yaml
            echo ""
            read -p "Press Enter to continue..."
            configure_settings
            ;;
        4)
            cp config/settings.yaml.example config/settings.yaml 2>/dev/null || true
            echo -e "${GREEN}Settings reset to defaults${NC}"
            read -p "Press Enter to continue..."
            configure_settings
            ;;
        5)
            show_menu
            ;;
        *)
            configure_settings
            ;;
    esac
}

# Show documentation
show_docs() {
    clear
    echo -e "${BLUE}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                   📖 DOCUMENTATION                        ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "${BOLD}Quick Start:${NC}"
    echo "  1. Set up your API keys (Menu option 6)"
    echo "  2. Run a test cycle (Menu option 4)"
    echo "  3. Start the bot (Menu option 1)"
    echo ""
    echo -e "${BOLD}Important Notes:${NC}"
    echo "  • The bot runs in PAPER TRADING mode by default (no real money)"
    echo "  • Get free API keys at: https://alpaca.markets"
    echo "  • View your trades at: https://app.alpaca.markets"
    echo ""
    echo -e "${BOLD}Available Strategies:${NC}"
    echo "  • daily_profit_taker - RSI/MACD swing trading"
    echo "  • mean_reversion     - Buy oversold, sell at mean"
    echo "  • momentum           - Trend following with EMAs"
    echo "  • breakout           - Trade price breakouts"
    echo "  • pairs_trading      - Statistical arbitrage"
    echo ""
    echo -e "${BOLD}Useful Links:${NC}"
    echo "  • GitHub: https://github.com/HackingPain/trading-bot"
    echo "  • Alpaca: https://alpaca.markets"
    echo ""
    read -p "Press Enter to return to menu..."
    show_menu
}

# Handle command line arguments
case "${1:-}" in
    run|start)
        if ! check_api_keys; then
            echo -e "${RED}Error: API keys not configured. Run ./start.sh to set up.${NC}"
            exit 1
        fi
        python -m src.bot
        ;;
    dashboard)
        streamlit run src/dashboard/app.py
        ;;
    backtest)
        shift
        python -m src.cli.backtest "$@"
        ;;
    test)
        python -m src.bot --once
        ;;
    setup)
        setup_wizard
        ;;
    *)
        show_menu
        ;;
esac
