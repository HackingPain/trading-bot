#!/bin/bash
#
# Trading Bot - One-Click Installer
#
# Usage: curl -sSL https://raw.githubusercontent.com/HackingPain/trading-bot/main/install.sh | bash
#    or: ./install.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

clear
echo -e "${BLUE}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║              📈 STOCK TRADING BOT INSTALLER 📈            ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# Check if we're in the trading-bot directory or need to clone
if [ -f "src/bot.py" ]; then
    INSTALL_DIR="$(pwd)"
    echo -e "${GREEN}✓${NC} Found trading bot in current directory"
else
    INSTALL_DIR="$HOME/trading-bot"
    echo -e "${YELLOW}→${NC} Will install to: $INSTALL_DIR"

    if [ -d "$INSTALL_DIR" ]; then
        echo -e "${YELLOW}→${NC} Directory exists, updating..."
        cd "$INSTALL_DIR"
        git pull origin main 2>/dev/null || true
    else
        echo -e "${YELLOW}→${NC} Downloading trading bot..."
        git clone https://github.com/HackingPain/trading-bot.git "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
fi

echo ""
echo -e "${BLUE}${BOLD}Step 1: Checking Python...${NC}"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
    else
        echo -e "${RED}✗${NC} Python 3.10+ required, found $PYTHON_VERSION"
        echo ""
        echo "Please install Python 3.10 or higher:"
        echo "  - Mac: brew install python@3.11"
        echo "  - Ubuntu/Debian: sudo apt install python3.11"
        echo "  - Windows: Download from python.org"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} Python 3 not found"
    echo ""
    echo "Please install Python 3.10 or higher first."
    exit 1
fi

echo ""
echo -e "${BLUE}${BOLD}Step 2: Setting up virtual environment...${NC}"

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment exists"
fi

# Activate and install dependencies
source venv/bin/activate
pip install --upgrade pip -q
echo -e "${YELLOW}→${NC} Installing dependencies (this may take a minute)..."
pip install -r requirements.txt -q
echo -e "${GREEN}✓${NC} Dependencies installed"

echo ""
echo -e "${BLUE}${BOLD}Step 3: Creating directories...${NC}"
mkdir -p data logs results
echo -e "${GREEN}✓${NC} Directories created"

echo ""
echo -e "${BLUE}${BOLD}Step 4: Setting up configuration...${NC}"

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Created .env file"
else
    echo -e "${GREEN}✓${NC} .env file exists"
fi

echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}                    ✓ INSTALLATION COMPLETE!                ${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}Next Steps:${NC}"
echo ""
echo -e "  ${YELLOW}1.${NC} Get your FREE Alpaca API keys (required for trading):"
echo -e "     ${BLUE}https://app.alpaca.markets/signup${NC}"
echo ""
echo -e "  ${YELLOW}2.${NC} Run the setup wizard to configure your bot:"
echo -e "     ${GREEN}cd $INSTALL_DIR && ./start.sh${NC}"
echo ""
echo -e "  ${YELLOW}3.${NC} Or manually edit .env with your API keys:"
echo -e "     ${GREEN}nano $INSTALL_DIR/.env${NC}"
echo ""
echo -e "${BOLD}Quick Commands:${NC}"
echo -e "  ${GREEN}./start.sh${NC}          - Start the bot (interactive menu)"
echo -e "  ${GREEN}./start.sh run${NC}      - Run trading bot"
echo -e "  ${GREEN}./start.sh dashboard${NC} - Open web dashboard"
echo -e "  ${GREEN}./start.sh backtest${NC}  - Run a backtest"
echo ""
