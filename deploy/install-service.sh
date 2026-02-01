#!/bin/bash
# Install Trading Bot as a systemd service
# Run with: sudo ./deploy/install-service.sh

set -e

# Configuration
BOT_USER="tradingbot"
BOT_GROUP="tradingbot"
INSTALL_DIR="/opt/trading-bot"
LOG_DIR="/var/log/trading-bot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Trading Bot Service Installer${NC}"
echo "================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo)${NC}"
    exit 1
fi

# Create service user
echo -e "${YELLOW}Creating service user...${NC}"
if ! id "$BOT_USER" &>/dev/null; then
    useradd --system --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin "$BOT_USER"
    echo -e "${GREEN}Created user: $BOT_USER${NC}"
else
    echo "User $BOT_USER already exists"
fi

# Create directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p "$INSTALL_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/logs"

# Copy project files
echo -e "${YELLOW}Copying project files...${NC}"
rsync -av --exclude='venv' --exclude='__pycache__' --exclude='.git' \
    "$PROJECT_DIR/" "$INSTALL_DIR/"

# Set up virtual environment
echo -e "${YELLOW}Setting up virtual environment...${NC}"
if [ ! -d "$INSTALL_DIR/venv" ]; then
    python3 -m venv "$INSTALL_DIR/venv"
fi
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# Set permissions
echo -e "${YELLOW}Setting permissions...${NC}"
chown -R "$BOT_USER:$BOT_GROUP" "$INSTALL_DIR"
chown -R "$BOT_USER:$BOT_GROUP" "$LOG_DIR"
chmod 750 "$INSTALL_DIR"
chmod 640 "$INSTALL_DIR/.env" 2>/dev/null || true

# Install systemd services
echo -e "${YELLOW}Installing systemd services...${NC}"
cp "$SCRIPT_DIR/trading-bot.service" /etc/systemd/system/
cp "$SCRIPT_DIR/trading-bot-scheduler.service" /etc/systemd/system/
cp "$SCRIPT_DIR/trading-bot-scheduler.timer" /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable services
echo -e "${YELLOW}Enabling services...${NC}"
systemctl enable trading-bot-scheduler.timer

# Create logrotate config
echo -e "${YELLOW}Setting up log rotation...${NC}"
cat > /etc/logrotate.d/trading-bot << 'EOF'
/var/log/trading-bot/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 640 tradingbot tradingbot
    postrotate
        systemctl reload trading-bot 2>/dev/null || true
    endscript
}
EOF

# Print summary
echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit /opt/trading-bot/.env with your API keys"
echo "  2. Review /opt/trading-bot/config/settings.yaml"
echo "  3. Start the services:"
echo ""
echo "     # For continuous running:"
echo "     sudo systemctl start trading-bot"
echo ""
echo "     # For scheduled runs (every 5 minutes):"
echo "     sudo systemctl start trading-bot-scheduler.timer"
echo ""
echo "  4. Check status:"
echo "     sudo systemctl status trading-bot"
echo "     sudo journalctl -u trading-bot -f"
echo ""
echo "  5. View logs:"
echo "     tail -f /var/log/trading-bot/bot.log"
echo ""
