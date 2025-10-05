#!/usr/bin/env bash
set -euo pipefail

# setup-gcloud.sh
# - Moves the repo to /opt/market-notifier (if not already there)
# - Installs docker if missing
# - Sets timezone to IST
# - Builds the Docker image
# - Copies systemd service & timer files into /etc/systemd/system/
# - Enables the timers so the service starts at 09:00 IST and stops at 16:00 IST Mon-Fri
#
# Usage: run from inside the cloned repo directory or anywhere as long as you pass the repo path.
# Example:
#   cd ~/market-notifier
#   sudo bash deploy/setup-gcloud.sh
#
# Important: create a private .env in the repo (copy .env.example -> .env) and fill TELEGRAM_BOT_TOKEN/CHAT_ID before running.

REPO_SRC_DIR="$(pwd)"
DEST_DIR="/opt/market-notifier"
SERVICE_DIR="/etc/systemd/system"

echo "Starting setup-gcloud.sh"
echo "Repo source: ${REPO_SRC_DIR}"
echo "Destination will be: ${DEST_DIR}"

# require .env in source repo
if [ ! -f "${REPO_SRC_DIR}/.env" ]; then
  echo "ERROR: .env file not found in repo root (${REPO_SRC_DIR}). Create it by copying .env.example and adding your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID. Aborting."
  exit 1
fi

# ensure /opt directory exists
sudo mkdir -p "${DEST_DIR}"
sudo chown "$(whoami):$(whoami)" "${DEST_DIR}"

# copy repo contents to /opt/market-notifier (overwrite if exists)
echo "Copying files to ${DEST_DIR} (may overwrite existing files there)..."
sudo rsync -a --delete "${REPO_SRC_DIR}/" "${DEST_DIR}/"
sudo chown -R root:root "${DEST_DIR}"
# make start/stop scripts executable
sudo chmod +x "${DEST_DIR}/start.sh" || true
sudo chmod +x "${DEST_DIR}/stop.sh" || true
sudo chmod 600 "${DEST_DIR}/.env" || true

# install docker if missing
if ! command -v docker >/dev/null 2>&1; then
  echo "Installing docker..."
  sudo apt-get update -y
  sudo apt-get install -y docker.io
  sudo systemctl enable --now docker
else
  echo "Docker already installed."
fi

# set timezone to IST so timers run in IST
echo "Setting timezone to Asia/Kolkata"
sudo timedatectl set-timezone Asia/Kolkata

# build docker image
echo "Building Docker image market-notifier:latest from ${DEST_DIR}..."
cd "${DEST_DIR}"
sudo docker build -t market-notifier:latest .

# copy systemd files (service + timers)
for f in market-notifier-start.service market-notifier-stop.service market-notifier-start.timer market-notifier-stop.timer; do
  if [ ! -f "deploy/${f}" ]; then
    echo "ERROR: deploy/${f} not found in repo. Aborting."
    exit 1
  fi
  echo "Installing ${f} -> ${SERVICE_DIR}/${f}"
  sudo cp "deploy/${f}" "${SERVICE_DIR}/${f}"
  sudo chmod 644 "${SERVICE_DIR}/${f}"
done

# reload systemd and enable timers
echo "Reloading systemd and enabling timers..."
sudo systemctl daemon-reload
sudo systemctl enable --now market-notifier-start.timer
sudo systemctl enable --now market-notifier-stop.timer

echo "Setup complete."
echo "Manual test commands:"
echo "  sudo systemctl start market-notifier-start.service   # start container now"
echo "  sudo docker ps --filter name=market-notifier"
echo "  sudo docker logs -f market-notifier"
echo "  sudo systemctl start market-notifier-stop.service    # stop container now"
echo ""
echo "Check timers with:"
echo "  systemctl list-timers --all | egrep 'market-notifier|NEXT'"
