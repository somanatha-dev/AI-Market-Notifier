#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# stop existing container if present
docker rm -f market-notifier || true

# load envfile if present
if [ -f "./.env" ]; then
  set -o allexport
  source ./.env
  set +o allexport
fi

# run container; bind health endpoint to localhost
docker run -d \
  --name market-notifier \
  --restart unless-stopped \
  --env-file "$(pwd)/.env" \
  -p 127.0.0.1:8080:8080 \
  market-notifier:latest
