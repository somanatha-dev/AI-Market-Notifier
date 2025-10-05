#!/usr/bin/env bash
set -e
docker stop market-notifier || true
docker rm -f market-notifier || true
