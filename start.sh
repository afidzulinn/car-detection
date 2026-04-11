#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Car Component AI Detector — Quick Start
#  Starts the backend and opens the web app in your browser.
# ─────────────────────────────────────────────────────────────
set -e

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
URL="http://localhost:${PORT}"

cd "$(dirname "$0")"

echo ""
echo "  Car Component AI Detector"
echo "  ─────────────────────────"
echo "  Backend : ${URL}"
echo "  Frontend: ${URL}"
echo "  Health  : ${URL}/health"
echo ""

# Wait a moment then open browser
(sleep 2 && xdg-open "${URL}" 2>/dev/null || open "${URL}" 2>/dev/null || echo "  Open ${URL} in your browser") &

# Start backend (blocks until Ctrl+C)
exec uvicorn backend.app:app --host "$HOST" --port "$PORT" --reload
