#!/usr/bin/env bash
set -euo pipefail

# Set the script to the project root (where this script file lives)
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "🚀 Initializing AI Heuristic Evaluation dev environment..."

if [ -f "scripts/setup_omniparser.py" ]; then
    echo "📦 Verifying OmniParser V2 weights..."
    python3 scripts/setup_omniparser.py
else
    echo "❌ Error: scripts/setup_omniparser.py not found."
    exit 1
fi

echo "🔥 Starting FastAPI backend (Port 8000)..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &

# Start firebase-functions / frontend if present
if [ -d "firebase-functions" ]; then
  echo "⚙️  Starting firebase-functions (if configured)..."
  (cd firebase-functions && npm run serve) &
else
  echo "⚠️  firebase-functions not found. Skipping frontend/emulator."
fi

# Keep the script alive so background processes continue
wait
