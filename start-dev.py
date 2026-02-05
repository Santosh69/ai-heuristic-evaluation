#!/usr/bin/env python3
"""Cross-platform dev environment launcher for AI Heuristic Evaluation."""

import os
import sys
import subprocess
import signal
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

print("Initializing AI Heuristic Evaluation dev environment...")

setup_script = PROJECT_ROOT / "scripts" / "setup_omniparser.py"

if setup_script.exists():
    print("Verifying OmniParser V2 weights...")
    subprocess.run([sys.executable, str(setup_script)], check=True)
else:
    print("Error: scripts/setup_omniparser.py not found.")
    sys.exit(1)

print("Starting FastAPI backend (Port 8000)...")

uvicorn_process = subprocess.Popen(
    [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ],
    cwd=PROJECT_ROOT
)

firebase_dir = PROJECT_ROOT / "firebase-functions"
firebase_process = None

if firebase_dir.is_dir():
    if shutil.which("npm") is None:
        print("npm not found in PATH. Skipping firebase-functions.")
    else:
        print("Starting firebase-functions (if configured)...")
        firebase_process = subprocess.Popen(
            ["npm", "run", "serve"],
            cwd=firebase_dir,
            shell=(os.name == "nt")
        )
else:
    print("firebase-functions not found. Skipping frontend/emulator.")

processes = [p for p in [uvicorn_process, firebase_process] if p is not None]

def shutdown_handler(signum, frame):
    """Terminate child processes on shutdown."""
    print("\nShutting down services...")
    for proc in processes:
        proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

print("Dev environment running. Press Ctrl+C to stop.")

try:
    uvicorn_process.wait()
except KeyboardInterrupt:
    shutdown_handler(None, None)
