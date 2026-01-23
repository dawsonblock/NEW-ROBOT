#!/bin/bash
# demo.sh - One-button demo script
set -e

echo "Running safety check..."
uv run python -c "import rfsn; print('OK')"

echo "Launching demo..."
uv run python run_demo.py

echo "Demo complete."
