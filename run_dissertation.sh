#!/bin/bash

# TSP RL Experiment Runner Script
# This script provides a convenient way to run experiments with proper environment setup

set -e  # Exit on any error

echo "=== TSP Reinforcement Learning Experiments ==="
echo "Starting experiment runner..."

\rm -rf results/*


# Keep system awake during long experiments (macOS)
if command -v caffeinate &> /dev/null; then
    echo "Using caffeinate to prevent system sleep..."
    caffeinate -s python run_dissertation.py
else
    # For other systems, run normally
    python run_dissertation.py
fi

echo "Experiment completed!"