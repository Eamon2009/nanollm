#!/bin/bash
# train_cpu.sh
# Train the LLM on CPU using nanollm/cpu/main.py
# Slow but works anywhere.

set -e

cd nanollm/cpu

echo "Starting LLM training on CPU..."
python main.py

echo "Training complete. Check logs/ and llm.pt"