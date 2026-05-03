#!/bin/bash
set -e

echo "=== Stage 1: Build adversarial prompt dataset ==="
python scripts/01_build_prompts.py

echo ""
echo "=== Stage 2: Adversarial search (4700 candidates → top 350) ==="
python scripts/02_adversarial_search.py

echo ""
echo "=== Stage 3a: Main benchmark (5 models × 1200 prompts @ temp=0.7) ==="
python scripts/03_run_eval.py --run main

echo ""
echo "=== Stage 3b: Temperature sweep (5 models × 350 prompts × 7 temps) ==="
python scripts/03_run_eval.py --run temp_sweep

echo ""
echo "=== Stage 4: Analysis and plots ==="
python scripts/04_analyze.py

echo ""
echo "=== Pipeline complete. Results in results/ ==="
