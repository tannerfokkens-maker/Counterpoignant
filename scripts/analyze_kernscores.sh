#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/analyze_kernscores.py "$@"
