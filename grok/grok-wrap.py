#!/usr/bin/env python3
"""grok-wrap — run Claude Code with xAI Grok (API key or `grok login`). See README.md and `grok-wrap --help`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.base_launcher import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.example.json")))
