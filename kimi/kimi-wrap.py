#!/usr/bin/env python3
"""kimi-wrap — run Claude Code with Moonshot Kimi (official Anthropic-compatible endpoint). See README.md and `kimi-wrap --help`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.base_launcher import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.example.json")))
