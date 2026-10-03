#!/usr/bin/env python3
"""codex-wrap — run Claude Code with OpenAI GPT (API key or ChatGPT/Codex login). See README.md and `codex-wrap --help`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.base_launcher import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.example.json")))
