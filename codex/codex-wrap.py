#!/usr/bin/env python3
"""codex-wrap launch claude — route Claude Code to OpenAI via API key or codex CLI bridge."""
import json
import os
import sys
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from shared.base_launcher import Launcher

manifest = json.loads((Path(__file__).parent / "config.example.json").read_text(encoding="utf-8"))
if __name__ == "__main__":
    sys.exit(Launcher(manifest).main(sys.argv[1:]))