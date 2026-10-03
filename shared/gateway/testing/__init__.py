"""Test support shipped with the package (mock upstreams, mock auth servers, fake CLI bins, the
real-Claude-Code E2E harness, golden fixtures). Nothing here is imported by the runtime gateway.

``fixture_path(name, version=CLAUDE_CODE_FIXTURES)``, ``load_fixture(name, version=...)`` load the
scrubbed golden requests recorded from the real Claude Code binary (see fixtures/<version>/).
Each fixture is ``{"method", "path", "headers" (subset, no auth), "body"}``; placeholders:
``<WORKDIR>``, ``<HOME>``, ``<SPIKEDIR>``, ``-WORKDIR`` (dash-encoded project dir), session id
``00000000-0000-4000-8000-000000000001``, device id ``"0"*64``.
"""

import json
import os

__all__ = ["FIXTURES_DIR", "CLAUDE_CODE_FIXTURES", "SENTINEL", "fixture_path", "load_fixture", "list_fixtures"]

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
CLAUDE_CODE_FIXTURES = "claude_code_2.1.288"
SENTINEL = "FAKEKEY-SENTINEL"


def fixture_path(name, version=CLAUDE_CODE_FIXTURES):
    return os.path.join(FIXTURES_DIR, version, name)


def load_fixture(name, version=CLAUDE_CODE_FIXTURES):
    with open(fixture_path(name, version), "r", encoding="utf-8") as f:
        return json.load(f)


def list_fixtures(version=CLAUDE_CODE_FIXTURES):
    d = os.path.join(FIXTURES_DIR, version)
    return sorted(n for n in os.listdir(d) if n.endswith(".json")) if os.path.isdir(d) else []
