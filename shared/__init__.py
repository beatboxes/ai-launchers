"""ai-launchers shared package: the launcher layer behind every ``<provider>-wrap`` command.

Modules:
  base_launcher.py  manifests, transports, ``launch``/``models``/``keys``/``doctor`` commands
  key_manager.py    credentials.json (0600) + secret-source helpers (env -> op:// -> credentials.json)
  utils.py          per-user paths (AIL_HOME, config, state, cache, logs) and atomic JSON I/O
  gateway/          stdlib Anthropic-Messages gateway (see gateway/DESIGN.md); vendored into fry
"""

from .utils import VERSION as __version__  # noqa: F401
