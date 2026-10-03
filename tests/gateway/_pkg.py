"""Single import root for the gateway tests.

``tools/vendor_gateway.py`` rewrites ONLY the ``PKG = ...`` line below when copying these tests into
fry-launch-claude (``shared.gateway`` -> ``fry_gateway``). Tests must import the package exclusively
through ``mod()`` / ``PKG`` so that rewrite is the only change needed.
"""

import importlib
import os
import sys

PKG = "shared.gateway"

#: repository root (tests/gateway/_pkg.py -> ../..)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def mod(name=""):
    """Import ``<PKG>`` or ``<PKG>.<name>`` (e.g. ``mod("events")``, ``mod("auth.static")``)."""
    return importlib.import_module(PKG + ("." + name if name else ""))


def package_dir():
    return os.path.dirname(os.path.abspath(mod().__file__))
