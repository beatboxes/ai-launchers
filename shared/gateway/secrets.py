"""Secret resolution (DESIGN §8.2 B). Values are NEVER logged, traced or put in exceptions.

``resolve_many(specs, timeout=15.0, environ=None, warnings=None) -> Dict[name, Optional[value]]``
    ``specs`` maps a secret name to an ordered list of sources; the first source yielding a
    non-empty value wins:
      ``"env:VAR"``                    environment variable (``environ`` or ``os.environ``)
      ``"op://vault/item/field"``      1Password CLI ``op read <ref>`` (stdin closed, ``timeout``)
      ``"literal:value"``              the value itself
      ``"json:<path>#<dotted.key>"``   string at ``a.b.0.c`` inside a JSON file (``~`` expanded)
    Every ``op read`` that may be needed (across all names) runs concurrently in a thread pool;
    results are memoized per process (``clear_cache()``). A missing ``op`` binary or a signed-out
    1Password CLI yields None plus ONE warning, never an exception. Values are whitespace-stripped.
    Human-readable warnings (no values) are appended to ``warnings`` when a list is given and
    logged on the ``ai_gateway`` logger.
``load_into(store, specs, timeout=15.0, environ=None, warnings=None) -> Dict[name, label]``
    Resolves and ``store.set``s every found secret; returns source labels such as
    ``"env:XAI_API_KEY"``, ``"op:<item>"``, ``"literal"``, ``"json:credentials.json"``.
``resolve_detailed(...) -> (values, labels, warnings)`` is the underlying single pass.
"""

import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import threading

__all__ = ["resolve_many", "load_into", "resolve_detailed", "source_label", "clear_cache"]

LOG = logging.getLogger("ai_gateway")

_OP_CACHE = {}  # (op binary, ref) -> (value or None, warning or None)
_OP_CACHE_LOCK = threading.Lock()
_OP_MAX_WORKERS = 8
_NOT_SIGNED_IN_RE = re.compile(
    r"not (currently )?signed in|no accounts? (configured|found)|sign ?in required|session expired|"
    r"authorization (prompt )?(dismissed|denied|timeout)|account is not signed in|connecting to desktop app",
    re.I)
_SIGNED_OUT_WARNING = "1Password CLI is not signed in (run `op signin`); op:// secrets were skipped"
_NO_OP_WARNING = "1Password CLI `op` not found on PATH; op:// secrets were skipped"


def clear_cache():
    """Forget memoized ``op read`` results (tests / after ``op signin``)."""
    with _OP_CACHE_LOCK:
        _OP_CACHE.clear()


def _op_item(ref):
    parts = [p for p in ref[len("op://"):].split("/") if p]
    return parts[1] if len(parts) >= 2 else (parts[0] if parts else "?")


def source_label(source):
    """Non-secret label for a source spec (``literal:`` values are never echoed)."""
    if source.startswith("env:"):
        return source
    if source.startswith("op://"):
        return "op:" + _op_item(source)
    if source.startswith("literal:"):
        return "literal"
    if source.startswith("json:"):
        path = source[len("json:"):].rpartition("#")[0] or source[len("json:"):]
        return "json:" + os.path.basename(path.replace("\\", "/"))
    return "unknown"


def _clean(value):
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def _json_lookup(source):
    spec = source[len("json:"):]
    path, sep, dotted = spec.rpartition("#")
    if not sep or not path or not dotted:
        return None, "malformed json: secret source (expected json:<path>#<key>)"
    path = os.path.expanduser(path)
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            node = json.load(f)
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as exc:
        return None, "cannot read %s: %s" % (os.path.basename(path), type(exc).__name__)
    for part in dotted.split("."):
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None, None
    return _clean(node), None


def _op_read(op, ref, timeout, environ):
    """(value or None, warning or None) for one ``op read`` — memoized per process."""
    key = (op, ref)
    with _OP_CACHE_LOCK:
        if key in _OP_CACHE:
            return _OP_CACHE[key]
    item = _op_item(ref)
    try:
        r = subprocess.run([op, "read", ref], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout, env=environ,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        result = (None, "1Password `op read` for item %r timed out after %gs" % (item, timeout))
    except OSError as exc:
        result = (None, "cannot run 1Password CLI: %s" % type(exc).__name__)
    else:
        err = r.stderr.decode("utf-8", "replace")
        if r.returncode == 0:
            result = (_clean(r.stdout.decode("utf-8", "replace")), None)
        elif _NOT_SIGNED_IN_RE.search(err):
            result = (None, _SIGNED_OUT_WARNING)
        else:
            result = (None, "1Password `op read` for item %r failed (exit %d)" % (item, r.returncode))
    with _OP_CACHE_LOCK:
        _OP_CACHE[key] = result
    return result


def _cheap(source, env):
    """(value, warning) for non-op sources."""
    if source.startswith("env:"):
        return _clean(env.get(source[len("env:"):])), None
    if source.startswith("literal:"):
        return _clean(source[len("literal:"):]), None
    if source.startswith("json:"):
        return _json_lookup(source)
    return None, "unsupported secret source kind"


def resolve_detailed(specs, timeout=15.0, environ=None):
    """-> (values: Dict[name, Optional[str]], labels: Dict[name, str], warnings: List[str])."""
    env = os.environ if environ is None else environ
    warnings = []
    cheap_results = {}

    def warn(msg):
        if msg and msg not in warnings:
            warnings.append(msg)

    def cheap(src):
        if src not in cheap_results:
            cheap_results[src] = _cheap(src, env)
        return cheap_results[src]

    # pass 1: which op refs could matter (those before the first cheap source that resolves)
    needed = []
    for name, sources in specs.items():
        for src in sources or ():
            if not isinstance(src, str):
                continue
            if src.startswith("op://"):
                if src not in needed:
                    needed.append(src)
                continue
            if cheap(src)[0] is not None:
                break
    op_results = {}
    if needed:
        op = shutil.which("op", path=env.get("PATH"))
        if op is None:
            warn(_NO_OP_WARNING)
            op_results = {ref: (None, None) for ref in needed}
        else:
            child_env = None if environ is None else dict(environ)
            workers = min(_OP_MAX_WORKERS, len(needed))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                                       thread_name_prefix="gw-op-read") as pool:
                futures = {ref: pool.submit(_op_read, op, ref, timeout, child_env) for ref in needed}
                for ref, fut in futures.items():
                    try:
                        op_results[ref] = fut.result(timeout=timeout + 5)
                    except Exception as exc:  # defensive: _op_read handles its own errors
                        op_results[ref] = (None, "1Password lookup failed: %s" % type(exc).__name__)
    # pass 2: first winning source per name, in order
    values, labels = {}, {}
    for name, sources in specs.items():
        values[name] = None
        for idx, src in enumerate(sources or ()):
            if not isinstance(src, str):
                warn("secret %r: source #%d is not a string" % (name, idx + 1))
                continue
            if src.startswith("op://"):
                value, msg = op_results.get(src, (None, None))
            else:
                value, msg = cheap(src)
                if msg:
                    msg = "secret %r: %s (source #%d)" % (name, msg, idx + 1)
            warn(msg)
            if value is not None:
                values[name] = value
                labels[name] = source_label(src)
                break
    for msg in warnings:
        LOG.warning("secrets: %s", msg)
    return values, labels, warnings


def resolve_many(specs, timeout=15.0, environ=None, warnings=None):
    """Name -> value (None when no source produced one). See module docstring."""
    values, _, warns = resolve_detailed(specs, timeout, environ)
    if warnings is not None:
        warnings.extend(warns)
    return values


def load_into(store, specs, timeout=15.0, environ=None, warnings=None):
    """Resolve ``specs`` into ``store`` (a ``config.SecretStore``); name -> source label."""
    values, labels, warns = resolve_detailed(specs, timeout, environ)
    if warnings is not None:
        warnings.extend(warns)
    for name, value in values.items():
        if value is not None:
            store.set(name, value)
    return labels
