"""Python 3.8 compatibility + hygiene for every .py in the gateway package, the tests and (whole-repo
scan) the rest of the repository.

Checks (DESIGN §7):
1. ``ast.parse(src, feature_version=(3, 8))`` (best effort: catches match/except*/…).
2. AST denylist for constructs the 3.8 grammar flag misses: ``X | Y`` / ``list[int]`` in annotations
   (unless ``from __future__ import annotations``), relaxed decorators, starred subscripts,
   ``isinstance(x, A | B)``, ``{..} | {..}`` dict-literal merges, 3.9+ stdlib APIs
   (``str.removeprefix``…, ``zoneinfo``, ``graphlib``, ``tomllib``, ``functools.cache``,
   ``dataclass(slots=/kw_only=)``, ``zip(strict=)``, ``datetime.UTC``, new ``typing`` names, …).
3. Token check for parenthesized context managers ``with (a as b, …):``.
4. No UTF-8 BOM in any text file.
5. Package-only: intra-package imports are relative; imports are stdlib-only (3.10+ runtime).
6. If a real Python 3.8 is available (``PY38`` env, ``python3.8`` on PATH, or ``uv python find 3.8``),
   ``py_compile`` every file with it and import every package module under it.
"""

import ast
import io
import os
import shutil
import subprocess
import sys
import tokenize
import unittest

from ._pkg import PKG, REPO_ROOT, package_dir

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist", ".mypy_cache", ".tox"}
TEXT_EXT = (".py", ".json", ".md", ".txt", ".cmd", ".bat", ".ps1", ".sh", ".toml", ".cfg", ".ini", ".sha256")

BUILTIN_GENERICS = {"list", "dict", "tuple", "set", "frozenset", "type"}
BANNED_MODULES = {"zoneinfo": "3.9", "graphlib": "3.9", "tomllib": "3.11"}
BANNED_FROM_IMPORTS = {
    ("functools", "cache"): "3.9",
    ("importlib.resources", "files"): "3.9",
    ("importlib.resources", "as_file"): "3.9",
    ("contextlib", "chdir"): "3.11",
    ("enum", "StrEnum"): "3.11",
    ("datetime", "UTC"): "3.11",
}
BANNED_TYPING = {"TypeAlias", "ParamSpec", "Concatenate", "TypeGuard", "Self", "LiteralString", "Never",
                 "Required", "NotRequired", "Unpack", "TypeVarTuple", "assert_never", "reveal_type",
                 "dataclass_transform", "override", "is_typeddict", "Annotated"}
BANNED_METHODS = {"removeprefix": "3.9", "removesuffix": "3.9", "is_relative_to": "3.9", "with_stem": "3.9",
                  "bit_count": "3.10", "randbytes": "3.9", "file_digest": "3.11", "to_thread": "3.9",
                  "waitstatus_to_exitcode": "3.9", "TaskGroup": "3.11", "lcm": "3.9", "nextafter": "3.9",
                  "ulp": "3.9", "cache": "3.9", "pairwise": "3.10", "batched": "3.12", "UTC": "3.11"}
METHOD_OWNER_EXCEPTIONS = {"compat"}  # our own py3.8 shims: compat.removeprefix(...)
BANNED_KWARGS = {"dataclass": {"slots", "kw_only", "match_args"}, "zip": {"strict"},
                 "run": {"process_group"}, "Popen": {"process_group"}}


def iter_files(root, exts=(".py",)):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            if f.endswith(exts):
                yield os.path.join(dirpath, f)


def _has_future_annotations(tree):
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(a.name == "annotations" for a in node.names):
                return True
    return False


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _ann_problems(ann):
    out = []
    for n in ast.walk(ann):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
            out.append("PEP 604 'X | Y' annotation (3.10)")
        elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in BUILTIN_GENERICS:
            out.append("builtin generic %s[...] annotation (3.9)" % n.value.id)
    return out


def _decorator_ok(node):
    target = node.func if isinstance(node, ast.Call) else node
    return _dotted(target) is not None


def ast_problems(src, path):
    """List of 'path:line: problem' strings for 3.9+ constructs."""
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as exc:
        return ["%s:%s: syntax error: %s" % (path, exc.lineno, exc.msg)]
    future = _has_future_annotations(tree)
    probs = []

    def add(node, msg):
        probs.append("%s:%s: %s" % (path, getattr(node, "lineno", "?"), msg))

    for node in ast.walk(tree):
        if not future:
            anns = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
                    if arg.annotation is not None:
                        anns.append(arg.annotation)
                if node.returns is not None:
                    anns.append(node.returns)
            elif isinstance(node, ast.AnnAssign):
                anns.append(node.annotation)
            for ann in anns:
                for p in _ann_problems(ann):
                    add(ann, p)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in node.decorator_list:
                if not _decorator_ok(d):
                    add(d, "relaxed decorator expression (3.9)")
        if isinstance(node, ast.Subscript):
            sl = node.slice
            inner = getattr(sl, "value", sl)  # ast.Index on 3.8
            if isinstance(inner, ast.Tuple) and any(isinstance(e, ast.Starred) for e in inner.elts):
                add(node, "starred expression in subscript (3.11)")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr) and (
                isinstance(node.left, (ast.Dict, ast.DictComp)) or isinstance(node.right, (ast.Dict, ast.DictComp))):
            add(node, "dict '|' merge (3.9)")
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.BitOr) and isinstance(node.value, ast.Dict):
            add(node, "dict '|=' update (3.9)")
        if isinstance(node, ast.Call):
            fname = _dotted(node.func) or ""
            short = fname.rsplit(".", 1)[-1]
            if short == "isinstance" and len(node.args) == 2 and isinstance(node.args[1], ast.BinOp):
                add(node, "isinstance with 'A | B' (3.10)")
            for kw in node.keywords:
                if kw.arg and kw.arg in BANNED_KWARGS.get(short, ()):
                    add(node, "%s(%s=...) (3.10+)" % (short, kw.arg))
        if isinstance(node, ast.Attribute) and node.attr in BANNED_METHODS:
            owner = _dotted(node.value)
            if owner not in METHOD_OWNER_EXCEPTIONS and not (node.attr == "UTC" and owner != "datetime") and \
                    not (node.attr in ("cache", "lcm", "ulp", "nextafter") and owner not in ("functools", "math")):
                add(node, ".%s (%s)" % (node.attr, BANNED_METHODS[node.attr]))
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in BANNED_MODULES:
                    add(node, "import %s (%s)" % (a.name, BANNED_MODULES[top]))
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top = node.module.split(".")[0]
            if top in BANNED_MODULES:
                add(node, "from %s import (%s)" % (node.module, BANNED_MODULES[top]))
            for a in node.names:
                if (node.module, a.name) in BANNED_FROM_IMPORTS:
                    add(node, "from %s import %s (%s)" % (node.module, a.name, BANNED_FROM_IMPORTS[(node.module, a.name)]))
                if node.module in ("typing", "typing_extensions") and a.name in BANNED_TYPING:
                    add(node, "from typing import %s (3.9+)" % a.name)
        if isinstance(node, ast.Attribute) and _dotted(node.value) == "typing" and node.attr in BANNED_TYPING:
            add(node, "typing.%s (3.9+)" % node.attr)
    return probs


def token_problems(src, path):
    probs = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        return ["%s: tokenize error: %s" % (path, exc)]
    sig = [t for t in toks if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                             tokenize.DEDENT)]
    for i, t in enumerate(sig):
        if t.type == tokenize.NAME and t.string == "with" and i + 1 < len(sig) and sig[i + 1].string == "(":
            depth, j, has_as = 0, i + 1, False
            while j < len(sig):
                s = sig[j].string
                if s in "([{" and sig[j].type == tokenize.OP:
                    depth += 1
                elif s in ")]}" and sig[j].type == tokenize.OP:
                    depth -= 1
                    if depth == 0:
                        break
                elif depth == 1 and sig[j].type == tokenize.NAME and s == "as":
                    has_as = True
                j += 1
            if has_as and j + 1 < len(sig) and sig[j + 1].string == ":":
                probs.append("%s:%d: parenthesized context managers (3.10)" % (path, t.start[0]))
    return probs


def find_python38():
    cand = os.environ.get("PY38")
    if cand and os.path.isfile(cand):
        return cand
    p = shutil.which("python3.8")
    if p:
        return p
    uv = shutil.which("uv")
    if uv:
        try:
            out = subprocess.run([uv, "python", "find", "3.8"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 timeout=20, universal_newlines=True)
            path = out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout.strip() else ""
            if path and os.path.isfile(path):
                return path
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return None


class Py38CompatTests(unittest.TestCase):
    def _roots(self):
        return [package_dir(), os.path.dirname(os.path.abspath(__file__))]

    def _all_repo_py(self):
        return list(iter_files(REPO_ROOT))

    def test_feature_version_parse(self):
        bad = []
        for path in self._all_repo_py():
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
            try:
                ast.parse(src, filename=path, feature_version=(3, 8))
            except SyntaxError as exc:
                bad.append("%s:%s: %s" % (path, exc.lineno, exc.msg))
        self.assertEqual(bad, [])

    def test_denylist(self):
        probs = []
        for path in self._all_repo_py():
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
            probs.extend(ast_problems(src, path))
            probs.extend(token_problems(src, path))
        self.assertEqual(probs, [], "\n".join(probs))

    def test_denylist_detects(self):
        samples = {
            "def f(x: int | None): pass\n": "PEP 604",
            "def f(x: list[int]) -> dict[str, int]: pass\n": "builtin generic",
            "@x[0].y\ndef f(): pass\n": "relaxed decorator",
            "x[*a]\n": "starred",
            "a = {} | {1: 2}\n": "dict '|' merge",
            "isinstance(x, int | str)\n": "isinstance",
            "'x'.removeprefix('y')\n": ".removeprefix",
            "import zoneinfo\n": "zoneinfo",
            "from functools import cache\n": "functools import cache",
            "from typing import TypeAlias\n": "typing import TypeAlias",
            "import dataclasses\n@dataclasses.dataclass(slots=True)\nclass A: pass\n": "dataclass(slots",
            "zip(a, b, strict=True)\n": "zip(strict",
            "import datetime\ndatetime.UTC\n": ".UTC",
        }
        for src, needle in samples.items():
            got = "\n".join(ast_problems(src, "s.py"))
            # on an older interpreter the sample may not even parse — that is a detection too
            self.assertTrue(needle in got or "syntax error" in got, "%r -> %r" % (src, got))
        self.assertTrue(token_problems("with (open('a') as f, open('b') as g):\n    pass\n", "s.py"))
        self.assertFalse(token_problems("with open('a') as f:\n    pass\nwith (yield):\n    pass\n", "s.py"))
        ok = ("from __future__ import annotations\ndef f(x: list[int] | None): pass\n"
              "import compat\ncompat.removeprefix('a', 'b')\n")
        self.assertEqual(ast_problems(ok, "s.py"), [])

    def test_no_bom(self):
        bad = []
        for path in iter_files(REPO_ROOT, TEXT_EXT):
            with open(path, "rb") as f:
                if f.read(3) == b"\xef\xbb\xbf":
                    bad.append(path)
        self.assertEqual(bad, [])

    def test_package_imports_relative_and_stdlib_only(self):
        pkg_root = package_dir()
        stdlib = getattr(sys, "stdlib_module_names", None)
        probs = []
        for path in iter_files(pkg_root):
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                for n in names:
                    top = n.split(".")[0]
                    if top in ("shared", "fry_gateway", PKG.split(".")[0]):
                        probs.append("%s:%d: absolute intra-package import %r (use relative)" % (path, node.lineno, n))
                    elif stdlib is not None and top not in stdlib and top != "__future__":
                        probs.append("%s:%d: non-stdlib import %r" % (path, node.lineno, n))
        self.assertEqual(probs, [], "\n".join(probs))

    def test_real_python38(self):
        py = find_python38()
        if not py:
            self.skipTest("no Python 3.8 interpreter found (set PY38 or install via `uv python install 3.8`)")
        ver = subprocess.run([py, "-c", "import sys; print(sys.version_info[:2])"], stdout=subprocess.PIPE,
                             universal_newlines=True, timeout=60).stdout.strip()
        self.assertEqual(ver, "(3, 8)")
        files = self._all_repo_py()
        code = ("import sys\nbad = []\nfor p in sys.argv[1:]:\n"
                "    try:\n        compile(open(p, encoding='utf-8').read(), p, 'exec', dont_inherit=True)\n"
                "    except SyntaxError as e:\n        bad.append('%s:%s: %s' % (p, e.lineno, e.msg))\n"
                "print('\\n'.join(bad)); sys.exit(1 if bad else 0)\n")
        r = subprocess.run([py, "-c", code] + files, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout)
        # import every package module under 3.8 (modules owned by other agents may not exist yet)
        imp = ("import importlib, pkgutil, sys\nsys.path.insert(0, %r)\npkg = importlib.import_module(%r)\n"
               "bad = []\nn = 0\nfor m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):\n"
               "    n += 1\n    try:\n        importlib.import_module(m.name)\n"
               "    except Exception as e:\n        bad.append('%%s: %%r' %% (m.name, e))\n"
               "print('\\n'.join(bad)); sys.exit(1 if bad or n < 10 else 0)\n") % (REPO_ROOT, PKG)
        r = subprocess.run([py, "-c", imp], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout)


if __name__ == "__main__":
    unittest.main()
