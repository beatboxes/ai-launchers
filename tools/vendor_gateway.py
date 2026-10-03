#!/usr/bin/env python3
"""Vendor the canonical gateway into fry-launch-claude.

    python tools/vendor_gateway.py <fry-repo> [--force] [--dry-run] [--check] [--src-root DIR]

Copies (byte-identical, excluding __pycache__/*.pyc and MANIFEST.sha256):
  <src>/shared/gateway/  ->  <fry>/fry_gateway/
  <src>/tests/gateway/   ->  <fry>/tests/gateway/   (only tests/gateway/_pkg.py is rewritten:
                                                    PKG = "shared.gateway" -> PKG = "fry_gateway")
Deletes stale files in both targets, regenerates MANIFEST.sha256 in <src>/shared/gateway,
<fry>/fry_gateway and <fry>/tests/gateway ("<sha256>  <posix path>" lines, sorted), and creates
<fry>/tests/__init__.py if missing.

Refuses (exit 2) when a target differs from its own MANIFEST.sha256 (local edits, added or deleted
files) or exists without a manifest — unless --force. --check exits 0 when the targets are already in
sync with the source, 1 otherwise (no writes). --dry-run prints the plan without writing.
Stdlib only; Python >= 3.8.
"""

import argparse
import hashlib
import os
import sys

MANIFEST = "MANIFEST.sha256"
SKIP_DIRS = {"__pycache__", ".mypy_cache", ".pytest_cache"}
SKIP_SUFFIXES = (".pyc", ".pyo")
SKIP_NAMES = {MANIFEST, ".DS_Store", "Thumbs.db"}
PKG_LINE_SRC = 'PKG = "shared.gateway"'
PKG_LINE_DST = 'PKG = "fry_gateway"'
DEFAULT_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VendorError(Exception):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def list_files(root):
    """{posix relpath: absolute path} of vendorable files under ``root``."""
    out = {}
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            if f in SKIP_NAMES or f.endswith(SKIP_SUFFIXES):
                continue
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            out[rel] = full
    return out


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def manifest_text(contents):
    """``contents``: {relpath: bytes} -> manifest text."""
    return "".join("%s  %s\n" % (sha256_bytes(contents[rel]), rel) for rel in sorted(contents))


def parse_manifest(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, rel = line.partition("  ")
        if len(digest) != 64 or not rel:
            raise VendorError("malformed manifest line: %r" % line)
        out[rel] = digest
    return out


def rewrite_tests(contents):
    """Rewrite the single import-root line of tests/gateway/_pkg.py."""
    if "_pkg.py" not in contents:
        raise VendorError("tests/gateway/_pkg.py not found")
    text = contents["_pkg.py"].decode("utf-8")
    lines = text.split("\n")
    hits = [i for i, l in enumerate(lines) if l.strip() == PKG_LINE_SRC]
    if len(hits) != 1:
        raise VendorError("expected exactly one line %r in _pkg.py, found %d" % (PKG_LINE_SRC, len(hits)))
    lines[hits[0]] = lines[hits[0]].replace(PKG_LINE_SRC, PKG_LINE_DST)
    out = dict(contents)
    out["_pkg.py"] = "\n".join(lines).encode("utf-8")
    return out


def local_edits(target):
    """Problems if ``target`` deviates from its manifest; [] if clean or absent/empty."""
    files = list_files(target)
    mpath = os.path.join(target, MANIFEST)
    if not os.path.isfile(mpath):
        return ["%s exists without %s" % (target, MANIFEST)] if files else []
    want = parse_manifest(read_bytes(mpath).decode("utf-8"))
    probs = []
    for rel in sorted(set(want) | set(files)):
        if rel not in files:
            probs.append("deleted locally: %s" % rel)
        elif rel not in want:
            probs.append("added locally: %s" % rel)
        elif sha256_bytes(read_bytes(files[rel])) != want[rel]:
            probs.append("modified locally: %s" % rel)
    return probs


def plan_sync(target, contents):
    """-> (writes, deletes): relpaths to (over)write and stale relpaths to delete."""
    existing = list_files(target)
    writes = [rel for rel in sorted(contents)
              if rel not in existing or read_bytes(existing[rel]) != contents[rel]]
    deletes = [rel for rel in sorted(existing) if rel not in contents]
    mpath = os.path.join(target, MANIFEST)
    mtext = manifest_text(contents).encode("utf-8")
    manifest_stale = not os.path.isfile(mpath) or read_bytes(mpath) != mtext
    return writes, deletes, manifest_stale


def write_file(path, data):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    tmp = path + ".vendor-tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def remove_empty_dirs(root):
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        try:
            entries = [e for e in os.listdir(dirpath) if e != "__pycache__"]
            if not entries:
                pyc = os.path.join(dirpath, "__pycache__")
                if os.path.isdir(pyc):
                    for f in os.listdir(pyc):
                        os.remove(os.path.join(pyc, f))
                    os.rmdir(pyc)
                os.rmdir(dirpath)
        except OSError:
            pass


def apply_sync(target, contents, writes, deletes):
    for rel in writes:
        write_file(os.path.join(target, *rel.split("/")), contents[rel])
    for rel in deletes:
        os.remove(os.path.join(target, *rel.split("/")))
    write_file(os.path.join(target, MANIFEST), manifest_text(contents).encode("utf-8"))
    remove_empty_dirs(target)


def vendor(fry_repo, src_root=DEFAULT_SRC_ROOT, force=False, dry_run=False, check=False, out=sys.stdout):
    """Returns process exit code (0 ok/in sync, 1 out of sync for --check, 2 refused)."""
    src_pkg = os.path.join(src_root, "shared", "gateway")
    src_tests = os.path.join(src_root, "tests", "gateway")
    if not os.path.isfile(os.path.join(src_pkg, "__init__.py")):
        raise VendorError("source package not found: %s" % src_pkg)
    if not os.path.isdir(fry_repo):
        raise VendorError("fry repo not found: %s" % fry_repo)
    pkg_contents = {rel: read_bytes(p) for rel, p in list_files(src_pkg).items()}
    test_contents = rewrite_tests({rel: read_bytes(p) for rel, p in list_files(src_tests).items()})
    targets = [(os.path.join(fry_repo, "fry_gateway"), pkg_contents),
               (os.path.join(fry_repo, "tests", "gateway"), test_contents)]

    if check:
        dirty = False
        for target, contents in targets:
            writes, deletes, mstale = plan_sync(target, contents)
            for rel in writes:
                out.write("out of sync: %s/%s\n" % (target, rel))
            for rel in deletes:
                out.write("stale: %s/%s\n" % (target, rel))
            if mstale:
                out.write("manifest out of date: %s\n" % os.path.join(target, MANIFEST))
            dirty = dirty or bool(writes or deletes or mstale)
        out.write("in sync\n" if not dirty else "")
        return 1 if dirty else 0

    problems = []
    for target, _ in targets:
        problems.extend(local_edits(target))
    if problems and not force:
        out.write("refusing to vendor: target has local edits (use --force to overwrite):\n")
        for p in problems:
            out.write("  %s\n" % p)
        return 2

    src_manifest = manifest_text(pkg_contents).encode("utf-8")
    total_w = total_d = 0
    for target, contents in targets:
        writes, deletes, _ = plan_sync(target, contents)
        total_w += len(writes)
        total_d += len(deletes)
        for rel in writes:
            out.write("%s %s/%s\n" % ("would write" if dry_run else "write", target, rel))
        for rel in deletes:
            out.write("%s %s/%s\n" % ("would delete" if dry_run else "delete", target, rel))
        if not dry_run:
            apply_sync(target, contents, writes, deletes)
    tests_init = os.path.join(fry_repo, "tests", "__init__.py")
    if not dry_run:
        if not os.path.isfile(tests_init):
            write_file(tests_init, b"")
        mpath = os.path.join(src_pkg, MANIFEST)
        if not os.path.isfile(mpath) or read_bytes(mpath) != src_manifest:
            write_file(mpath, src_manifest)
    out.write("%s: %d file(s) written, %d stale file(s) deleted\n" % ("dry run" if dry_run else "vendored",
                                                                       total_w, total_d))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Vendor shared/gateway into fry-launch-claude/fry_gateway")
    ap.add_argument("fry_repo", help="path to the fry-launch-claude checkout")
    ap.add_argument("--force", action="store_true", help="overwrite local edits in the target")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--check", action="store_true", help="exit 1 if the target is out of sync (no writes)")
    ap.add_argument("--src-root", default=DEFAULT_SRC_ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        return vendor(os.path.abspath(args.fry_repo), os.path.abspath(args.src_root), force=args.force,
                      dry_run=args.dry_run, check=args.check)
    except VendorError as exc:
        sys.stderr.write("vendor_gateway: %s\n" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
