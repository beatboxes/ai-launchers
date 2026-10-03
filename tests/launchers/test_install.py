"""install.sh / uninstall.sh in a sandbox HOME: absolute-python shims, tagged PATH line in the right rc
files (idempotent), fish hint, portable removal incl. the v0.1 untagged line (POSIX only)."""

import os
import shutil
import subprocess
import tempfile
import unittest

from ._util import REPO_ROOT

TAG = "# ai-launchers"


@unittest.skipIf(os.name == "nt" or not shutil.which("bash"), "POSIX shell scripts")
class InstallScriptTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="ail-install-")
        self.addCleanup(shutil.rmtree, self.home, True)
        self.bin_dir = os.path.join(self.home, ".ai-launchers", "bin")

    def sh(self, script, shell="/bin/bash"):
        env = {"HOME": self.home, "SHELL": shell, "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
        proc = subprocess.run(["bash", os.path.join(REPO_ROOT, script)], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stdout)
        return proc.stdout

    def rc(self, name, content=None):
        path = os.path.join(self.home, name)
        if content is not None:
            with open(path, "w") as f:
                f.write(content)
        return path

    def read(self, name):
        with open(os.path.join(self.home, name)) as f:
            return f.read()

    def test_shims_and_bashrc_idempotent(self):
        self.rc(".bashrc", "alias ll='ls -l'")             # no trailing newline
        self.sh("install.sh")
        out = self.sh("install.sh")
        self.assertIn("already in", out)
        text = self.read(".bashrc")
        self.assertEqual(text.count(TAG), 1)
        self.assertTrue(text.startswith("alias ll='ls -l'\nexport PATH="))
        self.assertFalse(os.path.exists(os.path.join(self.home, ".zshrc")))
        self.assertFalse(os.path.exists(os.path.join(self.home, ".profile")))
        for p in ("grok", "codex", "gemini", "deepseek", "kimi"):
            shim = os.path.join(self.bin_dir, "%s-wrap" % p)
            self.assertTrue(os.access(shim, os.X_OK), shim)
            with open(shim) as f:
                line = f.read().splitlines()[1]
            py = line.split('"')[1]
            self.assertTrue(os.path.isabs(py), line)
        proc = subprocess.run([os.path.join(self.bin_dir, "grok-wrap"), "--version"], stdout=subprocess.PIPE,
                              universal_newlines=True, timeout=60, env={"HOME": self.home, "PATH": "/usr/bin:/bin"})
        self.assertIn("grok-wrap 0.2.0", proc.stdout)

    def test_zsh_login_shell_creates_zshrc(self):
        self.rc(".bashrc", "")
        self.sh("install.sh", shell="/bin/zsh")
        self.assertEqual(self.read(".zshrc").count(TAG), 1)
        self.assertEqual(self.read(".bashrc").count(TAG), 1)

    def test_profile_fallbacks_and_fish(self):
        out = self.sh("install.sh", shell="/usr/bin/fish")
        self.assertIn("fish_add_path %s" % self.bin_dir, out)
        self.assertEqual(self.read(".profile").count(TAG), 1)
        os.remove(os.path.join(self.home, ".profile"))
        self.rc(".bash_profile", "")
        self.sh("install.sh")
        self.assertEqual(self.read(".bash_profile").count(TAG), 1)
        self.assertFalse(os.path.exists(os.path.join(self.home, ".profile")))

    def test_uninstall_removes_tagged_and_legacy_lines(self):
        legacy = 'export PATH="%s:$PATH"' % self.bin_dir
        self.rc(".bashrc", "keep-me\n%s\n" % legacy)
        self.rc(".zshrc", "zsh-keep\n")
        self.sh("install.sh", shell="/bin/zsh")
        self.sh("uninstall.sh")
        self.assertEqual(self.read(".bashrc"), "keep-me\n")
        self.assertEqual(self.read(".zshrc"), "zsh-keep\n")
        self.assertFalse(os.path.exists(self.bin_dir))
        leftovers = [n for n in os.listdir(self.home) if n.endswith((".bak", ".tmp"))]
        self.assertEqual(leftovers, [])
        self.sh("uninstall.sh")                              # idempotent


if __name__ == "__main__":
    unittest.main()
