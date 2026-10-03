"""cli dialect (DESIGN §3.5) with a fake ``opencode`` binary written by the test."""

import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from ._pkg import mod

ev = mod("events")
model = mod("model")
config = mod("config")
errors = mod("errors")
presets = mod("presets")
dbase = mod("dialects.base")
cli = mod("dialects.cli")

SECRET = "FAKEKEY-SENTINEL-cli-9876543210"
B = model.Block

FAKE_OPENCODE = r'''
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "behaviour.json"), "r", encoding="utf-8") as f:
    beh = json.load(f)
prompt = sys.stdin.buffer.read().decode("utf-8")
start = time.time()
mode = beh.get("mode", "ok")
if mode == "sleep":
    time.sleep(beh.get("seconds", 0.5))
rec = {"argv": sys.argv[1:], "stdin": prompt, "cwd": os.getcwd(), "start": start, "end": time.time(),
       "no_color": os.environ.get("NO_COLOR"),
       "secret_in_env": any(beh.get("secret", "\0") in v for v in os.environ.values())}
with open(os.path.join(here, "calls", "%d-%f.json" % (os.getpid(), start)), "w", encoding="utf-8") as f:
    json.dump(rec, f)
out = sys.stdout.buffer
if mode == "fail":
    sys.stderr.buffer.write(("\x1b[31mError: model not found\x1b[0m " + beh.get("secret", "")).encode("utf-8"))
    sys.exit(3)
if mode == "empty":
    sys.exit(0)
out.write(b"\x1b]0;opencode\x07\x1b[2K\r\x1b[0m\n> build \xc2\xb7 " + sys.argv[-1].encode("utf-8") + b"\n\n")
out.write(beh.get("text", "Hello \x1b[1mworld\x1b[22m ✓\n\nsecond line").encode("utf-8") + b"\n")
'''


def make_fake_opencode(directory, **behaviour):
    """Write the fake CLI into ``directory``; returns the path of the runnable binary."""
    os.makedirs(os.path.join(directory, "calls"), exist_ok=True)
    script = os.path.join(directory, "fake_opencode.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(FAKE_OPENCODE)
    set_behaviour(directory, **behaviour)
    if os.name == "nt":
        path = os.path.join(directory, "opencode.cmd")
        with open(path, "w", encoding="utf-8") as f:
            f.write('@"%s" "%s" %%*\r\n' % (sys.executable, script))
    else:
        path = os.path.join(directory, "opencode")
        with open(path, "w", encoding="utf-8") as f:
            f.write("#!%s\n" % sys.executable)
            f.write(FAKE_OPENCODE)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def set_behaviour(directory, **behaviour):
    behaviour.setdefault("secret", SECRET)
    with open(os.path.join(directory, "behaviour.json"), "w", encoding="utf-8") as f:
        json.dump(behaviour, f)


def calls(directory):
    out = []
    d = os.path.join(directory, "calls")
    for name in sorted(os.listdir(d)):
        with open(os.path.join(d, name), "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


class Resolution(object):
    def __init__(self, provider, model_id, spec):
        self.provider, self.model, self.model_spec, self.requested = provider, model_id, spec, model_id
        self.role, self.background = None, False


def transcript_request(max_tokens=32000, output_format=None):
    msgs = [
        model.Message("user", [B.of_text("<system-reminder>\nsecret reminder\n</system-reminder>\n"),
                               B.of_text("List the files"), B.of_image_base64("image/png", "AAAA")]),
        model.Message("assistant", [B.of_thinking("hidden thoughts", "fgw1.chat.0123456789ab"),
                                    B.of_text("Running ls"),
                                    B.of_tool_use("toolu_1", "Bash", {"command": "ls"})]),
        model.Message("user", [B.of_tool_result("toolu_1", "a.txt\nb.txt"),
                               B.of_tool_result("toolu_1", [B.of_image_base64("image/png", "AA")], is_error=True),
                               B.of_text("<system-reminder>more</system-reminder>What now?")]),
    ]
    return model.NormalizedRequest(model="claude-via-opencode,nemotron-3-ultra-free", max_tokens=max_tokens,
                                   system=["You are a helpful assistant.", "<system-reminder>x</system-reminder>"],
                                   messages=msgs, output_format=output_format)


class CliTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gw-cli-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bin = make_fake_opencode(self.tmp)

    def provider(self, **options):
        opts = {"cli_bin": self.bin}
        opts.update(options)
        return presets.provider_from_preset("opencode", {"options": opts})

    def ctx(self, provider, model_id="nemotron-3-ultra-free", req=None, secrets=None):
        req = req or transcript_request()
        spec = provider.model_spec(model_id)
        secrets = secrets if secrets is not None else config.SecretStore({"xai": SECRET})
        return dbase.RequestContext(req, Resolution(provider, model_id, spec), provider, spec,
                                    dbase.ProviderRuntime(provider.id), None, None, "sess", est_tokens=123,
                                    secrets=secrets)

    def run_dialect(self, ctx):
        return list(cli.CliDialect().execute(ctx))


class CliDialectTests(CliTestBase):
    def test_run_prompt_on_stdin_and_clean_output(self):
        events = self.run_dialect(self.ctx(self.provider()))  # SECRET lives only in the SecretStore
        self.assertEqual(events[0], ev.TextDelta(0, "Hello world ✓\n\nsecond line"))
        self.assertIsInstance(events[1], ev.Usage)
        self.assertEqual(events[1].input_tokens, 123)
        self.assertGreater(events[1].output_tokens, 0)
        self.assertEqual(events[2], ev.Finish("end_turn"))
        (call,) = calls(self.tmp)
        self.assertEqual(call["argv"], ["run", "--model", "opencode/nemotron-3-ultra-free"])
        prompt = call["stdin"]
        self.assertNotIn("system-reminder", prompt)
        self.assertNotIn("secret reminder", prompt)
        self.assertNotIn("hidden thoughts", prompt)
        for piece in ("System:\nYou are a helpful assistant.", "User:\nList the files\n\n[image]",
                      "Assistant:\nRunning ls\n\nTool call Bash: {\"command\":\"ls\"}",
                      "Tool result (Bash):\na.txt\nb.txt", "Tool result (Bash) [error]:\n[image]", "What now?"):
            self.assertIn(piece, prompt)
        self.assertTrue(prompt.startswith(cli.PREAMBLE))
        self.assertEqual(call["no_color"], "1")
        self.assertFalse(call["secret_in_env"], "gateway-held secret placed into the CLI environment")
        self.assertNotIn(SECRET, " ".join(call["argv"]) + prompt)
        self.assertNotEqual(os.path.realpath(call["cwd"]), os.path.realpath(os.getcwd()))

    def test_model_mapping_and_structured_output(self):
        self.assertEqual(cli.cli_model_id("nemotron-3-ultra-free"), "opencode/nemotron-3-ultra-free")
        self.assertEqual(cli.cli_model_id("anthropic/claude-x"), "anthropic/claude-x")
        self.assertEqual(cli.cli_model_id("big-model[1m]"), "opencode/big-model")
        self.assertEqual(cli.cli_model_id("m", prefix=""), "m")
        req = transcript_request(output_format={"type": "json_schema", "schema": {"type": "object"}})
        self.run_dialect(self.ctx(self.provider(), model_id="openrouter/some-model", req=req))
        call = calls(self.tmp)[-1]
        self.assertEqual(call["argv"][-1], "openrouter/some-model")
        self.assertIn('Respond with ONLY a JSON object matching this JSON schema: {"type":"object"}', call["stdin"])

    def test_nonzero_exit_is_502_with_redacted_stderr_tail(self):
        set_behaviour(self.tmp, mode="fail")
        with self.assertRaises(errors.GatewayError) as cm:
            self.run_dialect(self.ctx(self.provider()))
        e = cm.exception
        self.assertEqual((e.status, e.err_type, e.should_retry), (502, "api_error", False))
        self.assertIn("exited with status 3", e.message)
        self.assertIn("Error: model not found", e.message)
        self.assertNotIn("\x1b", e.message)
        self.assertNotIn(SECRET, e.message)

    def test_empty_output_is_an_error(self):
        set_behaviour(self.tmp, mode="empty")
        with self.assertRaises(errors.GatewayError) as cm:
            self.run_dialect(self.ctx(self.provider()))
        self.assertEqual(cm.exception.status, 502)

    def test_missing_binary(self):
        p = self.provider(cli_bin=os.path.join(self.tmp, "does-not-exist"))
        with self.assertRaises(errors.GatewayError) as cm:
            self.run_dialect(self.ctx(p))
        e = cm.exception
        self.assertEqual((e.status, e.should_retry), (503, False))
        self.assertIn("npm i -g opencode-ai", e.message)
        p = presets.provider_from_preset("opencode")
        with mock.patch.dict(os.environ, {"PATH": os.path.join(self.tmp, "empty")}):
            with self.assertRaises(errors.GatewayError) as cm:
                self.run_dialect(self.ctx(p))
        self.assertEqual(cm.exception.status, 503)

    def test_binary_found_on_path(self):
        p = presets.provider_from_preset("opencode")
        with mock.patch.dict(os.environ, {"PATH": self.tmp + os.pathsep + os.environ.get("PATH", "")}):
            events = self.run_dialect(self.ctx(p))
        self.assertEqual(events[-1], ev.Finish("end_turn"))
        self.assertEqual(len(calls(self.tmp)), 1)

    def test_probe_answered_locally(self):
        p = self.provider(cli_bin=os.path.join(self.tmp, "does-not-exist"))
        events = self.run_dialect(self.ctx(p, req=transcript_request(max_tokens=1)))
        self.assertEqual(events[0], ev.TextDelta(0, "ok"))
        self.assertEqual(events[-1], ev.Finish("end_turn"))
        self.assertEqual(calls(self.tmp), [])

    def test_timeout_kills_cli(self):
        set_behaviour(self.tmp, mode="sleep", seconds=30)
        t0 = time.time()
        with self.assertRaises(errors.GatewayError) as cm:
            self.run_dialect(self.ctx(self.provider(timeout=1)))
        self.assertLess(time.time() - t0, 15)
        self.assertIn("timed out", cm.exception.message)
        self.assertFalse(cm.exception.should_retry)

    def test_semaphore_limits_concurrency_to_two(self):
        set_behaviour(self.tmp, mode="sleep", seconds=0.6)
        p = self.provider()
        results, errs = [], []

        def worker():
            try:
                results.append(self.run_dialect(self.ctx(p)))
            except Exception as exc:  # pragma: no cover - surfaced below
                errs.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(errs, [])
        self.assertEqual(len(results), 5)
        spans = [(c["start"], c["end"]) for c in calls(self.tmp)]
        self.assertEqual(len(spans), 5)
        peak = max(sum(1 for s, e in spans if s <= t < e) for t, _ in spans)
        self.assertLessEqual(peak, cli.DEFAULT_CONCURRENCY)
        self.assertGreaterEqual(peak, 2)


class CleanOutputTests(unittest.TestCase):
    def test_strip_ansi_variants(self):
        raw = ("\x1b[38;5;196mred\x1b[0m \x1b]8;;https://x\x1b\\link\x1b]8;;\x1b\\ \x1bPq#0\x1b\\dcs "
               "\x9b1mc1\x1b(B charset \x1b=keypad\x07bell")
        self.assertEqual(cli.strip_ansi(raw), "red link dcs c1 charset keypadbell")

    def test_headers_and_carriage_returns(self):
        out = "\n> build · opencode/nemotron-3-ultra-free\n\nspinner\rFinal answer\n  indented code\n\n"
        self.assertEqual(cli.clean_output(out), "Final answer\n  indented code")
        self.assertEqual(cli.clean_output("> quoted markdown line"), "> quoted markdown line")


if __name__ == "__main__":
    unittest.main()
