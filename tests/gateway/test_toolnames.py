"""toolnames: ToolNameMap (shortening, sanitizing, collisions, reverse map) and tool-id encoding."""

import hashlib
import random
import re
import string
import unittest

from ._pkg import mod

tn = mod("toolnames")
mocks = mod("testing.mock_upstreams")
testing = mod("testing")
presets = mod("presets")

LONG_MCP = "mcp__stub__get_the_magic_number_for_an_extremely_long_tool_name_that_exceeds_limits_x"  # 85 chars
DEFAULT_RE = re.compile(tn.DEFAULT_TOOL_NAME_REGEX)
GEMINI_RE = re.compile(presets.GEMINI_TOOL_NAME_REGEX)


def sha8(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


class ToolNameMapTests(unittest.TestCase):
    def test_valid_names_unchanged(self):
        names = ["Bash", "Read", "mcp__srv__tool-1", "x" * 64]
        m = tn.ToolNameMap(names)
        for n in names:
            self.assertEqual(m.upstream(n), n)
            self.assertEqual(m.original(n), n)
            self.assertFalse(m.is_renamed(n))
        self.assertEqual(len(m), 4)

    def test_long_mcp_name_shortened_exactly(self):
        self.assertEqual(len(LONG_MCP), 85)
        m = tn.ToolNameMap(["Bash", LONG_MCP])
        up = m.upstream(LONG_MCP)
        self.assertEqual(up, LONG_MCP[:55] + "_" + sha8(LONG_MCP))
        self.assertEqual(len(up), 64)
        self.assertTrue(DEFAULT_RE.match(up))
        self.assertEqual(m.original(up), LONG_MCP)
        self.assertTrue(m.is_renamed(LONG_MCP))
        # the testing Brain recognizes the shortened form of its tool
        brain = mocks.Brain(tool=LONG_MCP)
        self.assertEqual(brain.find_tool(["Bash", up]), up)

    def test_invalid_chars_replaced_and_hashed(self):
        m = tn.ToolNameMap(["my.tool", "srv/tool name", "ünïcode"])
        self.assertEqual(m.upstream("my.tool"), "my_tool_" + sha8("my.tool"))
        self.assertEqual(m.upstream("srv/tool name"), "srv_tool_name_" + sha8("srv/tool name"))
        self.assertTrue(DEFAULT_RE.match(m.upstream("ünïcode")))
        for orig, up in m.items():
            self.assertEqual(m.original(up), orig)

    def test_custom_maxlen(self):
        name = "a" * 40
        m = tn.ToolNameMap([name], maxlen=32)
        up = m.upstream(name)
        self.assertEqual(len(up), 32)
        self.assertEqual(up, "a" * 23 + "_" + sha8(name))

    def test_gemini_regex_and_leading_letter(self):
        m = tn.ToolNameMap(["ns.tool:v1", "1st-tool", "-dash", LONG_MCP], regex=presets.GEMINI_TOOL_NAME_REGEX,
                           leading_letter=True)
        self.assertEqual(m.upstream("ns.tool:v1"), "ns.tool:v1")  # dots/colons allowed by Gemini
        for n in ("1st-tool", "-dash", LONG_MCP):
            up = m.upstream(n)
            self.assertTrue(GEMINI_RE.match(up), up)
            self.assertTrue(up[0].isalpha() or up[0] == "_")
            self.assertEqual(m.original(up), n)
        self.assertEqual(m.upstream("1st-tool"), "_1st-tool_" + sha8("1st-tool"))

    def test_collisions_extend_hash(self):
        a = "p" * 70 + "A"
        b = "p" * 70 + "B"
        m = tn.ToolNameMap([a, b])
        self.assertNotEqual(m.upstream(a), m.upstream(b))  # different hashes already
        # force a collision: a valid name equal to the shortened form of another name
        shortened = "my_tool_" + sha8("my.tool")
        m = tn.ToolNameMap(["my.tool", shortened])
        self.assertEqual(m.upstream(shortened), shortened)
        up = m.upstream("my.tool")
        self.assertNotEqual(up, shortened)
        self.assertTrue(up.startswith("my_tool_"))
        self.assertEqual(up, "my_tool_" + hashlib.sha1(b"my.tool").hexdigest()[:9])
        self.assertEqual(m.original(up), "my.tool")
        self.assertEqual(m.original(shortened), shortened)

    def test_unknown_names(self):
        m = tn.ToolNameMap(["Bash"])
        self.assertEqual(m.original("Hallucinated"), "Hallucinated")
        up = m.upstream("late.added")  # mapped on the fly, stays reversible
        self.assertEqual(m.original(up), "late.added")
        self.assertEqual(m.items()[-1], ("late.added", up))
        self.assertEqual(tn.ToolNameMap(None).upstream("x"), "x")
        self.assertEqual(tn.ToolNameMap(["", None, "a", "a"]).items(), [("a", "a")])

    def test_random_round_trips_are_unique_and_valid(self):
        rnd = random.Random(42)
        alphabet = string.ascii_letters + string.digits + "_-.:/ é"
        names = set()
        while len(names) < 300:
            names.add("".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 120))))
        names = sorted(names)
        for regex, lead in ((None, False), (presets.GEMINI_TOOL_NAME_REGEX, True)):
            m = tn.ToolNameMap(names, regex=regex, leading_letter=lead)
            ups = [m.upstream(n) for n in names]
            self.assertEqual(len(set(ups)), len(names))
            rx = re.compile(regex or tn.DEFAULT_TOOL_NAME_REGEX)
            for n, up in zip(names, ups):
                self.assertTrue(rx.match(up), up)
                self.assertLessEqual(len(up), 64)
                self.assertEqual(m.original(up), n)

    def test_fixture_tools(self):
        body = testing.load_fixture("turn1_mcp_long_tool_name_request.json")["body"]
        names = [t["name"] for t in body["tools"]]
        m = tn.ToolNameMap(names)
        renamed = [n for n in names if m.is_renamed(n)]
        self.assertEqual(renamed, [LONG_MCP])
        self.assertEqual(sorted(m.original(m.upstream(n)) for n in names), sorted(names))


class ToolIdTests(unittest.TestCase):
    def test_passthrough(self):
        for tid in ("call_abc123", "toolu_01ABCdef", "chatcmpl-tool-1", "a" * 128):
            self.assertEqual(tn.encode_tool_id(tid), tid)
            self.assertEqual(tn.decode_tool_id(tid), tid)

    def test_encoded_round_trip(self):
        for tid in ("functions.Bash:0", "call 1", "a" * 129, "ид/1", "x" * 300):
            enc = tn.encode_tool_id(tid)
            self.assertTrue(enc.startswith("toolu_x"))
            self.assertTrue(re.match(r"^[A-Za-z0-9_-]+$", enc), enc)
            self.assertEqual(tn.decode_tool_id(enc), tid)

    def test_missing_ids_generated(self):
        a, b = tn.encode_tool_id(None), tn.encode_tool_id("")
        for x in (a, b, tn.new_tool_id()):
            self.assertTrue(re.match(r"^toolu_[0-9a-f]{24}$", x), x)
        self.assertNotEqual(a, b)
        self.assertEqual(tn.encode_tool_id(7), "7")

    def test_decode_leaves_lookalikes_alone(self):
        # genuine ids that happen to start with the prefix, or would decode to a pass-through id
        for tid in ("toolu_xyz", "toolu_x", "toolu_x" + "YWJj", "toolu_x!!", None, 5):
            self.assertEqual(tn.decode_tool_id(tid), tid)
        # an encoded form whose decoded value would have passed through verbatim is not ours
        self.assertEqual(tn.decode_tool_id("toolu_xY2FsbF8x"), "toolu_xY2FsbF8x")  # b64("call_1")


if __name__ == "__main__":
    unittest.main()
