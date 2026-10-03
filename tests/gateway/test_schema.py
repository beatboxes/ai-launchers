"""schema.scrub: every mode, $ref cycles, root combinators, immutability, Claude Code fixture tools."""

import copy
import json
import unittest

from ._pkg import mod

schema = mod("schema")
config = mod("config")
testing = mod("testing")

SAMPLE = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:x",
    "type": "object",
    "properties": {
        "$schema": {"type": "string", "$id": "inner"},          # a property literally called "$schema"
        "items": {"type": "array", "items": {"type": "string", "$schema": "x"}},
        "mode": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["items"],
    "additionalProperties": False,
}


def walk(node):
    """All dict nodes of a JSON value."""
    out = []
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            out.append(n)
            stack.extend(n.values())
        elif isinstance(n, list):
            stack.extend(n)
    return out


def schema_nodes(node):
    """Schema objects (not property-name maps) of an OpenAPI-style schema."""
    out = [node]
    for sub in (node.get("properties") or {}).values():
        out.extend(schema_nodes(sub))
    if isinstance(node.get("items"), dict):
        out.extend(schema_nodes(node["items"]))
    for sub in node.get("anyOf") or []:
        out.extend(schema_nodes(sub))
    return out


class BasicModesTests(unittest.TestCase):
    def test_modes_constant_matches_config(self):
        self.assertEqual(tuple(schema.MODES), tuple(config.SCHEMA_MODES))
        with self.assertRaises(ValueError):
            schema.scrub({}, "bogus")

    def test_none_is_deep_copy(self):
        out = schema.scrub(SAMPLE, "none")
        self.assertEqual(out, SAMPLE)
        self.assertIsNot(out, SAMPLE)
        out["properties"]["items"]["type"] = "x"
        self.assertEqual(SAMPLE["properties"]["items"]["type"], "array")

    def test_basic_drops_keywords_only(self):
        before = copy.deepcopy(SAMPLE)
        out = schema.scrub(SAMPLE, "basic")
        self.assertEqual(SAMPLE, before)  # never mutates
        self.assertNotIn("$schema", out)
        self.assertNotIn("$id", out)
        self.assertIn("$schema", out["properties"])  # property name kept
        self.assertEqual(out["properties"]["$schema"], {"type": "string"})
        self.assertEqual(out["properties"]["items"]["items"], {"type": "string"})
        self.assertEqual(out["additionalProperties"], False)
        self.assertEqual(schema.scrub(SAMPLE, None), out)  # None -> basic
        self.assertEqual(schema.scrub("not a schema", "basic"), {})


class NoRootCombinatorTests(unittest.TestCase):
    def test_root_anyof_merged(self):
        s = {"$schema": "x", "description": "d", "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}, "kind": {"const": "A"}},
             "required": ["a", "kind"]},
            {"type": "object", "properties": {"b": {"type": "integer"}, "kind": {"const": "B"}},
             "required": ["kind", "b"]},
        ]}
        out = schema.scrub(s, "no_root_combinators")
        self.assertEqual(out["type"], "object")
        self.assertEqual(out["description"], "d")
        self.assertNotIn("anyOf", out)
        self.assertNotIn("$schema", out)
        self.assertEqual(list(out["properties"]), ["a", "kind", "b"])
        self.assertEqual(out["properties"]["kind"], {"anyOf": [{"const": "A"}, {"const": "B"}]})
        self.assertEqual(out["required"], ["kind"])  # intersection

    def test_oneof_with_refs_and_root_props(self):
        s = {"type": "object", "properties": {"common": {"type": "string"}}, "required": ["common"],
             "$defs": {"X": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}},
             "oneOf": [{"$ref": "#/$defs/X"}, {"$ref": "#/$defs/X", "description": "same"}]}
        out = schema.scrub(s, "no_root_combinators")
        self.assertEqual(out["required"], ["common", "x"])
        self.assertEqual(set(out["properties"]), {"common", "x"})
        self.assertIn("$defs", out)  # nested refs may still point there

    def test_allof_union_and_nested(self):
        s = {"allOf": [{"properties": {"a": {"type": "string"}}, "required": ["a"]},
                       {"anyOf": [{"properties": {"b": {"type": "string"}}, "required": ["b"]},
                                  {"properties": {"c": {"type": "string"}}, "required": ["c"]}]},
                       True]}
        out = schema.scrub(s, "no_root_combinators")
        self.assertEqual(out["required"], ["a"])
        self.assertEqual(set(out["properties"]), {"a", "b", "c"})
        self.assertFalse(any(k in out for k in ("allOf", "anyOf", "oneOf")))

    def test_nested_combinators_kept_and_plain_schema_untouched(self):
        plain = {"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}
        self.assertEqual(schema.scrub(plain, "no_root_combinators"), plain)


class GeminiJsonTests(unittest.TestCase):
    def test_inline_refs_and_drop_defs(self):
        s = {"type": "object", "$defs": {"Pos": {"type": "object", "properties": {"x": {"type": "number"}}}},
             "definitions": {"Name": {"type": "string", "minLength": 1}},
             "properties": {"p": {"$ref": "#/$defs/Pos", "description": "where"},
                            "n": {"$ref": "#/definitions/Name"},
                            "same": {"$ref": "#/properties/n"},
                            "ext": {"$ref": "https://example.com/s.json"}}}
        out = schema.scrub(s, "gemini_json")
        self.assertNotIn("$defs", out)
        self.assertNotIn("definitions", out)
        self.assertEqual(out["properties"]["p"], {"type": "object", "properties": {"x": {"type": "number"}},
                                                  "description": "where"})
        self.assertEqual(out["properties"]["n"], {"type": "string", "minLength": 1})
        self.assertEqual(out["properties"]["same"], {"type": "string", "minLength": 1})
        self.assertEqual(out["properties"]["ext"], {})
        self.assertFalse(any("$ref" in n for n in walk(out)))

    def test_ref_cycle_bounded(self):
        s = {"$defs": {"Node": {"type": "object", "properties": {
            "value": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
            "parent": {"$ref": "#"}}}},
             "$ref": "#/$defs/Node"}
        out = schema.scrub(s, "gemini_json")
        self.assertFalse(any("$ref" in n for n in walk(out)))
        depth, node = 0, out
        while node:
            node = node.get("properties", {}).get("children", {}).get("items")
            depth += 1
        self.assertLessEqual(depth, schema.MAX_REF_DEPTH + 1)
        self.assertGreaterEqual(depth, schema.MAX_REF_DEPTH)
        self.assertLess(len(json.dumps(out)), 2000000)

    def test_exponential_fanout_is_bounded(self):
        s = {"$defs": {"T": {"type": "object", "properties": dict(("k%d" % i, {"$ref": "#/$defs/T"})
                                                                   for i in range(6))}},
             "$ref": "#/$defs/T"}
        out = schema.scrub(s, "gemini_json")
        self.assertLess(len(walk(out)), 50000)

    def test_null_type_collapse(self):
        s = {"type": "object", "properties": {"a": {"type": ["string", "null"]}, "b": {"type": ["null", "integer"]},
                                              "c": {"type": ["string", "number"]},
                                              "d": {"type": ["string", "number", "null"]}}}
        out = schema.scrub(s, "gemini_json")
        props = out["properties"]
        self.assertEqual(props["a"]["type"], "string")
        self.assertEqual(props["b"]["type"], "integer")
        self.assertEqual(props["c"]["type"], ["string", "number"])
        self.assertEqual(props["d"]["type"], ["string", "number", "null"])


class GeminiOpenApiTests(unittest.TestCase):
    def test_allow_list_and_conversions(self):
        s = {"$schema": "x", "type": "object", "additionalProperties": False, "title": "T",
             "properties": {
                 "n": {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 10, "format": "int64",
                       "default": 3, "examples": [1]},
                 "s": {"type": ["string", "null"], "format": "uri", "pattern": "^a", "maxLength": 5},
                 "k": {"const": "fixed"},
                 "e": {"enum": ["a", "b", None]},
                 "num_enum": {"type": "integer", "enum": [1, 2]},
                 "d": {"type": "string", "format": "date-time"},
                 "o": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                 "maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                 "arr": {"type": "array", "items": [{"type": "string"}], "minItems": 1, "uniqueItems": True},
                 "tuple": {"prefixItems": [{"type": "number"}]},
             },
             "required": ["n", "missing"]}
        out = schema.scrub(s, "gemini_openapi")
        allowed = set(("type", "format", "description", "nullable", "enum", "properties", "required", "items",
                       "minItems", "maxItems", "minimum", "maximum", "minLength", "maxLength", "pattern", "anyOf",
                       "title"))
        p = out["properties"]
        self.assertEqual(set(out), {"type", "title", "properties", "required"})
        self.assertEqual(out["required"], ["n"])
        self.assertEqual(p["n"], {"type": "integer", "format": "int64", "minimum": 0, "maximum": 10})
        self.assertEqual(p["s"], {"type": "string", "nullable": True, "pattern": "^a", "maxLength": 5})
        self.assertEqual(p["k"], {"enum": ["fixed"]})
        self.assertEqual(p["e"], {"enum": ["a", "b"], "nullable": True})
        self.assertEqual(p["num_enum"], {"type": "integer"})
        self.assertEqual(p["d"], {"type": "string", "format": "date-time"})
        self.assertEqual(p["o"], {"anyOf": [{"type": "string"}, {"type": "integer"}]})
        self.assertEqual(p["maybe"], {"type": "string", "nullable": True})
        self.assertEqual(p["arr"], {"type": "array", "items": {"type": "string"}, "minItems": 1})
        self.assertEqual(p["tuple"], {"type": "array", "items": {"type": "number"}})
        for node in schema_nodes(out):
            self.assertTrue(set(node) <= allowed, node)

    def test_allof_merged_refs_inlined(self):
        s = {"type": "object", "$defs": {"Base": {"properties": {"id": {"type": "string"}}, "required": ["id"]}},
             "allOf": [{"$ref": "#/$defs/Base"}, {"properties": {"to": {"allOf": [
                 {"pattern": "^[^\\n\\r]*$"}, {"pattern": "^[\\s\\S]{0,300}$"}], "type": "string"}},
                 "required": ["to"]}]}
        out = schema.scrub(s, "gemini_openapi")
        self.assertEqual(out["required"], ["id", "to"])
        self.assertEqual(out["properties"]["id"], {"type": "string"})
        self.assertEqual(out["properties"]["to"], {"type": "string", "pattern": "^[^\\n\\r]*$"})


class FixtureToolsTests(unittest.TestCase):
    def test_all_modes_on_real_claude_code_tools(self):
        body = testing.load_fixture("turn1_mcp_long_tool_name_request.json")["body"]
        for tool in body["tools"]:
            original = copy.deepcopy(tool["input_schema"])
            for mode in schema.MODES:
                out = schema.scrub(tool["input_schema"], mode)
                self.assertIsInstance(out, dict)
                json.dumps(out)
                if mode != "none":
                    self.assertNotIn("$schema", out)
                    self.assertEqual(schema.scrub(out, mode), out, (tool["name"], mode))  # idempotent
                if mode == "no_root_combinators":
                    self.assertFalse(any(k in out for k in ("anyOf", "oneOf", "allOf")))
                if mode == "gemini_openapi":
                    self.assertFalse(any("allOf" in n or "$ref" in n for n in walk(out)))
            self.assertEqual(tool["input_schema"], original)


if __name__ == "__main__":
    unittest.main()
