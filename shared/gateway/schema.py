"""JSON-Schema scrubbing for provider tool parameters (DESIGN §3.0). ``scrub`` never mutates its input.

Modes (``config.SCHEMA_MODES``):

``none``                 deep copy, unchanged.
``basic``                drop ``$schema`` and ``$id`` (only where they are keywords — a property
                         literally named ``$schema`` is kept).
``no_root_combinators``  basic + flatten root ``oneOf``/``anyOf``/``allOf`` into one object schema:
                         branch properties are merged (conflicting definitions become a nested
                         ``anyOf``), ``required`` = intersection of the branches for ``anyOf``/
                         ``oneOf`` and union for ``allOf`` (plus the root's own), ``type: object``.
                         Root branches that are local ``$ref``s are resolved first. (xAI, grok proxy)
``gemini_json``          for Gemini ``parametersJsonSchema``: basic + inline local ``$ref``s
                         (``#``, ``#/$defs/…``, ``#/definitions/…``, any JSON pointer; ref nesting
                         deeper than 8 — i.e. recursive schemas — becomes ``{}``), drop
                         ``$defs``/``definitions``, collapse ``type: [T, "null"]`` to ``T``.
``gemini_openapi``       for Gemini's fallback ``parameters`` (OpenAPI 3.0 subset): refs inlined as
                         above, ``allOf`` merged into its parent, ``oneOf`` -> ``anyOf``, ``const`` ->
                         ``enum``, numeric ``exclusiveMinimum``/``exclusiveMaximum`` ->
                         ``minimum``/``maximum``, ``type: [T, "null"]`` -> ``T`` + ``nullable``, then
                         only these keys survive: type, format, description, nullable, enum,
                         properties, required, items, minItems, maxItems, minimum, maximum,
                         minLength, maxLength, pattern, anyOf, title. Non-string enums and formats
                         Gemini rejects are dropped; ``required`` only lists defined properties.
"""

import copy
from urllib.parse import unquote

__all__ = ["scrub", "MODES", "MAX_REF_DEPTH"]

MODES = ("none", "basic", "no_root_combinators", "gemini_json", "gemini_openapi")
MAX_REF_DEPTH = 8
_REF_BUDGET = 2000  # total ref expansions per scrub (guards exponential fan-out)

# keywords whose values are subschemas
_MAP_KEYS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_LIST_KEYS = ("allOf", "anyOf", "oneOf", "prefixItems")
_ONE_KEYS = ("additionalProperties", "items", "not", "if", "then", "else", "contains", "propertyNames",
             "unevaluatedProperties", "unevaluatedItems", "additionalItems")
_COMBINATORS = ("allOf", "anyOf", "oneOf")

_OPENAPI_KEYS = ("type", "format", "description", "nullable", "enum", "properties", "required", "items",
                 "minItems", "maxItems", "minimum", "maximum", "minLength", "maxLength", "pattern", "anyOf",
                 "title")
_OPENAPI_FORMATS = {"string": ("enum", "date-time"), "integer": ("int32", "int64"), "number": ("float", "double")}


def scrub(schema, mode):
    """Return a scrubbed deep copy of ``schema`` for ``mode`` (``None`` means ``basic``)."""
    mode = mode or "basic"
    if mode not in MODES:
        raise ValueError("unknown schema mode %r (expected one of %s)" % (mode, ", ".join(MODES)))
    if mode == "none":
        return copy.deepcopy(schema)
    if not isinstance(schema, dict):
        return {}
    base = _basic(schema)
    if mode == "basic":
        return base
    if mode == "no_root_combinators":
        return _flatten_root(base, base, 0)
    inlined = _Inliner(base).run()
    if mode == "gemini_json":
        return _collapse_null_types(inlined)
    return _openapi(inlined)


# ---------------------------------------------------------------------------------------
# generic traversal
# ---------------------------------------------------------------------------------------

def _map_subschemas(node, fn, skip=()):
    """Shallow copy of schema ``node`` with ``fn`` applied to each direct subschema."""
    out = {}
    for k, v in node.items():
        if k in skip:
            continue
        if k in _MAP_KEYS and isinstance(v, dict):
            out[k] = dict((name, fn(sub)) for name, sub in v.items())
        elif k in _LIST_KEYS and isinstance(v, list):
            out[k] = [fn(sub) for sub in v]
        elif k in _ONE_KEYS and isinstance(v, (dict, list)):
            out[k] = [fn(sub) for sub in v] if isinstance(v, list) else fn(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _basic(node):
    if not isinstance(node, dict):
        return copy.deepcopy(node)
    return _map_subschemas(node, _basic, skip=("$schema", "$id"))


def _resolve_pointer(root, ref):
    """Resolve a local ``$ref`` (``#``, ``#/a/b``) against ``root``; None if unresolvable."""
    if not isinstance(ref, str) or not ref.startswith("#"):
        return None
    pointer = unquote(ref[1:])
    if pointer == "":
        return root
    if not pointer.startswith("/"):
        return None  # anchors / $id-relative refs are not supported
    cur = root
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict) and token in cur:
            cur = cur[token]
        elif isinstance(cur, list) and token.isdigit() and int(token) < len(cur):
            cur = cur[int(token)]
        else:
            return None
    return cur


# ---------------------------------------------------------------------------------------
# no_root_combinators
# ---------------------------------------------------------------------------------------

def _flatten_root(node, root, depth):
    combos = [k for k in _COMBINATORS if isinstance(node.get(k), list)]
    if not combos:
        return node
    out = dict((k, v) for k, v in node.items() if k not in _COMBINATORS)
    variants = []        # [(name, [distinct schemas])] in first-seen order
    index = {}

    def add_prop(name, sub):
        if name not in index:
            index[name] = len(variants)
            variants.append((name, [sub]))
        elif sub not in variants[index[name]][1]:
            variants[index[name]][1].append(sub)

    for name, sub in (node.get("properties") or {}).items():
        add_prop(name, sub)
    required = [r for r in (node.get("required") or []) if isinstance(r, str)]
    for key in combos:
        sets, order = [], []
        for branch in node[key]:
            if isinstance(branch, dict) and isinstance(branch.get("$ref"), str) and depth < MAX_REF_DEPTH:
                target = _resolve_pointer(root, branch["$ref"])
                if isinstance(target, dict):
                    merged = dict(target)
                    merged.update((k, v) for k, v in branch.items() if k != "$ref")
                    branch = merged
            if not isinstance(branch, dict):
                sets.append(set())
                continue
            branch = _flatten_root(branch, root, depth + 1)
            for name, sub in (branch.get("properties") or {}).items():
                add_prop(name, copy.deepcopy(sub))
            req = [r for r in (branch.get("required") or []) if isinstance(r, str)]
            sets.append(set(req))
            order.extend(r for r in req if r not in order)
        if not sets:
            continue
        if key == "allOf":
            chosen = set().union(*sets)
        else:
            chosen = set.intersection(*sets)
        required.extend(r for r in order if r in chosen and r not in required)
    out["type"] = "object"
    out["properties"] = dict((name, subs[0] if len(subs) == 1 else {"anyOf": subs}) for name, subs in variants)
    if required:
        out["required"] = required
    else:
        out.pop("required", None)
    return out


# ---------------------------------------------------------------------------------------
# gemini_json
# ---------------------------------------------------------------------------------------

class _Inliner(object):
    """Inline local ``$ref``s; drop ``$defs``/``definitions`` from the result."""

    def __init__(self, root):
        self.root = root
        self.budget = _REF_BUDGET

    def run(self):
        return self._node(self.root, 0)

    def _node(self, node, depth):
        if not isinstance(node, dict):
            return {} if isinstance(node, bool) else copy.deepcopy(node)
        ref = node.get("$ref")
        if isinstance(ref, str):
            siblings = dict((k, v) for k, v in node.items() if k != "$ref")
            target = None
            if depth < MAX_REF_DEPTH and self.budget > 0:
                target = _resolve_pointer(self.root, ref)
            if isinstance(target, dict):
                self.budget -= 1
                out = self._node(target, depth + 1)
            else:
                out = {}
            if siblings:
                out = dict(out)
                out.update(self._node(siblings, depth))
            return out
        return _map_subschemas(node, lambda sub: self._node(sub, depth), skip=("$defs", "definitions"))


def _collapse_null_types(node):
    if not isinstance(node, dict):
        return node
    out = _map_subschemas(node, _collapse_null_types)
    t = out.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if len(non_null) == 1 and len(t) == 2:
            out["type"] = non_null[0]
    return out


# ---------------------------------------------------------------------------------------
# gemini_openapi
# ---------------------------------------------------------------------------------------

def _merge_all_of(node):
    """Merge ``allOf`` branches into ``node`` (properties/required unioned, first value wins)."""
    branches = node.get("allOf")
    if not isinstance(branches, list):
        return node
    out = dict((k, v) for k, v in node.items() if k != "allOf")
    for b in branches:
        if not isinstance(b, dict):
            continue
        b = _merge_all_of(b)
        for k, v in b.items():
            if k == "properties" and isinstance(v, dict):
                props = dict(out.get("properties") or {})
                for name, sub in v.items():
                    props.setdefault(name, sub)
                out["properties"] = props
            elif k == "required" and isinstance(v, list):
                req = list(out.get("required") or [])
                req.extend(r for r in v if r not in req)
                out["required"] = req
            elif k not in out:
                out[k] = v
    return out


def _openapi(node):
    if not isinstance(node, dict):
        return {}
    node = _merge_all_of(node)
    nullable = bool(node.get("nullable"))
    t = node.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        nullable = nullable or len(non_null) != len(t)
        t = non_null[0] if non_null else None
    if t is None:
        if isinstance(node.get("properties"), dict):
            t = "object"
        elif "items" in node or "prefixItems" in node:
            t = "array"
    out = {}
    if isinstance(t, str):
        out["type"] = t
    for k in ("title", "description", "pattern", "minItems", "maxItems", "minLength", "maxLength",
              "minimum", "maximum"):
        if k in node and not isinstance(node[k], (dict, list)):
            out[k] = copy.deepcopy(node[k])
    for ex, plain in (("exclusiveMinimum", "minimum"), ("exclusiveMaximum", "maximum")):
        v = node.get(ex)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and plain not in out:
            out[plain] = v
    fmt = node.get("format")
    if isinstance(fmt, str) and fmt in _OPENAPI_FORMATS.get(t, ()):
        out["format"] = fmt
    enum = node.get("enum")
    if "const" in node and enum is None:
        enum = [node["const"]]
    if isinstance(enum, list):
        if None in enum:
            nullable = True
        values = [v for v in enum if v is not None]
        if values and all(isinstance(v, str) for v in values):
            out["enum"] = values
    if isinstance(node.get("properties"), dict):
        out["properties"] = dict((name, _openapi(sub)) for name, sub in node["properties"].items())
    req = node.get("required")
    if isinstance(req, list):
        defined = out.get("properties") or {}
        req = [r for r in req if isinstance(r, str) and r in defined]
        if req:
            out["required"] = req
    items = node.get("items")
    if isinstance(items, list):
        items = items[0] if items else None
    if items is None and isinstance(node.get("prefixItems"), list) and node["prefixItems"]:
        items = node["prefixItems"][0]
    if isinstance(items, dict):
        out["items"] = _openapi(items)
    elif out.get("type") == "array":
        out["items"] = {}
    branches = []
    for key in ("anyOf", "oneOf"):
        for b in node.get(key) or []:
            if isinstance(b, dict) and b.get("type") == "null" and len(b) == 1:
                nullable = True
                continue
            conv = _openapi(b)
            if conv not in branches:
                branches.append(conv)
    if len(branches) == 1 and not set(branches[0]) & set(out):
        out.update(branches[0])
    elif branches:
        out["anyOf"] = branches
    if nullable:
        out["nullable"] = True
    return dict((k, out[k]) for k in _OPENAPI_KEYS if k in out)
