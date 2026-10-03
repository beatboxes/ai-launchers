"""Internal event model (DESIGN §2.1).

Every dialect's ``execute(ctx)`` yields these events; ``anthropic_out.SSEEmitter`` and
``anthropic_out.Aggregator`` consume them. Plain classes with ``__slots__``, value equality and a
readable ``repr``. ``key`` identifies a content block (any hashable, e.g. an output index); a change
of key closes the open block and opens a new one.

| Event               | Fields                                              |
|---------------------|-----------------------------------------------------|
| TextDelta           | key, text                                           |
| ThinkingDelta       | key, text  (``""`` just opens the thinking block)   |
| ThinkingSignature   | key, signature                                      |
| ToolCall            | id, name, input_json  (complete call; JSON object)  |
| Usage               | input_tokens, output_tokens, cache_read=0, cache_write=0 (last wins) |
| Finish              | stop_reason, stop_sequence=None                     |
| StreamError         | err_type, message, retryable                        |

Tool arguments are never live-streamed: dialects buffer them and emit one ``ToolCall`` per call.
"""

__all__ = [
    "Event", "TextDelta", "ThinkingDelta", "ThinkingSignature", "ToolCall", "Usage", "Finish",
    "StreamError", "STOP_REASONS",
]

STOP_REASONS = ("end_turn", "max_tokens", "tool_use", "stop_sequence", "pause_turn", "refusal")


class Event(object):
    """Base class: equality/hash/repr derived from ``__slots__``."""

    __slots__ = ()
    kind = "event"

    def _values(self):
        return tuple(getattr(self, name) for name in self.__slots__)

    def __eq__(self, other):
        return type(self) is type(other) and self._values() == other._values()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        try:
            return hash((type(self).__name__,) + self._values())
        except TypeError:
            return hash(type(self).__name__)

    def __repr__(self):
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join("%s=%r" % (name, getattr(self, name)) for name in self.__slots__),
        )

    def to_dict(self):
        """JSON-friendly dict (for tracing): ``{"event": kind, <fields>}``."""
        d = {"event": self.kind}
        for name in self.__slots__:
            d[name] = getattr(self, name)
        return d


class TextDelta(Event):
    __slots__ = ("key", "text")
    kind = "text"

    def __init__(self, key, text):
        self.key = key
        self.text = text


class ThinkingDelta(Event):
    __slots__ = ("key", "text")
    kind = "thinking"

    def __init__(self, key, text=""):
        self.key = key
        self.text = text


class ThinkingSignature(Event):
    __slots__ = ("key", "signature")
    kind = "signature"

    def __init__(self, key, signature):
        self.key = key
        self.signature = signature


class ToolCall(Event):
    __slots__ = ("id", "name", "input_json")
    kind = "tool_call"

    def __init__(self, id, name, input_json):  # noqa: A002 - field name mandated by spec
        self.id = id
        self.name = name
        self.input_json = input_json


class Usage(Event):
    __slots__ = ("input_tokens", "output_tokens", "cache_read", "cache_write")
    kind = "usage"

    def __init__(self, input_tokens, output_tokens, cache_read=0, cache_write=0):
        self.input_tokens = int(input_tokens or 0)
        self.output_tokens = int(output_tokens or 0)
        self.cache_read = int(cache_read or 0)
        self.cache_write = int(cache_write or 0)


class Finish(Event):
    __slots__ = ("stop_reason", "stop_sequence")
    kind = "finish"

    def __init__(self, stop_reason, stop_sequence=None):
        self.stop_reason = stop_reason
        self.stop_sequence = stop_sequence


class StreamError(Event):
    __slots__ = ("err_type", "message", "retryable")
    kind = "error"

    def __init__(self, err_type, message, retryable=False):
        self.err_type = err_type
        self.message = message
        self.retryable = bool(retryable)
