"""Converters between AG-UI and the two model formats: the OpenAI chat format and the Anthropic Messages format. A request comes out
as the SDK call's keyword arguments, and an answer goes in as the SDK's object or its dict.

    completion = client.chat.completions.create(model=os.environ["AGENT_MODEL"], **convert.to_openai(history, tools=tools))
    answer = convert.from_openai(completion)   # answer.text, answer.tool_calls (each with the model's id)

A reasoning or activity message is the agent's own record, not model input, so the converters leave it out.
"""
import json
from dataclasses import dataclass

from .run import _id

# Why a model's answer isn't finished, by its stop reason: Anthropic's stop_reason, or OpenAI's finish_reason. LangChain keeps both in
# response_metadata. _LIMIT: the answer stopped at the output limit. _REFUSAL: the model refused (Anthropic's safeguards, or a filter)
_LIMIT, _REFUSAL = "limit", "refusal"
_STOPS = {"stop_reason": {"max_tokens": _LIMIT, "refusal": _REFUSAL}, "finish_reason": {"length": _LIMIT, "content_filter": _REFUSAL}}


def _stop(meta):
    """Why an answer isn't finished, from the dict that holds its stop reason (an Anthropic message, an OpenAI choice, or LangChain's
    response_metadata). -> (_LIMIT or _REFUSAL, the key that says so), or (None, None) for a finished answer."""
    for key, why in _STOPS.items():
        reason = (meta or {}).get(key)
        if isinstance(reason, str) and reason in why:
            return why[reason], key
    return None, None


@dataclass(frozen=True)
class ToolCall:
    """One tool call the model asked for: its id (keep it, so the answer can name the call), its name, and its arguments as the JSON
    text the model wrote (it can be broken JSON). Send it with run.tool_call() or run.client_tool_call() and `tool_call_id=call.id`."""
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Answer:
    """The model's answer as library outputs: `text` ("" when none) for run.text(), and `tool_calls`. `at_limit` is True when the
    answer stopped at the output limit (Anthropic's stop_reason max_tokens, OpenAI's finish_reason length): then it isn't a finished
    answer, and its text or its last tool call can be cut."""
    text: str
    tool_calls: tuple = ()
    at_limit: bool = False

    def message(self):
        """The answer as an AG-UI assistant message, for the agent's own copy of the history within one AG-UI run."""
        msg = {"id": _id("msg"), "role": "assistant", **({"content": self.text} if self.text else {})}
        if self.tool_calls:
            msg["toolCalls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                                for c in self.tool_calls]
        return msg


def to_openai(messages, tools=None):
    """The AG-UI history (`run.messages`) and AG-UI tools as the keyword arguments of an OpenAI chat completion: `{"messages": [...],
    "tools": [...]}`, with `tools` left out when there are none. System and developer messages become system messages (an
    OpenAI-compatible server may not know the developer role). A tool message with `error` starts with "Error: ", since the format
    has no error flag."""
    out = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            out.append({"role": "system", "content": m.get("content") or ""})
        elif role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or None}
            if m.get("toolCalls"):
                msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["function"]["name"],
                                                                                       "arguments": c["function"]["arguments"]}}
                                     for c in m["toolCalls"]]
            out.append(msg)
        elif role == "tool":
            text = _tool_text(m)
            out.append({"role": "tool", "tool_call_id": m["toolCallId"], "content": f"Error: {text}" if m.get("error") else text})
    request = {"messages": out}
    if tools:
        request["tools"] = [openai_tool(t) for t in tools]
    return request


def openai_tool(tool):
    """An AG-UI tool (`name`, `description`, `parameters`) as a tool of an OpenAI chat completion, the function form. LangChain's
    bind_tools() and LiteLLM take this form too, in both model formats."""
    return {"type": "function", "function": {"name": tool["name"], "description": tool.get("description") or "",
                                             "parameters": _schema(tool)}}


def from_openai(completion):
    """An OpenAI chat completion (a dict, or the SDK's object) as an `Answer`: the first choice's text and tool calls, and whether it
    stopped at the output limit (finish_reason length)."""
    choice = _plain(completion)["choices"][0]
    msg = choice["message"]
    return Answer(msg.get("content") or "", tuple(ToolCall(c["id"], c["function"]["name"], c["function"]["arguments"])
                                                  for c in msg.get("tool_calls") or []), _stop(choice)[0] == _LIMIT)


def to_anthropic(messages, tools=None):
    """The AG-UI history (`run.messages`) and AG-UI tools as the keyword arguments of an Anthropic Messages call: `{"system": ...,
    "messages": [...], "tools": [...]}`, with `system` and `tools` left out when there are none. System and developer messages join
    into `system`. Each message's content is a list of blocks. Tool results go in user turns, as the format needs, and turns of one
    role merge into one. A tool call's arguments that are not a JSON object become `{}` (only a model in the other format can write
    them). The library keeps no thinking blocks, so a tool loop with extended thinking on is not supported."""
    system, out = [], []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role in ("system", "developer"):
            system += [content] if content else []
            continue
        if role == "user":
            turn, blocks = "user", [_text_block(content)] if isinstance(content, str) else list(content or [])
        elif role == "assistant":
            turn = "assistant"
            blocks = [_text_block(content)] + [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                                                "input": _object(c["function"]["arguments"])} for c in m.get("toolCalls") or []]
        elif role == "tool":
            turn, blocks = "user", [{"type": "tool_result", "tool_use_id": m["toolCallId"], "content": _tool_text(m),
                                     **({"is_error": True} if m.get("error") else {})}]
        else:
            continue
        blocks = [b for b in blocks if not (b.get("type") == "text" and not b.get("text"))]   # the API refuses empty text
        if not blocks:
            continue
        if out and out[-1]["role"] == turn:
            out[-1]["content"] += blocks
        else:
            out.append({"role": turn, "content": blocks})
    request = {**({"system": "\n\n".join(system)} if system else {}), "messages": out}
    if tools:
        request["tools"] = [{"name": t["name"], "description": t.get("description") or "", "input_schema": _schema(t)} for t in tools]
    return request


def from_anthropic(message):
    """An Anthropic Messages answer (a dict, or the SDK's object) as an `Answer`: its text blocks and its tool_use blocks, with each
    input as JSON text, and whether it stopped at the output limit (stop_reason max_tokens). Thinking blocks are left out."""
    message = _plain(message)
    blocks = message["content"]
    return Answer("".join(b["text"] for b in blocks if b["type"] == "text"),
                  tuple(ToolCall(b["id"], b["name"], json.dumps(b["input"])) for b in blocks if b["type"] == "tool_use"),
                  _stop(message)[0] == _LIMIT)


def _plain(answer):
    """A model SDK's answer object as a dict (both SDKs' objects are pydantic models)."""
    return answer.model_dump() if hasattr(answer, "model_dump") else answer


def _text_block(text):
    return {"type": "text", "text": text or ""}


def _object(arguments):
    try:
        value = json.loads(arguments or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _schema(tool):
    """An AG-UI tool's parameters (a JSON Schema, carried as-is); a tool with none takes no arguments."""
    return tool.get("parameters") or {"type": "object", "properties": {}}


def _tool_text(m):
    """A tool message's text: its error when the call failed, then its content (each said once)."""
    return "\n\n".join(dict.fromkeys(p for p in (m.get("error"), m.get("content")) if p))
