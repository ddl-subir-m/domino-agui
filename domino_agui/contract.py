"""The agent contract's shared parts: each ready-made agent uses them, and so can an author's own graph, flow or loop, so that every
framework calls the model, the relay and an own tool, and asks a question, the same way. The standard library only.

    model = contract.model_alias()                                       # AGENT_MODEL, in the format contract.alias_format() names
    tools = await contract.relay_tools() + run.client_tools + [contract.ASK_HUMAN_TOOL]   # AG-UI tools, for convert.to_openai()
    system = contract.system_prompt(run.context)                         # "" when the run has no context (a laptop)
    question, expires_at = contract.ask_human_arguments(call.arguments)  # the model's ask_human call
    reply = payload if resolved else contract.NO_ANSWER                  # ask_human's result

Own tools are plain functions that run in the agent, in the sandbox. They aren't gateway tools, so the gateway doesn't decide or
audit them:

    specs, own = contract.own_tools(tools)                               # when the entry point is built: NameClash for ask_human
    contract.check_names(own, gateway=relay_names, client=client_names)  # in each AG-UI run, before the model call
    result = await contract.call_tool(call.name, call.arguments, own)    # an own tool here, a gateway tool through the relay

The agent's own code (a node of its own graph, a step of its own flow) reads the task's facts, and calls a gateway tool, with no parse
of a text that is written for a model:

    facts = contract.task(run.context)                                   # a dict, or None outside a Job (no facts item)
    bundle = await contract.call_json("get_bundle", {"bundleId": bundle_id})   # the answer's JSON; ToolError for a denial
    result = await contract.gateway_call("get_bundle", {"bundleId": bundle_id})   # the ToolResult, with is_error for a denial
    data = result.json()                                                 # the JSON at the start of the answer (a note can follow)

Outside a Job (no AGENT_SERVICE_URL) there is no relay: gateway_call() and call_json() raise a RelayError that names the setting.
"""
import asyncio
import inspect
import json
import os
import types
import typing

from .relay import Relay, RelayError, ToolResult

ASK_HUMAN = "ask_human"
# ask_human, the agent's tool for a question to a person, as an AG-UI tool (name, description, parameters). The agent ends its AG-UI run
# with one interrupt for it: the question is the interrupt's message, and expires_at its expiresAt
ASK_HUMAN_TOOL = {"name": ASK_HUMAN, "description": "Ask the person who started the task a question, and wait for the answer.",
                  "parameters": {"type": "object", "required": ["question"], "properties": {
                      "question": {"type": "string", "description": "The question"},
                      "expires_at": {"type": "string", "description": "Optional: when the answer stops being useful (ISO 8601)"}}}}
# ask_human's result when the question expired (the resume is cancelled). It says only what happened: what the agent does then is the
# system prompt's rule
NO_ANSWER = "No answer came before the question expired, so it was cancelled."
# ask_human's result for a question that the agent doesn't ask: a second one in the same answer, or one next to a client-side call. An
# AG-UI run ends with one interrupt at most, and the ready-made handler and flow keep one open question across runs
ONE_QUESTION = "Ask one question at a time: ask this again once you have the answer you are waiting for."
# The message that the ready-made LangGraph agent and the loop give the model after an answer with no text and no tool call, one time
# in a run, as a user message. A second empty answer ends the run. (CrewAI's flow doesn't send it.)
EMPTY_ANSWER = "Your last answer was empty. Continue with the task, or give your result."
# The run's result, as its last text, when the model's last answer stopped at the output limit (MAX_TOKENS, or the limit of the
# author's own model): the answer isn't finished, so the result says so, and says how to give the model more room. The ready-made
# LangGraph agent and the loop send it
AT_LIMIT = ("The model's last answer stopped at its output limit, so it is not a finished answer. To give the model a larger limit, "
            "set AGENT_MAX_TOKENS, or max_tokens on the agent's own model.")
# The relay client's timeouts, in seconds: to connect, then for each answer. A relay call can stay open for up to 45 s, so the time
# for an answer must be at least 60 s
CONNECT_TIMEOUT, READ_TIMEOUT = 30, 300

# The framework of the Environment this runs in: each framework image sets it (an ENV line in its Dockerfile), and the sandbox keeps
# AGENT_* variables for the agent server. Not set on a laptop
FRAMEWORK_ENV = "AGENT_FRAMEWORK"
# The variable that sets MAX_TOKENS, in tokens (an AGENT_* variable, so the sandbox keeps it for the agent server)
MAX_TOKENS_ENV = "AGENT_MAX_TOKENS"


def _max_tokens():
    """MAX_TOKENS_ENV as a whole number above 0, or 16000 when it isn't set. Another value is a ValueError, at the import."""
    value = os.environ.get(MAX_TOKENS_ENV, "").strip()
    if not value:
        return 16000
    if not (value.isascii() and value.isdigit()) or int(value) < 1:
        raise ValueError(f"{MAX_TOKENS_ENV} is the model's output limit, a whole number of tokens above 0, and it is {value!r}")
    return int(value)


# The output limit of the platform's model, in tokens: each ready-made agent gives it to each model call (the loop's call_model(), the
# LangGraph module's chat_model() and the CrewAI flow's litellm_arguments()), unless the author's own model or option sets one. A model
# with no limit can get a small default (4,096 for an alias with no model profile), which cuts a long answer
MAX_TOKENS = _max_tokens()


def _output_limit(options, key="max_tokens"):
    """The output limit to add to a model call: {key: MAX_TOKENS}, or {} when its `options` set one (max_tokens or
    max_completion_tokens). The loop's call_model() and the CrewAI flow's litellm_arguments() use it."""
    return {} if {"max_tokens", "max_completion_tokens"} & set(options) else {key: MAX_TOKENS}

_relay = None    # the relay client for this agent server
_listed = None   # the relay's tools, listed once for each agent server, at the first run (no network at import)


def require_framework(module, framework):
    """A framework module's first step, at its import: an ImportError, which names the Environment to use, when this is the Environment
    of another framework (FRAMEWORK_ENV names it). Nothing when it is this framework's, or not set (a laptop)."""
    here = os.environ.get(FRAMEWORK_ENV)
    if here and here != framework:
        raise ImportError(f"{module} is the ready-made agent for the {framework} framework, and this is the agent-{here} Environment: "
                          f"run the agent on the agent-{framework} Environment")


def no_tool(name):
    """The result of a call to a tool that no list has (not an own tool, not the relay's, not client-side). The agent answers it, with
    no relay call."""
    return f"There is no tool named {name}."


def model_alias():
    """AGENT_MODEL, the model's alias at the model gateway, read when it is called (so an agent builds with no model variables)."""
    model = os.environ.get("AGENT_MODEL")
    if not model:
        raise RuntimeError("AGENT_MODEL isn't set: the platform sets it in a framework-lane Job. Elsewhere, set AGENT_MODEL to a model "
                           "name, and AGENT_MODEL_FORMAT to anthropic or openai")
    return model


def alias_format():
    """The model format of the alias: "anthropic" (Anthropic Messages) when AGENT_MODEL_FORMAT says so, else "openai" (OpenAI chat
    completions, which OpenAI-compatible models speak too)."""
    return "anthropic" if os.environ.get("AGENT_MODEL_FORMAT") == "anthropic" else "openai"


def relay():
    """The relay client for this agent server (MCP at $AGENT_SERVICE_URL/mcp, with $SESSION_TOKEN): connect timeout CONNECT_TIMEOUT,
    read timeout READ_TIMEOUT, never through the egress proxy. None outside a Job (no AGENT_SERVICE_URL): there, gateway_call() and
    call_json() raise a RelayError that says so."""
    global _relay
    if not os.environ.get("AGENT_SERVICE_URL"):
        return None
    if _relay is None:
        _relay = Relay(timeout=CONNECT_TIMEOUT, read_timeout=READ_TIMEOUT)
    return _relay


async def relay_tools():
    """Every tool the relay lists, under the relay's names, as AG-UI tools: listed at the first run, then kept (a failed listing is
    tried again at the next run). [] outside a Job."""
    global _listed
    client = relay()
    if client is None:
        return []
    if _listed is None:
        _listed = await client.list_tools()
    return _listed


async def call_tool(name, arguments, own=None):
    """A tool call that the agent runs, with the model's arguments (a dict or its JSON text). -> ToolResult: an own tool's (`own`:
    {name: function}), run here with call_function; a gateway tool's (the relay lists it), through the relay; else no_tool(name), with
    no relay call. A denial and an own tool's exception are error results too, for the model to read."""
    if name in (own or {}):
        return await call_function(own[name], arguments)
    if name in {t["name"] for t in await relay_tools()}:
        return await relay().call(name, arguments)
    return ToolResult(no_tool(name), True)


async def gateway_call(name, arguments):
    """A gateway tool call from the agent's own code (a node of its own graph, a step of its own flow), through the Job's relay.
    `arguments`: a dict, or its JSON text. -> the relay's ToolResult: a denial or a failed call is a result with is_error, as for the
    model. Outside a Job (no AGENT_SERVICE_URL: a laptop, a unit test) there is no relay, so it raises RelayError, which names the
    setting."""
    client = relay()
    if client is None:
        raise RelayError("AGENT_SERVICE_URL isn't set, so there is no relay for the gateway tools: the platform sets it in the Job of a "
                         "framework agent. Elsewhere (a laptop, a unit test), give your code a stand-in for the call, or call a relay "
                         "that you run with Relay(url=..., token=...)")
    return await client.call(name, arguments)


class ToolError(RuntimeError):
    """A gateway tool answered with an error (a denial, or a call that failed): call_json() raises it. `name` is the tool, and `result`
    its ToolResult (result.text: the gateway's text, which says why)."""

    def __init__(self, name, result):
        super().__init__(f"the gateway tool {name} answered with an error: {result.text}")
        self.name, self.result = name, result


async def call_json(name, arguments):
    """A gateway tool call from the agent's own code that gives the JSON of the answer (ToolResult.json(): a note after the JSON is
    left out). `arguments`: a dict, or its JSON text. A denial or a failed call raises ToolError, an answer that isn't JSON a
    ValueError that names the tool, and a call outside a Job a RelayError (as gateway_call())."""
    result = await gateway_call(name, arguments)
    if result.is_error:
        raise ToolError(name, result)
    try:
        return result.json()
    except ValueError as e:
        raise ValueError(f"{name}: {e}") from None


def own_tools(tools):
    """The own tools (plain functions), as the entry point is built: (their AG-UI tools, {name: function}). Each AG-UI tool comes from
    function_tool(); NameClash for one named ask_human, or two with one name (check_names)."""
    specs = [function_tool(fn) for fn in tools]
    check_names([s["name"] for s in specs])
    return specs, {s["name"]: fn for s, fn in zip(specs, tools)}


def ask_human_arguments(arguments):
    """The model's ask_human call's arguments (a dict, or the model's JSON text) as (question, expires_at): the question's text, and
    when its answer stops being useful (ISO 8601, or None). Arguments that aren't a JSON object are the question's text."""
    try:
        args = json.loads(arguments or "{}") if isinstance(arguments, str) else arguments
    except ValueError:
        args = None
    if not isinstance(args, dict):
        return str(arguments or ""), None
    expires_at = args.get("expires_at")
    return str(args.get("question") or ""), str(expires_at) if expires_at else None


class NameClash(ValueError):
    """An own tool has the name of ask_human, of a gateway tool or of a client-side tool."""


def check_names(own, *, gateway=(), client=()):
    """Raises NameClash when an own tool would hide another tool. `own`: the own tools' names. When the entry point is built: ask_human,
    and two own tools with one name. In each AG-UI run, also `gateway` (the names the relay lists) and `client` (the run's client-side
    tools, RunAgentInput.tools): an own tool must never hide a tool that the gateway decides and audits."""
    own = list(own)
    if ASK_HUMAN in own:
        raise NameClash(f"an own tool is named {ASK_HUMAN}, the agent's tool that asks a person a question: rename the function")
    twice = sorted({n for n in own if own.count(n) > 1})
    if twice:
        raise NameClash(f"two own tools have the name {', '.join(twice)}: give each one its own name")
    for names, what in ((set(gateway), "a gateway tool (the relay lists it)"), (set(client), "a client-side tool of this run")):
        hit = sorted(n for n in own if n in names)
        if hit:
            raise NameClash(f"the own tool {', '.join(hit)} has the name of {what}, which it would hide from the gateway's decisions "
                            "and audit: rename the function")


def function_tool(fn):
    """An own tool, a plain function (def or async def), as an AG-UI tool: its name, its docstring's first paragraph as the
    description, and a JSON schema of its parameters from their type hints."""
    name = getattr(fn, "__name__", "")
    if not (inspect.isfunction(fn) or inspect.ismethod(fn)) or not name.isidentifier():
        raise TypeError(f"an own tool is a function with a name (def or async def), and {fn!r} isn't one: write it with def")
    sig, hints = inspect.signature(fn), typing.get_type_hints(fn)
    properties, required = {}, []
    for p in sig.parameters.values():
        try:
            properties[p.name] = _schema(hints.get(p.name))
        except (KeyError, TypeError):
            raise TypeError(f"the own tool {name} has a parameter {p.name} of type {hints.get(p.name)!r}, which has no JSON schema here: "
                            "give it one of str, int, float, bool, list, dict, list[X], dict[str, X], Optional[X] (X | None) or "
                            "Literal[...]") from None
        if p.default is p.empty:
            required.append(p.name)
        elif _json(p.default):
            properties[p.name]["default"] = p.default
    doc = (inspect.getdoc(fn) or "").split("\n\n")[0]
    return {"name": fn.__name__, "description": " ".join(doc.split()),
            "parameters": {"type": "object", "properties": properties, "required": required}}


async def call_function(fn, arguments):
    """Runs an own tool with the model's arguments (a dict, or a model's JSON text) as its keyword arguments, in this process (the
    agent server, in the sandbox): an async def is awaited, a def runs in a thread. -> ToolResult: the answer as text (JSON when it
    isn't a str). An exception, or arguments the function can't take, is an error answer (is_error), as a gateway denial is: the model
    reads it, and the run goes on."""
    name = getattr(fn, "__name__", "tool")
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        args = None
    if not isinstance(args, dict):
        return ToolResult(f"the own tool {name} takes its arguments as a JSON object, and got {str(arguments)[:200]}", True)
    try:
        inspect.signature(fn).bind(**args)
    except TypeError as e:
        return ToolResult(f"the own tool {name} can't take these arguments: {e}", True)
    try:
        result = await fn(**args) if inspect.iscoroutinefunction(fn) else await asyncio.to_thread(fn, **args)
    except Exception as e:  # noqa: BLE001 (the model gets the error as the tool's answer)
        return ToolResult(f"the own tool {name} raised {type(e).__name__}: {e}", True)
    return ToolResult(result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False))


def _json(value):
    """The value is a JSON value (so it can be a schema's default)."""
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _schema(hint):
    """The JSON schema of one type hint (KeyError when it has none here)."""
    if hint is None or hint is typing.Any:   # no hint: any JSON value
        return {}
    origin, args = typing.get_origin(hint) or hint, typing.get_args(hint)
    if origin in (typing.Union, types.UnionType) and len(args) == 2 and type(None) in args:   # Optional[X], X | None: X or null
        out = _schema(next(a for a in args if a is not type(None)))
        if "type" in out:
            out["type"] = [out["type"], "null"]
        elif "enum" in out:
            out["enum"] = out["enum"] + [None]
        return out
    if origin is typing.Literal:
        kinds = {_TYPES.get(type(a)) for a in args}
        return {**({"type": kinds.pop()} if len(kinds) == 1 and None not in kinds else {}), "enum": list(args)}
    out = {"type": _TYPES[origin]}
    if origin is list and args:
        out["items"] = _schema(args[0])
    if origin is dict and len(args) == 2:
        out["additionalProperties"] = _schema(args[1])
    return out


# The description of the AG-UI context item with the task's facts. The agent service sends it in each AG-UI run of a framework agent,
# after the text item (the operating rules and the task brief): its value is the facts as compact JSON
FACTS = "Domino Agents: the task's facts (JSON)"
# The version of the facts that task() reads. A new field is an addition to this version; a rename or a removal is a new version
FACTS_VERSION = 1


def system_prompt(context):
    """The model's system prompt: the `value` of each AG-UI context entry (the agent's system prompt, the platform's operating rules
    and the task brief), in order, as one text. "" when there is none. An entry is a dict, or an object with the two attributes
    (AG-UI's Context), as for task(). The task's facts (the entry FACTS) are left out: they are for the agent's code (task()), and the
    task brief gives the model the same facts as text."""
    values = (str(value or "").strip() for description, value in (_entry(c) for c in context or []) if description != FACTS)
    return "\n\n".join(v for v in values if v)


def task(context):
    """The task's facts, from the AG-UI run's context (run.context; in LangGraph state["ag-ui"]["context"], in CrewAI state.context),
    as a dict, so that the agent's code reads a fact with no parse of the task brief. None when the context has no facts item (FACTS):
    outside a Job, or with an agent service that doesn't send it. Facts of another version than FACTS_VERSION, or an item that isn't a
    JSON object, raise a ValueError. The keys of version 1:

    version         1 (FACTS_VERSION)
    task            {id, key, title}
    project         {id, name, owner}: the task's project (owner: its owner's user name, or None), or None
    resources       the task's resources, except its project, in the task's order: {type, id, name, access, mount, from}. `access`
                    is "read" or "write", `mount` the path in the Job (a dataset or a volume; else None), `from` why the task has it:
                    "pinned" (the trigger's resources), "context" (the trigger's payload names it), "approval" (a person granted it) or
                    "caller" (the call that started the task, or a message on it)
    inputs          {name: value} for each named input of the agent: a resource {type, id, name} (a list for an input that takes
                    more than one), a text input's text, or None
    trigger         {id, type, name} of the subscription (type "event") or the schedule (type "schedule"), or None for a direct run
    payload         what the task got with it: an event's payload, a schedule's {"schedule": {...}}, or a direct call's context. None
                    when there is none, or when it was too large (then payloadOmitted is True)
    payloadOmitted  True when the payload was left out for its size. Its entities are still in `resources` (from "context")
    """
    value = next((v for d, v in (_entry(c) for c in context or []) if d == FACTS), None)
    if value is None:
        return None
    try:
        facts = json.loads(value) if isinstance(value, str) else value
    except ValueError as e:
        raise ValueError(f"the task's facts (the AG-UI context item {FACTS!r}) are not JSON: {e}") from None
    if not isinstance(facts, dict):
        raise ValueError(f"the task's facts (the AG-UI context item {FACTS!r}) are not a JSON object: {str(value)[:200]}")
    if facts.get("version") != FACTS_VERSION:
        raise ValueError(f"the task's facts have version {facts.get('version')!r}, and this domino-agui reads version {FACTS_VERSION}: "
                         "use a domino-agui that reads that version")
    return facts


def _entry(c):
    """An AG-UI context entry as (description, value): a dict, or an object with the two attributes (AG-UI's Context)."""
    if isinstance(c, dict):
        return c.get("description"), c.get("value")
    return getattr(c, "description", None), getattr(c, "value", None)
