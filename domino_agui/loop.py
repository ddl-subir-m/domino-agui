"""The ready-made handler for hand-written agents, and the agent contract's parts as helpers for a handler that owns its loop. An
agent's entry point can be one line:

    from domino_agui.loop import handler

    "framework": {"name": "domino-agui", "entrypoint": "agent/handler.py:handler"}

handler(run), which agent() makes, calls the model and does what each answer asks, until an answer has no tool call. One handler serves
both model formats. Each AG-UI run brings the whole thread (run.messages), so it keeps nothing between runs but the open question, in
the run's state:
- model_client(): the SDK client for AGENT_MODEL_FORMAT: AsyncAnthropic (Anthropic Messages) or AsyncOpenAI (OpenAI chat completions).
  It reads the base URL and the key from the environment that the platform sets (the Job's model proxy), and AGENT_MODEL names the
  model. call_model() calls it through the library's converters;
- agent_side_tools(run): the gateway tools that the relay lists (contract.relay_tools(): MCP at $AGENT_SERVICE_URL/mcp, with
  $SESSION_TOKEN), without the run's client-side tools. They run in this AG-UI run, through the relay (call_tool()), and a denial comes
  back as the tool's answer. Outside a Job (no AGENT_SERVICE_URL) there are none;
- run.client_tools: the client-side tools. The handler offers them and doesn't run them: a call to one ends the AG-UI run, and its
  answer comes as a tool message in a later run. A tool in both lists is client-side;
- ask_human(run, call): the agent's tool for a question to a person. The run ends with one interrupt (its message is the question, its
  expiresAt the model's expires_at), and the call waits in the run's state. answered_question(run) gives the next run the call and its
  result: the reply, or contract.NO_ANSWER when the resume is cancelled (no reply before expiresAt). It keeps both in the thread too
  (run.remember), so later runs have them;
- system_prompt(run): the AG-UI run's context (the agent's system prompt, the platform's operating rules and the task brief) as the
  model's system message.

Two options, your own model and your own tools, make another entry point of the same shape:

    handler = agent(model=Model(AsyncAnthropic(max_retries=5), max_tokens=1024), tools=[lookup_customer])

The client's type sets the format, and Model's options go to each model call (a client alone works too: model=AsyncAnthropic()). Own
tools are plain functions (def or async def, with type hints and a docstring). They run here, in the agent server, in the sandbox: they
aren't gateway tools, so the gateway doesn't decide or audit them. A name clash fails (contract.check_names).

An author who owns the loop writes `async def handler(run)` on the core and these helpers: agent() shows how they fit. This module
imports the model SDKs (openai and anthropic) when it is imported (the core doesn't), so it needs the agent-domino-agui Environment's
packages. It calls no host at import.
"""
import inspect

from . import contract, convert
from .relay import ToolResult

contract.require_framework(__name__, "domino-agui")   # in another framework's Environment: an ImportError that names this one

try:
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(f"domino_agui.loop needs the model SDKs openai and anthropic, and this Python has no {e.name or e}: run the agent "
                      "on the agent-domino-agui Environment") from None

__all__ = ["ASKED", "MAX_TOKENS", "Model", "agent", "agent_side_tools", "answered_question", "ask_human", "call_model", "call_tool",
           "handler", "model_client", "model_format", "system_prompt"]

MAX_TOKENS = contract.MAX_TOKENS   # each model call's output limit, unless its options give one: the contract's default
ASKED = "asked"     # the state's key for the open ask_human call


def model_client():
    """A new SDK client for AGENT_MODEL_FORMAT: AsyncAnthropic for "anthropic", else AsyncOpenAI (OpenAI-compatible models speak its
    format too). It reads its base URL and key from the environment. agent() makes it at the first run, so a handler builds with no
    model variables."""
    return AsyncAnthropic() if contract.alias_format() == "anthropic" else AsyncOpenAI()


def model_format(client):
    """The model format of an SDK client, by its type: "anthropic" (AsyncAnthropic) or "openai" (AsyncOpenAI). TypeError for another."""
    if isinstance(client, AsyncAnthropic):
        return "anthropic"
    if isinstance(client, AsyncOpenAI):
        return "openai"
    raise TypeError(f"the model is an SDK client, AsyncAnthropic or AsyncOpenAI (the async one: the handler awaits its calls), and this "
                    f"one is {type(client).__module__}.{type(client).__name__}")


class Model:
    """An own model with call options, for agent(model=...): an SDK client, and the options for each of its model calls, such as the
    maximum tokens (call_model's `options`).

        handler = agent(model=Model(AsyncOpenAI(max_retries=5), max_tokens=1024))

    `client`: AsyncAnthropic or AsyncOpenAI (its type sets the format), or None: model_client(), at the first run. `options`: keyword
    arguments of the SDK's create() call, as they are. Each SDK takes its own: the pinned anthropic SDK has no temperature. With a
    client, an option its create() doesn't take, or one that call_model() sets (model, messages, system, tools, stream), is a TypeError
    here, when the entry point is built."""

    def __init__(self, client=None, **options):
        if client is not None:
            anthropic = model_format(client) == "anthropic"
            takes = inspect.signature(client.messages.create if anthropic else client.chat.completions.create).parameters
            wrong = sorted(k for k in options if k in ("model", "messages", "system", "tools", "stream") or k not in takes)
            if wrong:
                raise TypeError(f"{type(client).__name__}'s model call doesn't take {', '.join(wrong)} as an option: call_model() sets "
                                "model, messages, system, tools and stream, and the others are the SDK's create() arguments"
                                + (" (the pinned anthropic SDK has no temperature)" if anthropic else ""))
        self.client, self.options = client, options


async def call_model(client, messages, tools, **options):
    """One model call for AGENT_MODEL, in the client's format, with the AG-UI history and tools (through the converters). Not streamed.
    `options` go to the SDK's create() call as they are, such as max_tokens=1024: each SDK takes its own (the pinned anthropic SDK has
    no temperature). The output limit is MAX_TOKENS (contract.MAX_TOKENS) unless they give one: an Anthropic call's max_tokens, an
    OpenAI call's max_completion_tokens (as LangChain's ChatOpenAI sends it; max_tokens in the options works too). -> convert.Answer,
    with answer.at_limit when the answer stopped at the limit."""
    model = contract.model_alias()
    if model_format(client) == "anthropic":
        return convert.from_anthropic(await client.messages.create(**{"model": model, "max_tokens": MAX_TOKENS, **options,
                                                                      **convert.to_anthropic(messages, tools)}))
    limit = {} if {"max_tokens", "max_completion_tokens"} & set(options) else {"max_completion_tokens": MAX_TOKENS}
    return convert.from_openai(await client.chat.completions.create(model=model, **limit, **options, **convert.to_openai(messages, tools)))


async def agent_side_tools(run):
    """The gateway tools the model can call in this run: the relay's, without the run's client-side tools (a tool in both lists is
    client-side). The agent runs them, with call_tool()."""
    client = {t["name"] for t in run.client_tools}
    return [t for t in await contract.relay_tools() if t["name"] not in client]


async def call_tool(call, own=None):
    """The result of a call that runs here (a convert.ToolCall), as a ToolResult: an own tool's (`own`: {name: function}), in this
    process; a gateway tool's, through the relay; else contract.no_tool(). A denial and an own tool's exception are error results too,
    for the model to read (contract.call_tool)."""
    return await contract.call_tool(call.name, call.arguments, own)


def system_prompt(run):
    """The run's context as one system message, for the head of the model's history ([] when there is none, as on a laptop). The
    converters make it the call's system prompt."""
    text = contract.system_prompt(run.context)
    return [{"id": "context", "role": "system", "content": text}] if text else []


def ask_human(run, call):
    """Ends the run with one interrupt for the model's ask_human call (a convert.ToolCall): its message is the question, its expiresAt
    the call's expires_at. The call waits in the run's state (ASKED), so that the next run can answer it. Returns the interrupt id."""
    question, expires_at = contract.ask_human_arguments(call.arguments)
    asked = {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
    run.state_snapshot({**run.state, ASKED: asked})
    return run.interrupt(question, reason=contract.ASK_HUMAN, expires_at=expires_at)


def answered_question(run):
    """When this run resumes the question that ended the last one: the ask_human call, taken out of the state, and its result, as
    AG-UI messages for the model's history. The result is the reply when the resume is resolved, contract.NO_ANSWER when it is
    cancelled. Else []. The run also keeps both in the thread (run.remember), so that a later run, after a client-side call or a later
    message, still has the question and its answer."""
    asked = run.state.get(ASKED)
    if not (run.resume and asked):
        return []
    reply = run.resume.payload if run.resume.status == "resolved" else contract.NO_ANSWER
    run.state = {k: v for k, v in run.state.items() if k != ASKED}
    run.state_snapshot(run.state)
    answered = [{"id": "asked", "role": "assistant", "toolCalls": [asked]},
                {"id": "answered", "role": "tool", "toolCallId": asked["id"], "content": "" if reply is None else str(reply)}]
    run.remember(*answered)
    return answered


def agent(model=None, tools=()):
    """The ready-made handler: `async def handler(run)`, the shape the domino-agui adapter serves. It calls the model with the run's
    history and every tool the model can use, sends the model's text, runs the calls of gateway tools and own tools, and calls the model
    again, until an answer has no tool call (its text is the task's result), a client-side call ends the run, or ask_human does.

    The two options:
    - model: Model(client, **options), for the options of each model call (such as max_tokens), or an SDK client alone, AsyncAnthropic
      or AsyncOpenAI, for the client's options (default headers, the timeout, the retries). The client's type sets the format. It still
      goes through the model proxy (the base URL and the key come from the environment), and AGENT_MODEL still names the model. None:
      model_client(), at the first run;
    - tools: own tools, plain functions (def or async def, with type hints and a docstring: contract.function_tool). They run here, in
      the agent server, in the sandbox: they aren't gateway tools, so the gateway doesn't decide or audit them. An own tool named
      ask_human raises NameClash here; one with a gateway tool's or a client-side tool's name fails the AG-UI run."""
    chosen = model if isinstance(model, Model) else Model(model)   # TypeError when the entry point is built, not at the first run
    specs, own = contract.own_tools(tools)
    client, options = chosen.client, chosen.options

    async def handler(run):
        nonlocal client
        client_side = {t["name"] for t in run.client_tools}
        listed = await contract.relay_tools()
        contract.check_names(own, gateway=[t["name"] for t in listed], client=client_side)
        gateway = [t for t in listed if t["name"] not in client_side]
        offered = run.client_tools + gateway + specs + [contract.ASK_HUMAN_TOOL]
        if client is None:
            client = model_client()
        history = system_prompt(run) + run.messages + answered_question(run)
        while True:
            answer = await call_model(client, history, offered, **options)
            asked = history[-1].get("content") == contract.EMPTY_ANSWER   # this answer came after the retry message
            if not (answer.text.strip() or answer.tool_calls or asked):   # no text and no tool call: ask the model again
                history.append({"id": "empty-answer", "role": "user", "content": contract.EMPTY_ANSWER})
                continue
            history.append(answer.message())
            if answer.text:
                run.text(answer.text)
            open_calls = any(c.name in client_side for c in answer.tool_calls)
            # one question at a time: the run ends with one interrupt, and a client-side call ends it first
            ask = None if open_calls else next((c for c in answer.tool_calls if c.name == contract.ASK_HUMAN), None)
            for call in answer.tool_calls:
                if call.name in client_side:   # the platform runs it: the call stays open
                    run.client_tool_call(call.name, call.arguments, tool_call_id=call.id)
                elif call is not ask:   # a gateway or own tool, or a question that isn't asked (contract.ONE_QUESTION)
                    result = (ToolResult(contract.ONE_QUESTION) if call.name == contract.ASK_HUMAN else await call_tool(call, own))
                    run.tool_call(call.name, call.arguments, result.text, tool_call_id=call.id)
                    history.append(result.message(call.id))
            if open_calls:
                return   # the run ends with the client-side calls open; the model can ask a person after their answers
            if ask:
                return ask_human(run, ask)   # the run ends with the question
            if not answer.tool_calls:
                return   # the last text is the task's result

    return handler


handler = agent()
