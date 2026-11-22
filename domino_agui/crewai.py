"""The ready-made CrewAI flow, and the agent contract's parts as CrewAI helpers. An agent's entry point can be one line:

    from domino_agui.crewai import build_flow

    "framework": {"name": "crewai", "entrypoint": "flow.py:build_flow"}

build_flow(), which agent() makes, returns a flow that calls the model and does what each answer asks, until an answer has no tool
call. The platform's adapter serves it with the bridge, ag-ui-crewai: each AG-UI run starts converse(). The parts:
- model_name(): the LiteLLM name of the model, for the alias in AGENT_MODEL, in the format of AGENT_MODEL_FORMAT: `openai/<alias>`
  (chat completions at OPENAI_BASE_URL) or `anthropic/<alias>` (Messages at ANTHROPIC_API_BASE). Each call streams, because the
  bridge sends text and tool calls (copilotkit_stream) only for a streamed response;
- relay_tools() and relay_call(): the gateway tools that the relay lists (contract.relay_tools(): MCP at $AGENT_SERVICE_URL/mcp, with
  $SESSION_TOKEN), in LiteLLM's form, and a call to one. A denial comes back as the tool's answer. Outside a Job (no AGENT_SERVICE_URL)
  there are none;
- client_side(state): the client-side tools, the AG-UI run's tools (RunAgentInput.tools), which the bridge puts in
  state.copilotkit.actions. The flow offers them to the model and doesn't run them: a call to one ends the AG-UI run, and its answer
  comes as a tool message in a later run. A tool in both lists is client-side;
- ask_human: the agent's tool for a question to a person, one at a time (contract.ONE_QUESTION). The flow's `@human_feedback` method
  returns question(...), which is {question, expiresAt}: question_provider makes the question the interrupt's message, and the
  platform's adapter puts the expiry on the interrupt. The reply is the tool's result; empty feedback (a cancelled resume: the
  question expired) gives contract.NO_ANSWER (answer());
- model_messages(state): the thread's history for the model, with the AG-UI run's context (the agent's system prompt, the platform's
  operating rules and the task brief) as the system prompt.

Two options, your own model and your own tools, make another entry point of the same shape:

    build_flow = agent(model=LLM(model="openai/" + os.environ["AGENT_MODEL"], max_tokens=1024), tools=[lookup_customer])

The model is a LiteLLM name, or a crewai.LLM for call options such as the maximum tokens (litellm_arguments()). Own tools are plain
functions (def or async def, with type hints and a docstring). They run here, in the agent server, in the sandbox: they aren't gateway
tools, so the gateway doesn't decide or audit them. A name clash fails (contract.check_names).

An author who needs their own flow builds one from these helpers: agent() shows how they fit. This module imports CrewAI when it is
imported (the core doesn't), so it needs the agent-crewai Environment's packages. It calls no host at import.
"""
import dataclasses
from typing import Optional

from . import contract, convert
from .relay import ToolResult

contract.require_framework(__name__, "crewai")   # in another framework's Environment: an ImportError that names this one

try:   # CrewAI first, so that in another framework's image the error names it
    from crewai.flow.flow import Flow, listen, or_, router, start
    from crewai.flow.human_feedback import human_feedback
    from ag_ui_crewai import CopilotKitState, agui_feedback_provider, copilotkit_emit_tool_result, copilotkit_stream
    from crewai.llms.base_llm import BaseLLM
    from litellm import acompletion
except ImportError as e:
    raise ImportError(f"domino_agui.crewai needs CrewAI and its packages, and this Python has no {e.name or e}: run the agent on the "
                      "agent-crewai Environment") from None

__all__ = ["QuestionProvider", "State", "agent", "answer", "build_flow", "client_side", "litellm_arguments", "model_messages", "model_name",
           "question", "question_from", "question_provider", "relay_call", "relay_tools"]

# A crewai LLM's call options that go to each model call, when the LLM sets them
_CALL_OPTIONS = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "seed", "presence_penalty", "frequency_penalty")


def model_name():
    """The LiteLLM name for AGENT_MODEL in AGENT_MODEL_FORMAT, read when it is called (so a flow builds with no model variables)."""
    return contract.alias_format() + "/" + contract.model_alias()


def litellm_arguments(model=None):
    """The model and call options of a LiteLLM acompletion() call, for agent(model=...): None, model_name() (read now); a LiteLLM name,
    that name; a crewai LLM (crewai.LLM(...), or a native provider that it makes, such as OpenAICompletion), its LiteLLM name and the
    call options it sets: temperature, top_p, max_tokens, max_completion_tokens, stop, seed, presence_penalty and frequency_penalty. The
    output limit is max_tokens=contract.MAX_TOKENS (16000, or AGENT_MAX_TOKENS) when the LLM sets neither max_tokens nor
    max_completion_tokens. The base URL and the key come from the environment (the Job's model proxy), never from the LLM."""
    limit = {"max_tokens": contract.MAX_TOKENS}
    if model is None:
        return {"model": model_name(), **limit}
    if isinstance(model, str):
        return {"model": model, **limit}
    name, provider = str(model.model), getattr(model, "provider", None)
    if not getattr(model, "is_litellm", False) and provider and not name.startswith(f"{provider}/"):
        name = f"{provider}/{name}"   # a native provider keeps the name without its prefix
    options = {k: v for k, v in ((k, getattr(model, k, None)) for k in _CALL_OPTIONS) if v not in (None, [], "")}
    return {"model": name, **({} if {"max_tokens", "max_completion_tokens"} & set(options) else limit), **options}


async def relay_tools():
    """Every tool the relay lists, under the relay's names (no prefix), as LiteLLM tools (convert.openai_tool). [] outside a Job (no
    AGENT_SERVICE_URL)."""
    return [convert.openai_tool(t) for t in await contract.relay_tools()]


async def relay_call(name, arguments):
    """Calls the gateway tool `name` through the relay, with the model's arguments (a dict or its JSON text). -> ToolResult: the answer
    as text, for the model. A denial is an answer too (is_error). Outside a Job there is no gateway tool: contract.no_tool()."""
    client = contract.relay()
    return await client.call(name, arguments) if client else ToolResult(contract.no_tool(name), True)


def client_side(state):
    """The names of the client-side tools of this AG-UI run (state.copilotkit.actions, as the bridge copies RunAgentInput.tools)."""
    return {t["function"]["name"] for t in state.copilotkit.actions}


def question(text, expires_at=None):
    """The output of a `@human_feedback` method that asks a person: {question, expiresAt}, plain strings. With question_provider, the
    person sees `text`, and the platform's adapter puts `expires_at` (ISO 8601) on the interrupt, so the question can expire.
    No expiresAt when expires_at is None."""
    return {"question": str(text or "") or "The agent has a question.", **({"expiresAt": str(expires_at)} if expires_at else {})}


def question_from(arguments):
    """question() for the model's ask_human call: its arguments, a dict or the model's JSON text ({question, expires_at})."""
    return question(*contract.ask_human_arguments(arguments))


class QuestionProvider:
    """The bridge's feedback provider (ag_ui_crewai.agui_feedback_provider), with the question as the interrupt's message:
    `@human_feedback`'s message is fixed per method, and the question is in the method's output, question(...)."""

    def request_feedback(self, context, flow):
        output = context.method_output
        text = output.get("question") if isinstance(output, dict) else output
        return agui_feedback_provider.request_feedback(dataclasses.replace(context, message=str(text)), flow)


question_provider = QuestionProvider()


def answer(feedback):
    """ask_human's result in the resumed run: the person's reply (`feedback` is flow.last_human_feedback, or its text), or
    contract.NO_ANSWER when it is empty. A cancelled resume (the question expired) reaches the flow as empty feedback: CrewAI has no
    cancel. That holds on this platform only, where a person can't send an empty reply."""
    text = getattr(feedback, "feedback", feedback)
    return (text.strip() if isinstance(text, str) else "") or contract.NO_ANSWER


def model_messages(state):
    """The thread's history (state.messages, as the bridge gives it) as chat messages for LiteLLM, with only the fields a model API
    takes, after the system prompt from the AG-UI run's context (state.context). No system message when there is no context."""
    system = contract.system_prompt(state.context)
    out = [{"role": "system", "content": system}] if system else []
    for m in state.messages:
        role = m.get("role")
        if role in ("user", "system", "developer"):
            out.append({"role": "system" if role == "developer" else role, "content": m.get("content") or ""})
        elif role == "assistant":
            calls = [{"id": c["id"], "type": "function", "function": {"name": c["function"]["name"],
                                                                     "arguments": c["function"].get("arguments") or ""}}
                     for c in m.get("tool_calls") or []]
            out.append({"role": "assistant", "content": m.get("content") or "", **({"tool_calls": calls} if calls else {})})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": m.get("content") or ""})
    return out


class State(CopilotKitState):
    """The flow's state: the bridge's (the messages, the client-side tools in copilotkit.actions, the context) and the open question."""
    asking: Optional[dict] = None   # the model's ask_human call, while it waits for the person's answer


def _tool_message(call_id, text):
    return {"id": f"result-{call_id}", "role": "tool", "tool_call_id": call_id, "content": text}


def agent(model=None, tools=()):
    """The ready-made flow's entry point: build_flow(), which returns the flow that the platform's adapter serves.

    The two options:
    - model: a LiteLLM model name, such as "openai/" + os.environ["AGENT_MODEL"], or a crewai.LLM for the options of each model call,
      such as LLM(model=..., max_tokens=1024): its name and its call options go to each streamed call (litellm_arguments). It still
      goes through the model proxy (the base URLs and the keys come from the environment). None: model_name(), at the first run;
    - tools: own tools, plain functions (def or async def, with type hints and a docstring: contract.function_tool). They run here, in
      the agent server, in the sandbox: they aren't gateway tools, so the gateway doesn't decide or audit them. An own tool named
      ask_human raises NameClash here; one with a gateway tool's or a client-side tool's name fails the AG-UI run.

    The flow class is made here, with its flow methods in its own body: crewai finds a flow's methods in its class's own __dict__ only,
    and the resume (Flow.from_pending) makes a new instance from the class. So the options live in the class, never on an instance."""
    if model is not None and not isinstance(model, (str, BaseLLM)):
        raise TypeError(f"model= is a LiteLLM model name, such as \"openai/<model>\" or \"anthropic/<model>\", or a crewai.LLM, and "
                        f"this one is a {type(model).__name__}")
    specs, own = contract.own_tools(tools)
    own_and_ask = [convert.openai_tool(s) for s in specs] + [convert.openai_tool(contract.ASK_HUMAN_TOOL)]

    class ReadyMadeFlow(Flow[State]):
        @start()
        async def converse(self):
            self.state.asking = None   # a new AG-UI run (not a resume) has no question open
            await self.follow_model()

        @router(or_("converse", "answered"))
        def next_step(self):
            return "ask" if self.state.asking else "done"

        @listen("ask")
        @human_feedback(message="The agent has a question.", provider=question_provider)
        def ask_person(self):
            """-> {question, expiresAt}: the model's question, and its expires_at (none when the model gave none)."""
            return question_from(self.state.asking["function"]["arguments"])

        @listen(ask_person)
        async def answered(self):
            """The resumed run: the person's answer, or contract.NO_ANSWER, is the ask_human call's result."""
            call, self.state.asking = self.state.asking, None
            self.state.messages.append(_tool_message(call["id"], answer(self.last_human_feedback)))
            await self.follow_model()

        async def follow_model(self):
            """Asks the model and does what each answer says, until an answer has no tool call, calls a client-side tool, or asks the
            person (then `asking` holds the call, and the router asks). An own tool with the name of a gateway tool or a client-side
            tool fails the run first (NameClash)."""
            client, relay = client_side(self.state), await relay_tools()
            contract.check_names(own, gateway=[t["function"]["name"] for t in relay], client=client)
            offered = list(self.state.copilotkit.actions) + [t for t in relay if t["function"]["name"] not in client] + own_and_ask
            while True:
                said = await copilotkit_stream(await acompletion(**litellm_arguments(model), messages=model_messages(self.state),
                                                                 tools=offered, stream=True))
                message = said.choices[0].message
                calls = [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments or ""}}
                         for c in message.tool_calls or []]
                self.state.messages.append({"id": said.id, "role": "assistant", "content": message.content or "",
                                            **({"tool_calls": calls} if calls else {})})
                names = {c["function"]["name"] for c in calls}
                for call in calls:
                    name, arguments = call["function"]["name"], call["function"]["arguments"]
                    if name in client:
                        continue   # stays open: the platform runs it, and answers in the next AG-UI run
                    if name == contract.ASK_HUMAN and not self.state.asking and not client & names:
                        self.state.asking = call   # answered after the person replies, in a later AG-UI run
                        continue
                    if name == contract.ASK_HUMAN:   # one interrupt per run, and a client-side call ends the run first
                        text = contract.ONE_QUESTION
                    else:   # an own tool here, a gateway tool through the relay, else no_tool(), never through the relay
                        text = (await contract.call_tool(name, arguments, own)).text
                    self.state.messages.append(_tool_message(call["id"], text))
                    await copilotkit_emit_tool_result(call["id"], text)   # the bridge sends a result only when the flow says so
                if not calls or self.state.asking or client & names:
                    return   # the method ends: the last text is the result, the client-side calls stay open, or the router asks

    def build_flow():
        """The ready-made flow."""
        return ReadyMadeFlow()

    return build_flow


build_flow = agent()
