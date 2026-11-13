"""The ready-made LangGraph graph, and the agent contract's parts as LangGraph helpers. An agent's entry point can be one line:

    from domino_agui.langgraph import build_graph

    "framework": {"name": "langgraph", "entrypoint": "graph.py:build_graph"}

build_graph(checkpointer), which agent() makes, compiles a graph that calls the model and does what each answer asks, until an answer
has no tool call:
- chat_model(): the model, for the alias in AGENT_MODEL, in the format of AGENT_MODEL_FORMAT: ChatAnthropic (Anthropic Messages) or
  ChatOpenAI (OpenAI chat completions). The base URLs and the keys come from the environment that the platform sets;
- relay_tools(): the gateway tools that the relay lists (MCP at $AGENT_SERVICE_URL/mcp, with $SESSION_TOKEN, the timeouts of
  contract.CONNECT_TIMEOUT and contract.READ_TIMEOUT), as LangChain tools. They run in this AG-UI run, and a denial comes back as the
  tool's answer. Outside a Job (no AGENT_SERVICE_URL) there are none;
- client_side(state): the client-side tools, the AG-UI run's tools (RunAgentInput.tools), which the bridge puts in state["tools"]. The
  graph offers them to the model and doesn't run them: a call to one ends the AG-UI run, and its answer comes as a tool message in a
  later run. A tool in both lists is client-side;
- ask_human: the agent's tool for a question to a person. The ask_human node ends the run with one interrupt (its message is the
  question, its expiresAt the model's expires_at). The reply is the tool's result; a cancelled resume (no reply before expiresAt) gives
  contract.NO_ANSWER;
- system_prompt(state): the AG-UI run's context (the agent's system prompt, the platform's operating rules and the task brief) as the
  model's system message. State keeps each context entry as a plain dict, which the checkpointer can store.

Two options, your own model and your own tools, make another entry point of the same shape:

    build_graph = agent(model=ChatOpenAI(model=os.environ["AGENT_MODEL"], use_responses_api=False), tools=[lookup_customer])

Own tools are plain functions (with type hints and a docstring) or LangChain tools. They run here, in the agent server, in the
sandbox: they aren't gateway tools, so the gateway doesn't decide or audit them. A name clash fails (contract.check_names).

An author who needs their own nodes builds a graph from these helpers: agent() shows how they fit. This module imports LangGraph when
it is imported (the core doesn't), so it needs the agent-langgraph Environment's packages. It calls no host at import.
"""
import os
from datetime import timedelta
from typing import Annotated

from . import contract, convert

contract.require_framework(__name__, "langgraph")   # in another framework's Environment: an ImportError that names this one

try:   # LangGraph first, so that in another framework's image the error names it
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import interrupt
    from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
    from langchain_core.tools import BaseTool, StructuredTool, ToolException
    from langchain_mcp_adapters.client import MultiServerMCPClient
    import httpx
    from typing_extensions import TypedDict   # as MessagesState's
except ImportError as e:
    raise ImportError(f"domino_agui.langgraph needs LangGraph and its packages, and this Python has no {e.name or e}: run the agent on "
                      "the agent-langgraph Environment") from None

__all__ = ["CANCELLED", "State", "agent", "ask_human", "build_graph", "chat_model", "client_side", "own_tool", "pending", "plain_context",
           "relay_tools", "route", "system_prompt"]

# What interrupt() returns when the platform cancels the question (the bridge's sentinel)
CANCELLED = "__agui_cancelled__"

_relay = None   # the relay's tools, listed once for each agent server, at the first run (no network at import)


def plain_context(_, new):
    """The reducer of the bridge's "ag-ui" keys: each context entry as a plain dict. The bridge passes AG-UI's Context objects, which
    the checkpointer can't store."""
    new = dict(new or {})
    new["context"] = [c if isinstance(c, dict) else {"description": getattr(c, "description", None), "value": getattr(c, "value", None)}
                      for c in new.get("context") or []]
    return new


class State(MessagesState, TypedDict("AgUi", {"ag-ui": Annotated[dict, plain_context]}, total=False)):
    """The graph's state: the messages, the bridge's "ag-ui" keys (the context), and the client-side tools."""
    tools: list   # the client-side tools: RunAgentInput.tools, as the bridge copies them


def system_prompt(state):
    """The AG-UI run's context as one system message for the model ([] when there is none)."""
    text = contract.system_prompt((state.get("ag-ui") or {}).get("context"))
    return [SystemMessage(text)] if text else []


def client_side(state):
    """The names of the client-side tools of this AG-UI run."""
    return {t["name"] for t in state.get("tools") or []}


def chat_model():
    """A chat model for AGENT_MODEL in AGENT_MODEL_FORMAT, made when it is called (so a graph builds with no model variables)."""
    model = contract.model_alias()
    if contract.alias_format() == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, use_responses_api=False)   # chat completions: the format OpenAI-compatible models speak too


def _direct(headers=None, timeout=None, auth=None):
    """The relay's HTTP client: the relay is local, so never through HTTP_PROXY (the egress proxy), whatever NO_PROXY says."""
    return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth, trust_env=False)


async def relay_tools():
    """Every tool the relay lists, under the relay's names (no prefix), as LangChain tools. [] outside a Job (no AGENT_SERVICE_URL)."""
    global _relay
    url = os.environ.get("AGENT_SERVICE_URL")
    if not url:
        return []
    if _relay is None:
        client = MultiServerMCPClient({"domino": {
            "transport": "streamable_http", "url": url.rstrip("/") + "/mcp",
            "headers": {"Authorization": "Bearer " + os.environ.get("SESSION_TOKEN", "")},
            # the read timeout is for each answer, the other one for connect, write and pool
            "timeout": timedelta(seconds=contract.CONNECT_TIMEOUT), "sse_read_timeout": timedelta(seconds=contract.READ_TIMEOUT),
            "httpx_client_factory": _direct}})
        _relay = await client.get_tools()
    return _relay


def pending(state):
    """The last answer's tool calls that have no result yet."""
    msgs = state["messages"]
    i = next((i for i in range(len(msgs) - 1, -1, -1) if isinstance(msgs[i], AIMessage)), None)
    if i is None:
        return []
    done = {m.tool_call_id for m in msgs[i + 1:] if isinstance(m, ToolMessage)}
    return [c for c in msgs[i].tool_calls if c["id"] not in done]


def route(state):
    """What runs next, by node name: "relay" for the open calls that run here (gateway tools and own tools), then "ask_human" for the
    question; END while client-side calls are open; else "agent" (the model) after a new message or result, or END after the model's
    answer."""
    calls, client = pending(state), client_side(state)
    if any(c["name"] not in client and c["name"] != contract.ASK_HUMAN for c in calls):
        return "relay"
    if any(c["name"] == contract.ASK_HUMAN for c in calls):
        return "ask_human"
    if calls:   # client-side calls only: the platform answers them in a later AG-UI run
        return END
    return END if isinstance(state["messages"][-1], AIMessage) else "agent"


def ask_human(state):
    """The ask_human node: the question, as one interrupt {reason: "ask_human", message, expiresAt}. On the resume this node runs again
    from its start, and interrupt() returns the reply, or the bridge's cancelled sentinel, which becomes contract.NO_ANSWER."""
    call = next(c for c in pending(state) if c["name"] == contract.ASK_HUMAN)
    question, expires_at = contract.ask_human_arguments(call["args"])
    answer = interrupt({"reason": contract.ASK_HUMAN, "message": question, **({"expiresAt": expires_at} if expires_at else {})})
    if isinstance(answer, dict) and answer.get(CANCELLED):
        answer = contract.NO_ANSWER
    return {"messages": [ToolMessage(str(answer), tool_call_id=call["id"], name=contract.ASK_HUMAN)]}


def own_tool(fn):
    """A plain function (def or async def) as a LangChain tool: the schema from its type hints and its docstring
    (contract.function_tool), and the call through contract.call_function, so that it runs as on every framework (an exception is an
    error answer). A LangChain tool, because the bridge sends a tool's result from its end event."""
    spec = contract.function_tool(fn)

    async def call(**arguments):
        result = await contract.call_function(fn, arguments)
        if result.is_error:
            raise ToolException(result.text)   # handled: an error ToolMessage, and the end event
        return result.text
    return StructuredTool(name=spec["name"], description=spec["description"], args_schema=spec["parameters"], coroutine=call,
                          handle_tool_error=True)


def agent(model=None, tools=()):
    """The ready-made graph's entry point: build_graph(checkpointer), which compiles the graph with the checkpointer that the platform's
    adapter gives (the thread lives in the state folder). The nodes: "agent" (one model call), "relay" (the open calls of gateway
    tools and own tools) and "ask_human"; route() picks the next one after each.

    The two options:
    - model: a LangChain chat model, such as ChatOpenAI(model=..., use_responses_api=False). It still goes through the model proxy
      (the base URL and the key come from the environment). None: chat_model(), at the first run;
    - tools: own tools, plain functions (def or async def, with type hints and a docstring: contract.function_tool) or LangChain tools.
      They run here, in the agent server, in the sandbox: they aren't gateway tools, so the gateway doesn't decide or audit them. An
      own tool named ask_human raises NameClash here; one with a gateway tool's or a client-side tool's name fails the AG-UI run."""
    langchain_tools = [t if isinstance(t, BaseTool) else own_tool(t) for t in tools]
    contract.check_names([t.name for t in langchain_tools])
    own = {t.name: t for t in langchain_tools}

    async def call_model(state):
        """One model call, with every tool the model can use: the relay's (without the client-side ones), the client-side ones, the own
        tools and ask_human. An own tool with the name of a gateway tool or a client-side tool fails the run first (NameClash)."""
        names, relay = client_side(state), await relay_tools()
        contract.check_names(own, gateway=[t.name for t in relay], client=names)
        offered = ([t for t in relay if t.name not in names] + [convert.openai_tool(t) for t in state.get("tools") or []]
                   + list(own.values()) + [convert.openai_tool(contract.ASK_HUMAN_TOOL)])
        llm = (model or chat_model()).bind_tools(offered)
        return {"messages": [await llm.ainvoke(system_prompt(state) + state["messages"])]}

    async def call_tools(state):
        """Each open call that runs here: an own tool's in this process, a gateway tool's through the relay. ainvoke with the call
        gives the ToolMessage (a tool error, such as a denial, is a ToolMessage too), and the bridge sends it as the call's result."""
        runnable, client = {**{t.name: t for t in await relay_tools()}, **own}, client_side(state)
        out = []
        for call in pending(state):
            if call["name"] in client or call["name"] == contract.ASK_HUMAN:
                continue
            if call["name"] in runnable:
                out.append(await runnable[call["name"]].ainvoke(call))
            else:
                out.append(ToolMessage(contract.no_tool(call["name"]), tool_call_id=call["id"], status="error"))
        return {"messages": out}

    def build_graph(checkpointer):
        """The compiled ready-made graph, with `checkpointer`."""
        g = StateGraph(State)
        g.add_node("agent", call_model)
        g.add_node("relay", call_tools)   # gateway and own tools; never name a node "tools": the bridge reads that name as a ToolNode's
        g.add_node("ask_human", ask_human)
        for node in (START, "agent", "relay", "ask_human"):
            g.add_conditional_edges(node, route, ["agent", "relay", "ask_human", END])
        return g.compile(checkpointer=checkpointer)

    return build_graph


build_graph = agent()
