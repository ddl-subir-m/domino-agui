# domino-agui

domino-agui is the helper library for framework agents on the Domino agent service. The agent service runs your agent's code in a
Domino Job, in a sandbox, and talks to it in [AG-UI](https://docs.ag-ui.com) 1.0. This library speaks AG-UI for your agent, and
keeps the agent contract: the model, the governed tools, the questions to a person, and the thread.

It has three parts:

- **A ready-made agent for each framework:** a LangGraph graph, a CrewAI flow, and an agent loop for the Anthropic and OpenAI SDKs.
  Your agent can be one line.
- **The helpers** that each ready-made agent is made from, for your own graph, flow or loop.
- **The core,** for a hand-written agent's handler: the server, the run's lifecycle rules, converters for the two model formats,
  and a client for the relay (the agent's governed tools). The core needs only the Python standard library.

## Install

```
pip install domino-agui
```

Python 3.10 or newer. The core has no dependencies. A framework module imports its framework only when you import that module, so
install the framework's packages too:

| Module | Its packages |
|---|---|
| `domino_agui.langgraph` | `langgraph`, `langchain-core`, `langchain-mcp-adapters`, `httpx`, and `langchain-openai` or `langchain-anthropic` for the model |
| `domino_agui.crewai` | `crewai`, `ag-ui-crewai`, `litellm` |
| `domino_agui.loop` | `openai`, `anthropic` |

On Domino, you don't install anything: the platform's certified Environments have this library and the framework's packages, and
they pin the framework versions.

## Quick start: a ready-made agent

Add one file with one line to your project. The agent's entry point is the file, a colon, and the name that the line imports:

| Framework | The file's one line | The entry point, for the file `agent/<name>.py` |
|---|---|---|
| LangGraph | `from domino_agui.langgraph import build_graph` | `agent/graph.py:build_graph` |
| CrewAI | `from domino_agui.crewai import build_flow` | `agent/flow.py:build_flow` |
| The loop (Anthropic and OpenAI SDKs) | `from domino_agui.loop import handler` | `agent/handler.py:handler` |

In the agent service, give the agent its framework (LangGraph, CrewAI or Domino AG-UI SDK for the loop), the entry point, and the
certified Environment of the framework.

Each ready-made agent does the same things:

- It calls the model in `AGENT_MODEL`, in the format of `AGENT_MODEL_FORMAT`, through the Job's model proxy. Each model call has an
  output limit of 16,000 tokens (`contract.MAX_TOKENS`), or the value of `AGENT_MAX_TOKENS`.
- It offers the model the agent-side tools that the relay lists, and runs them in the same AG-UI run. A denial comes back as the
  tool's answer, and the model reads it.
- It offers the client-side tools of the run, and doesn't run them. A call to one ends the AG-UI run. The agent service runs the
  tool, and the answer comes back in the next AG-UI run.
- It offers `ask_human`, the agent's tool for a question to a person. The AG-UI run ends with one interrupt, and the reply is the
  tool's result.
- It uses the run's context (the agent's system prompt, the platform's operating rules and the task brief) as the model's system
  prompt.
- It loops until the model's answer has no tool call. That last text is the task's result.
- An empty answer (no text and no tool call) doesn't end the run at once. The LangGraph graph and the loop give the model one more
  message, `contract.EMPTY_ANSWER`: "Your last answer was empty. Continue with the task, or give your result." They do this one time
  in a run. When the next answer is empty too, the run ends with no result. The CrewAI flow runs its own model loop, so it has the
  output limit, but not this retry.
- An answer that stopped at the output limit isn't a finished answer. When the last answer of the LangGraph graph or the loop stopped
  at the limit, the run's last text, and so the task's result, is `contract.AT_LIMIT`: it says that the answer isn't finished, and
  how to give the model a larger limit.

## The two options: `model=` and `tools=`

Each framework module has `agent()`, which makes its ready-made agent. `build_graph`, `build_flow` and `handler` are `agent()` with no
options. Two options make another entry point of the same shape:

```python
from domino_agui.loop import agent


def risk_score(customer_id: str, days: int = 30) -> str:
    """The customer's risk score for the last `days` days."""
    return f"{customer_id}: low risk in the last {days} days"


handler = agent(tools=[risk_score])
```

**`tools=`** takes your own tools: plain functions (`def` or `async def`), with type hints and a docstring. The LangGraph module also
takes LangChain tools.

- The tool's name is the function's name, its description is the docstring's first paragraph, and its parameters come from the
  type hints: `str`, `int`, `float`, `bool`, `list`, `dict`, `list[X]`, `dict[str, X]`, `Optional[X]` (or `X | None`) and
  `Literal[...]`. A parameter with a default is optional.
- It returns a `str`, or a value that JSON can encode. An exception is the tool's answer: the model reads it, and the run goes on.
- It runs in the agent server, in the sandbox. It isn't a gateway tool, so the gateway doesn't decide or audit its calls. A tool that
  needs a decision, an approval or an audit row must be a gateway tool.
- Its name must be its own. An own tool named `ask_human`, or two own tools with one name, fail when the entry point is built. An
  own tool with the name of a gateway tool or a client-side tool fails the AG-UI run.

**`model=`** takes the framework's own model type, with the options of each model call, such as the maximum tokens:

| Module | `model=` |
|---|---|
| `domino_agui.langgraph` | A LangChain chat model: `ChatOpenAI(model=os.environ["AGENT_MODEL"], use_responses_api=False, max_tokens=1024)` |
| `domino_agui.crewai` | A LiteLLM model name, `"openai/" + os.environ["AGENT_MODEL"]`, or a `crewai.LLM(model=..., max_tokens=1024)` |
| `domino_agui.loop` | `Model(AsyncAnthropic(), max_tokens=1024)`, or an SDK client alone: `AsyncOpenAI(max_retries=5)` |

The model still goes through the Job's model proxy: the base URL and the key come from the environment. Give `ChatOpenAI`
`use_responses_api=False`, so that it uses chat completions. To limit the length of an answer, use `max_tokens`. Don't set a
temperature: a gpt-5 model can refuse a temperature other than 1.

**The output limit.** With no `model=`, each model call has the limit `contract.MAX_TOKENS`: 16,000 tokens, or the value of
`AGENT_MAX_TOKENS` (a whole number above 0, read when the library is imported). To change the limit of the platform's model in
LangGraph, use `chat_model(max_tokens=...)`:

```python
from domino_agui.langgraph import agent, chat_model

build_graph = agent(model=chat_model(max_tokens=32000))
```

For the loop, `Model(max_tokens=32000)` keeps the platform's model and changes its limit. The CrewAI flow and the loop give the limit
to an own model that sets none. A LangChain chat model of your own keeps its own limit: give it `max_tokens`, because a model alias
with no model profile can get a small default (4,096 tokens), which cuts a long answer.

## A hand-written handler on the core

A handler is `async def handler(run)`. `run` has the AG-UI run's inputs (`run.messages`, the full history; `run.state`;
`run.resume`; `run.client_tools`; `run.context`) and its outputs (`run.text()`, `run.tool_call()`, `run.client_tool_call()`,
`run.state_snapshot()`, `run.interrupt()`, `run.finish()`). The library keeps the lifecycle rules: it sends `RUN_STARTED` first and
one final event last, and a handler that raises ends the run with `RUN_ERROR`.

This handler owns its loop. It calls an OpenAI-compatible model through the converters, and runs the gateway tools through the relay:

```python
from openai import AsyncOpenAI

from domino_agui import contract, convert

client = AsyncOpenAI()   # the base URL and the key come from the environment


async def handler(run):
    client_side = {t["name"] for t in run.client_tools}
    tools = run.client_tools + [t for t in await contract.relay_tools() if t["name"] not in client_side]
    system = contract.system_prompt(run.context)
    history = ([{"id": "context", "role": "system", "content": system}] if system else []) + run.messages
    while True:
        completion = await client.chat.completions.create(model=contract.model_alias(), **convert.to_openai(history, tools))
        answer = convert.from_openai(completion)
        history.append(answer.message())
        if answer.text:
            run.text(answer.text)
        for call in answer.tool_calls:
            if call.name in client_side:   # the agent service runs it: the call stays open, and the run ends
                run.client_tool_call(call.name, call.arguments, tool_call_id=call.id)
            else:   # a gateway tool, through the relay: a denial is an error result, not an exception
                result = await contract.call_tool(call.name, call.arguments)
                run.tool_call(call.name, call.arguments, result.text, tool_call_id=call.id)
                history.append(result.message(call.id))
        if not answer.tool_calls or any(c.name in client_side for c in answer.tool_calls):
            return   # the last text is the task's result, or the client-side calls stay open
```

For the Anthropic SDK, use `convert.to_anthropic()` and `convert.from_anthropic()` with `client.messages.create()`. To ask a person,
end the run with `run.interrupt(question)`: the answer is `run.resume` in the next run. The ready-made loop, `domino_agui.loop`, has
the whole loop with `ask_human`, and its helpers (`call_model()`, `ask_human()`, `answered_question()`) are public.

To try a handler on a laptop, serve it on a Unix socket and post an AG-UI run to it:

```
python -c 'import domino_agui, my_agent; domino_agui.serve(my_agent.handler, "agent.sock")' &
curl -sN --unix-socket agent.sock http://agent/agent -H 'Content-Type: application/json' \
  -d '{"threadId": "t1", "runId": "r1", "messages": [{"id": "m1", "role": "user", "content": "Say hello in five words."}]}'
```

The answer is the run's AG-UI events, one per `data:` line.

## The agent contract

The platform sets these variables in the agent server's environment, in a Job:

| Variable | What it is |
|---|---|
| `AGENT_MODEL` | The model's alias at the model gateway. The ready-made agents call this model. |
| `AGENT_MODEL_FORMAT` | The model's format: `anthropic` (Anthropic Messages), else `openai` (OpenAI chat completions, which OpenAI-compatible models speak too). |
| `AGENT_SERVICE_URL` | The agent service. Its relay is MCP at `$AGENT_SERVICE_URL/mcp`: the agent-side tools. Not set outside a Job, so there are no agent-side tools on a laptop. |
| `SESSION_TOKEN` | The relay's bearer token for this Job. |
| `AGENT_FRAMEWORK` | The framework of the Environment. A framework module of another framework fails at its import, with an error that names the Environment to use. Not set on a laptop. |

You can set one more variable in the Job's environment, for example as a project's environment variable: `AGENT_MAX_TOKENS`, the
output limit of the ready-made agents' model calls, in tokens (`contract.MAX_TOKENS`, 16,000 when it isn't set). The sandbox keeps
`AGENT_*` variables for the agent server. A value that isn't a whole number above 0 fails at the import, with an error that names it.

The platform also sets each model SDK's base URL and key, so that each model call goes through the Job's model proxy and the model
gateway. On a laptop, set `AGENT_MODEL`, `AGENT_MODEL_FORMAT` and your provider's key.

The relay's rules:

- Each relay call goes through the gateway's decision and audit. A denial or a failed call is a result with `is_error`, for the
  model to read. A `RelayError` means that the relay itself failed.
- The relay client of the ready-made agents connects within 30 seconds (`contract.CONNECT_TIMEOUT`), and waits up to 300 seconds
  for each answer (`contract.READ_TIMEOUT`). A relay call can stay open for up to 45 seconds, so keep a read timeout of 60 seconds or
  more. `Relay(url=None, token=None, *, timeout=120, read_timeout=None)`: `timeout` is the time to connect, and to wait for each
  answer unless `read_timeout` is given.
- The relay client connects to the relay directly, never through the egress proxy in `HTTP_PROXY`.

The run's rules:

- Each AG-UI run brings the full history and the last state. The agent server keeps nothing between runs: the agent service keeps
  the thread.
- A run ends with one interrupt at most, and an interrupt is a question, never an approval. A resume is `resolved`, with the reply
  text, or `cancelled`, when the answer came after the question's `expiresAt`.
- A run with open client-side calls ends as a success that names them. Their answers are tool messages in the next run.
- The agent server listens on a Unix socket: `POST /agent` takes a `RunAgentInput` and answers with server-sent events, and
  `GET /health` answers `{ok, framework, versions}`.

## Versions

The versions follow [semantic versioning](https://semver.org). The changes are in
[CHANGELOG.md](https://github.com/ddl-subir-m/domino-agui/blob/main/CHANGELOG.md).

## Report a problem

Open an issue on [GitHub](https://github.com/ddl-subir-m/domino-agui/issues). For a security problem, don't open an issue: read
[SECURITY.md](https://github.com/ddl-subir-m/domino-agui/blob/main/SECURITY.md).

## The checks

The checks need only the standard library. Most of them run against the installed package; `check_text.py` reads the repo's text:

```
pip install .
python tests/check_domino_agui.py
python tests/check_docstrings.py
python tests/check_release_tag.py
python tests/check_text.py
```

## License

Apache License 2.0. Copyright Domino Data Lab, Inc. See [LICENSE](https://github.com/ddl-subir-m/domino-agui/blob/main/LICENSE)
and [NOTICE](https://github.com/ddl-subir-m/domino-agui/blob/main/NOTICE).
