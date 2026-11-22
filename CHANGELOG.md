# Changelog

The changes in each release of domino-agui. The versions follow [semantic versioning](https://semver.org).

## 0.3.0

Helpers for an agent's own graph and for gateway calls from code, and a higher output limit.

Added:

- The task's facts as data: `contract.task(context)` gives the facts of the run's context as a dict, or None when the run has no
  facts. `contract.FACTS` is the context item's description, and `contract.FACTS_VERSION` is the version that `task()` reads.
- Gateway calls from your own code: `contract.gateway_call(name, args)` gives the relay's `ToolResult`, and `contract.call_json(name,
  args)` gives the answer as JSON. Outside a Job, `gateway_call()` raises `RelayError`, and the message names `AGENT_SERVICE_URL`.
  On a denial or a failed call, `call_json()` raises `ToolError`. `ToolResult.json()` gives the JSON at the start of a tool's answer.
- Helpers for your own LangGraph graph (`domino_agui.langgraph`): `finish(text)` makes a text the run's last message, so the task's
  result. `quiet(runnable)` keeps the inner model calls out of the AG-UI stream. `structured(model, schema, parse=None)` gives a typed
  answer with no forced tool choice, and asks the model one more time when an answer does not fit. It raises `Unusable`, or
  `Refused` when the model refuses. `chat_model()` takes `max_tokens=`.
- `contract.MAX_TOKENS`, the output limit of each model call: 16,000, or the value of `AGENT_MAX_TOKENS` in the Job.
  `contract.EMPTY_ANSWER` and `contract.AT_LIMIT` are the texts that the ready-made agents use.

Changed:

- The ready-made agents set the output limit `contract.MAX_TOKENS` on each model call, unless the agent's own model sets one. Before,
  a model alias with no profile got 4,096 output tokens, and the loop used 4,096.
- After an empty answer (no text and no tool call), the ready-made LangGraph agent and the loop ask the model one more time, one time
  in each AG-UI run. A second empty answer ends the run with no result.
- In the ready-made LangGraph agent and the loop, when the last answer stops at the output limit, the task's result is
  `contract.AT_LIMIT`, a blank line, then the cut text.
- `contract.system_prompt()` leaves the task's facts out of the system prompt. Code reads them with `task()`.

## 0.2.0

The first release on PyPI. Its API is the API of the library in the Domino agent service's certified Environments.

- The core, on the standard library only: the agent server on a Unix socket (`serve()`, `start_server()`), one AG-UI run with its
  lifecycle rules (`Run`, `Resume`, `LifecycleError`), the converters for the OpenAI chat format and the Anthropic Messages format
  (`domino_agui.convert`), and the relay client (`Relay`, `ToolResult`, `RelayError`).
- The agent contract's shared parts (`domino_agui.contract`): `ask_human` and its texts, the system prompt from the run's context, the
  model alias and its format, the relay's tools, and the own tools' schema, call and name check.
- A ready-made agent for each framework, with the helpers that it is made from: `domino_agui.langgraph` (`build_graph`),
  `domino_agui.crewai` (`build_flow`) and `domino_agui.loop` (`handler`, for the Anthropic and OpenAI SDKs). Each module's `agent()`
  takes two options, `model=` and `tools=`.
- Python 3.10, 3.11, 3.12 and 3.13.
