# Changelog

The changes in each release of domino-agui. The versions follow [semantic versioning](https://semver.org).

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
