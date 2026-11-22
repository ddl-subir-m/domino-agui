"""domino-agui: a hand-written agent's handler speaks AG-UI 1.0 through this library, and each framework gets a ready-made agent and
the agent contract's parts. The core (this package's import) is the standard library only.

    async def handler(run):
        run.text("Hello")
        run.finish(result="done")

    domino_agui.serve(handler, "/tmp/agent/agui.sock")

The core is the protocol side, so a handler can't break the contract: it owns the server (a Unix socket, POST /agent, GET /health)
and the lifecycle rules. The core has no loop: no model calls, no retries, and nothing kept between AG-UI runs. A hand-written agent's
code owns its loop, or names the ready-made one, `domino_agui.loop.handler`.

Modules: `server` (the transport), `run` (the lifecycle, inputs and outputs), `convert` (AG-UI to and from the OpenAI chat format
and the Anthropic Messages format), `relay` (the relay client: list and call gateway tools over MCP) and `contract` (the parts every
ready-made agent shares: ask_human, the expired question's text, the system prompt). A framework module, such as
`domino_agui.langgraph`, imports its framework when it is imported, never with the core. One step of an agent loop:

    answer = convert.from_openai(client.chat.completions.create(model=model, **convert.to_openai(history, tools=tools)))
    for call in answer.tool_calls:
        result = await relay.call(call.name, call.arguments)   # a denial is a result with is_error, not an exception
        run.tool_call(call.name, call.arguments, result.text, tool_call_id=call.id)
"""
__version__ = "0.3.0"
PROTOCOL_VERSION = "1.0"

from . import contract, convert  # noqa: E402
from .relay import Relay, RelayError, ToolResult  # noqa: E402
from .run import LifecycleError, Resume, Run  # noqa: E402
from .server import serve, start_server  # noqa: E402

__all__ = ["PROTOCOL_VERSION", "LifecycleError", "Relay", "RelayError", "Resume", "Run", "ToolResult", "contract", "convert", "serve",
           "start_server"]
