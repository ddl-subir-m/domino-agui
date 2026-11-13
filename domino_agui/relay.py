"""The relay client: a hand-written agent lists and calls gateway tools (its agent-side tools) with no MCP code of its own. The relay
is the job's MCP endpoint at `$AGENT_SERVICE_URL/mcp`; it takes `$SESSION_TOKEN` as a bearer token, and every call goes through the
gateway's decision and audit.

    relay = domino_agui.Relay()
    tools = await relay.list_tools()

The relay keeps no MCP session, so the client sends no `initialize`: each request is one JSON-RPC call.
"""
import asyncio
import http.client
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .run import _id


class _ReadTimeout:
    """An http.client connection that connects within its timeout, then waits up to `read_timeout` for each read."""
    def __init__(self, *args, read_timeout, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_timeout = read_timeout

    def connect(self):
        super().connect()
        self.sock.settimeout(self.read_timeout)


class _HTTP(_ReadTimeout, http.client.HTTPConnection):
    pass


class _HTTPS(_ReadTimeout, http.client.HTTPSConnection):
    pass


class _Timeouts(urllib.request.HTTPHandler, urllib.request.HTTPSHandler):
    """http and https with two timeouts: the request's timeout to connect, then `read` seconds for each read."""
    def __init__(self, read):
        super().__init__()
        self.read = read

    def http_open(self, req):
        return self.do_open(_HTTP, req, read_timeout=self.read)

    def https_open(self, req):
        return self.do_open(_HTTPS, req, context=self._context, read_timeout=self.read)


@dataclass(frozen=True)
class ToolResult:
    """A gateway tool's answer: `text` for the model, and `is_error` when the call was denied or failed. Send it with
    run.tool_call(name, call.arguments, result.text, tool_call_id=call.id)."""
    text: str
    is_error: bool = False

    def message(self, tool_call_id):
        """The result as an AG-UI tool message that answers `tool_call_id`, for the agent's own copy of the history within one AG-UI
        run (next to `Answer.message()`)."""
        return {"id": _id("msg"), "role": "tool", "toolCallId": tool_call_id, "content": self.text,
                **({"error": self.text} if self.is_error else {})}


class RelayError(RuntimeError):
    """The relay could not answer: it is not set up, can't be reached, refused the token, or did not answer in MCP. A gateway denial is
    not a RelayError: it is a result with `is_error`."""


class Relay:
    """The client for one Job's relay. `url` and `token` default to `$AGENT_SERVICE_URL` and `$SESSION_TOKEN`, which the platform sets
    in the agent server's environment. `timeout` (seconds) is the time to connect, and to wait for each answer unless `read_timeout`
    is given: a relay call can stay open for up to 45 s, so keep the time for an answer at 60 s or more."""

    def __init__(self, url=None, token=None, *, timeout=120, read_timeout=None):
        url, token = url or os.environ.get("AGENT_SERVICE_URL"), token or os.environ.get("SESSION_TOKEN")
        missing = [k for k, v in (("AGENT_SERVICE_URL", url), ("SESSION_TOKEN", token)) if not v]
        if missing:
            raise RelayError(f"{' and '.join(missing)} not set: the platform sets them in a framework-lane Job. Elsewhere, give "
                             "Relay(url=..., token=...) for a relay you run.")
        self.endpoint = url.rstrip("/") + "/mcp"
        self._token, self._timeout, self._ids = token, timeout, 0
        self._read = timeout if read_timeout is None else read_timeout
        # the relay is local: never through HTTP_PROXY (the egress proxy)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _Timeouts(self._read))

    async def list_tools(self):
        """The gateway tools the relay offers this agent, as AG-UI tools (`name`, `description`, `parameters`): give them to
        convert.to_openai() or convert.to_anthropic() with `run.client_tools`."""
        listed = await self._rpc("tools/list", {})
        return [{"name": t["name"], "description": t.get("description") or "", "parameters": t.get("inputSchema") or {}}
                for t in listed.get("tools") or []]

    async def call(self, name, args):
        """Calls the gateway tool `name`. `args` is a dict, or the JSON text the model wrote (`ToolCall.arguments`). Returns a
        `ToolResult`. A denial, a failed call, or arguments that are not a JSON object (the tool is then not called) come back as a
        result with `is_error`, for the model to read: an exception means the relay itself failed."""
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except ValueError as e:
                args = e
        if not isinstance(args, dict):
            return ToolResult(f"The arguments for {name} are not a JSON object ({args}), so the tool was not called. Call it again with "
                              "a JSON object.", True)
        done = await self._rpc("tools/call", {"name": name, "arguments": args})
        return ToolResult("\n".join(c.get("text", "") for c in done.get("content") or [] if c.get("type") == "text"),
                          bool(done.get("isError")))

    async def _rpc(self, method, params):
        self._ids += 1
        body = {"jsonrpc": "2.0", "id": self._ids, "method": method, "params": params}
        return await asyncio.to_thread(self._post, body)

    def _post(self, body):
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Accept": "application/json",
                                              "Authorization": f"Bearer {self._token}"})
        where = f"the relay at {self.endpoint}"
        try:
            with self._opener.open(req, timeout=self._timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            refused = ": it refused the session token" if e.code == 401 else ""
            raise RelayError(f"{where} answered {body['method']} with HTTP {e.code}{refused}: {e.read()[:300]!r}") from None
        except TimeoutError:   # connected, then no answer in time (a connect timeout comes as a URLError)
            raise RelayError(f"{where} did not answer {body['method']} within {self._read} s") from None
        except OSError as e:   # URLError, a timeout, a reset connection
            raise RelayError(f"{where} can't be reached ({getattr(e, 'reason', e)})") from None
        try:
            reply = json.loads(raw)
            return reply["result"]
        except (ValueError, KeyError, TypeError):
            raise RelayError(f"{where} did not answer {body['method']} with an MCP result: {raw[:300]!r}") from None
