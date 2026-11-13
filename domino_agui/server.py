"""The agent server's transport: HTTP/1.1 on a Unix socket, one request per connection.

  POST /agent   a RunAgentInput (JSON) → 200 text/event-stream, one AG-UI event per `data:` line
  GET  /health  → {ok, framework: "domino-agui", versions}

A Unix socket binds no port, so no other container in the pod can post runs to it.
"""
import asyncio
import json
import sys

from . import __version__
from .run import bad_input, play

FRAMEWORK = "domino-agui"


def _sse(event):
    """One AG-UI event as one SSE event: a single `data:` line of compact JSON, LF line endings."""
    return b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"


def _header(head, name):
    for line in head.split("\r\n")[1:]:
        k, _, v = line.partition(":")
        if k.strip().lower() == name:
            return v.strip()
    return None


def _response(writer, status, reason, body):
    data = json.dumps(body).encode()
    writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n"
                 .encode() + data)


async def _serve_one(handler, versions, reader, writer):
    try:
        head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
        method, target = (head.split("\r\n", 1)[0].split(" ") + ["", ""])[:2]
        if target == "/health" and method == "GET":
            _response(writer, 200, "OK", {"ok": True, "framework": FRAMEWORK, "versions": versions})
        elif target == "/agent" and method == "POST":
            try:
                body = json.loads(await reader.readexactly(int(_header(head, "content-length") or 0)))
                why = bad_input(body)
            except ValueError:
                why = "the body is not JSON, or Content-Length is missing or wrong"
            if why:
                _response(writer, 400, "Bad Request", {"error": f"not a RunAgentInput: {why}"})
            else:
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")
                await play(handler, body, lambda event: writer.write(_sse(event)))
        else:
            _response(writer, 404, "Not Found", {"error": f"no route {method} {target}: use POST /agent or GET /health"})
        await writer.drain()
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        pass   # the client went away, or sent no complete request: nothing to answer
    finally:
        writer.close()


async def start_server(handler, socket_path, *, versions=None):
    """Listens on `socket_path` and serves `handler` (`async def handler(run)`). `versions` adds to what /health reports."""
    reported = {"domino-agui": __version__, "python": sys.version.split()[0], **(versions or {})}
    return await asyncio.start_unix_server(lambda r, w: _serve_one(handler, reported, r, w), path=socket_path)


def serve(handler, socket_path, *, versions=None):
    """Serves `handler` on `socket_path` until the process stops."""
    async def main():
        server = await start_server(handler, socket_path, versions=versions)
        async with server:
            await server.serve_forever()
    asyncio.run(main())
