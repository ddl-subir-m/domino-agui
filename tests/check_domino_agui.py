"""domino-agui's own check: its public API is a contract with agent authors. The lifecycle and output cases serve a handler with the
library's own server on a real Unix socket and read the SSE answer the way a client does. The converter cases check each model format
as its API documents it. The relay client case talks to a small fake MCP server in this check. No model and no framework run anywhere,
and the check needs only the standard library.

It checks the installed package, not the source folder: CI builds the wheel, installs it in a clean venv, and runs this check there.

  pip install .
  python tests/check_domino_agui.py
"""
import asyncio, http.client, http.server, json, os, shutil, socket, subprocess, sys, tempfile, threading, time, typing
from pathlib import Path

import domino_agui

PACKAGE = Path(domino_agui.__file__).resolve().parent
print(f"domino_agui {domino_agui.__version__} from {PACKAGE}\n")
# The cases that run with the standard library only (python -I -S) put this folder on their import path: a copy of the installed
# package, alone, so that no other installed package can be imported there
LIB = tempfile.mkdtemp(prefix="domino-agui-")
shutil.copytree(PACKAGE, Path(LIB) / "domino_agui", ignore=shutil.ignore_patterns("__pycache__"))

failed = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        failed.append(name)


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=10)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(10)
        self.sock.connect(self.path)


def serve(handler, **kw):
    """Serves `handler` with the library's server on a fresh Unix socket, in a daemon thread; returns the socket path once it answers."""
    path = os.path.join(tempfile.mkdtemp(prefix="agui", dir="/tmp" if os.path.isdir("/tmp") else None), "agui.sock")
    threading.Thread(target=domino_agui.serve, args=(handler, path), kwargs=kw, daemon=True).start()
    for _ in range(200):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(path)
            return path
        except OSError:
            time.sleep(0.01)
    raise RuntimeError("the server did not listen on " + path)


def request(path, method, url, body=None):
    conn = UnixHTTPConnection(path)
    conn.request(method, url, body=None if body is None else json.dumps(body), headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, r.getheader("Content-Type") or "", data


def sse_events(data):
    """The `data:` field of each SSE event, parsed as JSON (the client side of the AG-UI SSE binding)."""
    out = []
    for block in data.decode().split("\n\n"):
        lines = [ln[5:].lstrip(" ") for ln in block.split("\n") if ln.startswith("data:")]
        if lines:
            out.append(json.loads("\n".join(lines)))
    return out


FINAL = ("RUN_FINISHED", "RUN_ERROR")


def lifecycle(events):
    """The AG-UI rules the library promises, checked from the stream alone. Returns the first broken rule, or ''."""
    if not events or events[0].get("type") != "RUN_STARTED":
        return "the stream does not start with RUN_STARTED"
    if events[0].get("protocolVersion") != "1.0":
        return "RUN_STARTED has no protocolVersion 1.0"
    finals = [i for i, e in enumerate(events) if e["type"] in FINAL]
    if len(finals) != 1 or finals[0] != len(events) - 1:
        return f"not exactly one final event, last: {[e['type'] for e in events]}"
    open_ = set()
    for e in events:
        t = e["type"]
        key = (t.rsplit("_", 1)[0], e.get("messageId") or e.get("toolCallId"))
        if t.endswith("_START") and t != "RUN_STARTED":
            open_.add(key)
        elif t.endswith("_END"):
            if key not in open_:
                return f"{t} without its _START"
            open_.discard(key)
    if open_:
        return f"a _START without its _END: {open_}"
    if any(has_null(e) for e in events):
        return "an optional field sent as null"
    return ""


def has_null(v):
    """A null value anywhere in `v` (not the text "null" inside a string, such as a tool's JSON answer)."""
    if isinstance(v, dict):
        return any(has_null(x) for x in v.values())
    if isinstance(v, list):
        return any(has_null(x) for x in v)
    return v is None


def run_input(messages=None, **kw):
    return {"threadId": "AGT-1", "runId": "seg1-1", "messages": messages or [{"id": "u1", "role": "user", "content": "hi"}], **kw}


def play(handler, body=None):
    """Serves `handler`, posts one AG-UI run, and returns its events (after checking the response is an SSE stream)."""
    sock = serve(handler)
    status, ctype, data = request(sock, "POST", "/agent", body or run_input())
    if status != 200 or not ctype.startswith("text/event-stream"):
        raise RuntimeError(f"POST /agent answered {status} {ctype}: {data[:300]!r}")
    return sse_events(data)


def types(events):
    return [e["type"] for e in events]


def of(events, t):
    return [e for e in events if e["type"] == t]


async def says_nothing(run):
    pass


# --- GET /health: {ok, framework, versions}
sock = serve(says_nothing)
status, ctype, data = request(sock, "GET", "/health")
health = json.loads(data or b"{}")
check("GET /health answers 200 with JSON", status == 200 and ctype.startswith("application/json"), (status, ctype))
check("/health is {ok: true, framework: domino-agui, versions} with the library's own version",
      health.get("ok") is True and health.get("framework") == "domino-agui"
      and health.get("versions", {}).get("domino-agui") == domino_agui.__version__, health)
sock = serve(says_nothing, versions={"openai": "2.45.0"})
health = json.loads(request(sock, "GET", "/health")[2])
check("the server adds the versions it is given (an adapter reports its model clients)",
      health["versions"].get("openai") == "2.45.0" and health["versions"].get("domino-agui") == domino_agui.__version__, health)

# --- POST /agent answers SSE; RUN_STARTED with protocolVersion 1.0 comes first, then one final event
async def says_hello(run):
    run.text("Hello")


ev = play(says_hello)
check("run.text(): RUN_STARTED, TEXT_MESSAGE_START/CONTENT/END, then RUN_FINISHED last; the stream keeps the lifecycle rules",
      lifecycle(ev) == "" and types(ev)[0] == "RUN_STARTED" and types(ev)[-1] == "RUN_FINISHED"
      and [t for t in types(ev) if t.startswith("TEXT_")] == ["TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END"],
      (lifecycle(ev), types(ev)))
check("RUN_STARTED and RUN_FINISHED carry the input's threadId and runId",
      (ev[0].get("threadId"), ev[0].get("runId"), ev[-1].get("threadId"), ev[-1].get("runId")) == ("AGT-1", "seg1-1", "AGT-1", "seg1-1"),
      (ev[0], ev[-1]))
start, content = of(ev, "TEXT_MESSAGE_START")[0], of(ev, "TEXT_MESSAGE_CONTENT")[0]
check("the text is one assistant message with one id across START/CONTENT/END",
      start.get("role") == "assistant" and content.get("delta") == "Hello"
      and start["messageId"] == content["messageId"] == of(ev, "TEXT_MESSAGE_END")[0]["messageId"], ev)

ev = play(says_nothing)
check("a handler that returns without output still gets RUN_STARTED and RUN_FINISHED", lifecycle(ev) == "", types(ev))

# --- a handler that raises ends the run with RUN_ERROR, and the reason says what went wrong
async def breaks(run):
    run.text("Working on it")
    raise ValueError("the account id is missing")


ev = play(breaks)
check("a handler that raises: RUN_ERROR is the one final event, with no RUN_FINISHED", lifecycle(ev) == "" and types(ev)[-1] == "RUN_ERROR"
      and not of(ev, "RUN_FINISHED"), (lifecycle(ev), types(ev)))
check("RUN_ERROR.message names the exception and its text, so a person can read the reason",
      "ValueError" in ev[-1].get("message", "") and "the account id is missing" in ev[-1].get("message", ""), ev[-1])
check("what the handler sent before it raised is still in the stream", of(ev, "TEXT_MESSAGE_CONTENT")[0].get("delta") == "Working on it", ev)

# --- inputs: run.messages (the full history), run.state, run.resume (the answer to an interrupt), run.client_tools, run.context
async def echoes_inputs(run):
    r = run.resume
    run.text(json.dumps({"messages": run.messages, "state": run.state, "client_tools": run.client_tools, "context": run.context,
                         "resume": r and {"interrupt_id": r.interrupt_id, "status": r.status, "payload": r.payload}}))


def echoed(body):
    return json.loads(of(play(echoes_inputs, body), "TEXT_MESSAGE_CONTENT")[0]["delta"])


history = [{"id": "u1", "role": "user", "content": "Triage alert 7"},
           {"id": "a1", "role": "assistant", "toolCalls": [{"id": "c1", "type": "function",
                                                             "function": {"name": "write_finding", "arguments": "{\"x\": 1}"}}]},
           {"id": "t1", "role": "tool", "toolCallId": "c1", "content": "saved"}]
tool = {"name": "write_finding", "description": "Write a governance finding", "parameters": {"type": "object", "properties": {}}}
got = echoed(run_input(history, state={"step": 2}, tools=[tool], context=[{"description": "Task brief", "value": "AML triage"}],
                       resume=[{"interruptId": "int-1", "status": "resolved", "payload": "account 42"}]))
check("run.messages is the full history, as sent", got["messages"] == history, got["messages"])
check("run.state is the input state", got["state"] == {"step": 2}, got["state"])
check("run.client_tools are the tools the client declared, with their definitions", got["client_tools"] == [tool], got["client_tools"])
check("run.context holds the operating rules and the task brief, for the agent's code to use or not",
      got["context"] == [{"description": "Task brief", "value": "AML triage"}], got["context"])
check("run.resume is the answer to the interrupt: interrupt_id, status, payload",
      got["resume"] == {"interrupt_id": "int-1", "status": "resolved", "payload": "account 42"}, got["resume"])
got = echoed(run_input())
check("absent inputs: state {}, no client tools, no context, run.resume is None",
      (got["state"], got["client_tools"], got["context"], got["resume"]) == ({}, [], [], None), got)

# --- an agent-side tool call: the agent ran it (through the relay) and reports the call with its result, in the same run
async def looks_up(run):
    returned = run.tool_call("search_datasets", {"query": "alerts"}, {"hits": 3}, tool_call_id="call_abc")
    run.text(f"returned {returned}")


ev = play(looks_up)
calls = [e for e in ev if e["type"].startswith("TOOL_CALL_")]
check("run.tool_call(): TOOL_CALL_START, ARGS, END, then RESULT, and the run keeps the lifecycle rules",
      lifecycle(ev) == "" and types(calls) == ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END", "TOOL_CALL_RESULT"], (lifecycle(ev), types(ev)))
check("the call keeps the agent's toolCallId (the model's id) and carries the tool's name",
      all(e.get("toolCallId") == "call_abc" for e in calls) and calls[0].get("toolCallName") == "search_datasets"
      and of(ev, "TEXT_MESSAGE_CONTENT")[0]["delta"] == "returned call_abc", calls)
check("the arguments are JSON text; a result that is not a string is sent as JSON text, from the tool role",
      json.loads(calls[1]["delta"]) == {"query": "alerts"} and json.loads(calls[3]["content"]) == {"hits": 3}
      and calls[3].get("role") == "tool" and calls[3].get("messageId"), calls)
check("a run with only agent-side calls has no pending tool calls", "pendingToolCallIds" not in json.dumps(ev[-1]), ev[-1])


async def raw_args(run):
    run.tool_call("search_datasets", '{"query": "al', "bad arguments")


calls = of(play(raw_args), "TOOL_CALL_ARGS")
check("arguments given as text are sent as written (a model's arguments can be broken JSON)", calls[0]["delta"] == '{"query": "al', calls)

# --- a client-side tool call: the agent service runs it, so the run ends with the call open and the answer comes in the next run
async def asks_client(run):
    run.text("Writing the finding")
    run.client_tool_call("write_finding", {"severity": "high"}, tool_call_id="call_1")
    run.client_tool_call("write_finding", {"severity": "low"}, tool_call_id="call_2")


ev = play(asks_client, run_input(tools=[tool]))
check("run.client_tool_call(): START, ARGS, END and no TOOL_CALL_RESULT; the run keeps the lifecycle rules",
      lifecycle(ev) == "" and not of(ev, "TOOL_CALL_RESULT") and len(of(ev, "TOOL_CALL_END")) == 2, (lifecycle(ev), types(ev)))
check("the run ends as a success that names the open calls, in order (outcome.pendingToolCallIds)",
      ev[-1]["type"] == "RUN_FINISHED" and ev[-1].get("outcome") == {"type": "success", "pendingToolCallIds": ["call_1", "call_2"]}, ev[-1])
check("the call carries its name and JSON arguments",
      of(ev, "TOOL_CALL_START")[0].get("toolCallName") == "write_finding"
      and json.loads(of(ev, "TOOL_CALL_ARGS")[0]["delta"]) == {"severity": "high"}, ev)

ev = play(asks_client, run_input())
check("a client-side call to a tool the client did not declare ends the run with RUN_ERROR that names the tool",
      lifecycle(ev) == "" and ev[-1]["type"] == "RUN_ERROR" and "write_finding" in ev[-1].get("message", "")
      and not of(ev, "TOOL_CALL_START"), (types(ev), ev[-1]))

# --- run.interrupt(): the agent asks a person; the run ends with an interrupt outcome, and the answer comes in the next run's resume
after_end = []


async def asks_person(run):
    returned = run.interrupt("Which account should I check?", interrupt_id="int-1", expires_at="2026-10-23T12:00:00Z")
    after_end.append(returned)
    try:
        run.text("one more thing")
    except domino_agui.LifecycleError as e:
        after_end.append(str(e))


ev = play(asks_person)
check("run.interrupt(): RUN_FINISHED with outcome interrupt, one interrupt with its id, reason, message and expiresAt",
      lifecycle(ev) == "" and ev[-1]["type"] == "RUN_FINISHED"
      and ev[-1].get("outcome") == {"type": "interrupt", "interrupts": [{"id": "int-1", "reason": "question",
                                                                        "message": "Which account should I check?",
                                                                        "expiresAt": "2026-10-23T12:00:00Z"}]},
      (lifecycle(ev), ev[-1]))
check("an output after the run ended raises LifecycleError and sends nothing (exactly one final event)",
      after_end[:1] == ["int-1"] and len(after_end) == 2 and not of(ev, "TEXT_MESSAGE_START"), (after_end, types(ev)))


async def interrupts_open_call(run):
    run.client_tool_call("write_finding", {})
    run.interrupt("Are you sure?")


ev = play(interrupts_open_call, run_input(tools=[tool]))
check("an interrupt while a client-side call is open ends the run with RUN_ERROR (a run with open calls ends as a success)",
      lifecycle(ev) == "" and ev[-1]["type"] == "RUN_ERROR" and "client-side" in ev[-1].get("message", "")
      and not of(ev, "RUN_FINISHED"), (types(ev), ev[-1]))


# --- run.state_snapshot(): the whole state, which the agent service keeps for the thread and sends back as the next run's state
async def saves_state(run):
    run.state_snapshot({"step": run.state.get("step", 0) + 1, "alert": 7})


ev = play(saves_state, run_input(state={"step": 2}))
check("run.state_snapshot(): one STATE_SNAPSHOT with the whole state, before the final event",
      lifecycle(ev) == "" and [e.get("snapshot") for e in of(ev, "STATE_SNAPSHOT")] == [{"step": 3, "alert": 7}]
      and types(ev).index("STATE_SNAPSHOT") < len(ev) - 1, (lifecycle(ev), ev))

# --- run.finish(result=...): the run's final result, which becomes the task's result
finished = []


async def finishes(run):
    run.text("Alert 7 is a false positive")
    run.finish(result={"verdict": "false_positive", "alert": 7})
    try:
        run.finish(result="again")
    except domino_agui.LifecycleError:
        finished.append("refused")


ev = play(finishes)
check("run.finish(result=...): RUN_FINISHED carries the result, as a success",
      lifecycle(ev) == "" and ev[-1].get("result") == {"verdict": "false_positive", "alert": 7}
      and ev[-1].get("outcome") == {"type": "success"}, (lifecycle(ev), ev[-1]))
check("a second finish raises LifecycleError, and the stream still has exactly one final event", finished == ["refused"], (finished, types(ev)))


async def finishes_bare(run):
    run.finish()


ev = play(finishes_bare)
check("run.finish() with no result leaves `result` out (optional fields are never null)", lifecycle(ev) == "" and "result" not in ev[-1], ev[-1])

# --- MESSAGES_SNAPSHOT: before the final event, the full history plus what this run said, so the thread that the agent service keeps
#     has the agent's own tool calls for the next run (a model rejects a tool answer whose call is not in the history)
async def works(run):
    run.text("Checking")
    run.tool_call("search_datasets", {"query": "alerts"}, "3 hits", tool_call_id="c1")
    run.client_tool_call("write_finding", {"severity": "high"}, tool_call_id="c2")


ev = play(works, run_input(history, tools=[tool]))
snap = of(ev, "MESSAGES_SNAPSHOT")
start = {e["toolCallId"]: e for e in of(ev, "TOOL_CALL_START")}
args = {e["toolCallId"]: e["delta"] for e in of(ev, "TOOL_CALL_ARGS")}
fn = lambda cid, name: {"id": cid, "type": "function", "function": {"name": name, "arguments": args[cid]}}
mid = of(ev, "TEXT_MESSAGE_START")[0]["messageId"]
want = history + [{"id": mid, "role": "assistant", "content": "Checking", "toolCalls": [fn("c1", "search_datasets")]},
                  {"id": of(ev, "TOOL_CALL_RESULT")[0]["messageId"], "role": "tool", "toolCallId": "c1", "content": "3 hits"},
                  {"id": start["c2"].get("parentMessageId"), "role": "assistant", "toolCalls": [fn("c2", "write_finding")]}]
check("one MESSAGES_SNAPSHOT, right before the final event", len(snap) == 1 and types(ev)[-2] == "MESSAGES_SNAPSHOT", types(ev))
check("the snapshot is the input history, then the assistant's text with its tool call, the tool's answer, and the open call",
      snap and snap[0].get("messages") == want, snap and snap[0].get("messages"))
check("each tool call names its assistant message (parentMessageId)",
      start["c1"].get("parentMessageId") == mid and start["c2"].get("parentMessageId") not in (None, mid), start)

ev = play(asks_person)
check("an interrupted run sends the snapshot too", types(ev)[-2:] == ["MESSAGES_SNAPSHOT", "RUN_FINISHED"], types(ev))

sock = serve(says_hello)
first = sse_events(request(sock, "POST", "/agent", run_input([{"id": "u1", "role": "user", "content": "one"}]))[2])
second = sse_events(request(sock, "POST", "/agent", run_input([{"id": "u2", "role": "user", "content": "two"}], runId="seg1-2"))[2])
msgs = of(second, "MESSAGES_SNAPSHOT")[0]["messages"]
check("nothing is kept between runs: the second run's snapshot is its own input and its own answer only",
      [m["id"] for m in msgs] == ["u2", of(second, "TEXT_MESSAGE_START")[0]["messageId"]] and lifecycle(first) == lifecycle(second) == "",
      msgs)

# --- run.remember(): messages the handler adds to its model's history itself go in the snapshot too, with no event, so the thread keeps
#     them for later runs (such as the answer to a question, which the resumed run gives the model: without it, the next run loses it)
q_call = {"id": "toolu_q", "type": "function", "function": {"name": "ask_human", "arguments": '{"question": "Which region?"}'}}
q_pair = [{"id": "asked", "role": "assistant", "toolCalls": [q_call]},
          {"id": "answered", "role": "tool", "toolCallId": "toolu_q", "content": "EMEA."}]


async def remembers(run):
    run.remember(*q_pair)
    run.text("Reporting on EMEA.")


ev = play(remembers, run_input(history))
snap = (of(ev, "MESSAGES_SNAPSHOT") or [{}])[0].get("messages")
check("run.remember(*messages): the snapshot is the input history, the remembered messages, then what the run said",
      lifecycle(ev) == "" and snap == history + q_pair + [{"id": (of(ev, "TEXT_MESSAGE_START") or [{}])[0].get("messageId"),
                                                          "role": "assistant", "content": "Reporting on EMEA."}], (lifecycle(ev), snap))
check("...and it sends no event of its own: the run's events are its text, the snapshot and the final event",
      types(ev)[1:] == ["TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END", "MESSAGES_SNAPSHOT", "RUN_FINISHED"], types(ev))

# ...and loop.answered_question() remembers the question's call and its answer, answered or cancelled. The loop imports the model SDKs,
# which this image doesn't have: stand-ins with the two class names are enough, because answered_question calls no model
ANSWERED = """import json, sys, types
sys.path.insert(0, sys.argv[1])
for name, cls in (("anthropic", "AsyncAnthropic"), ("openai", "AsyncOpenAI")):
    m = types.ModuleType(name); setattr(m, cls, type(cls, (), {})); sys.modules[name] = m
from domino_agui import contract, loop
from domino_agui.run import Run
asked = json.loads(sys.argv[2])
out = {}
for status in ("resolved", "cancelled"):
    events = []
    run = Run({"threadId": "AGT-1", "runId": "seg1-2", "messages": [{"id": "u1", "role": "user", "content": "Report"}],
               "state": {loop.ASKED: asked, "keep": 1},
               "resume": [{"interruptId": "int-1", "status": status, **({"payload": "EMEA."} if status == "resolved" else {})}]}, events.append)
    got = loop.answered_question(run)
    run.text("Reporting.")
    run.finish()
    out[status] = {"got": got, "snapshot": [e for e in events if e["type"] == "MESSAGES_SNAPSHOT"][0]["messages"],
                   "types": [e["type"] for e in events], "state": run.state, "noAnswer": contract.NO_ANSWER}
print(json.dumps(out))
"""
p = subprocess.run([sys.executable, "-I", "-S", "-c", ANSWERED, LIB, json.dumps(q_call)], capture_output=True, text=True,
                   env={**os.environ, "AGENT_FRAMEWORK": "domino-agui"})
res = json.loads(p.stdout) if p.returncode == 0 else {}
for status, reply in (("resolved", "EMEA."), ("cancelled", (res.get("cancelled") or {}).get("noAnswer"))):
    r = res.get(status) or {}
    pair = [{"id": "asked", "role": "assistant", "toolCalls": [q_call]},
            {"id": "answered", "role": "tool", "toolCallId": "toolu_q", "content": reply}]
    said = [m for m in r.get("snapshot") or [] if m.get("content") == "Reporting."]
    check(f"loop.answered_question, {status}: it gives the model the question's call and its result ({reply!r}), and the snapshot keeps "
          "both after the input history, so a later run (after a client-side call, or a later message) still has them",
          p.returncode == 0 and r.get("got") == pair and r.get("snapshot") == [{"id": "u1", "role": "user", "content": "Report"}] + pair + said
          and len(said) == 1, (p.stderr[-500:], r))
    check(f"...{status}: the question isn't sent again as a tool call, and the state no longer holds it (other keys stay)",
          "TOOL_CALL_START" not in (r.get("types") or ["TOOL_CALL_START"]) and r.get("state") == {"keep": 1}, r.get("types"))

# --- bad input gets an HTTP error status and no stream (AG-UI HTTP + SSE binding); the handler never runs
ran = []


async def records(run):
    ran.append(run.run_id)


sock = serve(records)
for name, body in [("no threadId", {"runId": "r", "messages": []}), ("no runId", {"threadId": "t", "messages": []}),
                   ("messages not a list", {"threadId": "t", "runId": "r", "messages": "hi"}),
                   ("two resume entries (the library opens one interrupt at a time)",
                    run_input(resume=[{"interruptId": "a", "status": "resolved"}, {"interruptId": "b", "status": "resolved"}])),
                   ("a resume entry with no interruptId", run_input(resume=[{"status": "resolved"}]))]:
    status, ctype, data = request(sock, "POST", "/agent", body)
    check(f"bad input ({name}): 400 with a JSON reason, no stream", status == 400 and ctype.startswith("application/json")
          and json.loads(data).get("error"), (status, ctype, data[:200]))
conn = UnixHTTPConnection(sock)
conn.request("POST", "/agent", body=b"{not json", headers={"Content-Type": "application/json"})
r = conn.getresponse()
check("bad input (not JSON): 400, no stream", r.status == 400 and not r.read().startswith(b"data:"), r.status)
check("the handler never ran for bad input", ran == [], ran)
check("an unknown route: 404 that names the routes", request(sock, "GET", "/runs")[0] == 404 and b"/agent" in request(sock, "GET", "/x")[2])

# --- the standard library only: it imports and serves with no site-packages at all
code = ("import sys, asyncio; sys.path.insert(0, sys.argv[1]); import domino_agui; "
        "asyncio.run(domino_agui.start_server(lambda run: None, sys.argv[2])); "
        "domino_agui.convert.to_openai([], tools=[domino_agui.contract.ASK_HUMAN_TOOL]); domino_agui.convert.to_anthropic([]); "
        "domino_agui.Relay('http://127.0.0.1:1', 'tok'); domino_agui.Relay('http://127.0.0.1:1', 'tok', timeout=30, read_timeout=300); "
        "domino_agui.contract.system_prompt([{'value': 'x'}]); "
        "c = domino_agui.contract; c.check_names(['f'], gateway=['g'], client=['h']); c.function_tool(c.system_prompt); "
        "asyncio.run(c.call_function(c.system_prompt, {'context': []})); c.own_tools([c.system_prompt]); c.ask_human_arguments('{}'); "
        "c.alias_format(); assert c.relay() is None and asyncio.run(c.call_tool('x', {})).is_error; "
        "domino_agui.convert.openai_tool(c.ASK_HUMAN_TOOL); "
        "print(domino_agui.__version__); print(sorted(m for m in sys.modules if m.startswith('domino_agui.')))")
p = subprocess.run([sys.executable, "-I", "-S", "-c", code, LIB,
                                  os.path.join(tempfile.mkdtemp(prefix="agui", dir="/tmp"), "s.sock")], capture_output=True, text=True)
out = p.stdout.split("\n")
check("domino_agui imports, starts its server, converts, makes a relay client and uses the contract's parts (the schema builder, the "
      "name check, an own tool's call and a tool call with no relay too) with the standard library only (python -I -S)", p.returncode == 0 and out[0] == domino_agui.__version__, p.stderr[-500:])
check("import domino_agui imports no framework module (domino_agui.langgraph, .crewai, .loop)",
      p.returncode == 0 and not {"domino_agui.langgraph", "domino_agui.crewai", "domino_agui.loop"} & set(json.loads(out[1].replace("'", '"'))),
      out[1:2])
# The 1-line entry point in another framework's image (here: no LangGraph at all) fails with an error that names the Environment to
# use. The agent server shows the exception's message as its last line
p = subprocess.run([sys.executable, "-I", "-S", "-c", "import sys; sys.path.insert(0, sys.argv[1]); from domino_agui.langgraph import build_graph",
                    LIB], capture_output=True, text=True)
last = (p.stderr.strip().splitlines() or [""])[-1]
check("from domino_agui.langgraph import build_graph, with no LangGraph installed: an ImportError that names the agent-langgraph "
      "Environment", p.returncode == 1 and last.startswith("ImportError: ") and "agent-langgraph Environment" in last, p.stderr[-500:])
p = subprocess.run([sys.executable, "-I", "-S", "-c", "import sys; sys.path.insert(0, sys.argv[1]); from domino_agui.crewai import build_flow",
                    LIB], capture_output=True, text=True)
last = (p.stderr.strip().splitlines() or [""])[-1]
check("from domino_agui.crewai import build_flow, with no CrewAI installed: an ImportError that names CrewAI and the agent-crewai "
      "Environment", p.returncode == 1 and last.startswith("ImportError: ") and "crewai" in last and "agent-crewai Environment" in last,
      p.stderr[-500:])
p = subprocess.run([sys.executable, "-I", "-S", "-c", "import sys; sys.path.insert(0, sys.argv[1]); from domino_agui.loop import handler",
                    LIB], capture_output=True, text=True)
last = (p.stderr.strip().splitlines() or [""])[-1]
check("from domino_agui.loop import handler, with no model SDK installed: an ImportError that names the SDKs and the agent-domino-agui "
      "Environment", p.returncode == 1 and last.startswith("ImportError: ") and "agent-domino-agui Environment" in last
      and "openai" in last and "anthropic" in last, p.stderr[-500:])
# ...and in another framework's image that has the packages (the loop's SDKs are in agent-langgraph and agent-crewai too): each image
# says which framework it is for (AGENT_FRAMEWORK, an ENV line in its Dockerfile; the sandbox keeps AGENT_* variables), and a framework
# module of another framework fails at its import, before it imports any package, with an error that names the Environment to use


def import_in(framework, module, attr):
    """`from domino_agui.<module> import <attr>` with the standard library only, in the image of `framework` (AGENT_FRAMEWORK, or none
    when it is None). -> (exit code, the error's last line)"""
    env = {k: v for k, v in os.environ.items() if k != "AGENT_FRAMEWORK"} | ({"AGENT_FRAMEWORK": framework} if framework else {})
    p = subprocess.run([sys.executable, "-I", "-S", "-c", f"import sys; sys.path.insert(0, sys.argv[1]); from domino_agui.{module} import {attr}",
                        LIB], capture_output=True, text=True, env=env)
    return p.returncode, (p.stderr.strip().splitlines() or [""])[-1]


for module, attr, own, other in (("langgraph", "build_graph", "langgraph", "crewai"), ("crewai", "build_flow", "crewai", "domino-agui"),
                                 ("loop", "handler", "domino-agui", "langgraph")):
    rc, last = import_in(other, module, attr)
    check(f"from domino_agui.{module} import {attr} in the agent-{other} image (AGENT_FRAMEWORK={other}): an ImportError that names the "
          f"agent-{own} Environment, and this one", rc == 1 and last.startswith("ImportError: ") and f"agent-{own} Environment" in last
          and f"agent-{other} Environment" in last, last)
rc, last = import_in("domino-agui", "loop", "handler")
check("...in its own framework's image (AGENT_FRAMEWORK=domino-agui), the import goes on to the SDKs (here: the ImportError that names "
      "them)", rc == 1 and "openai" in last and "anthropic" in last and "agent-domino-agui Environment" in last, last)

# --- the converters: AG-UI messages and tools to the request of each model format, and the model's answer back
#     to library outputs. The expected values are written out as each API documents its format.
from domino_agui import convert  # noqa: E402

thread = [
    {"id": "s1", "role": "system", "content": "You triage AML alerts."},
    {"id": "d1", "role": "developer", "content": "Answer in English."},
    {"id": "u1", "role": "user", "content": "Triage alert 7"},
    {"id": "a1", "role": "assistant", "content": "Checking the alert", "toolCalls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_dataset", "arguments": "{\"datasetId\": \"ds1\"}"}},
        {"id": "call_2", "type": "function", "function": {"name": "write_governance_finding", "arguments": "{\"severity\": \"high\"}"}}]},
    {"id": "t1", "role": "tool", "toolCallId": "call_1", "content": "{\"name\": \"alerts\"}"},
    {"id": "t2", "role": "tool", "toolCallId": "call_2", "content": "{\"error\": \"denied\"}", "error": "write_governance_finding was denied"},
    {"id": "r1", "role": "reasoning", "content": "The finding was denied, so I report it."},
    {"id": "u2", "role": "user", "content": [{"type": "text", "text": "Also check alert 8"}]},
    {"id": "a2", "role": "assistant", "toolCalls": [
        {"id": "call_3", "type": "function", "function": {"name": "write_dataset_description", "arguments": "{\"datasetId\": "}}]},
    {"id": "t3", "role": "tool", "toolCallId": "call_3", "content": "saved"},
    {"id": "a3", "role": "assistant", "content": "Alert 7 is a false positive."},
]
fn_call = lambda cid, name, arguments: {"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}
got = convert.to_openai(thread)
check("to_openai: the history as OpenAI chat messages; system and developer become system messages, a reasoning message is left out",
      got == {"messages": [
          {"role": "system", "content": "You triage AML alerts."},
          {"role": "system", "content": "Answer in English."},
          {"role": "user", "content": "Triage alert 7"},
          {"role": "assistant", "content": "Checking the alert",
           "tool_calls": [fn_call("call_1", "get_dataset", "{\"datasetId\": \"ds1\"}"),
                          fn_call("call_2", "write_governance_finding", "{\"severity\": \"high\"}")]},
          {"role": "tool", "tool_call_id": "call_1", "content": "{\"name\": \"alerts\"}"},
          {"role": "tool", "tool_call_id": "call_2", "content": "Error: write_governance_finding was denied\n\n{\"error\": \"denied\"}"},
          {"role": "user", "content": [{"type": "text", "text": "Also check alert 8"}]},
          {"role": "assistant", "content": None, "tool_calls": [fn_call("call_3", "write_dataset_description", "{\"datasetId\": ")]},
          {"role": "tool", "tool_call_id": "call_3", "content": "saved"},
          {"role": "assistant", "content": "Alert 7 is a false positive."}]}, got)
schema = {"type": "object", "properties": {"datasetId": {"type": "string"}}, "required": ["datasetId"]}
agui_tools = [{"name": "get_dataset", "description": "Get a dataset", "parameters": schema}, {"name": "ping", "description": "Check the link"}]
got = convert.to_openai(thread[2:3], tools=agui_tools)
check("to_openai with tools: each AG-UI tool as an OpenAI function tool; a tool with no parameters takes no arguments",
      got.get("tools") == [{"type": "function", "function": {"name": "get_dataset", "description": "Get a dataset", "parameters": schema}},
                           {"type": "function", "function": {"name": "ping", "description": "Check the link",
                                                             "parameters": {"type": "object", "properties": {}}}}], got)
check("to_openai with no tools leaves `tools` out (the API refuses an empty list)",
      "tools" not in convert.to_openai(thread[2:3], tools=[]) and "tools" not in convert.to_openai(thread[2:3]))


def openai_completion(text, *calls):
    """A chat completion as the OpenAI API answers it; each call is (id, name, arguments text)."""
    msg = {"role": "assistant", "content": text, "refusal": None,
           **({"tool_calls": [fn_call(*c) for c in calls]} if calls else {})}
    return {"id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-4o", "usage": {"prompt_tokens": 9, "completion_tokens": 3},
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}]}


class SDKObject:
    """Stands for a model SDK's answer object (pydantic): the converters read it through model_dump()."""
    def __init__(self, data):
        self.data = data

    def model_dump(self, **kw):
        return self.data


answer = convert.from_openai(openai_completion("Checking", ("call_9", "get_dataset", "{\"datasetId\": \"ds1\"}"),
                                               ("call_10", "write_governance_finding", "{\"sev")))
check("from_openai: the answer's text and its tool calls, each with the model's id, name and arguments as written",
      answer.text == "Checking" and [(c.id, c.name, c.arguments) for c in answer.tool_calls]
      == [("call_9", "get_dataset", "{\"datasetId\": \"ds1\"}"), ("call_10", "write_governance_finding", "{\"sev")], answer)
answer = convert.from_openai(SDKObject(openai_completion(None, ("call_9", "get_dataset", "{}"))))
check("from_openai reads the SDK's answer object too; no content is empty text",
      answer.text == "" and [c.id for c in answer.tool_calls] == ["call_9"], answer)
check("from_openai: a text answer has no tool calls", convert.from_openai(openai_completion("Done")).tool_calls == (), "")
msg = convert.from_openai(openai_completion("Checking", ("call_9", "get_dataset", "{}"))).message()
check("answer.message(): the answer as an AG-UI assistant message (id, content, toolCalls), for the agent's own history",
      msg.get("id") and {k: v for k, v in msg.items() if k != "id"}
      == {"role": "assistant", "content": "Checking", "toolCalls": [fn_call("call_9", "get_dataset", "{}")]}, msg)
msg = convert.from_openai(openai_completion(None, ("call_9", "get_dataset", "{}"))).message()
check("answer.message() with no text leaves `content` out (optional fields are never null)", "content" not in msg, msg)

got = convert.to_anthropic(thread, tools=agui_tools)
text_block = lambda text: {"type": "text", "text": text}
check("to_anthropic: system and developer messages join into `system`; content is blocks; tool calls are tool_use blocks with the "
      "parsed arguments ({} when broken); tool results are tool_result blocks, with is_error for an error; turns of one role merge",
      {k: v for k, v in got.items() if k != "tools"} == {
          "system": "You triage AML alerts.\n\nAnswer in English.",
          "messages": [
              {"role": "user", "content": [text_block("Triage alert 7")]},
              {"role": "assistant", "content": [
                  text_block("Checking the alert"),
                  {"type": "tool_use", "id": "call_1", "name": "get_dataset", "input": {"datasetId": "ds1"}},
                  {"type": "tool_use", "id": "call_2", "name": "write_governance_finding", "input": {"severity": "high"}}]},
              {"role": "user", "content": [
                  {"type": "tool_result", "tool_use_id": "call_1", "content": "{\"name\": \"alerts\"}"},
                  {"type": "tool_result", "tool_use_id": "call_2", "content": "write_governance_finding was denied\n\n{\"error\": \"denied\"}",
                   "is_error": True},
                  text_block("Also check alert 8")]},
              {"role": "assistant", "content": [{"type": "tool_use", "id": "call_3", "name": "write_dataset_description", "input": {}}]},
              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_3", "content": "saved"}]},
              {"role": "assistant", "content": [text_block("Alert 7 is a false positive.")]}]}, got)
check("to_anthropic with tools: each AG-UI tool with its input_schema; a tool with no parameters takes no arguments",
      got.get("tools") == [{"name": "get_dataset", "description": "Get a dataset", "input_schema": schema},
                           {"name": "ping", "description": "Check the link", "input_schema": {"type": "object", "properties": {}}}], got)
got = convert.to_anthropic(thread[2:3])
check("to_anthropic with no system message and no tools leaves `system` and `tools` out (the SDK refuses null)",
      got == {"messages": [{"role": "user", "content": [text_block("Triage alert 7")]}]}, got)


def anthropic_message(text, *calls):
    """A Messages answer as the Anthropic API sends it; each call is (id, name, input)."""
    blocks = [{"type": "thinking", "thinking": "Look up the dataset first.", "signature": "c2ln"}] + ([text_block(text)] if text else [])
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5", "usage": {"input_tokens": 9, "output_tokens": 3},
            "content": blocks + [{"type": "tool_use", "id": i, "name": n, "input": a} for i, n, a in calls],
            "stop_reason": "tool_use" if calls else "end_turn"}


answer = convert.from_anthropic(anthropic_message("Checking", ("toolu_01", "get_dataset", {"datasetId": "ds1"}),
                                                  ("toolu_02", "ping", {})))
check("from_anthropic: the text blocks as the text, each tool_use as a tool call with the model's id and its input as JSON text; "
      "a thinking block is not text",
      answer.text == "Checking" and [(c.id, c.name, json.loads(c.arguments)) for c in answer.tool_calls]
      == [("toolu_01", "get_dataset", {"datasetId": "ds1"}), ("toolu_02", "ping", {})], answer)
answer = convert.from_anthropic(SDKObject(anthropic_message("Done")))
check("from_anthropic reads the SDK's answer object too; a text answer has no tool calls", answer.text == "Done" and answer.tool_calls == (), answer)

# --- the output limit: contract.MAX_TOKENS is the platform's model's default output limit, 16,000 tokens, or AGENT_MAX_TOKENS. The
#     contract reads the variable at its import, so each case imports it in a new Python, with the standard library only
from domino_agui import contract  # noqa: E402

LIMIT = """import sys
sys.path.insert(0, sys.argv[1])
try:
    from domino_agui import contract
    print(contract.MAX_TOKENS_ENV, contract.MAX_TOKENS)
except Exception as e:
    print(type(e).__name__ + ": " + str(e))
"""


def limit_with(value):
    """What the contract's import gives with AGENT_MAX_TOKENS set to `value` (None: not set): its name and MAX_TOKENS, or the error."""
    env = {k: v for k, v in os.environ.items() if k != "AGENT_MAX_TOKENS"}
    p = subprocess.run([sys.executable, "-I", "-S", "-c", LIMIT, LIB], capture_output=True, text=True,
                       env={**env, **({"AGENT_MAX_TOKENS": value} if value is not None else {})})
    return (p.stdout + p.stderr).strip()


check("contract.MAX_TOKENS: 16000 when AGENT_MAX_TOKENS isn't set or is empty (contract.MAX_TOKENS_ENV names the variable)",
      limit_with(None) == "AGENT_MAX_TOKENS 16000" and limit_with("") == "AGENT_MAX_TOKENS 16000", (limit_with(None), limit_with("")))
check("...AGENT_MAX_TOKENS sets it, in tokens", limit_with("32000") == "AGENT_MAX_TOKENS 32000" and limit_with(" 2048 ") == "AGENT_MAX_TOKENS 2048",
      (limit_with("32000"), limit_with(" 2048 ")))
bad = [limit_with(v) for v in ("lots", "0", "-5", "1.5")]
check("...a value that isn't a whole number above 0 fails at the import, with a ValueError that names the variable and the value",
      all(b.startswith("ValueError: AGENT_MAX_TOKENS ") for b in bad) and "'lots'" in bad[0] and "'0'" in bad[1], bad)

# --- an answer that stopped at the output limit: Anthropic's stop_reason max_tokens, OpenAI's finish_reason length. The other stop
#     reasons are a finished answer
cut = lambda answer, **stop: {**answer, **stop} if "stop_reason" in stop else {**answer, "choices": [{**answer["choices"][0], **stop}]}
got = [convert.from_anthropic(cut(anthropic_message("The report is cu"), stop_reason="max_tokens")).at_limit,
       convert.from_anthropic(SDKObject(cut(anthropic_message(""), stop_reason="max_tokens"))).at_limit,
       convert.from_anthropic(anthropic_message("Done")).at_limit,
       convert.from_anthropic(anthropic_message("", ("toolu_01", "ping", {}))).at_limit]
check("from_anthropic: answer.at_limit is True for stop_reason max_tokens (the SDK's object too), False for end_turn and tool_use",
      got == [True, True, False, False], got)
got = [convert.from_openai(cut(openai_completion("The report is cu"), finish_reason="length")).at_limit,
       convert.from_openai(SDKObject(cut(openai_completion(None), finish_reason="length"))).at_limit,
       convert.from_openai(openai_completion("Done")).at_limit,
       convert.from_openai(openai_completion(None, ("call_9", "ping", "{}"))).at_limit]
check("from_openai: answer.at_limit is True for finish_reason length (the SDK's object too), False for stop and tool_calls",
      got == [True, True, False, False], got)
check("convert.Answer(text) is a finished answer unless it says so: at_limit is False", convert.Answer("Done").at_limit is False)
check("contract.EMPTY_ANSWER is the message that the ready-made LangGraph agent and the loop give the model, one time in a run, after "
      "an answer with no text and no tool call", contract.EMPTY_ANSWER == "Your last answer was empty. Continue with the task, or give "
      "your result.", contract.EMPTY_ANSWER)
check("contract.AT_LIMIT is the run's result when the model's last answer stopped at the output limit: it says that the answer is not "
      "finished, and how to give the model more room", contract.AT_LIMIT == "The model's last answer stopped at its output limit, so it "
      "is not a finished answer. To give the model a larger limit, set AGENT_MAX_TOKENS, or max_tokens on the agent's own model.",
      contract.AT_LIMIT)

# --- the agent contract's shared parts: each ready-made agent, and an author's own graph, flow or loop, uses them the same way on
#     every framework. The expected values are written out: a model reads these texts, so a change to one is a change to the contract
from domino_agui import contract  # noqa: E402

check("contract.ASK_HUMAN_TOOL is ask_human as an AG-UI tool: `question` (a string, required) and `expires_at` (ISO 8601, optional)",
      contract.ASK_HUMAN_TOOL == {"name": "ask_human", "description": "Ask the person who started the task a question, and wait for the answer.",
                                  "parameters": {"type": "object", "required": ["question"], "properties": {
                                      "question": {"type": "string", "description": "The question"},
                                      "expires_at": {"type": "string", "description": "Optional: when the answer stops being useful (ISO 8601)"}}}},
      contract.ASK_HUMAN_TOOL)
got = (convert.to_openai([], tools=[contract.ASK_HUMAN_TOOL])["tools"], convert.to_anthropic([], tools=[contract.ASK_HUMAN_TOOL])["tools"])
check("the converters take contract.ASK_HUMAN_TOOL as it is, in each format",
      got[0][0]["function"]["name"] == "ask_human" and got[0][0]["function"]["parameters"]["required"] == ["question"]
      and got[1][0]["name"] == "ask_human" and got[1][0]["input_schema"]["required"] == ["question"], got)
check("contract.NO_ANSWER is the expired question's text: it says only what happened (the system prompt says what to do then)",
      contract.NO_ANSWER == "No answer came before the question expired, so it was cancelled.", contract.NO_ANSWER)
context = [{"description": "Domino Agents: the operating rules and the task brief", "value": "You triage alerts.\n\nRules: 1. Ask first. "},
           {"description": "empty", "value": ""}, {"description": "a brief", "value": "Alert 7."}]
check("contract.system_prompt(context): the AG-UI context's values, in order, as one text (an empty value adds nothing)",
      contract.system_prompt(context) == "You triage alerts.\n\nRules: 1. Ask first.\n\nAlert 7.", repr(contract.system_prompt(context)))
check("contract.system_prompt: no context (a laptop run) gives no text", contract.system_prompt([]) == "" and contract.system_prompt(None) == "",
      (contract.system_prompt([]), contract.system_prompt(None)))
got = [contract.ask_human_arguments(a) for a in ('{"question": "Close A-1?", "expires_at": "2026-11-11T09:00:00Z"}', {"question": "Why?"},
                                                   "Close it?", None)]
check("contract.ask_human_arguments: the model's ask_human arguments (a dict, or its JSON text) as (question, expires_at); text that "
      "isn't a JSON object is the question", got == [("Close A-1?", "2026-11-11T09:00:00Z"), ("Why?", None), ("Close it?", None), ("", None)], got)
check("contract.ONE_QUESTION and contract.no_tool(name): the answers to a question that isn't asked, and to a tool no list has",
      contract.ONE_QUESTION == "Ask one question at a time: ask this again once you have the answer you are waiting for."
      and contract.no_tool("nope") == "There is no tool named nope.", (contract.ONE_QUESTION, contract.no_tool("nope")))
saved = {k: os.environ.pop(k, None) for k in ("AGENT_MODEL", "AGENT_MODEL_FORMAT")}
try:
    contract.model_alias()
    no_alias = None
except RuntimeError as e:
    no_alias = str(e)
os.environ.update({"AGENT_MODEL": "haiku", "AGENT_MODEL_FORMAT": "anthropic"})
alias = (contract.model_alias(), contract.alias_format())
os.environ["AGENT_MODEL_FORMAT"] = "openai-compatible"
alias += (contract.alias_format(),)
for k, v in saved.items():
    os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
check("contract.model_alias() and alias_format(): AGENT_MODEL, else a RuntimeError that says the platform sets it; \"anthropic\" for "
      "AGENT_MODEL_FORMAT anthropic, else \"openai\"", no_alias and "AGENT_MODEL isn't set" in no_alias and alias == ("haiku", "anthropic", "openai"),
      (no_alias, alias))
tool = convert.openai_tool({"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}})
check("convert.openai_tool(tool): an AG-UI tool in the function form (to_openai's, and LangChain's and LiteLLM's); no parameters takes none",
      tool == {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}}
      and convert.openai_tool({"name": "g"})["function"]["parameters"] == {"type": "object", "properties": {}}
      and convert.to_openai([], tools=[{"name": "g"}])["tools"] == [convert.openai_tool({"name": "g"})], tool)

# --- the schema builder for own tools: a plain function, sync or async, becomes an AG-UI tool: its name, its docstring's first
#     paragraph, and a JSON schema from its type hints. Each ready-made agent offers it so, on every framework


def lookup(word: str) -> str:
    """The code for a word.

    The code is made here, in the agent, never through the relay."""
    return f"code-{word}"


check("contract.function_tool(f): the function's name, its docstring's first paragraph, and a str parameter with no default is a "
      "required string", contract.function_tool(lookup) == {
          "name": "lookup", "description": "The code for a word.",
          "parameters": {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]}},
      contract.function_tool(lookup) if hasattr(contract, "function_tool") else "no contract.function_tool")


def typed(count: int, ratio: float, ok: bool, items: list, table: dict, names: list[str], scores: dict[str, int]):
    """Each type."""


got = contract.function_tool(typed)["parameters"]
check("contract.function_tool: int, float, bool, list, dict, list[str] and dict[str, int] are their JSON schema types",
      got["properties"] == {"count": {"type": "integer"}, "ratio": {"type": "number"}, "ok": {"type": "boolean"}, "items": {"type": "array"},
                            "table": {"type": "object"}, "names": {"type": "array", "items": {"type": "string"}},
                            "scores": {"type": "object", "additionalProperties": {"type": "integer"}}}
      and got["required"] == ["count", "ratio", "ok", "items", "table", "names", "scores"], got)


def nullable(region: typing.Optional[str], limit: int | None, level: typing.Literal["low", "high"], size: typing.Literal[1, 2, 3]):
    """Optional and literal types."""


got = contract.function_tool(nullable)["parameters"]
check("contract.function_tool: Optional[X] and X | None are X or null; Literal[...] is an enum of its values, typed when they are of one "
      "type", got["properties"] == {"region": {"type": ["string", "null"]}, "limit": {"type": ["integer", "null"]},
                                    "level": {"type": "string", "enum": ["low", "high"]}, "size": {"type": "integer", "enum": [1, 2, 3]}}
      and got["required"] == ["region", "limit", "level", "size"], got)


def defaulted(alert: str, days: int = 7, region: typing.Optional[str] = None, *, strict: bool = False):
    """Defaults."""


got = contract.function_tool(defaulted)["parameters"]
check("contract.function_tool: a parameter with a default is optional (not in `required`), and its schema has the default",
      got["properties"] == {"alert": {"type": "string"}, "days": {"type": "integer", "default": 7},
                            "region": {"type": ["string", "null"], "default": None}, "strict": {"type": "boolean", "default": False}}
      and got["required"] == ["alert"], got)


async def fetch(word: str, page: int = 1) -> str:
    """Fetch a word's page.

    Async."""
    return word


check("contract.function_tool: an async def is a tool the same way as a def", contract.function_tool(fetch) == {
    "name": "fetch", "description": "Fetch a word's page.", "parameters": {
        "type": "object", "properties": {"word": {"type": "string"}, "page": {"type": "integer", "default": 1}}, "required": ["word"]}},
      contract.function_tool(fetch))


def untyped(note, extra: typing.Any = None):
    """No hints."""


check("contract.function_tool: a parameter with no type hint, or typing.Any, takes any JSON value",
      contract.function_tool(untyped)["parameters"] == {"type": "object", "properties": {"note": {}, "extra": {"default": None}},
                                                       "required": ["note"]}, contract.function_tool(untyped))


def refused(fn):
    try:
        contract.function_tool(fn)
    except TypeError as e:
        return str(e)
    return "no TypeError"


def dated(day: time.struct_time):
    """A type with no JSON schema."""


check("contract.function_tool: a type it can't describe is a TypeError that names the function, the parameter and the types it takes",
      all(w in refused(dated) for w in ("dated", "day", "struct_time", "str, int, float, bool, list, dict")), refused(dated))
check("contract.function_tool: a lambda (no name) or a callable that isn't a function is a TypeError that says to use def",
      "def" in refused(lambda word: word) and "def" in refused(len.__call__) and "def" in refused(str), (refused(lambda word: word), refused(str)))

# --- running an own tool: in the agent, never through the relay. The model's arguments (a dict, or the JSON text a model sends) are
#     the function's keyword arguments; the result is the tool's answer as text, a ToolResult like a relay call's


def ran(fn, arguments):
    r = asyncio.run(contract.call_function(fn, arguments))
    return (r.text, r.is_error)


check("contract.call_function: a def and an async def each run with the model's arguments (a dict or JSON text), and a str result is "
      "the answer as it is", ran(lookup, {"word": "heron"}) == ("code-heron", False) and ran(fetch, '{"word": "kite"}') == ("kite", False),
      (ran(lookup, {"word": "heron"}), ran(fetch, '{"word": "kite"}')))


def scored(days: int = 7) -> dict:
    """A result that isn't text."""
    return {"days": days, "flagged": ["a-1"], "where": Path("/mnt/data/alerts")}


check("contract.call_function: a result that isn't a str is the answer as JSON text (a value with no JSON form as its str)",
      json.loads(ran(scored, {})[0])["days"] == 7 and json.loads(ran(scored, {"days": 3})[0])["flagged"] == ["a-1"]
      and json.loads(ran(scored, {})[0])["where"] == "/mnt/data/alerts" and ran(lambda: None, {}) == ("null", False), ran(scored, {}))


async def broken(alert: str) -> str:
    """Raises."""
    raise ValueError(f"no alert {alert}")


got = {"raised": ran(broken, {"alert": "A-7"}), "unknown": ran(lookup, {"word": "x", "page": 2}), "missing": ran(lookup, {}),
       "not an object": ran(lookup, "[1, 2]"), "broken JSON": ran(lookup, '{"word": ')}
check("contract.call_function: an exception, or arguments the function can't take (not a JSON object, an unknown or a missing "
      "parameter), is an error answer that says what went wrong, never an exception: the model reads it and the run goes on",
      got["raised"] == ("the own tool broken raised ValueError: no alert A-7", True)
      and all(err and "lookup" in text for k, (text, err) in got.items() if k != "raised") and "page" in got["unknown"][0]
      and "word" in got["missing"][0]
      and "JSON object" in got["not an object"][0] and "JSON object" in got["broken JSON"][0], got)

# --- a name clash: an own tool never hides a governed tool. Each ready-made agent calls contract.check_names(own) when its entry point
#     is built, and check_names(own, gateway=..., client=...) in each AG-UI run


def clash(*args, **kw):
    try:
        contract.check_names(*args, **kw)
    except contract.NameClash as e:
        return str(e)
    return None


check("contract.check_names: an own tool named ask_human is a NameClash that names it (the entry point isn't built)",
      issubclass(contract.NameClash, ValueError) and "ask_human" in (clash(["lookup", "ask_human"]) or "")
      and clash(["lookup", "fetch"]) is None, clash(["lookup", "ask_human"]))
got = {"gateway": clash(["lookup", "get_project"], gateway=["get_project", "list_jobs"], client=["write_finding"]),
       "client": clash(["lookup", "write_finding"], gateway=["get_project"], client={"write_finding"}),
       "twice": clash(["lookup", "lookup"]),
       "none": clash(["lookup"], gateway=["get_project"], client=["write_finding"])}
check("contract.check_names, in an AG-UI run: an own tool with the name of a gateway tool or of a client-side tool is a NameClash that "
      "names the tool and says which; two own tools with one name clash too; other names pass",
      "get_project" in (got["gateway"] or "") and "gateway tool" in (got["gateway"] or "") and "list_jobs" not in (got["gateway"] or "")
      and "write_finding" in (got["client"] or "") and "client-side tool" in (got["client"] or "")
      and "lookup" in (got["twice"] or "") and got["none"] is None, got)
specs, own = contract.own_tools([lookup])
check("contract.own_tools(tools): each function's AG-UI tool (function_tool) and {name: function}, after the build's name check",
      specs == [contract.function_tool(lookup)] and own == {"lookup": lookup} and "ask_human" in (clash(["ask_human"]) or ""), (specs, own))
saved = os.environ.pop("AGENT_SERVICE_URL", None)
got = (contract.relay(), asyncio.run(contract.relay_tools()), asyncio.run(contract.call_tool("lookup", {"word": "heron"}, own)),
       asyncio.run(contract.call_tool("get_project", "{}", own)))
if saved is not None:
    os.environ["AGENT_SERVICE_URL"] = saved
check("contract.call_tool with no relay (no AGENT_SERVICE_URL, a laptop): an own tool runs here; any other name is an error result, "
      "contract.no_tool(name), with no relay call; relay() is None and relay_tools() is []",
      got[0] is None and got[1] == [] and got[2] == domino_agui.ToolResult("code-heron")
      and got[3] == domino_agui.ToolResult("There is no tool named get_project.", True), got)

# --- the relay client: it lists and calls the agent's gateway tools over MCP at $AGENT_SERVICE_URL/mcp, with $SESSION_TOKEN as a bearer
#     token, one JSON-RPC call for each request. A small fake MCP server stands for the relay here: it answers tools/list and
#     tools/call (a JSON-RPC error for a tool it doesn't have), refuses any other token with HTTP 401, and can wait before it answers
TOKEN = "relay-tok"


def fake_mcp(wait=0.0):
    """Starts a fake MCP server in a daemon thread. -> (its URL, the requests it received: (path, Authorization header, JSON-RPC body))."""
    received = []

    class MCP(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"null")
            received.append((self.path, self.headers.get("Authorization"), body))
            if self.path != "/mcp":
                return self.answer(404, {"error": f"no route {self.path}"})
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                return self.answer(401, {"error": "unknown session token"})
            time.sleep(wait)
            name = body["params"].get("name")
            if body["method"] == "tools/list":
                result = {"tools": [{"name": "get_dataset", "description": "Get a dataset", "inputSchema": schema}, {"name": "ping"}]}
            elif name == "get_dataset":
                result = {"content": [{"type": "text", "text": json.dumps({"id": body["params"]["arguments"]["datasetId"]})},
                                      {"type": "image", "data": "aGk=", "mimeType": "image/png"}, {"type": "text", "text": "1 row"}]}
            elif name == "write_dataset_description":
                result = {"content": [{"type": "text", "text": f"{name} was denied: the dataset is read-only"}], "isError": True}
            else:
                return self.answer(200, {"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32602, "message": f"Unknown tool: {name}"}})
            self.answer(200, {"jsonrpc": "2.0", "id": body["id"], "result": result})

        def answer(self, status, body):
            data = json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass   # the client stopped waiting (a timeout case)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MCP)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", received


def relay_error(make):
    """'RelayError: <text>' when `make()` raises one (making the client, or awaiting what it returns), else what happened instead."""
    try:
        asyncio.run(make())
    except domino_agui.RelayError as e:
        return f"RelayError: {e}"
    except Exception as e:
        return f"not a RelayError: {type(e).__name__}: {e}"
    return "no error"


def without_id(requests):
    """The requests as (path, Authorization header, JSON-RPC body), with each body's id left out (the client numbers its calls)."""
    return [(path, auth, {k: v for k, v in body.items() if k != "id"}) for path, auth, body in requests]


mcp_url, received = fake_mcp()
os.environ.update({"AGENT_SERVICE_URL": mcp_url, "SESSION_TOKEN": TOKEN})
relay = domino_agui.Relay()
listed = asyncio.run(relay.list_tools())
check("relay.list_tools(): the server's tools as AG-UI tools (name, description, parameters), for the converters; a tool with no "
      "description or inputSchema gets \"\" and {}", listed == [{"name": "get_dataset", "description": "Get a dataset", "parameters": schema},
                                                                {"name": "ping", "description": "", "parameters": {}}], listed)
check("...with one POST to $AGENT_SERVICE_URL/mcp that has $SESSION_TOKEN as a bearer token: a JSON-RPC 2.0 tools/list call with a "
      "number id, and no initialize call first",
      without_id(received) == [("/mcp", f"Bearer {TOKEN}", {"jsonrpc": "2.0", "method": "tools/list", "params": {}})]
      and all(isinstance(body.get("id"), int) for _, _, body in received), received)
res = asyncio.run(relay.call("get_dataset", {"datasetId": "ds1"}))
check("relay.call(): a tools/call with the tool's name and arguments; the answer's text parts, one for each line (an image part is left "
      "out), come back as a ToolResult with is_error False",
      res == domino_agui.ToolResult('{"id": "ds1"}\n1 row')
      and without_id(received[-1:]) == [("/mcp", f"Bearer {TOKEN}", {"jsonrpc": "2.0", "method": "tools/call",
                                                                    "params": {"name": "get_dataset", "arguments": {"datasetId": "ds1"}}})],
      (res, received[-1:]))
res = asyncio.run(relay.call("get_dataset", '{"datasetId": "ds2"}'))
check("relay.call() takes the model's arguments as JSON text too (ToolCall.arguments), and sends them as a JSON object",
      res == domino_agui.ToolResult('{"id": "ds2"}\n1 row') and received[-1][2].get("params", {}).get("arguments") == {"datasetId": "ds2"},
      (res, received[-1:]))
res = asyncio.run(relay.call("write_dataset_description", {"datasetId": "ds1", "description": "Alerts"}))
check("a result with isError (a denied call) comes back as a ToolResult with is_error and the server's text, not as an exception",
      res == domino_agui.ToolResult("write_dataset_description was denied: the dataset is read-only", True), res)
before = len(received)
bad = [asyncio.run(relay.call("get_dataset", a)) for a in ('{"datasetId": ', '["ds1"]')]
check("arguments that are not a JSON object: an error result that tells the model to send a JSON object, and no request goes out",
      all(r.is_error and "JSON object" in r.text and "get_dataset" in r.text for r in bad) and len(received) == before,
      (bad, received[before:]))
why = relay_error(lambda: relay.call("nope", {}))
check("a JSON-RPC error in place of a result raises RelayError that shows the error",
      why.startswith("RelayError: ") and "did not answer tools/call with an MCP result" in why and "Unknown tool: nope" in why, why)

why = relay_error(lambda: domino_agui.Relay(token="not-the-token").list_tools())
check("a token the server refuses (HTTP 401) raises RelayError that says the relay refused the session token",
      why.startswith("RelayError: ") and "HTTP 401" in why and "refused the session token" in why
      and received[-1][1] == "Bearer not-the-token", (why, received[-1:]))
why = relay_error(lambda: domino_agui.Relay(url=mcp_url + "/not-the-relay").list_tools())
check("a URL whose /mcp is not an MCP route raises RelayError with the HTTP status (404)",
      why.startswith("RelayError: ") and "HTTP 404" in why and received[-1][0] == "/not-the-relay/mcp", (why, received[-1:]))
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    closed_port = s.getsockname()[1]
env = {k: v for k, v in os.environ.items() if k.lower() not in ("no_proxy", "http_proxy", "https_proxy", "all_proxy")}
env["HTTP_PROXY"] = env["http_proxy"] = f"http://127.0.0.1:{closed_port}"   # from the start: urllib reads it once per process
code = ("import sys, asyncio; sys.path.insert(0, sys.argv[1]); import domino_agui; "
        "print(asyncio.run(domino_agui.Relay().list_tools())[0]['name'])")
p = subprocess.run([sys.executable, "-I", "-c", code, LIB], env=env, capture_output=True, text=True, timeout=30)
check("the relay client connects to the relay directly: a proxy in HTTP_PROXY (an egress proxy) never sees a relay call",
      p.returncode == 0 and p.stdout.strip() == "get_dataset", p.stderr[-500:])
why = relay_error(lambda: domino_agui.Relay(url=f"http://127.0.0.1:{closed_port}").list_tools())
check("a server that can't be reached (nothing listens) raises RelayError that names the endpoint",
      why.startswith("RelayError: ") and f"http://127.0.0.1:{closed_port}/mcp" in why and "can't be reached" in why, why)
for k in ("AGENT_SERVICE_URL", "SESSION_TOKEN"):
    os.environ.pop(k)
why = relay_error(lambda: domino_agui.Relay().list_tools())
check("with no $AGENT_SERVICE_URL and no $SESSION_TOKEN (outside a Job), making the client raises RelayError that names both and says "
      "to give url= and token=", why.startswith("RelayError: ") and "AGENT_SERVICE_URL and SESSION_TOKEN" in why
      and "url=" in why and "token=" in why, why)
listed = asyncio.run(domino_agui.Relay(url=mcp_url + "/", token=TOKEN).list_tools())
check("Relay(url=..., token=...) needs no environment: the token is the bearer token, and a / at the end of the URL is fine",
      [t["name"] for t in listed] == ["get_dataset", "ping"] and received[-1][:2] == ("/mcp", f"Bearer {TOKEN}"), (listed, received[-1:]))

slow_url, _ = fake_mcp(wait=1.5)
why = relay_error(lambda: domino_agui.Relay(url=slow_url, token=TOKEN, timeout=0.5).list_tools())
check("Relay(timeout=...) alone is the time to connect and to wait for each answer: an answer slower than it raises RelayError that "
      "says the relay did not answer in time", why.startswith("RelayError: ") and "did not answer tools/list within 0.5 s" in why, why)
outcome = relay_error(lambda: domino_agui.Relay(url=slow_url, token=TOKEN, timeout=0.5, read_timeout=5).list_tools())
check("Relay(timeout=..., read_timeout=...): the connect timeout doesn't limit the wait for an answer, so an answer slower than the "
      "connect timeout and quicker than the read timeout comes back", outcome == "no error", outcome)
why = relay_error(lambda: domino_agui.Relay(url=slow_url, token=TOKEN, timeout=5, read_timeout=0.5).list_tools())
check("...and an answer slower than the read timeout raises RelayError that says the relay did not answer within it",
      why.startswith("RelayError: ") and "did not answer tools/list within 0.5 s" in why, why)

print("\nall checks passed" if not failed else f"\n{len(failed)} failed: {failed}")
sys.exit(1 if failed else 0)
