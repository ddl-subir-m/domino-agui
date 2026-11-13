"""One AG-UI run: what the handler reads and what it can send, with the lifecycle rules kept for it.

The library sends RUN_STARTED (protocolVersion "1.0") before the handler starts and exactly one final event after it: RUN_FINISHED
(finish(), interrupt(), or the handler returning) or RUN_ERROR (the handler raised). Each output sends its *_START and *_END together,
so no pair is left open. Before RUN_FINISHED it sends MESSAGES_SNAPSHOT: the input history plus what this run said (and what it
remembered: run.remember()), which the agent service keeps as the thread, so the next run's history has the agent's own messages and
tool calls.
"""
import copy
import json
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from . import PROTOCOL_VERSION


def _id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def _text(v):
    return v if isinstance(v, str) else json.dumps(v)


class LifecycleError(RuntimeError):
    """The handler asked for an output that would break the AG-UI contract. Uncaught, it ends the run with RUN_ERROR."""


@dataclass(frozen=True)
class Resume:
    """The answer to the interrupt that ended the previous AG-UI run. `status` is "resolved" or "cancelled" (an answer that came
    after `expiresAt`); `payload` is the reply text when resolved."""
    interrupt_id: str
    status: str
    payload: Any = None


class Run:
    """The handler's view of one AG-UI run.

    Inputs, as the client sent them (AG-UI's camelCase dicts): `messages` (the full history: nothing is kept between runs), `state`
    (the last state, `{}` when absent), `resume` (a `Resume`, or None), `client_tools` (the tools the agent service runs, with their
    definitions), `context` (the operating rules and the task brief: the agent's code decides whether the model sees them).
    """

    def __init__(self, body, emit):
        self.thread_id = body["threadId"]
        self.run_id = body["runId"]
        self.messages = body["messages"]
        self.state = body.get("state") or {}
        self.client_tools = body.get("tools") or []
        self.context = body.get("context") or []
        r = (body.get("resume") or [None])[0]
        self.resume = r and Resume(r["interruptId"], r["status"], r.get("payload"))
        self._emit = emit
        self._history = copy.deepcopy(self.messages)   # the snapshot's base, whatever the handler does to run.messages
        self._said = []      # this run's messages, for the snapshot
        self._pending = []   # open client-side calls, in order
        self._done = False

    def text(self, content):
        """Sends one assistant text message. Returns its message id."""
        self._open()
        mid = _id("msg")
        self._emit({"type": "TEXT_MESSAGE_START", "messageId": mid, "role": "assistant"})
        self._emit({"type": "TEXT_MESSAGE_CONTENT", "messageId": mid, "delta": content})
        self._emit({"type": "TEXT_MESSAGE_END", "messageId": mid})
        self._said.append({"id": mid, "role": "assistant", "content": content})
        return mid

    def tool_call(self, name, args, result, *, tool_call_id=None):
        """Sends a tool call that the agent ran in this run, with its result: an agent-side tool's (through the relay) or an own
        tool's. `args` and a result that is not a string are sent as JSON text; `args` given as text is sent as written. Returns the tool
        call id."""
        cid = self._call(name, args, tool_call_id)
        rid, content = _id("msg"), _text(result)
        self._emit({"type": "TOOL_CALL_RESULT", "messageId": rid, "toolCallId": cid, "content": content, "role": "tool"})
        self._said.append({"id": rid, "role": "tool", "toolCallId": cid, "content": content})
        return cid

    def client_tool_call(self, name, args, *, tool_call_id=None):
        """Sends a call to a client-side tool (one of `client_tools`). The agent service runs it through the gateway, so the call stays
        open: the run ends as a success that names it in `pendingToolCallIds`, and the answer is a tool message in the next run's
        `messages`. Returns the tool call id."""
        declared = [t.get("name") for t in self.client_tools]
        if name not in declared:
            raise LifecycleError(f"{name!r} is not a client-side tool of this run (declared: {declared or 'none'}). Run an agent-side "
                                 "tool yourself and send it with run.tool_call().")
        cid = self._call(name, args, tool_call_id)
        self._pending.append(cid)
        return cid

    def remember(self, *messages):
        """Keeps AG-UI messages in the thread without sending them: they go in this run's MESSAGES_SNAPSHOT, after the input history and
        what the run said before, so later runs' `messages` have them. For the messages a handler puts in its model's history itself,
        such as a question's call and its answer (loop.answered_question). The run sends no event for them."""
        self._open()
        self._said.extend(copy.deepcopy(list(messages)))

    def state_snapshot(self, state):
        """Sends the whole state (STATE_SNAPSHOT replaces it, nothing merges). The agent service keeps the last one for the thread and
        sends it back as the next run's `state`. Don't put secrets in it: it leaves the agent server on every run."""
        self._open()
        self._emit({"type": "STATE_SNAPSHOT", "snapshot": state})

    def interrupt(self, message, *, reason="question", interrupt_id=None, expires_at=None):
        """Asks a person a question and ends the run with an interrupt outcome (one interrupt). The answer comes back as `run.resume`
        in the next run: "resolved" with the reply text, or "cancelled" when it came after `expires_at` (ISO 8601). An interrupt is a
        question, never an approval: only the gateway holds a tool call for approval. Returns the interrupt id."""
        self._open()
        if self._pending:
            raise LifecycleError("a run with open client-side calls ends as a success, not with an interrupt: ask the person after the "
                                 "calls are answered, in the next run")
        it = {"id": interrupt_id or _id("int"), "reason": reason, "message": message, **({"expiresAt": expires_at} if expires_at else {})}
        self._finished({"outcome": {"type": "interrupt", "interrupts": [it]}})
        return it["id"]

    def finish(self, result=None):
        """Ends the run as a success. `result` (any JSON value) becomes the task's result; without one, the task's result is the
        agent's last text. A handler that returns without finish() or interrupt() finishes with no result."""
        self._open()
        outcome = {"type": "success", **({"pendingToolCallIds": self._pending} if self._pending else {})}
        self._finished({**({"result": result} if result is not None else {}), "outcome": outcome})

    def _open(self):
        if self._done:
            raise LifecycleError("the run has ended (its final event was sent): return from the handler")

    def _call(self, name, args, tool_call_id):
        """TOOL_CALL_START, ARGS, END. The call belongs to the assistant message this run just said, else to a new one."""
        self._open()
        cid, arguments = tool_call_id or _id("call"), _text(args)
        if not (self._said and self._said[-1]["role"] == "assistant"):
            self._said.append({"id": _id("msg"), "role": "assistant"})
        parent = self._said[-1]
        self._emit({"type": "TOOL_CALL_START", "toolCallId": cid, "toolCallName": name, "parentMessageId": parent["id"]})
        self._emit({"type": "TOOL_CALL_ARGS", "toolCallId": cid, "delta": arguments})
        self._emit({"type": "TOOL_CALL_END", "toolCallId": cid})
        parent.setdefault("toolCalls", []).append({"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}})
        return cid

    def _finished(self, fields):
        self._emit({"type": "MESSAGES_SNAPSHOT", "messages": self._history + self._said})
        self._end({"type": "RUN_FINISHED", "threadId": self.thread_id, "runId": self.run_id, **fields})

    def _end(self, event):
        self._done = True
        self._emit(event)


def bad_input(body):
    """Why `body` isn't a RunAgentInput this library can run, or '' when it is. Bad input gets an HTTP error and no stream."""
    if not isinstance(body, dict):
        return "the body is not a JSON object"
    for k in ("threadId", "runId"):
        if not isinstance(body.get(k), str) or not body[k]:
            return f"{k} must be a non-empty string"
    if not isinstance(body.get("messages"), list):
        return "messages must be a list (the full history)"
    for k, kind in (("state", dict), ("tools", list), ("context", list), ("resume", list)):
        if body.get(k) is not None and not isinstance(body[k], kind):
            return f"{k} must be a {'JSON object' if kind is dict else 'list'}"
    resume = body.get("resume") or []
    if len(resume) > 1:
        return "resume has more than one entry: a domino-agui run ends with one interrupt at most"
    if resume and not (isinstance(resume[0], dict) and isinstance(resume[0].get("interruptId"), str)
                       and resume[0].get("status") in ("resolved", "cancelled")):
        return 'the resume entry needs interruptId and a status of "resolved" or "cancelled"'
    return ""


async def play(handler, body, emit):
    """Runs `handler` for one RunAgentInput, sending every event through `emit`, with RUN_STARTED first and one final event last."""
    run = Run(body, emit)
    emit({"type": "RUN_STARTED", "threadId": run.thread_id, "runId": run.run_id, "protocolVersion": PROTOCOL_VERSION})
    try:
        await handler(run)
    except Exception as e:
        traceback.print_exc()   # stderr: the agent service shows its last lines when a run fails
        if not run._done:
            run._end({"type": "RUN_ERROR", "message": f"the handler raised {type(e).__name__}: {e}"})
        return
    if not run._done:
        run.finish()
