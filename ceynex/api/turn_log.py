"""The frame log a conversational turn writes to — D12, amended for resume.

SRS 3.4.1 sets the response-time budget a stream exists to make bearable, and
SRS 3.3.1 the availability a dropped connection should not cost. Until this, a
turn *was* its HTTP response: the frames went straight from the orchestration
into the socket, so a connection that dropped mid-turn — a flaky network, a
suspended laptop, a proxy blip — took the answer with it. The server cancelled
the fan-out it had been paying for, and the reader was left to guess whether to
ask again.

Now a turn writes every frame it produces into a log, with a monotonic `seq`,
and an HTTP response is only a *reader* of that log. A reader that drops can
come back with the last `seq` it saw and pick up exactly where it left off. The
turn itself neither knows nor cares how many readers it has.

**Two stores, one authority.**

- `LocalTurnRegistry` holds the log in the process that runs the turn. It is the
  authority: the connection that started a turn is always served from here, so
  the live path never depends on anything but this process's own memory.
- `RedisTurnMirror` copies each frame to a Redis stream, best-effort. It exists
  for one case only — a reconnect that lands on the *other* uvicorn worker
  (`ceynex-infra/backend/Dockerfile` runs two, and the kernel decides which one
  accepts a connection). Without `REDIS_URL` there is no mirror, and a resume
  works only on the same worker, which locally is always the case.

A mirror write that fails is logged and dropped, never raised: a replica that is
down must not stall the answer it is replicating. The same fail-open direction
`rate_limit.py` takes for the same reason.

**The resume token is the `request_id`**, a uuid4 no one can guess, checked
against the owner the turn was started for. Knowing someone's request id is not
enough; being them is also required.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ceynex import settings

log = logging.getLogger(__name__)

#: How long a finished turn stays replayable. Long enough to survive a laptop
#: lid closing through a coffee, short enough that the log is not a second,
#: unbounded transcript. The durable record is the persisted message.
TURN_TTL_S = 600

#: A hard ceiling on turns held in one process, finished or not, so a burst of
#: abandoned turns cannot grow memory without bound. Only finished turns are
#: ever evicted to make room.
MAX_LOCAL_TURNS = 512

#: Frames kept per turn in the Redis stream. A real turn is ~25 frames plus a
#: handful of answer sentences; this bounds a pathological one.
MIRROR_MAXLEN = 2000

KEY_PREFIX = "ceynex:turn"


@dataclass(frozen=True)
class TurnFrame:
    """One SSE frame, numbered in the order the turn produced it."""

    seq: int
    event: str
    data: dict[str, Any]

    def as_sse(self) -> str:
        """The wire form. `id:` is what makes a frame resumable: a reader that
        drops reports the last one it saw. `data` is one line — `json.dumps`
        never emits a raw newline."""
        return f"id: {self.seq}\nevent: {self.event}\ndata: {json.dumps(self.data, default=str)}\n\n"


@dataclass
class LocalTurn:
    """A turn's log in the process running it, plus the task producing it."""

    request_id: str
    owner: str | None
    conversation_id: int | None = None
    frames: list[TurnFrame] = field(default_factory=list)
    done: bool = False
    finished_at: float | None = None
    task: asyncio.Task | None = None
    _changed: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)

    @property
    def last_seq(self) -> int:
        return self.frames[-1].seq if self.frames else 0

    async def append(self, event: str, data: dict[str, Any]) -> TurnFrame:
        async with self._changed:
            frame = TurnFrame(seq=self.last_seq + 1, event=event, data=data)
            self.frames.append(frame)
            if event == "done":
                self.done = True
                self.finished_at = time.monotonic()
            self._changed.notify_all()
        return frame

    async def finish(self) -> None:
        """Mark the log closed even if no `done` frame was ever written, so a
        reader waiting on a turn whose task died does not wait forever."""
        async with self._changed:
            if not self.done:
                self.done = True
                self.finished_at = time.monotonic()
            self._changed.notify_all()

    def after(self, seq: int) -> list[TurnFrame]:
        # `seq` is 1-based and dense, so frame N sits at index N-1.
        return self.frames[max(seq, 0):]

    async def wait_after(self, seq: int, timeout_s: float) -> list[TurnFrame]:
        """Frames past `seq`, waiting up to `timeout_s` for one to arrive.

        Returns `[]` on timeout — the caller's cue to consider a heartbeat and
        check whether its reader is still there.
        """
        async with self._changed:
            if self.last_seq <= seq and not self.done:
                # `asyncio.timeout`, not `asyncio.wait_for`, for the reason in
                # `turn_runner._pump`: on 3.11, wait_for can swallow a
                # cancellation, and it runs the wait in a separate task, so the
                # Condition's lock is released in one task and re-taken in
                # another. This keeps both in this task, as 3.12's wait_for does.
                try:
                    async with asyncio.timeout(timeout_s):
                        await self._changed.wait_for(lambda: self.last_seq > seq or self.done)
                except TimeoutError:
                    return []
            return self.after(seq)


class LocalTurnRegistry:
    """Every turn this process is running or has recently finished."""

    def __init__(self, ttl_s: float = TURN_TTL_S, max_turns: int = MAX_LOCAL_TURNS) -> None:
        self.ttl_s = ttl_s
        self.max_turns = max_turns
        self._turns: dict[str, LocalTurn] = {}

    def start(self, request_id: str, owner: str | None, conversation_id: int | None) -> LocalTurn:
        self._evict()
        turn = LocalTurn(request_id=request_id, owner=owner, conversation_id=conversation_id)
        self._turns[request_id] = turn
        return turn

    def get(self, request_id: str) -> LocalTurn | None:
        self._evict()
        return self._turns.get(request_id)

    def __len__(self) -> int:
        return len(self._turns)

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [
            rid
            for rid, turn in self._turns.items()
            if turn.done and turn.finished_at is not None and now - turn.finished_at > self.ttl_s
        ]
        for rid in expired:
            del self._turns[rid]

        # Over the ceiling: drop the oldest *finished* turns. A running turn is
        # never evicted — its task still holds work someone asked for.
        overflow = len(self._turns) - self.max_turns
        if overflow > 0:
            finished = sorted(
                (turn for turn in self._turns.values() if turn.done),
                key=lambda turn: turn.finished_at or 0.0,
            )
            for turn in finished[:overflow]:
                del self._turns[turn.request_id]


class RedisTurnMirror:
    """A best-effort replica of each turn's log, readable from any worker.

    Stream entry ids are `0-<seq>`, so a frame's Redis id and its SSE id are the
    same number and a resume from `seq` is a plain range read. Every method
    swallows its own failures: a mirror is never allowed to stall or fail the
    turn it copies.
    """

    def __init__(self, client: Any, ttl_s: int = TURN_TTL_S, maxlen: int = MIRROR_MAXLEN):
        self._redis = client
        self.ttl_s = ttl_s
        self.maxlen = maxlen

    @staticmethod
    def _key(request_id: str, suffix: str = "") -> str:
        return f"{KEY_PREFIX}:{request_id}{suffix}"

    async def open(self, request_id: str, owner: str | None) -> None:
        try:
            meta = self._key(request_id, ":meta")
            await self._redis.hset(meta, mapping={"owner": owner or "", "status": "running"})
            await self._redis.expire(meta, self.ttl_s)
        except Exception as exc:  # noqa: BLE001 - a replica must not fail the turn
            log.warning("turn mirror unavailable (open %s): %s", request_id, exc)

    async def append(self, request_id: str, frame: TurnFrame) -> None:
        try:
            key = self._key(request_id)
            await self._redis.xadd(
                key,
                {"event": frame.event, "data": json.dumps(frame.data, default=str)},
                id=f"0-{frame.seq}",
                maxlen=self.maxlen,
                approximate=True,
            )
            await self._redis.expire(key, self.ttl_s)
        except Exception as exc:  # noqa: BLE001 - a replica must not fail the turn
            log.warning("turn mirror unavailable (append %s#%d): %s", request_id, frame.seq, exc)

    async def close(self, request_id: str) -> None:
        try:
            await self._redis.hset(self._key(request_id, ":meta"), "status", "done")
        except Exception as exc:  # noqa: BLE001
            log.warning("turn mirror unavailable (close %s): %s", request_id, exc)

    async def owner(self, request_id: str) -> str | None:
        """The turn's owner, `""` for an anonymous turn, None when unknown."""
        try:
            value = await self._redis.hget(self._key(request_id, ":meta"), "owner")
        except Exception as exc:  # noqa: BLE001
            log.warning("turn mirror unavailable (owner %s): %s", request_id, exc)
            return None
        if value is None:
            return None
        return value.decode() if isinstance(value, bytes) else str(value)

    async def read(self, request_id: str, after: int, block_ms: int) -> list[TurnFrame]:
        """Frames past `after`, blocking up to `block_ms` for the first."""
        try:
            response = await self._redis.xread(
                {self._key(request_id): f"0-{after}"}, count=100, block=block_ms
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("turn mirror unavailable (read %s): %s", request_id, exc)
            return []
        frames: list[TurnFrame] = []
        for _stream, entries in response or []:
            for entry_id, fields in entries:
                raw_id = entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
                event = fields.get(b"event", fields.get("event"))
                data = fields.get(b"data", fields.get("data"))
                if isinstance(event, bytes):
                    event = event.decode()
                if isinstance(data, bytes):
                    data = data.decode()
                frames.append(
                    TurnFrame(seq=int(raw_id.split("-")[1]), event=str(event), data=json.loads(data))
                )
        return frames

    async def request_cancel(self, request_id: str) -> bool:
        try:
            await self._redis.set(self._key(request_id, ":cancel"), "1", ex=self.ttl_s)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("turn mirror unavailable (cancel %s): %s", request_id, exc)
            return False

    async def cancel_requested(self, request_id: str) -> bool:
        try:
            return bool(await self._redis.get(self._key(request_id, ":cancel")))
        except Exception as exc:  # noqa: BLE001
            log.warning("turn mirror unavailable (cancel check %s): %s", request_id, exc)
            return False


def build_turn_mirror() -> RedisTurnMirror | None:
    """The mirror when `REDIS_URL` is set, otherwise None — and says which."""
    url = settings.redis_url()
    if not url:
        log.info("turn mirror: no REDIS_URL, a resume is served by the same worker only")
        return None
    try:
        from redis.asyncio import from_url  # noqa: PLC0415 - optional at import time
    except ImportError:
        log.warning("turn mirror: redis client not installed; resume is same-worker only")
        return None
    return RedisTurnMirror(from_url(url))


_registry = LocalTurnRegistry()
_mirror: RedisTurnMirror | None = None
_mirror_built = False


def registry() -> LocalTurnRegistry:
    return _registry


def mirror() -> RedisTurnMirror | None:
    """Built on first use, not at import — `build_turn_mirror` reads REDIS_URL."""
    global _mirror, _mirror_built  # noqa: PLW0603 - one process-lifetime object
    if not _mirror_built:
        _mirror = build_turn_mirror()
        _mirror_built = True
    return _mirror


def set_registry(value: LocalTurnRegistry | None) -> None:
    """Test seam. Production never calls this."""
    global _registry  # noqa: PLW0603
    _registry = value if value is not None else LocalTurnRegistry()


def set_mirror(value: RedisTurnMirror | None) -> None:
    """Test seam. Production never calls this."""
    global _mirror, _mirror_built  # noqa: PLW0603
    _mirror = value
    _mirror_built = True


__all__ = [
    "MAX_LOCAL_TURNS",
    "TURN_TTL_S",
    "LocalTurn",
    "LocalTurnRegistry",
    "RedisTurnMirror",
    "TurnFrame",
    "build_turn_mirror",
    "mirror",
    "registry",
    "set_mirror",
    "set_registry",
]
