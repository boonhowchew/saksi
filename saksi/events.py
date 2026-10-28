# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Append-only, hash-chained event log.

Every action in a Saksi project — by a human author, a reviewer or an AI
agent — is one immutable event. Each event carries the hash of the event
before it, so any later edit, deletion or reordering breaks the chain and
is detected by `verify()`.

This one log is the source for three things the OSAV white paper treats
separately: the Methods & Analysis Log, the Contribution Ledger and the
review (CVE) record. Project state is never stored; it is rebuilt by
replaying the log (see workflow.py).
"""

from __future__ import annotations

import fcntl
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .canonical import GENESIS_HASH, canonical_json, sha256_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Event:
    seq: int
    ts: str
    project_id: str
    actor_id: str
    type: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str = field(default="")

    def body(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("hash")
        return d

    def compute_hash(self) -> str:
        return sha256_text(canonical_json(self.body()))

    def to_json(self) -> str:
        return canonical_json(asdict(self))


@dataclass
class VerifyResult:
    ok: bool
    events: int
    head: str
    error: str | None = None
    bad_seq: int | None = None


class EventLog:
    """One JSON-lines file per project. Single writer, guarded by a file lock."""

    def __init__(self, path: str | Path, project_id: str, clock: Callable[[], str] = utc_now):
        self.path = Path(path)
        self.project_id = project_id
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    # -- reading -----------------------------------------------------------
    def __iter__(self) -> Iterator[Event]:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield Event(**json.loads(line))

    def events(self) -> list[Event]:
        return list(self)

    def head(self) -> tuple[int, str]:
        last = None
        for last in self:
            pass
        return (last.seq, last.hash) if last else (0, GENESIS_HASH)

    # -- writing -----------------------------------------------------------
    def append(self, actor_id: str, type: str, payload: dict[str, Any]) -> Event:
        # Round-trip through JSON so the stored payload is exactly what is hashed.
        payload = json.loads(canonical_json(payload))
        with open(self.path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                seq, prev = self.head()
                ev = Event(
                    seq=seq + 1,
                    ts=self.clock(),
                    project_id=self.project_id,
                    actor_id=actor_id,
                    type=type,
                    payload=payload,
                    prev_hash=prev,
                )
                ev = Event(**{**asdict(ev), "hash": ev.compute_hash()})
                f.write(ev.to_json() + "\n")
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return ev

    # -- verification ------------------------------------------------------
    def verify(self) -> VerifyResult:
        prev, n = GENESIS_HASH, 0
        try:
            for n, ev in enumerate(self, start=1):
                if ev.seq != n:
                    return VerifyResult(False, n, prev, f"sequence gap: expected {n}, found {ev.seq}", ev.seq)
                if ev.project_id != self.project_id:
                    return VerifyResult(False, n, prev, "event belongs to another project", ev.seq)
                if ev.prev_hash != prev:
                    return VerifyResult(False, n, prev, "chain broken: prev_hash does not match", ev.seq)
                if ev.compute_hash() != ev.hash:
                    return VerifyResult(False, n, prev, "event content altered after it was written", ev.seq)
                prev = ev.hash
        except (json.JSONDecodeError, TypeError) as e:
            return VerifyResult(False, n, prev, f"unreadable event: {e}", n + 1)
        return VerifyResult(True, n, prev)
