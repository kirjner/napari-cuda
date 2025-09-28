"""Shared helpers for control command sessions."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class PendingControl:
    seq: int
    value: Any
    phase: str
    timestamp: float


@dataclass
class ControlSession:
    key: str
    pending: List[PendingControl] = field(default_factory=list)
    target_value: Any = None
    confirmed_value: Any = None
    last_sent_seq: Optional[int] = None
    last_confirmed_seq: Optional[int] = None
    last_send_ts: float = 0.0
    dirty: bool = False
    interaction_id: Optional[str] = None
    awaiting_commit: bool = False
    commit_in_flight: bool = False
    last_server_seq: Optional[int] = None
    last_commit_seq: Optional[int] = None

    def mark_target(self, value: Any) -> None:
        self.target_value = value
        self.dirty = True

    def ensure_interaction_id(self) -> str:
        if self.interaction_id is None:
            self.interaction_id = uuid.uuid4().hex
        return self.interaction_id

    def push_pending(self, seq: int, value: Any, *, phase: str) -> None:
        now = time.perf_counter()
        self.pending.append(PendingControl(seq=seq, value=value, phase=phase, timestamp=now))
        self.last_sent_seq = seq
        self.dirty = False
        self.last_send_ts = now

    def pop_pending(self, seq: Optional[int]) -> Optional[PendingControl]:
        if not self.pending:
            return None
        if seq is None:
            return self.pending.pop(0)
        for idx, item in enumerate(self.pending):
            if item.seq == seq:
                return self.pending.pop(idx)
        return None

    def clear_pending(self) -> None:
        self.pending.clear()
        self.last_sent_seq = None

    def mark_confirmed(
        self,
        value: Any,
        seq: Optional[int],
        *,
        server_seq: Optional[int] = None,
    ) -> None:
        self.confirmed_value = value
        if seq is not None:
            self.last_confirmed_seq = seq
        self.dirty = False
        if server_seq is not None:
            self.last_server_seq = server_seq

    def reset_interaction(self) -> None:
        self.interaction_id = None
        self.awaiting_commit = False
        self.commit_in_flight = False
        self.last_commit_seq = None


__all__ = ["ControlSession", "PendingControl"]
