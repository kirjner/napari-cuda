"""Resumable history storage for control-channel notify lanes."""

from __future__ import annotations

import time
import uuid
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from napari_cuda.protocol import FeatureResumeState


@dataclass(frozen=True)
class EnvelopeSnapshot:
    """Persistent description of a resumable notify envelope."""

    seq: int
    delta_token: str
    frame_id: str
    timestamp: float
    payload: Mapping[str, Any]
    intent_id: Optional[str] = None


@dataclass
class ResumableRetention:
    """Retention policy for a resumable topic."""

    min_deltas: int = 0
    max_deltas: Optional[int] = None
    max_age_s: Optional[float] = None


@dataclass
class ResumableTopicHistory:
    """In-memory history for a resumable topic."""

    topic: str
    retention: ResumableRetention
    snapshot: Optional[EnvelopeSnapshot] = None
    deltas: deque[EnvelopeSnapshot] = field(default_factory=deque)

    def latest_cursor(self) -> Optional[FeatureResumeState]:
        if self.deltas:
            entry = self.deltas[-1]
            return FeatureResumeState(seq=entry.seq, delta_token=entry.delta_token)
        if self.snapshot is not None:
            snap = self.snapshot
            return FeatureResumeState(seq=snap.seq, delta_token=snap.delta_token)
        return None


class ResumeDecision(str, Enum):
    REPLAY = "replay"
    RESET = "reset"
    REJECT = "reject"


@dataclass(frozen=True)
class ResumePlan:
    """Outcome of validating a client-provided resume token."""

    topic: str
    decision: ResumeDecision
    deltas: list[EnvelopeSnapshot]


class ResumableHistoryStore:
    """Store and replay resumable notify payloads across sessions."""

    def __init__(
        self,
        retention: Mapping[str, ResumableRetention],
    ) -> None:
        self._retention: dict[str, ResumableRetention] = dict(retention)
        self._topics: dict[str, ResumableTopicHistory] = {
            name: ResumableTopicHistory(topic=name, retention=policy)
            for name, policy in self._retention.items()
        }
        self._seq_state: dict[str, int] = {name: -1 for name in retention}

    # ------------------------------------------------------------------
    # Snapshot / delta recording
    # ------------------------------------------------------------------

    def snapshot_envelope(
        self,
        topic: str,
        *,
        payload: Mapping[str, Any],
        timestamp: Optional[float] = None,
        frame_id: Optional[str] = None,
        intent_id: Optional[str] = None,
    ) -> EnvelopeSnapshot:
        history = self._topic(topic)
        ts = float(timestamp) if timestamp is not None else time.time()
        token = self._new_token()
        frame = EnvelopeSnapshot(
            seq=0,
            delta_token=token,
            frame_id=frame_id or self._new_frame_id(topic),
            timestamp=ts,
            payload=dict(payload),
            intent_id=intent_id,
        )
        history.snapshot = frame
        history.deltas.clear()
        self._seq_state[topic] = 0
        return frame

    def delta_envelope(
        self,
        topic: str,
        *,
        payload: Mapping[str, Any],
        timestamp: Optional[float] = None,
        frame_id: Optional[str] = None,
        intent_id: Optional[str] = None,
    ) -> EnvelopeSnapshot:
        history = self._topic(topic)
        if history.snapshot is None:
            # Treat the first emission as the snapshot baseline so resumable
            # state exists even if we have not sent an explicit snapshot yet.
            return self.snapshot_envelope(
                topic,
                payload=payload,
                timestamp=timestamp,
                frame_id=frame_id,
                intent_id=intent_id,
            )
        seq = self._seq_state[topic] + 1
        self._seq_state[topic] = seq
        ts = float(timestamp) if timestamp is not None else time.time()
        entry = EnvelopeSnapshot(
            seq=seq,
            delta_token=self._new_token(),
            frame_id=frame_id or self._new_frame_id(topic),
            timestamp=ts,
            payload=dict(payload),
            intent_id=intent_id,
        )
        history.deltas.append(entry)
        self._prune(history)
        return entry

    # ------------------------------------------------------------------
    # Resume handling
    # ------------------------------------------------------------------

    def latest_resume_state(self, topic: str) -> Optional[FeatureResumeState]:
        history = self._topic(topic)
        return history.latest_cursor()

    def plan_resume(self, topic: str, token: Optional[str]) -> ResumePlan:
        history = self._topic(topic)
        if token is not None and not isinstance(token, str):
            return ResumePlan(topic=topic, decision=ResumeDecision.REJECT, deltas=[])

        if history.snapshot is None:
            return ResumePlan(topic=topic, decision=ResumeDecision.RESET, deltas=list(history.deltas))

        if not token:
            return ResumePlan(topic=topic, decision=ResumeDecision.RESET, deltas=list(history.deltas))

        if token == history.snapshot.delta_token:
            return ResumePlan(topic=topic, decision=ResumeDecision.REPLAY, deltas=list(history.deltas))

        for idx, entry in enumerate(history.deltas):
            if entry.delta_token == token:
                if idx + 1 >= len(history.deltas):
                    return ResumePlan(topic=topic, decision=ResumeDecision.REPLAY, deltas=[])
                return ResumePlan(
                    topic=topic,
                    decision=ResumeDecision.REPLAY,
                    deltas=list(history.deltas)[idx + 1 :],
                )

        return ResumePlan(topic=topic, decision=ResumeDecision.RESET, deltas=list(history.deltas))

    def apply_replay(self, topic: str, cursors: Iterable[EnvelopeSnapshot]) -> None:
        history = self._topic(topic)
        history.deltas.extend(cursors)
        self._prune(history)

    def current_snapshot(self, topic: str) -> Optional[EnvelopeSnapshot]:
        return self._topic(topic).snapshot

    def all_deltas(self, topic: str) -> list[EnvelopeSnapshot]:
        return list(self._topic(topic).deltas)

    def reset_epoch(
        self,
        topic: str,
        *,
        timestamp: Optional[float] = None,
        payload: Mapping[str, Any] | None = None,
    ) -> EnvelopeSnapshot:
        history = self._topic(topic)
        ts = float(timestamp) if timestamp is not None else time.time()
        token = self._new_token()
        frame = EnvelopeSnapshot(
            seq=0,
            delta_token=token,
            frame_id=self._new_frame_id(topic),
            timestamp=ts,
            payload=dict(payload) if payload is not None else {},
        )
        history.snapshot = frame
        history.deltas.clear()
        self._seq_state[topic] = -1
        return frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _topic(self, topic: str) -> ResumableTopicHistory:
        if topic not in self._topics:
            policy = self._retention.get(topic, ResumableRetention())
            self._topics[topic] = ResumableTopicHistory(topic=topic, retention=policy)
            self._seq_state[topic] = -1
        return self._topics[topic]

    @staticmethod
    def _new_frame_id(topic: str) -> str:
        return f"{topic.replace('.', '-')}-{uuid.uuid4().hex}"

    @staticmethod
    def _new_token() -> str:
        return uuid.uuid4().hex

    def _prune(self, history: ResumableTopicHistory) -> None:
        policy = history.retention
        if not history.deltas:
            return
        now = time.time()
        if policy.max_age_s is not None:
            cutoff = now - policy.max_age_s
            while (
                history.deltas
                and history.deltas[0].timestamp < cutoff
                and len(history.deltas) > policy.min_deltas
            ):
                history.deltas.popleft()
        if policy.max_deltas is not None and policy.max_deltas >= 0:
            while len(history.deltas) > policy.max_deltas:
                history.deltas.popleft()
