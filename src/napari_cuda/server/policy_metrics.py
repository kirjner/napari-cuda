"""Policy decision metrics helpers for the EGL worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional
import time


@dataclass
class PolicyMetrics:
    """Track and snapshot multiscale policy decisions."""

    _sequence: int = 0
    _last_decision: Dict[str, object] = field(default_factory=dict)

    def reset(self) -> None:
        self._sequence = 0
        self._last_decision.clear()

    def record(
        self,
        *,
        policy: str,
        intent_level: Optional[int],
        selected_level: Optional[int],
        desired_level: int,
        applied_level: int,
        reason: str,
        idle_ms: float,
        oversampling: Mapping[int, float] | None,
        downgraded: bool,
        from_level: Optional[int] = None,
    ) -> None:
        overs = {int(k): float(v) for k, v in (oversampling or {}).items()}
        self._sequence += 1
        self._last_decision = {
            "timestamp_ms": time.time() * 1000.0,
            "seq": int(self._sequence),
            "policy": str(policy),
            "intent_level": int(intent_level) if intent_level is not None else None,
            "selected_level": int(selected_level) if selected_level is not None else None,
            "desired_level": int(desired_level),
            "applied_level": int(applied_level),
            "from_level": int(from_level) if from_level is not None else None,
            "reason": str(reason),
            "idle_ms": float(idle_ms),
            "oversampling": overs,
            "downgraded": bool(downgraded),
        }

    def snapshot(
        self,
        *,
        policy: str,
        active_level: int,
        downgraded: bool,
    ) -> Dict[str, object]:
        return {
            "last_decision": dict(self._last_decision),
            "policy": str(policy),
            "active_level": int(active_level),
            "level_downgraded": bool(downgraded),
        }


__all__ = ["PolicyMetrics"]
