"""Control-channel command helpers and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class CommandResult:
    """Result payload returned from command handlers."""

    result: Any | None = None
    idempotency_key: str | None = None


class CommandRejected(RuntimeError):
    """Raised by command handlers to signal transport-friendly errors."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details) if details else None
        self.idempotency_key = idempotency_key


__all__ = ["CommandRejected", "CommandResult"]
