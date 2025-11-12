"""Mailbox helpers shared between the runtime controller and worker."""

from .worker_intent import WorkerIntentMailbox

__all__ = [
    "WorkerIntentMailbox",
]
