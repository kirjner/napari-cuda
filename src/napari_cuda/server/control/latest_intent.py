"""
Thread-safe latest-intent store for continuous control domains (e.g., dims).

Design
- No classes; simple module-level functions and state.
- Latest-wins by caller-provided monotonically increasing `seq` per (scope, key).
- Keys are normalized to strings to avoid accidental tuple/int mismatches.
- No try/except; internal invariants guarded by asserts.

API
- set_intent(scope, key, value, seq) -> None
- get_intent(scope, key) -> tuple[int, object] | None
- get_all(scope) -> dict[str, tuple[int, object]]
- last_seq(scope, key) -> int | None
- clear_scope(scope) -> None
- clear_all() -> None
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Dict, Tuple

# Internal store shape:
#   _store[scope][key] = (seq, value)
_store: Dict[str, Dict[str, Tuple[int, Any]]] = {}
_lock = RLock()


def _norm_key(key: Any) -> str:
    # Normalize user keys (ints/tuples/etc.) to stable string keys
    return str(key)


def set_intent(scope: str, key: Any, value: Any, seq: int) -> None:
    """Set latest intent value for (scope, key) if `seq` is newer or equal.

    Latest-wins semantics: replace only when incoming `seq` >= stored `seq`.
    """
    assert isinstance(scope, str) and scope != "", "scope must be non-empty string"
    assert isinstance(seq, int), "seq must be int"

    k = _norm_key(key)
    with _lock:
        bucket = _store.get(scope)
        if bucket is None:
            bucket = {}
            _store[scope] = bucket

        current = bucket.get(k)
        if current is None or seq >= current[0]:
            bucket[k] = (seq, value)


def get_intent(scope: str, key: Any) -> Tuple[int, Any] | None:
    """Return (seq, value) for (scope, key), or None if missing."""
    assert isinstance(scope, str) and scope != "", "scope must be non-empty string"
    k = _norm_key(key)
    with _lock:
        bucket = _store.get(scope)
        if bucket is None:
            return None
        return bucket.get(k)


def get_all(scope: str) -> Dict[str, Tuple[int, Any]]:
    """Return a shallow copy of the scope bucket mapping key -> (seq, value)."""
    assert isinstance(scope, str) and scope != "", "scope must be non-empty string"
    with _lock:
        bucket = _store.get(scope)
        if bucket is None:
            return {}
        # Shallow copy to avoid external mutation
        return dict(bucket)


def last_seq(scope: str, key: Any) -> int | None:
    """Return last sequence for (scope, key), or None if missing."""
    assert isinstance(scope, str) and scope != "", "scope must be non-empty string"
    k = _norm_key(key)
    with _lock:
        bucket = _store.get(scope)
        if bucket is None:
            return None
        item = bucket.get(k)
        if item is None:
            return None
        return item[0]


def clear_scope(scope: str) -> None:
    """Clear all intents for a scope."""
    assert isinstance(scope, str) and scope != "", "scope must be non-empty string"
    with _lock:
        if scope in _store:
            del _store[scope]


def clear_all() -> None:
    """Clear the entire store."""
    with _lock:
        _store.clear()


__all__ = [
    "set_intent",
    "get_intent",
    "get_all",
    "last_seq",
    "clear_scope",
    "clear_all",
]
