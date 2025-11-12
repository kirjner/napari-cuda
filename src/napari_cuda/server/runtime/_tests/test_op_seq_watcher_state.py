from __future__ import annotations

from napari_cuda.server.runtime.worker.lifecycle import _OpSeqWatcherState
from napari_cuda.server.utils.signatures import SignatureToken


def _tokens(value: int = 0) -> dict[str, SignatureToken]:
    return {
        "view": SignatureToken(("view", value)),
        "axes": SignatureToken(("axes", value)),
        "index": SignatureToken(("index", value)),
        "lod": SignatureToken(("lod", value)),
        "camera": SignatureToken(("camera", value)),
        "layers": SignatureToken(("layers", value)),
    }


def test_op_seq_watcher_state_tracks_changes() -> None:
    watcher = _OpSeqWatcherState()
    first = watcher.observe(1, **_tokens(1))
    assert first

    # No change when op_seq and tokens are identical.
    assert watcher.observe(1, **_tokens(1)) is False

    # Token change triggers an update even if op_seq stays constant.
    updated = _tokens(1)
    updated["camera"] = SignatureToken(("camera", 2))
    assert watcher.observe(1, **updated) is True

    # Bumping op_seq also reports a change and updates cached seq.
    assert watcher.observe(2, **updated) is True
