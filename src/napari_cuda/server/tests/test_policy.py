from __future__ import annotations

import pytest

from napari_cuda.server.data import policy


class DummyLevel:
    def __init__(self, index: int) -> None:
        self.index = index


def make_ctx(
    current: int,
    overs: dict[int, float],
    *,
    intent: int | None = None,
    thresholds: dict[int, float] | None = None,
    hysteresis: float = 0.1,
):
    levels = [DummyLevel(i) for i in sorted(overs.keys())]
    return policy.LevelSelectionContext(
        levels=levels,
        current_level=current,
        requested_level=intent,
        level_oversampling=overs,
        thresholds=thresholds,
        hysteresis=hysteresis,
    )


def test_selects_finest_within_thresholds():
    ctx = make_ctx(current=2, overs={0: 0.9, 1: 1.8, 2: 3.2})
    assert policy.select_by_oversampling(ctx) == 0


def test_uses_coarser_when_fine_exceeds():
    ctx = make_ctx(current=2, overs={0: 1.6, 1: 2.1, 2: 2.9})
    assert policy.select_by_oversampling(ctx) == 1


def test_falls_back_to_current_when_none_fit():
    ctx = make_ctx(current=2, overs={0: 5.0, 1: 3.5, 2: 2.9})
    assert policy.select_by_oversampling(ctx) == 2


def test_hysteresis_favors_staying_on_current():
    ctx = make_ctx(current=1, overs={0: 1.19, 1: 2.55, 2: 4.0})
    assert policy.select_by_oversampling(ctx) == 1


def test_context_threshold_override_wins():
    ctx = make_ctx(current=2, overs={0: 1.17, 1: 2.6, 2: 3.0}, thresholds={1: 2.8})
    assert policy.select_by_oversampling(ctx) == 1


@pytest.mark.parametrize(
    'hysteresis,expected',
    [
        (0.1, 1),
        (0.0, 0),
    ],
)
def test_hysteresis_override_applies(hysteresis: float, expected: int):
    ctx = make_ctx(current=1, overs={0: 1.19, 1: 2.55}, hysteresis=hysteresis)
    assert policy.select_by_oversampling(ctx) == expected
