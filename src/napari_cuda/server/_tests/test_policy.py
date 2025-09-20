from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from napari_cuda.server import policy


class DummyLevel:
    def __init__(self, index: int) -> None:
        self.index = index


@contextmanager
def override_env(var: str, value: str | None):
    original = os.environ.get(var)
    if value is not None:
        os.environ[var] = value
    elif var in os.environ:
        del os.environ[var]
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = original


def make_ctx(current: int, overs: dict[int, float], *, intent: int | None = None, thresholds: dict[int, float] | None = None):
    levels = [DummyLevel(i) for i in sorted(overs.keys())]
    return policy.LevelSelectionContext(
        levels=levels,
        current_level=current,
        intent_level=intent,
        level_oversampling=overs,
        thresholds=thresholds,
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


def test_env_override():
    ctx = make_ctx(current=2, overs={0: 0.9, 1: 1.8, 2: 3.2})
    with override_env('NAPARI_CUDA_LEVEL_THRESHOLDS', '0:0.8,1:1.7'):
        assert policy.select_by_oversampling(ctx) == 2


def test_context_threshold_override_wins():
    ctx = make_ctx(current=2, overs={0: 1.17, 1: 2.6, 2: 3.0}, thresholds={1: 2.8})
    assert policy.select_by_oversampling(ctx) == 1


@pytest.mark.parametrize(
    'overrides,expected',
    [
        ('0:1.0', 1),
        ('1:10.0', 0),
        ('invalid', 0),
    ],
)
def test_env_parsing_gracefully_handles_bad_tokens(overrides: str, expected: int):
    ctx = make_ctx(current=1, overs={0: 1.1, 1: 2.2})
    with override_env('NAPARI_CUDA_LEVEL_THRESHOLDS', overrides):
        assert policy.select_by_oversampling(ctx) == expected
