from __future__ import annotations

from napari_cuda.server import policy
from napari_cuda.server.zarr_source import LevelDescriptor


def _make_levels(count: int) -> list[LevelDescriptor]:
    levels: list[LevelDescriptor] = []
    for idx in range(count):
        shape = (100, 200 // max(1, 2**idx), 100 // max(1, 2**idx))
        down = tuple(float(2**idx) for _ in shape)
        scale = tuple(float(2**idx) for _ in shape)
        levels.append(
            LevelDescriptor(
                index=idx,
                path=f"level_{idx:02d}",
                shape=shape,
                downsample=down,
                scale=scale,
            )
        )
    return levels


def _ctx(
    *,
    intent: int | None,
    current: int,
    overs: dict[int, float],
    metrics: policy.LevelMetricsWindow | None = None,
    primed: dict[int, policy.PrimedLevelMetrics] | None = None,
) -> policy.LevelSelectionContext:
    level_count = max(1, len(overs))
    return policy.LevelSelectionContext(
        levels=_make_levels(level_count),
        viewport_px=(1920, 1080),
        physical_scale=(1.0, 1.0, 1.0),
        oversampling=overs.get(current, 1.0),
        current_level=current,
        idle_ms=0.0,
        use_volume=False,
        ms_state={},
        metrics=metrics or policy.LevelMetricsWindow(),
        intent_level=intent,
        level_oversampling=overs,
        primed_metrics=primed,
    )


def test_level_metrics_window_mean_and_latest() -> None:
    window = policy.LevelMetricsWindow(window=3)
    window.observe_time(0, 10.0)
    window.observe_time(0, 20.0)
    window.observe_time(0, 30.0)
    window.observe_time(0, 40.0)  # pushes out the first entry
    assert window.mean_time_ms(0) == 30.0

    window.observe_bytes(0, 100)
    window.observe_bytes(0, 150)
    assert window.mean_bytes(0) == 125.0

    window.observe_oversampling(0, 1.5)
    window.observe_oversampling(0, 2.0)
    assert window.latest_oversampling(0) == 2.0


def test_level_metrics_snapshot() -> None:
    window = policy.LevelMetricsWindow(window=4)
    window.observe_time(1, 12.0)
    window.observe_chunks(1, 4)
    window.observe_bytes(1, 256)
    window.observe_oversampling(1, 1.4)
    snap = window.snapshot()
    assert 1 in snap
    stats = snap[1]
    assert stats['mean_time_ms'] == 12.0
    assert stats['mean_chunks'] == 4.0
    assert stats['latest_oversampling'] == 1.4
    # stats_for_level reuses the same datapoints
    level_stats = window.stats_for_level(1)
    assert level_stats['last_bytes'] == 256.0


def test_latency_policy_prefers_measured_level_within_budget() -> None:
    overs = {0: 3.0, 1: 1.2, 2: 0.6}
    metrics = policy.LevelMetricsWindow()
    metrics.observe_time(1, 12.0)
    metrics.observe_bytes(1, 200_000)
    ctx = _ctx(intent=None, current=0, overs=overs, metrics=metrics)
    assert policy.select_latency_aware(ctx) == 1


def test_latency_policy_uses_prediction_when_unmeasured() -> None:
    overs = {0: 3.0, 1: 1.1, 2: 0.6}
    metrics = policy.LevelMetricsWindow()
    metrics.observe_time(0, 30.0)
    metrics.observe_bytes(0, 1_000_000)
    metrics.observe_bytes(1, 200_000)
    metrics.observe_bytes(2, 100_000)
    ctx = _ctx(intent=None, current=0, overs=overs, metrics=metrics)
    # Predicted time for level 1 should be 6 ms (within 18 budget)
    assert policy.select_latency_aware(ctx) == 1


def test_latency_policy_falls_back_to_intent_when_no_data() -> None:
    overs = {0: 2.5, 1: 1.4}
    ctx = _ctx(intent=1, current=0, overs=overs)
    assert policy.select_latency_aware(ctx) == 1


def test_latency_policy_uses_primed_latency_when_no_live_samples() -> None:
    overs = {0: 3.0, 1: 1.1, 2: 0.6}
    primed = {
        1: policy.PrimedLevelMetrics(
            latency_ms=12.0,
            bytes_est=200_000,
            chunks=128,
            oversampling=overs[1],
            viewport_px=(1920, 1080),
            timestamp_ms=0.0,
        )
    }
    ctx = _ctx(intent=None, current=0, overs=overs, primed=primed)
    assert policy.select_latency_aware(ctx) == 1


def test_latency_policy_uses_primed_bytes_for_prediction() -> None:
    overs = {0: 3.0, 1: 1.2, 2: 0.7}
    metrics = policy.LevelMetricsWindow()
    metrics.observe_time(0, 30.0)
    metrics.observe_bytes(0, 1_000_000)
    primed = {
        1: policy.PrimedLevelMetrics(
            latency_ms=0.0,
            bytes_est=200_000,
            chunks=64,
            oversampling=overs[1],
            viewport_px=(1920, 1080),
            timestamp_ms=0.0,
        ),
        2: policy.PrimedLevelMetrics(
            latency_ms=0.0,
            bytes_est=600_000,
            chunks=32,
            oversampling=overs[2],
            viewport_px=(1920, 1080),
            timestamp_ms=0.0,
        ),
    }
    ctx = _ctx(intent=None, current=0, overs=overs, metrics=metrics, primed=primed)
    assert policy.select_latency_aware(ctx) == 1
