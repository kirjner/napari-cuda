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
) -> policy.LevelSelectionContext:
    return policy.LevelSelectionContext(
        levels=_make_levels(len(overs) or 1),
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


def test_policy_registry_round_trip() -> None:
    names = list(policy.available_policies())
    assert "budget" in names

    overs = {0: 3.0, 1: 1.4, 2: 0.7}
    ctx = _ctx(intent=None, current=0, overs=overs)

    screen = policy.resolve_policy("screen_pixel")
    assert screen(ctx) == 1

    coarse = policy.resolve_policy("coarse_first")
    assert coarse(ctx) == 1

    metrics = policy.LevelMetricsWindow()
    metrics.observe_time(0, 25.0)
    metrics.observe_time(1, 12.0)
    metrics.observe_time(2, 8.0)
    lat_ctx = _ctx(intent=None, current=0, overs=overs, metrics=metrics)
    latency = policy.resolve_policy("latency")
    assert latency(lat_ctx) == 1

    budget = policy.resolve_policy("budget")
    assert budget(_ctx(intent=5, current=0, overs=overs)) == 5
