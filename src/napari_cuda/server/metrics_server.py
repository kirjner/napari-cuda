"""Server metrics helpers (dashboard + policy gauges)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

from .metrics_core import Metrics
from .server_scene import ServerSceneData


def start_metrics_dashboard(
    host: str,
    port: int,
    metrics: Metrics,
    refresh_ms: int,
):
    """Start the Dash metrics dashboard and return the runner handle."""

    from .dash_dashboard import start_dash_dashboard  # type: ignore

    return start_dash_dashboard(host, int(port), metrics, int(refresh_ms))


def stop_metrics_dashboard(_runner) -> None:
    """Stop the metrics dashboard (no-op for current daemon thread)."""

    return None


def update_policy_metrics(
    scene: ServerSceneData,
    metrics: Metrics,
    snapshot: Mapping[str, object],
    *,
    dump_path: Optional[Path] = Path("tmp/policy_metrics_latest.json"),
) -> None:
    """Update server-side policy metrics and gauges from a worker snapshot."""

    scene.policy_metrics_snapshot = dict(snapshot)

    scene.multiscale_state["prime_complete"] = bool(snapshot.get("prime_complete"))
    if "active_level" in snapshot:
        scene.multiscale_state["current_level"] = int(snapshot["active_level"])  # type: ignore[index]
    scene.multiscale_state["downgraded"] = bool(snapshot.get("level_downgraded"))

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(snapshot, indent=2))

    levels = snapshot.get("levels")
    if isinstance(levels, Mapping):
        for level, stats in levels.items():
            if not isinstance(stats, Mapping):
                raise TypeError("policy metrics levels entries must be mappings")
            lvl = int(level)
            prefix = f"napari_cuda_policy_level_{lvl}"
            metrics.set(f"{prefix}_mean_time_ms", float(stats.get("mean_time_ms", 0.0)))
            metrics.set(f"{prefix}_last_time_ms", float(stats.get("last_time_ms", 0.0)))
            metrics.set(f"{prefix}_oversampling", float(stats.get("latest_oversampling", 0.0)))
            metrics.set(f"{prefix}_mean_bytes", float(stats.get("mean_bytes", 0.0)))
            metrics.set(f"{prefix}_samples", float(stats.get("samples", 0.0)))

    metrics.set(
        "napari_cuda_ms_prime_complete",
        1.0 if scene.multiscale_state.get("prime_complete") else 0.0,
    )
    metrics.set(
        "napari_cuda_ms_active_level",
        float(scene.multiscale_state.get("current_level", 0)),
    )

    decision = snapshot.get("last_decision")
    if isinstance(decision, Mapping) and decision:
        metrics.set("napari_cuda_policy_intent_level", float(decision.get("intent_level", -1.0)))
        metrics.set("napari_cuda_policy_desired_level", float(decision.get("desired_level", -1.0)))
        metrics.set("napari_cuda_policy_applied_level", float(decision.get("applied_level", -1.0)))
        metrics.set("napari_cuda_policy_idle_ms", float(decision.get("idle_ms", 0.0)))
        metrics.set("napari_cuda_policy_downgraded", 1.0 if decision.get("downgraded") else 0.0)
