"""Drive multiscale policy intents against a running napari-cuda server."""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from napari_cuda.client.streaming.controllers import StateController


logger = logging.getLogger(__name__)


def _await_connection(channel, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if getattr(channel, "_out_q", None) is not None:
            logger.info("state channel ready")
            return
        time.sleep(0.05)
    raise RuntimeError("state channel did not become ready")


def _fetch_metrics(url: str, timeout: float = 5.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # type: ignore[arg-type]
            return json.load(resp)
    except Exception as exc:  # pragma: no cover - network failure
        return {"error": f"metrics fetch failed: {exc}"}


def run_harness(
    host: str,
    state_port: int,
    metrics_url: str,
    policies: List[str],
    levels: List[int],
    zoom_factors: List[float],
    output_prefix: Path,
    idle_delay: float = 0.25,
    level_delay: float = 0.0,
) -> None:
    policies = [str(p).lower() for p in policies if str(p).lower() == "latency"]
    if not policies:
        policies = ["latency"]

    scenes: List[dict] = []
    dims_updates: List[dict] = []

    def on_scene(spec) -> None:
        try:
            scenes.append(spec.to_dict())
        except Exception:
            scenes.append({'error': 'scene.to_dict failed'})

    def on_dims(meta: dict) -> None:
        dims_updates.append(meta)
        logger.info("received dims.update seq=%s", meta.get('seq'))

    ctrl = StateController(host, state_port, on_scene_spec=on_scene, on_dims_update=on_dims)
    channel, thread = ctrl.start()
    _await_connection(channel)

    seq_counter = 1
    results: List[Dict[str, object]] = []
    last_active_level: Optional[int] = None

    def maybe_log_level_change(tag: str) -> None:
        nonlocal last_active_level
        metrics = _fetch_metrics(metrics_url)
        if not isinstance(metrics, dict):
            return
        gauges = metrics.get('gauges')
        level_val: Optional[float] = None
        if isinstance(gauges, dict):
            lvl = gauges.get('napari_cuda_policy_applied_level')
            if lvl is None:
                lvl = gauges.get('napari_cuda_ms_active_level')
            if isinstance(lvl, (int, float)):
                level_val = float(lvl)
        policy_metrics = metrics.get('policy_metrics')
        reason = None
        applied_stats = None
        if level_val is None and isinstance(policy_metrics, dict):
            lvl = policy_metrics.get('active_level')
            if isinstance(lvl, (int, float)):
                level_val = float(lvl)
        if isinstance(policy_metrics, dict):
            last_decision = policy_metrics.get('last_decision')
            if isinstance(last_decision, dict):
                reason = last_decision.get('reason')
                applied_stats = last_decision
        if level_val is None:
            return
        current = int(level_val)
        if last_active_level is None:
            last_active_level = current
            return
        if current != last_active_level:
            extra = ""
            if applied_stats:
                intent_level = applied_stats.get('intent_level')
                selected_level = applied_stats.get('selected_level')
                desired_level = applied_stats.get('desired_level')
                reason_text = reason or applied_stats.get('reason')
                extra = (
                    f" intent={int(intent_level)}" if isinstance(intent_level, (int, float)) else ""
                )
                if isinstance(selected_level, (int, float)):
                    extra += f" selected={int(selected_level)}"
                if isinstance(desired_level, (int, float)):
                    extra += f" desired={int(desired_level)}"
                if reason_text:
                    extra += f" reason={reason_text}"
            print(f"[switch] {tag}: level {last_active_level} -> {current}{extra}")
            last_active_level = current

    def send(payload: Dict[str, object]) -> None:
        nonlocal seq_counter
        full = dict(payload)
        full.update(
            {
                "client_id": "policy-harness",
                "client_seq": seq_counter,
                "origin": "harness",
            }
        )
        ok = channel.send_json(full)
        results.append(
            {
                "seq": seq_counter,
                "type": full.get("type"),
                "ok": bool(ok),
                "payload": full,
            }
        )
        print(f"[{seq_counter:03d}] {full['type']} -> {ok}")
        seq_counter += 1
        time.sleep(0.12)
        t = full.get('type')
        if t in {'camera.zoom_at', 'multiscale.intent.set_level'}:
            tag = f"{t}({full.get('level', full.get('factor'))})"
            maybe_log_level_change(tag)

    for policy_name in policies:
        send({"type": "multiscale.intent.set_policy", "policy": policy_name})
        for level in levels:
            send({"type": "multiscale.intent.set_level", "level": int(level)})
            if level_delay > 0:
                time.sleep(level_delay)
        for idx, factor in enumerate(zoom_factors):
            anchor = [960.0, 540.0] if (idx % 2 == 0) else [540.0, 300.0]
            send({"type": "camera.zoom_at", "factor": float(factor), "anchor_px": anchor})
            if idx in (2, len(zoom_factors) - 3):  # deep zoom in and beginning of zoom out
                send({"type": "multiscale.intent.set_level", "level": 1})
            if idle_delay > 0:
                time.sleep(idle_delay)
        send({"type": "dims.intent.set_index", "axis": 0, "value": 0})
        send({"type": "dims.intent.set_index", "axis": 0, "value": 10})
        send({"type": "view.intent.set_ndisplay", "ndisplay": 3})
        send({"type": "view.intent.set_ndisplay", "ndisplay": 2})

    print("Waiting for server to process intents...")
    time.sleep(3.0)

    metrics = _fetch_metrics(metrics_url)

    prefix = output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    intent_path = prefix.parent / f"{prefix.name}_intents.json"
    metrics_path = prefix.parent / f"{prefix.name}_metrics.json"
    scenes_path = prefix.parent / f"{prefix.name}_scenes.json"
    dims_path = prefix.parent / f"{prefix.name}_dims.json"
    intent_path.write_text(json.dumps(results, indent=2))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    if scenes:
        scenes_path.write_text(json.dumps(scenes, indent=2))
    if dims_updates:
        dims_path.write_text(json.dumps(dims_updates, indent=2))
    print(f"Wrote {intent_path} and {metrics_path}")

    try:
        ctrl.stop(channel, thread, timeout=2.0)
    except Exception:
        logger.debug("policy harness: failed to stop state controller", exc_info=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Exercise napari-cuda multiscale policies via state intents")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (state channel)")
    parser.add_argument("--state-port", type=int, default=8081, help="State channel port")
    parser.add_argument(
        "--metrics-url",
        default="http://127.0.0.1:8083/metrics.json",
        help="Metrics endpoint (default http://127.0.0.1:8083/metrics.json)",
    )
    parser.add_argument(
        "--policies",
        nargs="*",
        default=["latency"],
        help="Policies to cycle through (latency only)",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=[0, 1, 2, 3],
        help="Level indices to request per policy",
    )
    parser.add_argument(
        "--zoom-factors",
        nargs="*",
        type=float,
        default=[0.6, 0.9, 1.2, 1.8, 2.5, 4.0, 6.0, 9.0],
        help="Zoom factors to sweep for each policy",
    )
    parser.add_argument(
        "--output-prefix",
        default="tmp/policy_harness",
        help="File prefix for JSON outputs",
    )
    parser.add_argument(
        "--idle-delay",
        type=float,
        default=0.25,
        help="Sleep (seconds) between successive zoom intents",
    )
    parser.add_argument(
        "--level-delay",
        type=float,
        default=0.0,
        help="Additional sleep (seconds) after each multiscale level intent",
    )

    args = parser.parse_args()
    run_harness(
        host=args.host,
        state_port=args.state_port,
        metrics_url=args.metrics_url,
        policies=list(args.policies),
        levels=list(args.levels),
        zoom_factors=list(args.zoom_factors),
        output_prefix=Path(args.output_prefix),
        idle_delay=float(args.idle_delay),
        level_delay=float(args.level_delay),
    )


if __name__ == "__main__":
    main()
